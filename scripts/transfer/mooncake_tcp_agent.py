from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Set
import argparse
import json
import math
import os
import signal
import shutil
import socket
import socketserver
import time
import traceback


@dataclass
class AgentConfig:
    mode: str
    metadata_server: str
    protocol: str
    local_server_name: str
    peer_server_name: str
    device_name: str
    chunk_bytes: int
    max_inflight: int
    source_dir: str
    destination_dir: str
    destination_tmp_dir: str
    control_host: str
    control_port: int
    timeout_seconds: int
    session_id: str


class MooncakeRuntimeError(RuntimeError):
    pass


MANIFEST_FILE_NAME = "checkpoint_manifest.json"


def checksum_algorithm() -> str:
    return "sha256"


def backend_name(protocol: str) -> str:
    label = protocol.strip().lower()
    return f"mooncake_{label}" if label else "mooncake_unknown"


def parse_args() -> AgentConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["receiver", "sender"], required=True)
    parser.add_argument("--metadata-server", required=True)
    parser.add_argument("--protocol", default="tcp")
    parser.add_argument("--local-server-name", required=True)
    parser.add_argument("--peer-server-name", required=True)
    parser.add_argument("--device-name", default="")
    parser.add_argument("--chunk-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-inflight", type=int, default=2)
    parser.add_argument("--source-dir", default="")
    parser.add_argument("--destination-dir", default="")
    parser.add_argument("--destination-tmp-dir", default="")
    parser.add_argument("--control-host", required=True)
    parser.add_argument("--control-port", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--session-id", default="")
    args = parser.parse_args()
    return AgentConfig(
        args.mode,
        args.metadata_server,
        args.protocol,
        args.local_server_name,
        args.peer_server_name,
        args.device_name,
        args.chunk_bytes,
        args.max_inflight,
        args.source_dir,
        args.destination_dir,
        args.destination_tmp_dir,
        args.control_host,
        args.control_port,
        args.timeout_seconds,
        str(args.session_id),
    )


def import_transfer_engine() -> Any:
    try:
        from mooncake.engine import TransferEngine
    except ImportError as exc:
        raise MooncakeRuntimeError("mooncake.engine not available; install mooncake-transfer-engine on both hosts") from exc
    return TransferEngine


def checked_call(value: int, action: str) -> None:
    if value == 0:
        return
    raise MooncakeRuntimeError(f"{action} failed with code {value}")


def local_path(path: str) -> Path:
    return Path(path).expanduser()


def state_path(tmp_dir: Path) -> Path:
    return tmp_dir / ".mooncake_transfer_state.json"


def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"files": {}}
    return json.loads(path.read_text())


def save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def file_entry(state: Dict[str, Any], relative_path: str, file_size: int, file_sha: str) -> Dict[str, Any]:
    files = state.setdefault("files", {})
    current = files.get(relative_path, {})
    if int(current.get("size_bytes", -1)) != file_size or str(current.get("sha256", "")) != file_sha:
        current = {"size_bytes": file_size, "sha256": file_sha, "completed_chunks": []}
    files[relative_path] = current
    return current


def completed_chunks(entry: Dict[str, Any]) -> Set[int]:
    return {int(value) for value in entry.get("completed_chunks", [])}


def clear_completed_chunks(entry: Dict[str, Any]) -> None:
    entry["completed_chunks"] = []


def file_matches_entry(path: Path, file_size: int, file_sha: str) -> bool:
    return path.exists() and int(path.stat().st_size) == file_size and checksum(path) == file_sha


def checksum(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def truncate_file(path: Path, size_bytes: int) -> None:
    ensure_parent(path)
    with path.open("ab"):
        pass
    with path.open("r+b") as handle:
        handle.truncate(size_bytes)


class ReceiverState:
    def __init__(self, config: AgentConfig, engine: Any, slots: List[int], slot_bytes: int, init_record: Dict[str, Any], registered_buffers: int):
        self.config = config
        self.engine = engine
        self.slots = slots
        self.tmp_dir = local_path(config.destination_tmp_dir)
        self.final_dir = local_path(config.destination_dir)
        self.transfer_state_path = state_path(self.tmp_dir)
        self.transfer_state = load_state(self.transfer_state_path)
        self.lock = Lock()
        self.transfer_server_name = runtime_server_name(config.local_server_name, engine)
        self.slot_bytes = slot_bytes
        self.session_id = config.session_id
        self.init_record = init_record
        self.registered_buffers = registered_buffers


class ReceiverHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        line = self.rfile.readline().decode("utf-8")
        if not line.strip():
            return
        payload = json.loads(line)
        payload["_client"] = str(self.client_address[0])
        response = dispatch(payload, self.server.receiver_state)
        self.wfile.write((json.dumps(response) + "\n").encode("utf-8"))


class ReceiverServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

    def __init__(self, host_port: tuple, handler: Any, receiver_state: ReceiverState):
        self.receiver_state = receiver_state
        super().__init__(host_port, handler)


def dispatch(payload: Dict[str, Any], state: ReceiverState) -> Dict[str, Any]:
    command = payload.get("command")
    print(json.dumps({"event": "receiver_command", "command": command, "client": payload.get("_client", ""), "session_id": str(payload.get("session_id", ""))}))
    try:
        if not session_allowed(payload, state):
            return {"ok": False, "error": "session mismatch"}
        if command == "hello":
            return hello(state)
        if command == "progress":
            return progress(state, payload)
        if command == "commit":
            return commit(state, payload)
        if command == "finalize":
            return finalize(state, payload)
        if command == "shutdown":
            return shutdown(state)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": False, "error": f"unknown command: {command}"}


def hello(state: ReceiverState) -> Dict[str, Any]:
    with state.lock:
        state.tmp_dir.mkdir(parents=True, exist_ok=True)
    return {
        "ok": True,
        "slot_addresses": state.slots,
        "chunk_bytes": state.slot_bytes,
        "transfer_server_name": state.transfer_server_name,
        "session_id": state.session_id,
        "registered_destination_buffers": state.registered_buffers,
        "destination_initialize_call": state.init_record,
    }


def session_allowed(payload: Dict[str, Any], state: ReceiverState) -> bool:
    expected = str(state.session_id)
    provided = str(payload.get("session_id", ""))
    if not expected:
        return True
    if payload.get("command") == "hello" and not provided:
        return True
    return provided == expected


def runtime_server_name(configured_server_name: str, engine: Any) -> str:
    host = configured_server_name.split(":", 1)[0]
    return f"{host}:{int(engine.get_rpc_port())}"


def progress(state: ReceiverState, payload: Dict[str, Any]) -> Dict[str, Any]:
    relpath = str(payload["relative_path"])
    size_bytes = int(payload["size_bytes"])
    file_sha = str(payload["sha256"])
    with state.lock:
        entry = file_entry(state.transfer_state, relpath, size_bytes, file_sha)
        path = target_file(state, relpath)
        if completed_chunks(entry) and not file_matches_entry(path, size_bytes, file_sha):
            clear_completed_chunks(entry)
        save_state(state.transfer_state_path, state.transfer_state)
    return {"ok": True, "completed_chunks": sorted(completed_chunks(entry))}


def read_slot_bytes(state: ReceiverState, slot_id: int, length: int) -> bytes:
    address = int(state.slots[slot_id])
    return bytes(state.engine.read_bytes_from_buffer(address, int(length)))


def target_file(state: ReceiverState, relative_path: str) -> Path:
    return state.tmp_dir / relative_path


def write_chunk(path: Path, offset: int, data: bytes) -> None:
    ensure_parent(path)
    with path.open("r+b" if path.exists() else "wb") as handle:
        handle.seek(offset)
        handle.write(data)


def commit(state: ReceiverState, payload: Dict[str, Any]) -> Dict[str, Any]:
    chunks = list(payload.get("chunks", []))
    with state.lock:
        for chunk in chunks:
            relpath = str(chunk["relative_path"])
            size_bytes = int(chunk["size_bytes"])
            file_sha = str(chunk["sha256"])
            offset = int(chunk["offset"])
            length = int(chunk["length"])
            chunk_index = int(chunk["chunk_index"])
            slot_id = int(chunk["slot_id"])
            entry = file_entry(state.transfer_state, relpath, size_bytes, file_sha)
            truncate_file(target_file(state, relpath), size_bytes)
            if chunk_index in completed_chunks(entry):
                continue
            write_chunk(target_file(state, relpath), offset, read_slot_bytes(state, slot_id, length))
            entry["completed_chunks"] = sorted({*completed_chunks(entry), chunk_index})
        save_state(state.transfer_state_path, state.transfer_state)
    return {"ok": True}


def verify_manifest(state: ReceiverState, manifest: Dict[str, Any]) -> Dict[str, Any]:
    mismatches = []
    per_file = []
    files = list(manifest.get("files", []))
    for item in files:
        relpath = str(item["relative_path"])
        expected_size = int(item["size_bytes"])
        source_sha = str(item["sha256"])
        path = target_file(state, relpath)
        if not path.exists():
            mismatches.append({"relative_path": relpath, "error": "missing"})
            per_file.append({"relative_path": relpath, "bytes": expected_size, "source_sha256": source_sha, "destination_sha256": "", "verified": False})
            continue
        actual_size = int(path.stat().st_size)
        destination_sha = checksum(path)
        size_match = actual_size == expected_size
        hash_match = source_sha == destination_sha
        verified = bool(size_match and hash_match)
        per_file.append({"relative_path": relpath, "bytes": actual_size, "source_sha256": source_sha, "destination_sha256": destination_sha, "verified": verified})
        if not verified:
            mismatches.append(
                {
                    "relative_path": relpath,
                    "expected_size": expected_size,
                    "actual_size": actual_size,
                    "source_sha256": source_sha,
                    "destination_sha256": destination_sha,
                }
            )
    verified_files = sum(1 for item in per_file if bool(item["verified"]))
    return {"ok": not mismatches, "mismatches": mismatches, "per_file": per_file, "sha256_verified_files": verified_files}


def move_verified_directory(state: ReceiverState) -> None:
    if state.final_dir.exists():
        shutil.rmtree(state.final_dir)
    state.tmp_dir.rename(state.final_dir)


def finalize(state: ReceiverState, payload: Dict[str, Any]) -> Dict[str, Any]:
    manifest = dict(payload["manifest"])
    atomic_rename_completed = False
    with state.lock:
        verified = verify_manifest(state, manifest)
        if not verified["ok"]:
            return {
                "ok": False,
                "checksum_ok": False,
                "mismatches": verified["mismatches"],
                "per_file": verified["per_file"],
                "sha256_verified_files": verified["sha256_verified_files"],
                "atomic_rename_completed": atomic_rename_completed,
                "checksum_algorithm": checksum_algorithm(),
                "published_directory": str(state.final_dir),
                "temporary_directory": str(state.tmp_dir),
                "registered_destination_buffers": state.registered_buffers,
                "destination_initialize_call": state.init_record,
            }
        move_verified_directory(state)
        atomic_rename_completed = True
    return {
        "ok": True,
        "checksum_ok": True,
        "mismatches": [],
        "per_file": verified["per_file"],
        "sha256_verified_files": verified["sha256_verified_files"],
        "atomic_rename_completed": atomic_rename_completed,
        "checksum_algorithm": checksum_algorithm(),
        "published_directory": str(state.final_dir),
        "temporary_directory": str(state.tmp_dir),
        "registered_destination_buffers": state.registered_buffers,
        "destination_initialize_call": state.init_record,
    }


def shutdown(state: ReceiverState) -> Dict[str, Any]:
    state.server_should_exit = True
    return {"ok": True}


def json_rpc(host: str, port: int, payload: Dict[str, Any], timeout_seconds: int) -> Dict[str, Any]:
    with socket.create_connection((host, port), timeout=timeout_seconds) as sock:
        sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        response = sock.makefile("rb").readline().decode("utf-8")
    return json.loads(response)


def receiver_hello(config: AgentConfig) -> Dict[str, Any]:
    last_error = ""
    for _ in range(30):
        try:
            result = json_rpc(config.control_host, config.control_port, {"command": "hello", "session_id": config.session_id}, config.timeout_seconds)
            if result.get("ok"):
                return result
            last_error = str(result)
        except Exception as exc:
            last_error = str(exc)
        time.sleep(1)
    raise MooncakeRuntimeError(f"receiver hello failed at {config.control_host}:{config.control_port}: {last_error}")


def load_manifest(source_dir: Path) -> Dict[str, Any]:
    path = source_dir / "checkpoint_manifest.json"
    if path.exists():
        return normalized_manifest(json.loads(path.read_text()))
    return normalized_manifest(build_manifest(source_dir))


def normalized_manifest(payload: Dict[str, Any]) -> Dict[str, Any]:
    files = [item for item in list(payload.get("files", [])) if str(item.get("relative_path", "")) != MANIFEST_FILE_NAME]
    payload["files"] = files
    payload["file_count"] = len(files)
    payload["total_bytes"] = sum(int(item.get("size_bytes", 0)) for item in files)
    return payload


def build_manifest(source_dir: Path) -> Dict[str, Any]:
    files = []
    for file_path in sorted(path for path in source_dir.rglob("*") if path.is_file()):
        relative = str(file_path.relative_to(source_dir))
        if relative == MANIFEST_FILE_NAME:
            continue
        files.append({"relative_path": relative, "size_bytes": file_path.stat().st_size, "sha256": checksum(file_path)})
    return {"checkpoint_dir": str(source_dir), "file_count": len(files), "total_bytes": sum(int(item["size_bytes"]) for item in files), "files": files}


def chunks_for_file(size_bytes: int, chunk_bytes: int) -> int:
    return max(1, int(math.ceil(size_bytes / chunk_bytes)))


def chunk_offset(chunk_index: int, chunk_bytes: int) -> int:
    return chunk_index * chunk_bytes


def chunk_length(size_bytes: int, offset: int, chunk_bytes: int) -> int:
    return max(0, min(chunk_bytes, size_bytes - offset))


def initialize_record(config: AgentConfig) -> Dict[str, Any]:
    return {
        "function": "TransferEngine.initialize",
        "parameters": {
            "local_server_name": config.local_server_name,
            "metadata_server": config.metadata_server,
            "protocol": config.protocol,
            "device_name": config.device_name,
        },
    }


def initialize_engine(engine: Any, config: AgentConfig) -> Dict[str, Any]:
    call = initialize_record(config)
    start = time.perf_counter()
    checked_call(int(engine.initialize(config.local_server_name, config.metadata_server, config.protocol, config.device_name)), "initialize")
    call["elapsed_seconds"] = time.perf_counter() - start
    print(json.dumps({"event": "transfer_engine_initialize", "call": call}))
    return call


def transfer_manifest_path(session_id: str) -> Path:
    token = session_id if session_id else str(int(time.time() * 1000))
    return Path("/tmp") / f"mooncake_transfer_manifest_{token}.json"


def persist_transfer_manifest(payload: Dict[str, Any], session_id: str) -> str:
    path = transfer_manifest_path(session_id)
    path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"event": "transfer_manifest_saved", "path": str(path)}))
    return str(path)


def sender_slots(engine: Any, chunk_bytes: int, max_inflight: int) -> tuple[List[int], int]:
    return adaptive_slots(engine, chunk_bytes, max_inflight)


def adaptive_slots(engine: Any, requested_bytes: int, max_inflight: int) -> tuple[List[int], int]:
    size = requested_bytes
    min_size = 1024 * 1024
    while size >= min_size:
        slots = allocate_slot_batch(engine, size, max_inflight)
        if len(slots) == max_inflight:
            return slots, size
        release_slots(engine, slots, size)
        size = size // 2
    raise MooncakeRuntimeError(f"allocate_managed_buffer failed for requested chunk_bytes={requested_bytes}")


def allocate_slot_batch(engine: Any, size: int, count: int) -> List[int]:
    slots: List[int] = []
    for _ in range(count):
        address = int(engine.allocate_managed_buffer(size))
        if address == 0:
            return slots
        slots.append(address)
    return slots


def release_slots(engine: Any, slots: List[int], chunk_bytes: int) -> None:
    for address in slots:
        checked_call(int(engine.free_managed_buffer(address, chunk_bytes)), "free_managed_buffer")


def requires_memory_registration(protocol: str) -> bool:
    return protocol.strip().lower() != "tcp"


def free_sender_slots(engine: Any, slots: List[int], chunk_bytes: int, protocol: str) -> None:
    needs_registration = requires_memory_registration(protocol)
    for address in slots:
        if needs_registration:
            checked_call(int(engine.unregister_memory(address)), "unregister_memory")
        checked_call(int(engine.free_managed_buffer(address, chunk_bytes)), "free_managed_buffer")


def pending_chunks(total_chunks: int, done: Set[int]) -> List[int]:
    return [idx for idx in range(total_chunks) if idx not in done]


def chunk_batch(indices: List[int], width: int) -> List[List[int]]:
    return [indices[pos : pos + width] for pos in range(0, len(indices), width)]


def read_chunk(path: Path, offset: int, length: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(offset)
        return handle.read(length)


def chunk_bytes_for_indices(size_bytes: int, chunk_bytes: int, indices: Set[int]) -> int:
    return sum(chunk_length(size_bytes, chunk_offset(index, chunk_bytes), chunk_bytes) for index in indices)


def merge_chunk_metrics(files: List[Dict[str, Any]], transfers: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged = []
    for item in files:
        relative_path = str(item["relative_path"])
        transfer = transfers.get(relative_path)
        if transfer is None:
            raise MooncakeRuntimeError(f"missing chunk metrics for {relative_path}")
        merged.append({**item, **transfer})
    return merged


def assert_chunk_counts(files: List[Dict[str, Any]], chunk_count: int) -> None:
    per_file_total = sum(int(item.get("chunk_count", 0)) for item in files)
    if per_file_total != chunk_count:
        raise MooncakeRuntimeError(f"chunk_count mismatch: top_level={chunk_count} per_file_sum={per_file_total}")


def send_file(
    config: AgentConfig,
    engine: Any,
    target_server_name: str,
    chunk_bytes: int,
    local_slots: List[int],
    remote_slots: List[int],
    item: Dict[str, Any],
) -> Dict[str, Any]:
    relative_path = str(item["relative_path"])
    size_bytes = int(item["size_bytes"])
    digest = str(item["sha256"])
    source_file = local_path(config.source_dir) / relative_path
    response = json_rpc(config.control_host, config.control_port, {"command": "progress", "relative_path": relative_path, "size_bytes": size_bytes, "sha256": digest, "session_id": config.session_id}, config.timeout_seconds)
    if not response.get("ok"):
        raise MooncakeRuntimeError(f"progress failed for {relative_path}: {response}")
    total_chunks = chunks_for_file(size_bytes, chunk_bytes)
    done = {int(value) for value in response.get("completed_chunks", [])}
    all_chunks = pending_chunks(total_chunks, done)
    done_bytes = chunk_bytes_for_indices(size_bytes, chunk_bytes, done)
    expected_read_bytes = chunk_bytes_for_indices(size_bytes, chunk_bytes, set(all_chunks))
    payload_bytes = 0
    chunk_count = 0
    batch_count = 0
    transfer_engine_seconds = 0.0
    for group in chunk_batch(all_chunks, min(config.max_inflight, len(local_slots))):
        local_addrs: List[int] = []
        remote_addrs: List[int] = []
        lengths: List[int] = []
        descriptors: List[Dict[str, Any]] = []
        for slot_idx, chunk_idx in enumerate(group):
            offset = chunk_offset(chunk_idx, chunk_bytes)
            length = chunk_length(size_bytes, offset, chunk_bytes)
            payload = read_chunk(source_file, offset, length)
            write_start = time.perf_counter()
            checked_call(int(engine.write_bytes_to_buffer(local_slots[slot_idx], payload, length)), "write_bytes_to_buffer")
            transfer_engine_seconds += time.perf_counter() - write_start
            local_addrs.append(int(local_slots[slot_idx]))
            remote_addrs.append(int(remote_slots[slot_idx]))
            lengths.append(length)
            descriptors.append(
                {
                    "relative_path": relative_path,
                    "size_bytes": size_bytes,
                    "sha256": digest,
                    "offset": offset,
                    "length": length,
                    "chunk_index": chunk_idx,
                    "slot_id": slot_idx,
                }
            )
        batch_start = time.perf_counter()
        batch_id = int(engine.batch_transfer_async_write(target_server_name, local_addrs, remote_addrs, lengths))
        if batch_id <= 0:
            raise MooncakeRuntimeError(f"batch_transfer_async_write failed for {relative_path} with code={batch_id}")
        batch_status = int(engine.get_batch_transfer_status([batch_id]))
        transfer_engine_seconds += time.perf_counter() - batch_start
        if batch_status != 0:
            raise MooncakeRuntimeError(f"get_batch_transfer_status failed for {relative_path} with code={batch_status} batch_id={batch_id}")
        commit_result = json_rpc(config.control_host, config.control_port, {"command": "commit", "chunks": descriptors, "session_id": config.session_id}, config.timeout_seconds)
        if not commit_result.get("ok"):
            raise MooncakeRuntimeError(f"commit failed for {relative_path}: {commit_result}")
        payload_bytes += sum(lengths)
        chunk_count += len(lengths)
        batch_count += 1
    if payload_bytes != expected_read_bytes:
        raise MooncakeRuntimeError(f"chunked read bytes mismatch for {relative_path}: expected_read_bytes={expected_read_bytes} actual_read_bytes={payload_bytes}")
    if payload_bytes + done_bytes != size_bytes:
        raise MooncakeRuntimeError(f"chunk coverage mismatch for {relative_path}: size_bytes={size_bytes} covered_bytes={payload_bytes + done_bytes}")
    return {
        "relative_path": relative_path,
        "payload_bytes": payload_bytes,
        "chunk_count": chunk_count,
        "batch_count": batch_count,
        "transfer_engine_seconds": transfer_engine_seconds,
        "chunk_size_bytes": chunk_bytes,
        "bytes_read_via_chunks": payload_bytes,
        "file_size_bytes": size_bytes,
    }


def run_sender(config: AgentConfig) -> Dict[str, Any]:
    TransferEngine = import_transfer_engine()
    engine = TransferEngine()
    source_initialize_call = initialize_engine(engine, config)
    should_shutdown = False
    hello_result = receiver_hello(config)
    should_shutdown = True
    target_server_name = str(hello_result.get("transfer_server_name", config.peer_server_name))
    probe_start = time.perf_counter()
    if int(engine.send_probe(target_server_name)) != 0:
        raise MooncakeRuntimeError(f"send_probe failed for {target_server_name}")
    transfer_engine_seconds = time.perf_counter() - probe_start + float(source_initialize_call["elapsed_seconds"])
    remote_slots = [int(value) for value in hello_result.get("slot_addresses", [])]
    remote_chunk = int(hello_result.get("chunk_bytes", config.chunk_bytes))
    local_slots, local_chunk = sender_slots(engine, config.chunk_bytes, config.max_inflight)
    chunk_bytes = min(local_chunk, remote_chunk)
    needs_registration = requires_memory_registration(config.protocol)
    registered_source_buffers = 0
    if needs_registration:
        for address in local_slots:
            checked_call(int(engine.register_memory(address, local_chunk)), "register_memory")
            registered_source_buffers += 1
    operation_start = time.perf_counter()
    payload_bytes = 0
    chunk_count = 0
    batch_count = 0
    file_count = 0
    manifest_files: List[Dict[str, Any]] = []
    transfer_files: Dict[str, Dict[str, Any]] = {}
    try:
        manifest = load_manifest(local_path(config.source_dir))
        manifest_files = list(manifest.get("files", []))
        file_count = len(manifest_files)
        for item in manifest_files:
            file_result = send_file(config, engine, target_server_name, chunk_bytes, local_slots, remote_slots, item)
            payload_bytes += int(file_result["payload_bytes"])
            chunk_count += int(file_result["chunk_count"])
            batch_count += int(file_result["batch_count"])
            transfer_engine_seconds += float(file_result["transfer_engine_seconds"])
            transfer_files[str(file_result["relative_path"])] = {
                "chunk_size_bytes": int(file_result["chunk_size_bytes"]),
                "chunk_count": int(file_result["chunk_count"]),
                "bytes_read_via_chunks": int(file_result["bytes_read_via_chunks"]),
                "file_size_bytes": int(file_result["file_size_bytes"]),
            }
        verify = json_rpc(config.control_host, config.control_port, {"command": "finalize", "manifest": manifest, "session_id": config.session_id}, config.timeout_seconds)
        checksum_ok = bool(verify.get("checksum_ok", False))
        success = bool(verify.get("ok", False)) and checksum_ok
        operation_seconds = time.perf_counter() - operation_start
        receiver_init = hello_result.get("destination_initialize_call", verify.get("destination_initialize_call", {}))
        files = merge_chunk_metrics(list(verify.get("per_file", [])), transfer_files)
        assert_chunk_counts(files, chunk_count)
        payload = {
            "success": success,
            "backend": backend_name(config.protocol),
            "protocol": config.protocol,
            "source": config.local_server_name,
            "destination": target_server_name,
            "source_directory": str(local_path(config.source_dir)),
            "temporary_directory": str(local_path(config.destination_tmp_dir)),
            "published_directory": str(local_path(config.destination_dir)),
            "file_count": file_count,
            "payload_bytes": payload_bytes,
            "chunk_size_bytes": chunk_bytes,
            "chunk_count": chunk_count,
            "registered_source_buffers": registered_source_buffers,
            "registered_destination_buffers": int(verify.get("registered_destination_buffers", hello_result.get("registered_destination_buffers", 0))),
            "transfer_mode": "batch_transfer_async_write",
            "batch_count": batch_count,
            "sha256_verified_files": int(verify.get("sha256_verified_files", 0)),
            "atomic_rename_completed": bool(verify.get("atomic_rename_completed", False)),
            "transfer_engine_seconds": transfer_engine_seconds,
            "operation_seconds": operation_seconds,
            "checksum_ok": checksum_ok,
            "checksum_algorithm": checksum_algorithm(),
            "files": files,
            "initialize_calls": {"sender": source_initialize_call, "receiver": receiver_init},
            "mismatches": list(verify.get("mismatches", [])),
            "bytes_transferred": payload_bytes,
            "elapsed_seconds": operation_seconds,
            "throughput_bytes_per_second": 0.0 if operation_seconds <= 0 else payload_bytes / operation_seconds,
        }
        payload["manifest_path"] = persist_transfer_manifest(payload, config.session_id)
        return payload
    finally:
        try:
            free_sender_slots(engine, local_slots, local_chunk, config.protocol)
        finally:
            try:
                if should_shutdown:
                    json_rpc(config.control_host, config.control_port, {"command": "shutdown", "session_id": config.session_id}, config.timeout_seconds)
            except Exception:
                pass


def run_receiver(config: AgentConfig) -> None:
    TransferEngine = import_transfer_engine()
    engine = TransferEngine()
    destination_initialize_call = initialize_engine(engine, config)
    slots, slot_bytes = adaptive_slots(engine, config.chunk_bytes, config.max_inflight)
    needs_registration = requires_memory_registration(config.protocol)
    registered_destination_buffers = 0
    for address in slots:
        if needs_registration:
            checked_call(int(engine.register_memory(address, slot_bytes)), "register_memory")
            registered_destination_buffers += 1
    receiver_state = ReceiverState(config, engine, slots, slot_bytes, destination_initialize_call, registered_destination_buffers)
    install_signal_handlers(receiver_state)
    server = ReceiverServer((config.control_host, config.control_port), ReceiverHandler, receiver_state)
    server.timeout = 1
    receiver_state.server_should_exit = False
    print(json.dumps({"event": "receiver_ready", "control_host": config.control_host, "control_port": config.control_port, "transfer_server_name": receiver_state.transfer_server_name}))
    try:
        while not receiver_state.server_should_exit:
            try:
                server.handle_request()
            except Exception as exc:
                print(json.dumps({"event": "receiver_loop_exception", "error": str(exc)}))
                print(traceback.format_exc())
                raise
    finally:
        print(json.dumps({"event": "receiver_stopping", "control_host": config.control_host, "control_port": config.control_port}))
        for address in slots:
            if needs_registration:
                checked_call(int(engine.unregister_memory(address)), "unregister_memory")
            checked_call(int(engine.free_managed_buffer(address, slot_bytes)), "free_managed_buffer")


def install_signal_handlers(state: ReceiverState) -> None:
    signal.signal(signal.SIGTERM, lambda *_: stop_from_signal(state, "SIGTERM"))
    signal.signal(signal.SIGHUP, lambda *_: stop_from_signal(state, "SIGHUP"))


def stop_from_signal(state: ReceiverState, name: str) -> None:
    print(json.dumps({"event": "receiver_signal", "signal": name}))
    state.server_should_exit = True


def main() -> None:
    config = parse_args()
    if config.mode == "receiver":
        run_receiver(config)
        return
    payload = run_sender(config)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
