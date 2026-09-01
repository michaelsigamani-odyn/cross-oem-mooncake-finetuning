from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import argparse
import json


@dataclass
class ManifestConfig:
    mode: str
    checkpoint_a: str
    checkpoint_b: str
    output: str


def parse_args() -> ManifestConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["manifest", "assert"], required=True)
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    return ManifestConfig(args.mode, args.checkpoint_a, args.checkpoint_b, args.output)


def list_files(checkpoint_dir: str) -> List[Path]:
    root = Path(checkpoint_dir)
    return sorted([path for path in root.rglob("*") if path.is_file()])


def entry(root: Path, path: Path) -> Dict[str, Any]:
    return {"relative_path": str(path.relative_to(root)), "suffix": path.suffix}


def manifest(checkpoint_dir: str) -> Dict[str, Any]:
    root = Path(checkpoint_dir)
    files = [entry(root, path) for path in list_files(checkpoint_dir)]
    return {"checkpoint_dir": str(root), "file_count": len(files), "files": files}


def read_manifest(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def write_manifest(path: str, payload: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2))


def paths_set(payload: Dict[str, Any]) -> set:
    return {item["relative_path"] for item in payload.get("files", [])}


def assert_structurally_identical(manifest_a: Dict[str, Any], manifest_b: Dict[str, Any]) -> None:
    a_paths = paths_set(manifest_a)
    b_paths = paths_set(manifest_b)
    assert a_paths == b_paths, f"manifest mismatch: only_a={sorted(a_paths - b_paths)} only_b={sorted(b_paths - a_paths)}"


def assert_mode(cfg: ManifestConfig) -> None:
    left = read_manifest(cfg.checkpoint_a)
    right = read_manifest(cfg.checkpoint_b)
    assert_structurally_identical(left, right)
    write_manifest(cfg.output, {"status": "ok", "left": cfg.checkpoint_a, "right": cfg.checkpoint_b})


def manifest_mode(cfg: ManifestConfig) -> None:
    write_manifest(cfg.output, manifest(cfg.checkpoint_a))


def main() -> None:
    cfg = parse_args()
    manifest_mode(cfg) if cfg.mode == "manifest" else assert_mode(cfg)


if __name__ == "__main__":
    main()
