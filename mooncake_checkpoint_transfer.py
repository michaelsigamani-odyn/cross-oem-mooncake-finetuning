from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import argparse
import shutil
import subprocess


@dataclass
class TransferConfig:
    mode: str
    checkpoint_dir: str
    artifact_id: str
    staging_dir: str
    store_uri: Optional[str]
    put_cmd: Optional[str]
    get_cmd: Optional[str]


def parse_args() -> TransferConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["put", "get"], required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--staging-dir", default="/tmp")
    parser.add_argument("--store-uri", default=None)
    parser.add_argument("--put-cmd", default=None)
    parser.add_argument("--get-cmd", default=None)
    args = parser.parse_args()
    return TransferConfig(args.mode, args.checkpoint_dir, args.artifact_id, args.staging_dir, args.store_uri, args.put_cmd, args.get_cmd)


def archive_path(cfg: TransferConfig) -> Path:
    return Path(cfg.staging_dir) / f"{cfg.artifact_id}.tar.gz"


def create_archive(cfg: TransferConfig) -> str:
    src = Path(cfg.checkpoint_dir)
    out = archive_path(cfg)
    return shutil.make_archive(str(out).replace(".tar.gz", ""), "gztar", str(src.parent), src.name)


def extract_archive(cfg: TransferConfig) -> None:
    target = Path(cfg.checkpoint_dir)
    target.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(archive_path(cfg)), str(target))


def run_cmd(template: str, src: str, dst: str) -> None:
    command = template.replace("{src}", src).replace("{dst}", dst)
    subprocess.run(command, shell=True, check=True)


def copy_to_store(cfg: TransferConfig) -> None:
    if cfg.put_cmd:
        run_cmd(cfg.put_cmd, str(archive_path(cfg)), cfg.artifact_id)
        return
    assert cfg.store_uri, "store-uri required when put-cmd is not set"
    shutil.copy2(str(archive_path(cfg)), f"{cfg.store_uri}/{cfg.artifact_id}.tar.gz")


def copy_from_store(cfg: TransferConfig) -> None:
    if cfg.get_cmd:
        run_cmd(cfg.get_cmd, cfg.artifact_id, str(archive_path(cfg)))
        return
    assert cfg.store_uri, "store-uri required when get-cmd is not set"
    shutil.copy2(f"{cfg.store_uri}/{cfg.artifact_id}.tar.gz", str(archive_path(cfg)))


def put_checkpoint(cfg: TransferConfig) -> None:
    create_archive(cfg)
    copy_to_store(cfg)


def get_checkpoint(cfg: TransferConfig) -> None:
    copy_from_store(cfg)
    extract_archive(cfg)


def main() -> None:
    cfg = parse_args()
    put_checkpoint(cfg) if cfg.mode == "put" else get_checkpoint(cfg)


if __name__ == "__main__":
    main()
