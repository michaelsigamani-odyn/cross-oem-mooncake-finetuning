import json

from cross_oem_migration.config import load_settings


def test_load_settings_reads_run_and_machines_json():
    settings = load_settings()
    assert settings.run_id
    assert settings.source_host in {m.hostname for m in settings.machine_catalog.values()} or settings.source_host
    assert "nvidia_control" in settings.machine_catalog


def test_ssh_password_is_not_read_from_committed_config():
    settings = load_settings()
    assert settings.ssh_password in (None, "")


def test_loader_ignores_run_json_ssh_password_and_uses_env(tmp_path, monkeypatch):
    run = {
        "common": {"local_root_dir": "/tmp/local", "remote_root_dir": "~/tmp/remote"},
        "finetuning_sequential": {
            "source_host": "michael@odyn-dgx3",
            "target_host": "runpod-mi300x",
            "checkpoint_step": 10,
            "final_step": 20,
            "run_id": "test-run",
            "asset_paths": ["data/quant_mentor_500_alpaca.jsonl"],
            "command_retries": 1,
            "min_extra_steps": 1,
            "max_loss_delta": 1.0,
            "ssh_password": "leaked-secret",
            "transfer": {
                "source": {"server_name": "a", "control_host": "1.1.1.1", "control_port": 1111},
                "destination": {"server_name": "b", "control_host": "2.2.2.2", "control_port": 2222},
            },
        },
    }
    (tmp_path / "run.json").write_text(json.dumps(run))
    (tmp_path / "machines.json").write_text("{}")
    monkeypatch.delenv("CROSS_OEM_SSH_PASSWORD", raising=False)
    assert load_settings(tmp_path).ssh_password is None
    monkeypatch.setenv("CROSS_OEM_SSH_PASSWORD", "from-env")
    assert load_settings(tmp_path).ssh_password == "from-env"
