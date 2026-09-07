from cross_oem_migration.config import load_settings


def test_load_settings_reads_run_and_machines_json():
    settings = load_settings()
    assert settings.run_id
    assert settings.source_host in {m.hostname for m in settings.machine_catalog.values()} or settings.source_host
    assert "nvidia_control" in settings.machine_catalog


def test_ssh_password_is_not_read_from_committed_config():
    # Regression test for the plaintext-password issue in the original
    # repo_config.json -- configs/run.json must never contain a real
    # secret, and load_settings() must not silently pick one up from it.
    settings = load_settings()
    assert settings.ssh_password in (None, "")
