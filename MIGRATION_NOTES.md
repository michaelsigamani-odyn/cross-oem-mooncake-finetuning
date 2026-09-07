# Migration notes

This refactor was verified, not just written: `configs/run.json` +
`configs/machines.json` were generated directly from the original
`repo_config.json` (minus the secret), `pytest tests/` passes (7/7,
covering config loading and the energy/cost formula cross-checks), and
`cross_oem_migration.orchestration.dagster.definitions:defs` loads
successfully under a real installed Dagster 1.13, resolving to 9 assets
with correct lineage and 2 asset checks attached to `published_manifest`.
That's the honest extent of verification possible without the actual
NVIDIA/AMD hosts, SSH access, and GPUs this pipeline targets -- I did not
run an end-to-end training + transfer + resume cycle against real
hardware, and you should not treat this as production-ready until you
have.

## Fully ported (same behavior, relocated + given an interface)

- **Config parsing** -- `config/models.py` + `config/loader.py` are a
  straight port of `PortabilitySettings` / `MachineSpec` / `load_settings`
  / `machine_catalog` / `machine_spec` (original lines 16-421), with one
  deliberate change: `ssh_password` no longer comes from the committed
  JSON file. See "Security fix" below.
- **SSH execution** -- `execution/ssh.py` ports `run_shell` / `ssh_cmd` /
  `scp_cmd` / `ssh_options` / `sshpass_env` (lines 430-511) unchanged,
  behind the `Executor` interface.
- **GPU preflight probe** -- `hardware/_preflight_script.py` is a
  byte-for-byte copy of `kernel_preflight_script` (lines 2030-2142). It's
  shared rather than duplicated per vendor because it self-detects vendor
  by design (you don't know what's on a host until you probe it).
- **Vendor telemetry** -- `hardware/nvidia.py` / `hardware/amd.py` port
  `query_nvidia_smi_sample` / `query_amd_smi_sample` / `parse_amd_smi_row`
  / the JSON-flattening helpers (train_lora_migration.py lines 337-536).
- **Transfer backends** -- `transfer/scp_backend.py` and
  `transfer/mooncake_backend.py` port `transfer_checkpoint_scp` /
  `transfer_checkpoint_mooncake` and all their helpers (lines 1619-1833,
  2760-2826), including the manifest-mismatch checksum verification and
  the mooncake receiver/sender wire protocol and retry logic.
- **Training command construction** -- `workloads/finetuning.py` ports
  `source_train_command` and `source_checkpoint_reusable` (lines
  1558-1616) for the single-host and single-node-multi-GPU cases.
- **Energy/cost math + formula cross-checks** -- `reporting/metrics.py`
  ports `energy_cost_gbp`, `cost_per_million_tokens`, `end_to_end_summary`,
  and `formula_validation` verbatim (lines 800-870, 2958-3051). This is
  the part of the pipeline that exists specifically to catch a wrong
  number before it reaches a report, so it got the most direct,
  line-by-line port and the most unit test coverage.
- **Data/artifact plumbing** -- `data/filesystem.py` ports
  `sync_asset_to_host` / `save_asset_from_host` / `output_asset_paths`
  (lines 550-609).
- **Publishing + asset checks** -- `orchestration/dagster/assets/published.py`
  ports `save_assets` / `saved_assets_exist_check` /
  `saved_assets_supported_type_check` (lines 3470-3525), but see the
  Dagster-API-shape change below.
- **The four already-standalone scripts** (`train_lora_migration.py`,
  `mooncake_tcp_agent.py`, `validate_resume.py`, `checkpoint_io.py`) are
  moved into `scripts/` **unmodified**. They already ran as independent
  remote processes in the original repo -- that was good design that
  didn't need refactoring, only relocating next to `scripts/finetuning`,
  `scripts/transfer`, `scripts/validation` for discoverability.

## Deliberately simplified or stubbed -- check against the original before relying on these

- **Multi-host distributed (DDP) training and rendezvous** --
  `FineTuningJobSpec` in `workloads/finetuning.py` has `nnodes` /
  `node_rank` / `rdzv_endpoint` fields and builds a `torch.distributed.run`
  invocation, but this is a reduced-fidelity port of
  `standalone_training_command` / `distributed_train_invocation` /
  `run_distributed_training` (original lines 2275-2760), which also
  handled per-rank PID/log tracking, NCCL interface binding
  (`nccl_socket_ifname`), and hang detection across ranks. **Do not run
  a real multi-node job against this without reviewing that section of
  the original file first.**
- **Standalone single-machine benchmarking** (`run_standalone_benchmarks`,
  `benchmark_row`, `render_standalone_table`, original lines 2562-2760)
  was **not ported at all**. It's a separate, large feature (per-vendor
  baseline benchmarking independent of the migration flow) and isn't
  referenced anywhere in the new asset graph. If you use it, it needs its
  own `workloads/benchmark.py` + asset, built the same way
  `FineTuningWorkload` was.
- **`verify_transfer` as a separate step** -- the original had
  `copy_checkpoint` and `verify_transfer` as two ops (lines 2829-2879).
  Here, `transferred_checkpoint` calls `backend.verify(result)` inline
  rather than as a separate asset. For `scp` this is a no-op (verification
  already happened via manifest checksums during transfer); for
  `mooncake_tcp` it's also currently a no-op passthrough. If you want
  transfer and verification as separately materialized/retryable assets,
  split `transferred_checkpoint` into two `@asset`s.
- **`prepare_hosts` / `sync_and_install` / `preflight_torch_transformers`**
  (original lines 1461-1520, 2220-2255: remote pip installs, torch/
  transformers version + architecture compatibility assertions) were not
  ported into a dedicated asset. The torch-install-command piece lives in
  `hardware/base.py`'s `default_torch_install_command`, but nothing in
  `orchestration/dagster/assets/` currently calls it -- you'd want a
  `prepared_hosts` asset upstream of `source_gpu_preflight` before running
  this against fresh machines.
- **`record_transfer_tests`** (iperf3/rsync baseline transfer benchmarks
  alongside the mooncake transfer, original lines 613-830, 1324-1359) was
  not ported. It's independent instrumentation, not required for the
  migration to function, and can be added as its own asset the same way.

## Genuine additions (not in the original repo)

- **`data/metrics_db.py` (`RunMetricsDatabase`, SQLite)** and the
  `run_metrics_db` asset. The original only ever wrote metrics to
  per-run JSON files -- there was no queryable history across runs. You
  asked for materialized assets "including databases," so this is a real
  addition, not a port. It's stdlib SQLite for zero new dependencies; swap
  it for Postgres/DuckDB if you need concurrent writers.
- **`workloads/inference.py` (`InferenceWorkload`)** is an explicit,
  `NotImplementedError`-raising stub. The original README says inference
  was intentionally out of scope ("We could later add disaggregated
  prefill for inference"). I did not invent inference behavior that was
  never in the source repo -- this class only defines the shape a real
  one would take.

## Security fix (not a refactor, a fix)

The original `repo_config.json` contained `"ssh_password": "michael"` in
plaintext, and the README repeated the same password for both the NVIDIA
and AMD demo hosts. Both are gone here:

- `configs/run.json` has no `ssh_password` field.
- `config/loader.py` reads a password (if genuinely needed) from the
  `CROSS_OEM_SSH_PASSWORD` environment variable only.
- `tests/test_config.py::test_ssh_password_is_not_read_from_committed_config`
  is a regression test asserting `load_settings()` never picks up a secret
  from the committed file.

If those demo credentials are still valid on real machines, rotate them --
they were public in this repo's git history regardless of what this
refactor does going forward.
