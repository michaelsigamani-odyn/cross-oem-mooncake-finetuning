"""SQLite-backed store for per-run metrics.

NOTE: this is a genuine ADDITION, not a port -- the original repo only
ever wrote metrics to JSON files on disk (cross_oem_metrics_report.json,
run_summary.json, etc). You explicitly asked for assets "including
databases" to be materialized and visible in the Dagster Web UI, and a
flat JSON file doesn't give you a queryable history across runs. SQLite
is stdlib (zero new dependency), gives you a real queryable table, and
the orchestration layer materializes it as a `db_asset` with a row-count
+ preview so it shows up properly in the UI's Asset catalog. If you'd
rather point this at Postgres/DuckDB for concurrent access, swap this
one class -- everything else is unaffected.
"""
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List


class RunMetricsDatabase:
    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS run_metrics (
                    run_id TEXT NOT NULL,
                    source_host TEXT,
                    target_host TEXT,
                    status TEXT,
                    checkpoint_step INTEGER,
                    final_step INTEGER,
                    final_loss REAL,
                    tokens_per_second REAL,
                    gpu_energy_kwh REAL,
                    energy_cost_gbp REAL,
                    recorded_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, recorded_at)
                )
                """
            )

    def record(self, run_id: str, row: Dict[str, Any], recorded_at: str) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO run_metrics
                    (run_id, source_host, target_host, status, checkpoint_step, final_step,
                     final_loss, tokens_per_second, gpu_energy_kwh, energy_cost_gbp, recorded_at, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, row.get("source_host"), row.get("target_host"), row.get("status"),
                    row.get("checkpoint_step"), row.get("final_step"), row.get("final_loss"),
                    row.get("tokens_per_second"), row.get("gpu_energy_kwh"), row.get("energy_cost_gbp"),
                    recorded_at, json.dumps(row),
                ),
            )

    def all_rows(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute("SELECT * FROM run_metrics ORDER BY recorded_at DESC")]

    def row_count(self) -> int:
        with sqlite3.connect(self._db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM run_metrics").fetchone()[0]
