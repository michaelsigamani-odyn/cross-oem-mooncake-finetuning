"""Inference workload -- placeholder satisfying ask #2 ("potentially
inference"). The original repo (per its README) explicitly scoped
inference out ("we could later add disaggregated prefill for inference").
Rather than invent behavior that was never in the source repo, this stub
defines the *shape* a real inference workload would take, so plugging one
in later is additive (new file + new registry entry), not a rewrite of
the orchestration layer. Delete the NotImplementedError once a real
inference script (e.g. scripts/inference/serve.py) exists.
"""
from dataclasses import dataclass
from typing import Any, Dict

from .base import Workload


@dataclass
class InferenceJobSpec:
    model_id: str
    checkpoint_path: str
    prompts_path: str
    output_path: str
    max_new_tokens: int = 256


class InferenceWorkload(Workload):
    name = "inference"
    script_relpath = "scripts/inference/serve.py"  # does not exist yet

    def build_command(self, *, python_cmd: str, remote_root: str, spec: InferenceJobSpec, **kwargs: Any) -> str:
        raise NotImplementedError(
            "InferenceWorkload is a scaffold, not a shipped feature. The original repo intentionally "
            "scoped inference out. Implement scripts/inference/serve.py and this method together."
        )

    def parse_run_summary(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("see build_command docstring")
