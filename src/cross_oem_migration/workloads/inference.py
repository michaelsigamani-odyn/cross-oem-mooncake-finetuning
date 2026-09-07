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
