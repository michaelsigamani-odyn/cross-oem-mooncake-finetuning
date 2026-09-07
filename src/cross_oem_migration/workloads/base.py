"""Workload abstraction (ask #2): a Workload knows how to build the
remote command that runs it and how to parse its own result artifact. It
does NOT know about Dagster, SSH, or GPU vendors -- those are supplied to
it (or it's run via an Executor by the orchestration layer). This is the
seam that lets you add an InferenceWorkload, an EvalWorkload, etc. later
without touching the pipeline scaffold, and lets you run any workload
locally in a unit test without spinning up Dagster at all.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RunHandle:
    """Everything the orchestrator needs to locate a workload's outputs
    after it's been dispatched to a host."""
    host: str
    run_root: str
    summary_path: str


class Workload(ABC):
    name: str

    @abstractmethod
    def build_command(self, *, python_cmd: str, remote_root: str, **kwargs: Any) -> str:
        """Return the shell command to launch this workload on a remote
        host. Vendor/hardware specifics (CUDA_VISIBLE_DEVICES, torchrun
        wrapping) are the caller's job via hardware/execution, not this
        method's -- this method only knows about ITS OWN CLI contract."""

    @abstractmethod
    def parse_run_summary(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize the workload's own run_summary.json into whatever
        shape the reporting layer expects."""
