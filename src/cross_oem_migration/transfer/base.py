"""Transfer backend abstraction. `scp` and Mooncake RDMA-over-TCP are two
interchangeable strategies for moving a checkpoint from source_host to
target_host; nothing in the pipeline should care which one is active
beyond calling `backend.transfer(...)`. This replaces the
`transfer_checkpoint` dispatcher + `transfer_checkpoint_scp` /
`transfer_checkpoint_mooncake` pair (original lines 1619-1704), which is
already close to this shape but was a free-function dispatch rather than
a registered strategy -- adding a third backend (e.g. NCCL send/recv,
S3 relay) meant editing the `if/elif` in `transfer_checkpoint`.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class TransferResult:
    success: bool
    backend: str
    bytes_transferred: int
    elapsed_seconds: float
    throughput_bytes_per_second: float
    checksum_ok: bool
    resume_validation_ok: bool
    details: Dict[str, Any]


class TransferBackend(ABC):
    name: str

    @abstractmethod
    def transfer(self, *, source_host: str, source_path: str, destination_host: str, destination_path: str) -> TransferResult:
        ...

    @abstractmethod
    def verify(self, result: TransferResult) -> TransferResult:
        """Return a (possibly updated) TransferResult with checksum_ok set."""
