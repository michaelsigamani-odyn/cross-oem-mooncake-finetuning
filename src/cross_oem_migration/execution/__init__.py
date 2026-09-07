from .base import Executor, ExecResult
from .ssh import SSHExecutor
from .local import LocalExecutor

__all__ = ["Executor", "ExecResult", "SSHExecutor", "LocalExecutor"]
