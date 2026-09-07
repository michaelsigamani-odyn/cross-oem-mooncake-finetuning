"""Vendor lookup (Open/Closed Principle): add a new vendor by adding one
adapter class + one line here. Nothing else in the codebase changes."""
from typing import Dict, Type

from .amd import AmdAdapter
from .base import GpuVendorAdapter
from .nvidia import NvidiaAdapter

_ADAPTERS: Dict[str, Type[GpuVendorAdapter]] = {
    "nvidia": NvidiaAdapter,
    "amd": AmdAdapter,
}


def get_adapter(vendor: str) -> GpuVendorAdapter:
    try:
        return _ADAPTERS[vendor.strip().lower()]()
    except KeyError as exc:
        raise ValueError(f"No hardware adapter registered for vendor={vendor!r}. Known vendors: {list(_ADAPTERS)}") from exc


def register_adapter(vendor: str, adapter_cls: Type[GpuVendorAdapter]) -> None:
    """Escape hatch for plugging in a new vendor (e.g. Intel Gaudi) without
    editing this file, if you'd rather register from a plugin package."""
    _ADAPTERS[vendor.strip().lower()] = adapter_cls
