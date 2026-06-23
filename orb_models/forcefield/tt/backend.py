"""TT device/session management for Orb forcefield inference."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

import torch

if TYPE_CHECKING:
    from orb_models.forcefield.tt.ops.runner import TTOpsRunner

TILE_SIZE = 32


class TTBackend(StrEnum):
    """Execution backend for Tenstorrent-accelerated inference."""

    SIMULATOR = "simulator"
    HARDWARE = "hardware"

    @classmethod
    def from_env(cls, default: TTBackend | None = None) -> TTBackend:
        raw = os.environ.get("ORB_TT_BACKEND", "").strip().lower()
        if raw in {"sim", "simulator"}:
            return cls.SIMULATOR
        if raw in {"hw", "hardware", "device"}:
            return cls.HARDWARE
        if default is not None:
            return default
        return cls.HARDWARE if tt_device_available() else cls.SIMULATOR

    @classmethod
    def auto(cls) -> TTBackend:
        return cls.from_env()


def tt_device_available() -> bool:
    """Return True when a Tenstorrent device node is present."""
    return os.path.exists("/dev/tenstorrent/0")


def pad_dim(size: int, tile: int = TILE_SIZE) -> int:
    return ((size + tile - 1) // tile) * tile


class TTDevice:
    """Owns a TTNN device handle and a shared ops runner."""

    def __init__(self, backend: TTBackend | Literal["auto"] = "auto", device_id: int = 0) -> None:
        if backend == "auto":
            backend = TTBackend.auto()
        self.backend = TTBackend(backend)
        self.device_id = device_id
        self._device: Any | None = None
        self._runner: TTOpsRunner | None = None

    @property
    def runner(self) -> TTOpsRunner:
        if self._runner is None:
            from orb_models.forcefield.tt.ops.runner import TTOpsRunner

            self._runner = TTOpsRunner(self.backend, device_id=self.device_id)
        return self._runner

    def open(self) -> None:
        if self._device is None:
            self._device = self.runner.open_device()

    def close(self) -> None:
        if self._runner is not None:
            self._runner.close_device()
        self._device = None

    def __enter__(self) -> TTDevice:
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@contextmanager
def tt_session(
    backend: TTBackend | Literal["auto"] = "auto", device_id: int = 0
) -> Iterator[TTDevice]:
    session = TTDevice(backend=backend, device_id=device_id)
    try:
        session.open()
        yield session
    finally:
        session.close()


def to_bfloat16_cpu(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.bfloat16:
        return tensor.detach().cpu()
    return tensor.detach().to(device="cpu", dtype=torch.bfloat16)
