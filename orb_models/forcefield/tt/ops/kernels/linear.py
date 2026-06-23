"""TT-Lang matmul kernel used for linear layers on the functional simulator."""

from __future__ import annotations

from typing import Any

from orb_models.forcefield.tt.backend import TILE_SIZE

try:
    from ttl.sim import ttl as _sim_ttl
except ImportError:  # pragma: no cover - optional dependency
    _sim_ttl = None

_SIM_LINEAR_KERNEL: Any | None = None


def _register_sim_kernel() -> Any:
    global _SIM_LINEAR_KERNEL
    if _SIM_LINEAR_KERNEL is not None:
        return _SIM_LINEAR_KERNEL
    if _sim_ttl is None:
        raise ImportError("tt-lang is required for TT simulator execution.")

    ttl = _sim_ttl

    @ttl.operation(grid=(1, 1))
    def _linear_matmul(x_tensor, w_tensor, y_tensor) -> None:
        m_tiles = x_tensor.shape[0] // TILE_SIZE
        n_tiles = w_tensor.shape[1] // TILE_SIZE
        k_tiles = x_tensor.shape[1] // TILE_SIZE

        x_dfb = ttl.make_dataflow_buffer_like(x_tensor, shape=(1, 1), block_count=2)
        w_dfb = ttl.make_dataflow_buffer_like(w_tensor, shape=(1, 1), block_count=2)
        acc_dfb = ttl.make_dataflow_buffer_like(y_tensor, shape=(1, 1), block_count=2)
        y_dfb = ttl.make_dataflow_buffer_like(y_tensor, shape=(1, 1), block_count=2)

        @ttl.datamovement()
        def read() -> None:
            for m_tile in range(m_tiles):
                for n_tile in range(n_tiles):
                    for k_tile in range(k_tiles):
                        with x_dfb.reserve() as x_blk, w_dfb.reserve() as w_blk:
                            tx_x = ttl.copy(x_tensor[m_tile, k_tile], x_blk)
                            tx_w = ttl.copy(w_tensor[k_tile, n_tile], w_blk)
                            tx_x.wait()
                            tx_w.wait()

        @ttl.compute()
        def compute() -> None:
            for _ in range(m_tiles):
                for _ in range(n_tiles):
                    with acc_dfb.reserve() as acc_blk:
                        acc_blk.store(ttl.block.fill(0, shape=acc_blk.shape))
                    for _ in range(k_tiles):
                        with (
                            x_dfb.wait() as x_blk,
                            w_dfb.wait() as w_blk,
                            acc_dfb.wait() as prev_acc_blk,
                            acc_dfb.reserve() as next_acc_blk,
                        ):
                            next_acc_blk.store(prev_acc_blk + x_blk @ w_blk)
                    with acc_dfb.wait() as acc_blk, y_dfb.reserve() as y_blk:
                        y_blk.store(acc_blk)

        @ttl.datamovement()
        def write() -> None:
            for m_tile in range(m_tiles):
                for n_tile in range(n_tiles):
                    with y_dfb.wait() as y_blk:
                        tx = ttl.copy(y_blk, y_tensor[m_tile, n_tile])
                        tx.wait()

    _SIM_LINEAR_KERNEL = _linear_matmul
    return _SIM_LINEAR_KERNEL


def linear_matmul_sim(x_tensor: Any, w_tensor: Any, y_tensor: Any) -> None:
    kernel = _register_sim_kernel()
    kernel(x_tensor, w_tensor, y_tensor)
