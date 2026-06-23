"""Host-side runner that executes TT-Lang kernels for Orb models."""

from __future__ import annotations

import os
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any

import torch

from orb_models.forcefield.tt.backend import TTBackend, pad_dim, to_bfloat16_cpu


@dataclass(frozen=True)
class _WeightCacheKey:
    data_ptr: int
    in_features: int
    out_features: int
    in_pad: int
    out_pad: int


@dataclass(frozen=True)
class _BiasCacheKey:
    data_ptr: int
    out_features: int
    out_pad: int


@dataclass
class TTLinearProfileRecord:
    calls: int = 0
    input_to_tt_ns: int = 0
    weight_to_tt_ns: int = 0
    bias_to_tt_ns: int = 0
    matmul_ns: int = 0
    activation_ns: int = 0
    output_to_torch_ns: int = 0
    bias_ns: int = 0
    input_bytes: int = 0
    weight_bytes: int = 0
    bias_bytes: int = 0
    output_bytes: int = 0
    weight_cache_misses: int = 0
    bias_cache_misses: int = 0


class TTOpsRunner:
    """Runs tile-padded linear layers on the TT simulator or hardware."""

    def __init__(self, backend: TTBackend, device_id: int = 0) -> None:
        self.backend = backend
        self.device_id = device_id
        self._ttnn: Any | None = None
        self._device: Any | None = None
        self._weight_cache: dict[_WeightCacheKey, Any] = {}
        self._bias_cache: dict[_BiasCacheKey, Any] = {}
        self.profile_enabled = os.environ.get("ORB_TT_PROFILE", "").lower() in {
            "1",
            "true",
            "yes",
        }
        self._profile: dict[tuple[int, int, int], TTLinearProfileRecord] = {}

    def _import_ttnn(self) -> Any:
        if self._ttnn is not None:
            return self._ttnn
        if self.backend is TTBackend.SIMULATOR:
            from ttl.sim import ttnn  # type: ignore[import-not-found]
        else:
            os.environ.pop("TTLANG_SIM_ONLY", None)
            import ttnn  # type: ignore[import-not-found]
        self._ttnn = ttnn
        return ttnn

    def open_device(self) -> Any:
        if self._device is not None:
            return self._device
        ttnn = self._import_ttnn()
        self._device = ttnn.open_device(device_id=self.device_id)
        return self._device

    def close_device(self) -> None:
        if self._device is None:
            return
        ttnn = self._import_ttnn()
        ttnn.close_device(self._device)
        self._device = None
        self._weight_cache.clear()
        self._bias_cache.clear()

    def clear_profile(self) -> None:
        self._profile.clear()

    def profile_summary(self) -> dict[str, dict[str, float | int]]:
        summary: dict[str, dict[str, float | int]] = {}
        for (rows, in_features, out_features), record in sorted(self._profile.items()):
            key = f"{rows}x{in_features}->{out_features}"
            total_ns = (
                record.input_to_tt_ns
                + record.weight_to_tt_ns
                + record.bias_to_tt_ns
                + record.matmul_ns
                + record.activation_ns
                + record.output_to_torch_ns
                + record.bias_ns
            )
            summary[key] = {
                "calls": record.calls,
                "total_ms": total_ns / 1_000_000,
                "input_to_tt_ms": record.input_to_tt_ns / 1_000_000,
                "weight_to_tt_ms": record.weight_to_tt_ns / 1_000_000,
                "bias_to_tt_ms": record.bias_to_tt_ns / 1_000_000,
                "matmul_ms": record.matmul_ns / 1_000_000,
                "activation_ms": record.activation_ns / 1_000_000,
                "output_to_torch_ms": record.output_to_torch_ns / 1_000_000,
                "bias_ms": record.bias_ns / 1_000_000,
                "input_mb": record.input_bytes / 1_000_000,
                "weight_mb": record.weight_bytes / 1_000_000,
                "bias_mb": record.bias_bytes / 1_000_000,
                "output_mb": record.output_bytes / 1_000_000,
                "weight_cache_misses": record.weight_cache_misses,
                "bias_cache_misses": record.bias_cache_misses,
            }
        return summary

    def _from_torch(self, tensor: torch.Tensor) -> Any:
        ttnn = self._import_ttnn()
        device = self.open_device()
        return ttnn.from_torch(
            tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _to_torch(self, tensor: Any) -> torch.Tensor:
        ttnn = self._import_ttnn()
        return ttnn.to_torch(tensor)

    def _prepare_weight(self, weight: torch.Tensor, in_pad: int, out_pad: int) -> torch.Tensor:
        # PyTorch Linear stores [out, in]; matmul uses [in, out].
        out_features, in_features = weight.shape
        padded = torch.zeros(in_pad, out_pad, dtype=torch.bfloat16)
        padded[:in_features, :out_features] = weight.detach().to(dtype=torch.bfloat16).T
        return padded.contiguous()

    def _get_weight_tensor(
        self,
        weight: torch.Tensor,
        in_pad: int,
        out_pad: int,
        record: TTLinearProfileRecord | None = None,
    ) -> Any:
        if weight.ndim != 2:
            raise ValueError(f"TT linear weights must be 2D, got shape {tuple(weight.shape)}")

        key = _WeightCacheKey(
            data_ptr=weight.data_ptr(),
            in_features=weight.shape[1],
            out_features=weight.shape[0],
            in_pad=in_pad,
            out_pad=out_pad,
        )
        cached = self._weight_cache.get(key)
        if cached is not None:
            return cached

        start = perf_counter_ns()
        prepared = self._prepare_weight(weight, in_pad=in_pad, out_pad=out_pad)
        tensor = self._from_torch(prepared)
        if record is not None:
            record.weight_to_tt_ns += perf_counter_ns() - start
            record.weight_bytes += prepared.numel() * prepared.element_size()
            record.weight_cache_misses += 1
        self._weight_cache[key] = tensor
        return tensor

    def _prepare_bias(self, bias: torch.Tensor, out_pad: int) -> torch.Tensor:
        padded = torch.zeros(1, out_pad, dtype=torch.bfloat16)
        padded[0, : bias.shape[0]] = bias.detach().to(dtype=torch.bfloat16)
        return padded.contiguous()

    def _get_bias_tensor(
        self,
        bias: torch.Tensor,
        out_pad: int,
        record: TTLinearProfileRecord | None = None,
    ) -> Any:
        key = _BiasCacheKey(
            data_ptr=bias.data_ptr(),
            out_features=bias.shape[0],
            out_pad=out_pad,
        )
        cached = self._bias_cache.get(key)
        if cached is not None:
            return cached

        start = perf_counter_ns()
        prepared = self._prepare_bias(bias, out_pad=out_pad)
        tensor = self._from_torch(prepared)
        if record is not None:
            record.bias_to_tt_ns += perf_counter_ns() - start
            record.bias_bytes += prepared.numel() * prepared.element_size()
            record.bias_cache_misses += 1
        self._bias_cache[key] = tensor
        return tensor

    def _profile_record(
        self,
        rows: int,
        in_features: int,
        out_features: int,
    ) -> TTLinearProfileRecord | None:
        if not self.profile_enabled:
            return None
        record = self._profile.setdefault(
            (rows, in_features, out_features),
            TTLinearProfileRecord(),
        )
        record.calls += 1
        return record

    def _stage_input_tensor(
        self,
        x: torch.Tensor,
        rows_pad: int,
        in_pad: int,
        record: TTLinearProfileRecord | None = None,
    ) -> Any:
        start = perf_counter_ns()
        x_cpu = to_bfloat16_cpu(x)
        x_pad = torch.zeros(rows_pad, in_pad, dtype=torch.bfloat16)
        x_pad[: x.shape[0], : x.shape[1]] = x_cpu
        tensor = self._from_torch(x_pad)
        if record is not None:
            record.input_to_tt_ns += perf_counter_ns() - start
            record.input_bytes += x.numel() * x.element_size()
        return tensor

    def _to_torch_output(
        self,
        tensor: Any,
        rows: int,
        out_features: int,
        dtype: torch.dtype,
        record: TTLinearProfileRecord | None = None,
    ) -> torch.Tensor:
        start = perf_counter_ns()
        out = self._to_torch(tensor)[:rows, :out_features].to(dtype=dtype)
        if record is not None:
            record.output_to_torch_ns += perf_counter_ns() - start
            record.output_bytes += out.numel() * out.element_size()
        return out

    def _linear_simulator(
        self,
        x_pad: torch.Tensor,
        weight: torch.Tensor,
        in_pad: int,
        out_pad: int,
        record: TTLinearProfileRecord | None = None,
    ) -> torch.Tensor:
        from orb_models.forcefield.tt.ops.kernels.linear import linear_matmul_sim

        start = perf_counter_ns()
        x_tensor = self._from_torch(x_pad)
        if record is not None:
            record.input_to_tt_ns += perf_counter_ns() - start

        w_tensor = self._get_weight_tensor(weight, in_pad=in_pad, out_pad=out_pad, record=record)
        y_pad = torch.zeros(x_pad.shape[0], out_pad, dtype=torch.bfloat16)
        y_tensor = self._from_torch(y_pad)
        matmul_start = perf_counter_ns()
        linear_matmul_sim(x_tensor, w_tensor, y_tensor)
        if record is not None:
            record.matmul_ns += perf_counter_ns() - matmul_start
        start = perf_counter_ns()
        out = self._to_torch(y_tensor)
        if record is not None:
            record.output_to_torch_ns += perf_counter_ns() - start
        return out

    def _linear_hardware_tensor(
        self,
        x_tensor: Any,
        weight: torch.Tensor,
        in_pad: int,
        out_pad: int,
        record: TTLinearProfileRecord | None = None,
    ) -> Any:
        ttnn = self._import_ttnn()
        w_tensor = self._get_weight_tensor(weight, in_pad=in_pad, out_pad=out_pad, record=record)
        start = perf_counter_ns()
        y_tensor = ttnn.matmul(x_tensor, w_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if record is not None:
            record.matmul_ns += perf_counter_ns() - start
        return y_tensor

    def _linear_hardware(
        self,
        x_pad: torch.Tensor,
        weight: torch.Tensor,
        in_pad: int,
        out_pad: int,
        record: TTLinearProfileRecord | None = None,
    ) -> torch.Tensor:
        start = perf_counter_ns()
        x_tensor = self._from_torch(x_pad)
        if record is not None:
            record.input_to_tt_ns += perf_counter_ns() - start
        y_tensor = self._linear_hardware_tensor(
            x_tensor,
            weight,
            in_pad=in_pad,
            out_pad=out_pad,
            record=record,
        )
        start = perf_counter_ns()
        out = self._to_torch(y_tensor)
        if record is not None:
            record.output_to_torch_ns += perf_counter_ns() - start
        return out

    def _apply_bias_tensor(
        self,
        tensor: Any,
        bias: torch.Tensor,
        out_pad: int,
        record: TTLinearProfileRecord | None = None,
    ) -> Any:
        ttnn = self._import_ttnn()
        bias_tensor = self._get_bias_tensor(bias, out_pad=out_pad, record=record)
        start = perf_counter_ns()
        out = ttnn.add(tensor, bias_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if record is not None:
            record.bias_ns += perf_counter_ns() - start
        return out

    def _apply_activation_tensor(
        self,
        tensor: Any,
        activation: str,
        record: TTLinearProfileRecord | None = None,
    ) -> Any:
        if activation == "identity":
            return tensor

        ttnn = self._import_ttnn()
        start = perf_counter_ns()
        if activation == "silu":
            out = ttnn.silu(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        elif activation == "gelu":
            out = ttnn.gelu(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        elif activation == "softplus":
            out = ttnn.softplus(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            raise ValueError(f"Unsupported TT activation {activation!r}")
        if record is not None:
            record.activation_ns += perf_counter_ns() - start
        return out

    def linear(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute ``x @ weight.T (+ bias)`` on the TT simulator or hardware."""

        if weight.ndim != 2:
            raise ValueError(f"TT linear weights must be 2D, got shape {tuple(weight.shape)}")
        if x.ndim != 2:
            raise ValueError(f"TT linear expects a 2D input, got shape {tuple(x.shape)}")

        batch, in_features = x.shape
        out_features = weight.shape[0]
        in_pad = pad_dim(in_features)
        out_pad = pad_dim(out_features)
        rows_pad = pad_dim(batch)
        record = self._profile_record(batch, in_features, out_features)

        x_cpu = to_bfloat16_cpu(x)
        x_pad = torch.zeros(rows_pad, in_pad, dtype=torch.bfloat16)
        x_pad[:batch, :in_features] = x_cpu
        if record is not None:
            record.input_bytes += x.numel() * x.element_size()

        if self.backend is TTBackend.SIMULATOR:
            y_cpu = self._linear_simulator(
                x_pad,
                weight,
                in_pad=in_pad,
                out_pad=out_pad,
                record=record,
            )
        else:
            y_cpu = self._linear_hardware(
                x_pad,
                weight,
                in_pad=in_pad,
                out_pad=out_pad,
                record=record,
            )

        out = y_cpu[:batch, :out_features].to(dtype=x.dtype)
        if record is not None:
            record.output_bytes += out.numel() * out.element_size()
        if bias is not None:
            bias_start = perf_counter_ns()
            out = out + bias.to(device=out.device, dtype=out.dtype)
            if record is not None:
                record.bias_ns += perf_counter_ns() - bias_start
        return out

    def mlp(
        self,
        x: torch.Tensor,
        layers: list[tuple[torch.Tensor, torch.Tensor | None, str]],
    ) -> torch.Tensor:
        """Run a linear/activation stack while keeping intermediate tensors on TT hardware."""

        if self.backend is TTBackend.SIMULATOR:
            raise RuntimeError("TT resident MLP execution is hardware-only")
        if x.ndim != 2:
            raise ValueError(f"TT MLP expects a 2D input, got shape {tuple(x.shape)}")
        if not layers:
            return x

        batch, in_features = x.shape
        rows_pad = pad_dim(batch)
        first_weight = layers[0][0]
        first_record = self._profile_record(batch, in_features, first_weight.shape[0])
        tensor = self._stage_input_tensor(
            x,
            rows_pad=rows_pad,
            in_pad=pad_dim(in_features),
            record=first_record,
        )

        current_features = in_features
        final_record: TTLinearProfileRecord | None = None
        final_out_features = current_features
        for layer_index, (weight, bias, activation) in enumerate(layers):
            if weight.ndim != 2:
                raise ValueError(f"TT linear weights must be 2D, got shape {tuple(weight.shape)}")
            out_features = weight.shape[0]
            if weight.shape[1] != current_features:
                raise ValueError(
                    "TT MLP layer shape mismatch: "
                    f"expected {current_features} input features, got {weight.shape[1]}"
                )
            record = first_record if layer_index == 0 else self._profile_record(
                batch,
                current_features,
                out_features,
            )
            tensor = self._linear_hardware_tensor(
                tensor,
                weight,
                in_pad=pad_dim(current_features),
                out_pad=pad_dim(out_features),
                record=record,
            )
            if bias is not None:
                tensor = self._apply_bias_tensor(
                    tensor,
                    bias,
                    out_pad=pad_dim(out_features),
                    record=record,
                )
            tensor = self._apply_activation_tensor(tensor, activation, record=record)
            current_features = out_features
            final_out_features = out_features
            final_record = record

        return self._to_torch_output(
            tensor,
            batch,
            final_out_features,
            x.dtype,
            record=final_record,
        )
