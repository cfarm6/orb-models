"""Torch modules that route selected ops through Tenstorrent."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from orb_models.forcefield.tt.ops.runner import TTOpsRunner
from orb_models.forcefield.tt.backend import TTBackend

MIN_TT_LINEAR_IN_FEATURES = 256
MIN_TT_LINEAR_OUT_FEATURES = 256
MIN_TT_MLP_IN_FEATURES = 128


def _is_tt_linear_candidate(linear: nn.Linear) -> bool:
    """Return True when a linear layer is large enough to amortize TT transfer overhead."""

    return (
        linear.weight.ndim == 2
        and linear.in_features >= MIN_TT_LINEAR_IN_FEATURES
        and linear.out_features >= MIN_TT_LINEAR_OUT_FEATURES
    )


def _activation_name(module: nn.Module) -> str | None:
    if isinstance(module, nn.Identity):
        return "identity"
    if isinstance(module, nn.SiLU):
        return "silu"
    if isinstance(module, nn.GELU):
        return "gelu"
    return None


def _linear_activation_pairs(module: nn.Sequential) -> list[tuple[nn.Linear, nn.Module, str]] | None:
    children = list(module.children())
    if not children or len(children) % 2 != 0:
        return None

    pairs: list[tuple[nn.Linear, nn.Module, str]] = []
    for linear, activation in zip(children[0::2], children[1::2], strict=True):
        if not isinstance(linear, nn.Linear):
            return None
        activation_name = _activation_name(activation)
        if activation_name is None:
            return None
        pairs.append((linear, activation, activation_name))
    return pairs


def _is_tt_mlp_candidate(module: nn.Sequential) -> bool:
    pairs = _linear_activation_pairs(module)
    if pairs is None or len(pairs) < 2:
        return False
    return any(
        linear.weight.ndim == 2
        and linear.in_features >= MIN_TT_MLP_IN_FEATURES
        and linear.out_features >= MIN_TT_LINEAR_OUT_FEATURES
        for linear, _, _ in pairs
    )


class TTLinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` that uses TT-Lang matmul."""

    def __init__(self, linear: nn.Linear, runner: TTOpsRunner) -> None:
        super().__init__()
        if not _is_tt_linear_candidate(linear):
            raise ValueError(
                f"Cannot wrap Linear with shape {tuple(linear.weight.shape)} for TT execution."
            )
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        self._runner = runner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            leading = x.shape[:-1]
            flat = x.reshape(-1, x.shape[-1])
            out = self._runner.linear(flat, self.weight, self.bias)
            return out.reshape(*leading, self.out_features)
        return self._runner.linear(x, self.weight, self.bias)




class TTMLP(nn.Module):
    """Drop-in replacement for build_mlp Sequential modules with TT-resident intermediates."""

    def __init__(self, module: nn.Sequential, runner: TTOpsRunner) -> None:
        super().__init__()
        pairs = _linear_activation_pairs(module)
        if pairs is None or not _is_tt_mlp_candidate(module):
            raise ValueError("Cannot wrap Sequential as a TT MLP.")
        self._cpu = module
        self._runner = runner
        self._linears = nn.ModuleList(linear for linear, _, _ in pairs)
        self._activation_names = [activation_name for _, _, activation_name in pairs]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._runner.backend is TTBackend.SIMULATOR:
            return self._cpu(x)

        layers = [
            (linear.weight, linear.bias, activation_name)
            for linear, activation_name in zip(
                self._linears,
                self._activation_names,
                strict=True,
            )
        ]
        if x.ndim > 2:
            leading = x.shape[:-1]
            flat = x.reshape(-1, x.shape[-1])
            out = self._runner.mlp(flat, layers)
            return out.reshape(*leading, self._linears[-1].out_features)
        return self._runner.mlp(x, layers)
def patch_linears(module: nn.Module, runner: TTOpsRunner) -> int:
    """Replace child ``nn.Linear`` and MLP modules with TT wrappers."""

    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Sequential) and _is_tt_mlp_candidate(child):
            pairs = _linear_activation_pairs(child)
            assert pairs is not None
            setattr(module, name, TTMLP(child, runner))
            replaced += len(pairs)
        elif isinstance(child, nn.Linear) and _is_tt_linear_candidate(child):
            setattr(module, name, TTLinear(child, runner))
            replaced += 1
        else:
            replaced += patch_linears(child, runner)
    return replaced


def maybe_tt_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """CPU fallback used when a layer is not patched for TT."""

    return F.linear(x, weight, bias)
