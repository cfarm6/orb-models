"""TT-accelerated loaders for direct Orb forcefield models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from orb_models.forcefield import pretrained as cpu_pretrained
from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter
from orb_models.forcefield.models.direct_regressor import DirectForcefieldRegressor
from orb_models.forcefield.tt import TTBackend, TTDirectForcefieldRegressor

TT_DIRECT_PRETRAINED_MODELS: dict[str, Callable[..., tuple[DirectForcefieldRegressor, ForcefieldAtomsAdapter]]] = {
    "orb-v3-direct-omol": cpu_pretrained.orb_v3_direct_omol,
    "orbmol-v1-direct": cpu_pretrained.orbmol_v1_direct,
    "orb-v3-direct-20-omat": cpu_pretrained.orb_v3_direct_20_omat,
    "orb-v3-direct-inf-omat": cpu_pretrained.orb_v3_direct_inf_omat,
    "orb-v3-direct-20-mpa": cpu_pretrained.orb_v3_direct_20_mpa,
    "orb-v3-direct-inf-mpa": cpu_pretrained.orb_v3_direct_inf_mpa,
    "separate-d3-3layer": cpu_pretrained.separate_d3_direct_3layer,
    "separate-d3-5layer": cpu_pretrained.separate_d3_direct_5layer,
    "separate-d4-3layer": cpu_pretrained.separate_d4_direct_3layer,
    "separate-d4-5layer": cpu_pretrained.separate_d4_direct_5layer,
    "orb-v2": cpu_pretrained.orb_v2,
    "orb-mptraj-only-v2": cpu_pretrained.orb_mptraj_only_v2,
    "orb-d3-v2": cpu_pretrained.orb_d3_v2,
    "orb-d3-sm-v2": cpu_pretrained.orb_d3_sm_v2,
    "orb-d3-xs-v2": cpu_pretrained.orb_d3_xs_v2,
}


def load_tt_direct_model(
    model_name: str,
    *,
    backend: TTBackend | Literal["auto"] = "auto",
    device_id: int = 0,
    precision: str = "float32-high",
    compile: bool | None = False,
    **loader_kwargs,
) -> tuple[TTDirectForcefieldRegressor, ForcefieldAtomsAdapter]:
    """Load a direct pretrained Orb model and wrap it for TT inference."""

    if model_name not in TT_DIRECT_PRETRAINED_MODELS:
        known = ", ".join(sorted(TT_DIRECT_PRETRAINED_MODELS))
        raise KeyError(f"Unknown direct TT model {model_name!r}. Known models: {known}")
    loader = TT_DIRECT_PRETRAINED_MODELS[model_name]
    return TTDirectForcefieldRegressor.from_pretrained(
        loader,
        backend=backend,
        device_id=device_id,
        precision=precision,
        compile=compile,
        **loader_kwargs,
    )


def orbmol_v1_direct_tt(
    *,
    backend: TTBackend | Literal["auto"] = "auto",
    device_id: int = 0,
    precision: str = "float32-high",
    compile: bool | None = False,
    **loader_kwargs,
) -> tuple[TTDirectForcefieldRegressor, ForcefieldAtomsAdapter]:
    return load_tt_direct_model(
        "orbmol-v1-direct",
        backend=backend,
        device_id=device_id,
        precision=precision,
        compile=compile,
        **loader_kwargs,
    )

def orb_v3_direct_20_omat_tt(
    *,
    backend: TTBackend | Literal["auto"] = "auto",
    device_id: int = 0,
    precision: str = "float32-high",
    compile: bool | None = False,
    **loader_kwargs,
) -> tuple[TTDirectForcefieldRegressor, ForcefieldAtomsAdapter]:
    return load_tt_direct_model(
        "orb-v3-direct-20-omat",
        backend=backend,
        device_id=device_id,
        precision=precision,
        compile=compile,
        **loader_kwargs,
    )


def orb_v3_direct_inf_omat_tt(
    *,
    backend: TTBackend | Literal["auto"] = "auto",
    device_id: int = 0,
    precision: str = "float32-high",
    compile: bool | None = False,
    **loader_kwargs,
) -> tuple[TTDirectForcefieldRegressor, ForcefieldAtomsAdapter]:
    return load_tt_direct_model(
        "orb-v3-direct-inf-omat",
        backend=backend,
        device_id=device_id,
        precision=precision,
        compile=compile,
        **loader_kwargs,
    )
