"""Tenstorrent extension entrypoints for Orb models."""

from orb_models.extensions.tt.pretrained import (
    TT_DIRECT_PRETRAINED_MODELS,
    load_tt_direct_model,
    orb_v3_direct_20_omat_tt,
    orb_v3_direct_inf_omat_tt,
)

__all__ = [
    "TT_DIRECT_PRETRAINED_MODELS",
    "load_tt_direct_model",
    "orb_v3_direct_20_omat_tt",
    "orb_v3_direct_inf_omat_tt",
]
