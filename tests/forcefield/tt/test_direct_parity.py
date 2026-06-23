import copy

import pytest
import torch


def test_tt_direct_registry_includes_orbmol_v1_direct():
    from orb_models.extensions.tt import TT_DIRECT_PRETRAINED_MODELS, orbmol_v1_direct_tt
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.tt import DIRECT_PRETRAINED_MODELS

    assert TT_DIRECT_PRETRAINED_MODELS["orbmol-v1-direct"] is pretrained.orbmol_v1_direct
    assert DIRECT_PRETRAINED_MODELS["orbmol-v1-direct"] is pretrained.orbmol_v1_direct
    assert callable(orbmol_v1_direct_tt)


@pytest.mark.parametrize("graph_name", ["tt_single_graph", "tt_batch"])
def test_direct_regressor_tt_sim_matches_cpu(direct_regressor, graph_name, request):
    from orb_models.forcefield.tt import TTBackend, TTDirectForcefieldRegressor
    from orb_models.forcefield.tt.backend import TTDevice

    graph = request.getfixturevalue(graph_name)
    model = copy.deepcopy(direct_regressor)
    model.eval()

    cpu_out = model.predict(graph)

    tt_model = TTDirectForcefieldRegressor(
        model=model,
        tt_device=TTDevice(backend=TTBackend.SIMULATOR),
    )
    try:
        tt_out = tt_model.predict(graph)
    finally:
        tt_model.close()

    for key in cpu_out:
        torch.testing.assert_close(
            tt_out[key],
            cpu_out[key],
            atol=5e-2,
            rtol=5e-2,
            msg=f"Mismatch for {key!r}",
        )


@pytest.mark.tt_hardware
@pytest.mark.parametrize("graph_name", ["tt_single_graph"])
def test_direct_regressor_tt_hardware_matches_cpu(direct_regressor, graph_name, request):
    from orb_models.forcefield.tt import TTBackend, TTDirectForcefieldRegressor
    from orb_models.forcefield.tt.backend import TTDevice

    graph = request.getfixturevalue(graph_name)
    model = copy.deepcopy(direct_regressor)
    model.eval()
    cpu_out = model.predict(graph)

    tt_model = TTDirectForcefieldRegressor(
        model=model,
        tt_device=TTDevice(backend=TTBackend.HARDWARE),
    )
    try:
        tt_out = tt_model.predict(graph)
    finally:
        tt_model.close()

    for key in cpu_out:
        torch.testing.assert_close(
            tt_out[key],
            cpu_out[key],
            atol=1e-1,
            rtol=1e-1,
            msg=f"Mismatch for {key!r}",
        )
