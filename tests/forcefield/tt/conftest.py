import copy

import pytest

from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from tests.forcefield.conftest import get_batch_from_ase_with_latents

# Build graph fixtures before importing TT stack (ttnn init breaks warp neighbor lists).
_TT_SINGLE_GRAPH = get_batch_from_ase_with_latents()
_TT_BATCH = AtomGraphs.batch(
    [copy.deepcopy(_TT_SINGLE_GRAPH), copy.deepcopy(_TT_SINGLE_GRAPH)]
)


@pytest.fixture
def tt_single_graph():
    return copy.deepcopy(_TT_SINGLE_GRAPH)


@pytest.fixture
def tt_batch():
    return copy.deepcopy(_TT_BATCH)


@pytest.fixture
def tt_sim_device():
    from orb_models.forcefield.tt.backend import TTBackend, TTDevice

    device = TTDevice(backend=TTBackend.SIMULATOR)
    device.open()
    try:
        yield device
    finally:
        device.close()


@pytest.fixture
def tt_hw_device():
    from orb_models.forcefield.tt.backend import TTBackend, TTDevice

    device = TTDevice(backend=TTBackend.HARDWARE)
    device.open()
    try:
        yield device
    finally:
        device.close()
