import pytest

pytest.importorskip("torch_sim", reason="torch_sim is required for these tests")

from copy import deepcopy

import torch
import torch_sim as ts
from torch_sim.elastic import full_3x3_to_voigt_6_stress
from torch_sim.models.interface import validate_model_outputs
from torch_sim.testing import (
    CONSISTENCY_SIMSTATES,
    SIMSTATE_GENERATORS,
    assert_model_calculator_consistency,
)

from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter
from orb_models.forcefield.inference.calculator import ORBCalculator
from orb_models.forcefield.inference.orb_torchsim import OrbTorchSimModel

DEVICE = torch.device("cpu")
DTYPE = torch.float64


# Use the TorchSim consistency check harness for external models
@pytest.mark.parametrize("sim_state_name", CONSISTENCY_SIMSTATES)
@pytest.mark.parametrize(
    "edge_method",
    ["knn_scipy", "knn_alchemi"],
)
def test_orb_torchsim_consistency(sim_state_name, edge_method, conservative_regressor):
    adapter = ForcefieldAtomsAdapter(6.0, 120)
    calculator = ORBCalculator(
        model=conservative_regressor,
        atoms_adapter=adapter,
        edge_method=edge_method,
        device=DEVICE,
    )
    sim_model = OrbTorchSimModel(
        conservative_regressor,
        adapter,
        edge_method=edge_method,
        device=DEVICE,
    )

    sim_state = SIMSTATE_GENERATORS[sim_state_name](DEVICE, DTYPE)
    assert_model_calculator_consistency(sim_model, calculator, sim_state)


def test_orb_torchsim_validate_outputs(conservative_regressor):
    adapter = ForcefieldAtomsAdapter(6.0, 120)
    sim_model = OrbTorchSimModel(conservative_regressor, adapter, dtype=DTYPE, device=DEVICE)
    validate_model_outputs(sim_model, device=DEVICE, dtype=DTYPE)


def test_orb_torchsim_batch_matches_ase_calculator(conservative_regressor, mptraj_10_systems_db):
    atoms_list = [mptraj_10_systems_db.get_atoms(i) for i in [1, 2]]
    adapter = ForcefieldAtomsAdapter(6.0, 120)
    sim_model = OrbTorchSimModel(
        deepcopy(conservative_regressor),
        adapter,
        edge_method="knn_alchemi",
        device=DEVICE,
    )
    calculator = ORBCalculator(
        model=deepcopy(conservative_regressor),
        atoms_adapter=adapter,
        edge_method="knn_alchemi",
        device=DEVICE,
    )

    sim_state = ts.io.atoms_to_state(atoms_list, DEVICE, DTYPE)
    batch_results = sim_model(sim_state)

    offset = 0
    for system_idx, atoms in enumerate(atoms_list):
        calculator.calculate(atoms)
        n_atoms = len(atoms)

        torch.testing.assert_close(
            batch_results["energy"][system_idx],
            torch.tensor(calculator.results["energy"], dtype=batch_results["energy"].dtype),
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            batch_results["forces"][offset : offset + n_atoms],
            torch.tensor(calculator.results["forces"], dtype=batch_results["forces"].dtype),
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            full_3x3_to_voigt_6_stress(batch_results["stress"])[system_idx],
            torch.tensor(calculator.results["stress"], dtype=batch_results["stress"].dtype),
            rtol=1e-5,
            atol=1e-5,
            equal_nan=True,
        )
        offset += n_atoms


def test_orb_torchsim_returns_latent_charges(conservative_regressor):
    adapter = ForcefieldAtomsAdapter(6.0, 120)
    sim_model = OrbTorchSimModel(conservative_regressor, adapter, dtype=DTYPE, device=DEVICE)
    sim_state = ts.SimState(
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=DTYPE),
        masses=torch.tensor([1.0, 1.0, 16.0], dtype=DTYPE),
        cell=torch.diag(torch.tensor([5.0, 5.0, 5.0], dtype=DTYPE)).unsqueeze(0),
        pbc=True,
        atomic_numbers=torch.tensor([1, 1, 8]),
        charge=torch.tensor([0.0], dtype=DTYPE),
        spin=torch.tensor([1.0], dtype=DTYPE),
        region_mask=torch.tensor([0, 0, 1]),
        region_charges=torch.tensor([[0.5, -0.5]], dtype=DTYPE),
    )

    results = sim_model(sim_state)

    assert "latent_charges" in results
    assert "coulomb_energy" in results
    assert results["coulomb_energy"].shape == (1,)
    torch.testing.assert_close(results["latent_charges"][:2].sum(), torch.tensor(0.5, dtype=DTYPE))
    torch.testing.assert_close(results["latent_charges"][2:].sum(), torch.tensor(-0.5, dtype=DTYPE))


def test_conservative_stress_disabled(conservative_regressor, mptraj_10_systems_db):
    conservative_regressor.disable_stress()
    atoms_list = [mptraj_10_systems_db.get_atoms(1)]
    adapter = ForcefieldAtomsAdapter(6.0, 120)
    sim_state = ts.io.atoms_to_state(atoms_list, "cpu", torch.get_default_dtype())
    sim_model = OrbTorchSimModel(conservative_regressor, adapter, device=DEVICE)
    results = sim_model(sim_state)
    assert "stress" not in results
    assert "forces" in results


def test_conservative_stress_enabled(conservative_regressor, mptraj_10_systems_db):
    conservative_regressor.disable_stress()
    conservative_regressor.enable_stress()
    atoms_list = [mptraj_10_systems_db.get_atoms(1)]
    adapter = ForcefieldAtomsAdapter(6.0, 120)
    sim_state = ts.io.atoms_to_state(atoms_list, "cpu", torch.get_default_dtype())
    sim_model = OrbTorchSimModel(conservative_regressor, adapter, device=DEVICE)
    results = sim_model(sim_state)
    assert "stress" in results
    assert "forces" in results


def test_direct_stress_disabled(direct_regressor, mptraj_10_systems_db):
    direct_regressor.disable_stress()
    atoms_list = [mptraj_10_systems_db.get_atoms(1)]
    adapter = ForcefieldAtomsAdapter(6.0, 120)
    sim_state = ts.io.atoms_to_state(atoms_list, "cpu", torch.get_default_dtype())
    sim_model = OrbTorchSimModel(direct_regressor, adapter, device=DEVICE)
    results = sim_model(sim_state)
    assert "stress" not in results
    assert "forces" in results


def test_direct_stress_enabled(direct_regressor, mptraj_10_systems_db):
    direct_regressor.disable_stress()
    direct_regressor.enable_stress()
    atoms_list = [mptraj_10_systems_db.get_atoms(1)]
    adapter = ForcefieldAtomsAdapter(6.0, 120)
    sim_state = ts.io.atoms_to_state(atoms_list, "cpu", torch.get_default_dtype())
    sim_model = OrbTorchSimModel(direct_regressor, adapter, device=DEVICE)
    results = sim_model(sim_state)
    assert "stress" in results
    assert "forces" in results
