import copy

import numpy as np
import pytest

from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter


def test_tt_direct_calculator_matches_model_predict(direct_regressor, mptraj_10_systems_db):
    from orb_models.forcefield.inference.tt_calculator import TTDirectCalculator
    from orb_models.forcefield.tt import TTBackend, TTDirectForcefieldRegressor
    from orb_models.forcefield.tt.backend import TTDevice

    atoms = mptraj_10_systems_db.get_atoms(1)
    model = copy.deepcopy(direct_regressor)
    model.eval()

    tt_model = TTDirectForcefieldRegressor(
        model=model,
        tt_device=TTDevice(backend=TTBackend.SIMULATOR),
    )
    calc = TTDirectCalculator(
        tt_model,
        atoms_adapter=ForcefieldAtomsAdapter(6.0, 20),
    )

    try:
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress(voigt=True)

        graph = calc.adapter.from_ase_atoms(atoms, device="cpu").to("cpu")
        out = calc.model.predict(graph)

        assert np.allclose(
            energy,
            out["energy"].detach().cpu().numpy(),
            atol=5e-2,
            rtol=5e-2,
        )
        assert np.allclose(
            forces,
            out["forces"].detach().cpu().numpy(),
            atol=5e-2,
            rtol=5e-2,
        )
        assert np.allclose(
            stress,
            out["stress"].detach().cpu().numpy().squeeze(),
            atol=5e-2,
            rtol=5e-2,
        )
        assert set(calc.implemented_properties) == {
            "energy",
            "forces",
            "stress",
            "free_energy",
        }
    finally:
        calc.close()


def test_tt_direct_calculator_from_pretrained_smoke():
    from orb_models.forcefield.inference.tt_calculator import TTDirectCalculator

    calc = TTDirectCalculator.from_pretrained(
        "orb-v3-direct-20-omat",
        backend="simulator",
        compile=False,
    )
    try:
        assert calc.device == "cpu"
        assert "forces" in calc.implemented_properties
    finally:
        calc.close()


@pytest.mark.tt_hardware
def test_tt_direct_calculator_hardware_smoke(mptraj_10_systems_db):
    from orb_models.forcefield.inference.tt_calculator import TTDirectCalculator

    atoms = mptraj_10_systems_db.get_atoms(1)
    calc = TTDirectCalculator.from_pretrained(
        "orb-v3-direct-20-omat",
        backend="hardware",
        compile=False,
    )
    try:
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        assert np.isfinite(energy)
        assert forces.shape == (len(atoms), 3)
    finally:
        calc.close()
