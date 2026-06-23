"""ASE calculator for Tenstorrent-accelerated direct Orb forcefield models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from ase.calculators.calculator import Calculator, all_changes

from orb_models.common.atoms.abstract_atoms_adapter import AbstractAtomsAdapter
from orb_models.common.atoms.graph_featurization import EdgeCreationMethod
from orb_models.common.models.nn_util import ChargeSpinConditioner
from orb_models.common.torch_utils import to_numpy
from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter
from orb_models.forcefield.tt.backend import TTBackend

if TYPE_CHECKING:
    from orb_models.forcefield.tt.direct_regressor import TTDirectForcefieldRegressor


class TTDirectCalculator(Calculator):
    """ASE calculator backed by a Tenstorrent direct Orb forcefield model.

    Graph featurization stays on CPU; eligible linear/MLP layers run on TT
    hardware or the TT simulator through :class:`TTDirectForcefieldRegressor`.
    """

    def __init__(
        self,
        model: TTDirectForcefieldRegressor,
        atoms_adapter: AbstractAtomsAdapter,
        *,
        edge_method: EdgeCreationMethod | None = None,
        max_num_neighbors: int | None = None,
        half_supercell: bool | None = None,
        directory: str = ".",
    ) -> None:
        Calculator.__init__(self, directory=directory)
        self.results: dict = {}
        self.model = model
        self.adapter = atoms_adapter
        self.max_num_neighbors = max_num_neighbors
        self.edge_method = edge_method
        self.half_supercell = half_supercell
        self.device = "cpu"
        self._closed = False

        conditioner = getattr(self.model.model.model, "conditioner", None)
        self.expects_charge_and_spin = (conditioner is not None) and isinstance(
            conditioner, ChargeSpinConditioner
        )
        self.implemented_properties = list(self.model.properties)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        backend: TTBackend | Literal["auto"] = "auto",
        device_id: int = 0,
        precision: str = "float32-high",
        compile: bool | None = False,
        atoms_adapter: ForcefieldAtomsAdapter | None = None,
        edge_method: EdgeCreationMethod | None = None,
        max_num_neighbors: int | None = None,
        half_supercell: bool | None = None,
        directory: str = ".",
        **loader_kwargs,
    ) -> TTDirectCalculator:
        """Load a direct pretrained model on TT and return an ASE calculator."""

        from orb_models.extensions.tt import load_tt_direct_model

        model, loaded_adapter = load_tt_direct_model(
            model_name,
            backend=backend,
            device_id=device_id,
            precision=precision,
            compile=compile,
            **loader_kwargs,
        )
        return cls(
            model,
            atoms_adapter=atoms_adapter or loaded_adapter,
            edge_method=edge_method,
            max_num_neighbors=max_num_neighbors,
            half_supercell=half_supercell,
            directory=directory,
        )

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """Calculate energy, forces, and stress for ``atoms``."""

        Calculator.calculate(self, atoms)

        if self.expects_charge_and_spin and (
            ("charge" not in atoms.info) or ("spin" not in atoms.info)
        ):
            raise ValueError("atoms.info must contain both 'charge' and 'spin'")

        batch = self.adapter.from_ase_atoms(
            atoms=atoms,
            max_num_neighbors=self.max_num_neighbors,
            edge_method=self.edge_method,
            half_supercell=self.half_supercell,
            device=self.device,
        )
        batch = batch.to(self.device)
        out = self.model.predict(batch)
        self._update_results(out)

    def _update_results(self, out: dict[str, torch.Tensor]) -> None:
        """Populate ``self.results`` in the ASE-expected layout."""

        self.results = {}
        model = self.model.model
        no_direct_energy_head = "energy" not in model.heads
        no_direct_force_head = "forces" not in model.heads
        no_direct_stress_head = "stress" not in model.heads

        for property_name in self.implemented_properties:
            if property_name == "free_energy" and no_direct_energy_head:
                continue
            if property_name == "forces" and no_direct_force_head:
                continue
            if property_name == "stress" and no_direct_stress_head:
                continue

            output_key = "energy" if property_name == "free_energy" else property_name
            if property_name == "stress":
                self.results[property_name] = to_numpy(out[output_key].squeeze())
            else:
                self.results[property_name] = to_numpy(out[output_key])

    def close(self) -> None:
        """Release the Tenstorrent device held by the wrapped model."""

        if self._closed:
            return
        self.model.close()
        self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
