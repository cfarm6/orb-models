# ruff: noqa: E402, I001
"""Unsafe internal-cutoff diagnostic for the NcQEq spacing behavior.

This script records a failed attempt to preserve inter-fragment interactions by
changing both the adapter radius and the GNS cutoff function to 10 Å at inference
time. It is not a proposed fix: it changes model behavior far outside the loaded
checkpoint's trained/settings-consistent configuration and produces large energy
shifts.

Run from the repository root:

    python examples/ncqeq/failed_attempts/internal_cutoff_diagnostic.py --device cpu
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.io import read

sys.path.insert(0, str(Path(__file__).parents[1]))

from plot_ncqeq_scan import (
    DEFAULT_TEMPLATE,
    N_FIRST_REGION,
    _batched_sim_state,
    _electrostatic_energy,
    _energy,
    _with_fragment_spacing,
)
from orb_models.common.models import gns, nn_util
from orb_models.forcefield import pretrained
from orb_models.forcefield.inference.orb_torchsim import OrbTorchSimModel

SPACINGS = np.arange(2.5, 7.5, step=0.5)
CASES = {
    "total_charge": None,
    "neutral_regional": (0.0, 0.0),
    "ionic_regional": (1.0, -1.0),
}
SCENARIOS = {
    "baseline_adapter6_internal6": {"adapter_radius": 6.0, "internal_cutoff": None},
    "adapter10_internal6": {"adapter_radius": 10.0, "internal_cutoff": 6.0},
    "adapter10_internal10": {"adapter_radius": 10.0, "internal_cutoff": 10.0},
}


def _write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, float | str]], path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True, constrained_layout=True)
    baseline = {
        (row["case"], float(row["spacing_angstrom"])): float(row["energy_ev"])
        for row in rows
        if row["scenario"] == "baseline_adapter6_internal6"
    }
    for scenario in SCENARIOS:
        scenario_rows = [
            row for row in rows if row["scenario"] == scenario and row["case"] == "ionic_regional"
        ]
        axes[0].plot(
            [float(row["spacing_angstrom"]) for row in scenario_rows],
            [float(row["energy_ev"]) for row in scenario_rows],
            "o-",
            label=scenario,
        )
        axes[1].plot(
            [float(row["spacing_angstrom"]) for row in scenario_rows],
            [
                float(row["energy_ev"]) - baseline[(row["case"], float(row["spacing_angstrom"]))]
                for row in scenario_rows
            ],
            "o-",
            label=scenario,
        )
    axes[0].set_ylabel("Ionic energy (eV)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Fragment spacing (Å)")
    axes[1].set_ylabel("Ionic energy diff vs baseline (eV)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.savefig(path)


def _predict_scenario(
    *,
    scenario_name: str,
    adapter_radius: float,
    internal_cutoff: float | None,
    atoms_list,
    device_arg: str,
) -> list[dict[str, float | str]]:
    original_get_cutoff = gns.get_cutoff
    if internal_cutoff is None:
        gns.get_cutoff = original_get_cutoff
    else:
        gns.get_cutoff = lambda distances: nn_util.get_cutoff(distances, r_max=internal_cutoff)
    try:
        model, atoms_adapter = pretrained.orbmol_v2(
            device=device_arg,
            compile=False,
            precision="float64",
        )
        atoms_adapter.radius = adapter_radius
        calculator = OrbTorchSimModel(
            model,
            atoms_adapter,
            device=device_arg,
            dtype=torch.get_default_dtype(),
        )
        device = calculator.device
        dtype = calculator.dtype or torch.get_default_dtype()
        n_atoms = len(atoms_list[0])
        rows = []
        for case_name, regional_charges in CASES.items():
            state = _batched_sim_state(
                atoms_list,
                device=device,
                dtype=dtype,
                regional_charges=regional_charges,
            )
            with torch.enable_grad():
                results = calculator(state)
            energies = _energy(results).detach().cpu().reshape(-1).numpy()
            electrostatic = _electrostatic_energy(results).detach().cpu().reshape(-1).numpy()
            charges = results["latent_charges"].detach().cpu().reshape(len(atoms_list), n_atoms)
            first_region_charge = charges[:, :N_FIRST_REGION].sum(dim=1).numpy()
            for index, spacing in enumerate(SPACINGS):
                rows.append(
                    {
                        "scenario": scenario_name,
                        "adapter_radius_angstrom": adapter_radius,
                        "internal_cutoff_angstrom": internal_cutoff or 6.0,
                        "case": case_name,
                        "spacing_angstrom": spacing,
                        "energy_ev": energies[index],
                        "electrostatic_energy_ev": electrostatic[index],
                        "first_region_charge_e": first_region_charge[index],
                    }
                )
        return rows
    finally:
        gns.get_cutoff = original_get_cutoff


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    template = read(args.template)
    atoms_list = [_with_fragment_spacing(template, spacing) for spacing in SPACINGS]
    rows = []
    for scenario_name, settings in SCENARIOS.items():
        rows.extend(
            _predict_scenario(
                scenario_name=scenario_name,
                adapter_radius=settings["adapter_radius"],
                internal_cutoff=settings["internal_cutoff"],
                atoms_list=atoms_list,
                device_arg=args.device,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "internal_cutoff_diagnostic.csv"
    plot_path = args.output_dir / "internal_cutoff_diagnostic.svg"
    _write_csv(rows, csv_path)
    _plot(rows, plot_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
