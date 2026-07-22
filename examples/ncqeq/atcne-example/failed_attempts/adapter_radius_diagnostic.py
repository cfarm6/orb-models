# ruff: noqa: E402, I001
"""Diagnose whether increasing ForcefieldAtomsAdapter.radius fixes NcQEq spacing behavior.

This script intentionally records a failed inference-only attempt: increasing the
adapter radius preserves more cross-fragment graph edges, but OrbMol-v2 outputs
remain unchanged because the model's internal learned cutoff still gates those
edges at 6 Å.

Run from the repository root:

    python examples/ncqeq/failed_attempts/adapter_radius_diagnostic.py --device cpu
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
from orb_models.forcefield import pretrained
from orb_models.forcefield.inference.orb_torchsim import OrbTorchSimModel

SPACINGS = [5.0, 5.2, 5.4, 5.5, 5.6, 5.8, 6.0, 6.2]
RADII = [6.0, 6.5, 8.0, 10.0]
CASES = {
    "total_charge": None,
    "neutral_regional": (0.0, 0.0),
    "ionic_regional": (1.0, -1.0),
}


def _cross_fragment_edge_counts(batch, n_atoms: int, n_systems: int) -> list[int]:
    senders = batch.senders.detach().cpu().numpy()
    receivers = batch.receivers.detach().cpu().numpy()
    counts = []
    for system_idx in range(n_systems):
        start = system_idx * n_atoms
        stop = start + n_atoms
        in_system = (senders >= start) & (senders < stop)
        local_senders = senders[in_system] - start
        local_receivers = receivers[in_system] - start
        cross_edges = ((local_senders < N_FIRST_REGION) & (local_receivers >= N_FIRST_REGION)) | (
            (local_senders >= N_FIRST_REGION) & (local_receivers < N_FIRST_REGION)
        )
        counts.append(int(cross_edges.sum()))
    return counts


def _write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, float | str]], path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True, constrained_layout=True)

    for radius in RADII:
        edge_rows = [
            row
            for row in rows
            if row["case"] == "ionic_regional" and float(row["adapter_radius_angstrom"]) == radius
        ]
        axes[0].plot(
            [float(row["spacing_angstrom"]) for row in edge_rows],
            [float(row["cross_fragment_edges"]) for row in edge_rows],
            "o-",
            label=f"adapter radius {radius:g} Å",
        )

    baseline = {
        float(row["spacing_angstrom"]): float(row["energy_ev"])
        for row in rows
        if row["case"] == "ionic_regional" and float(row["adapter_radius_angstrom"]) == 6.0
    }
    for radius in RADII:
        energy_rows = [
            row
            for row in rows
            if row["case"] == "ionic_regional" and float(row["adapter_radius_angstrom"]) == radius
        ]
        axes[1].plot(
            [float(row["spacing_angstrom"]) for row in energy_rows],
            [
                float(row["energy_ev"]) - baseline[float(row["spacing_angstrom"])]
                for row in energy_rows
            ],
            "o-",
            label=f"adapter radius {radius:g} Å",
        )

    axes[0].set_ylabel("Cross-fragment directed edges")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Fragment spacing (Å)")
    axes[1].set_ylabel("Ionic energy diff vs 6 Å adapter (eV)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.savefig(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    template = read(args.template)
    atoms_list = [_with_fragment_spacing(template, spacing) for spacing in SPACINGS]
    n_atoms = len(atoms_list[0])

    model, atoms_adapter = pretrained.orbmol_v2(
        device=args.device,
        compile=False,
        precision="float64",
    )
    rows = []
    for radius in RADII:
        atoms_adapter.radius = radius
        calculator = OrbTorchSimModel(
            model,
            atoms_adapter,
            device=args.device,
            dtype=torch.get_default_dtype(),
        )
        device = calculator.device
        dtype = calculator.dtype or torch.get_default_dtype()
        for case_name, regional_charges in CASES.items():
            state = _batched_sim_state(
                atoms_list,
                device=device,
                dtype=dtype,
                regional_charges=regional_charges,
            )
            batch = atoms_adapter.from_torchsim_state(
                state,
                device=device,
                output_dtype=dtype,
            )
            cross_edges = _cross_fragment_edge_counts(batch, n_atoms, len(atoms_list))
            with torch.enable_grad():
                results = calculator(state)
            energies = _energy(results).detach().cpu().reshape(-1).numpy()
            electrostatic = _electrostatic_energy(results).detach().cpu().reshape(-1).numpy()
            charges = results["latent_charges"].detach().cpu().reshape(len(atoms_list), n_atoms)
            first_region_charge = charges[:, :N_FIRST_REGION].sum(dim=1).numpy()
            for index, spacing in enumerate(SPACINGS):
                rows.append(
                    {
                        "adapter_radius_angstrom": radius,
                        "case": case_name,
                        "spacing_angstrom": spacing,
                        "cross_fragment_edges": cross_edges[index],
                        "energy_ev": energies[index],
                        "electrostatic_energy_ev": electrostatic[index],
                        "first_region_charge_e": first_region_charge[index],
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "adapter_radius_diagnostic.csv"
    plot_path = args.output_dir / "adapter_radius_diagnostic.svg"
    _write_csv(rows, csv_path)
    _plot(rows, plot_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
