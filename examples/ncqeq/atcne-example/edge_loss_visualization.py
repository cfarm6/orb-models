"""Visualize which fragment atoms lose cross-fragment graph edges near 6 Å.

Run from the repository root:

    PYTHONPATH=$PWD python examples/ncqeq/atcne-example/edge_loss_visualization.py

The figure projects the structures onto the fragment-separation axis and a
perpendicular principal axis. Orange lines are unique cross-fragment graph edges
from the OrbMol-v2/ForcefieldAtomsAdapter graph. Marker sizes scale with the
number of directed cross-fragment edges touching each atom.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.io import read
from plot_ncqeq_scan import (
    DEFAULT_TEMPLATE,
    N_FIRST_REGION,
    _batched_sim_state,
    _with_fragment_spacing,
)

from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter

DEFAULT_SPACINGS = [5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0]
DEFAULT_OUTPUT = Path(__file__).with_name("edge_loss_visualization.svg")
DEFAULT_PNG_OUTPUT = Path(__file__).with_name("edge_loss_visualization.png")
DEFAULT_CSV = Path(__file__).with_name("edge_loss_visualization_edges.csv")


def _parse_spacings(values: Iterable[str]) -> list[float]:
    spacings = [float(value) for value in values]
    if not spacings:
        raise ValueError("At least one spacing is required.")
    return spacings


def _projection_basis(atoms) -> tuple[np.ndarray, np.ndarray]:
    positions = atoms.positions
    first_centroid = positions[:N_FIRST_REGION].mean(axis=0)
    second_centroid = positions[N_FIRST_REGION:].mean(axis=0)
    separation_axis = second_centroid - first_centroid
    separation_axis /= np.linalg.norm(separation_axis)

    centered = positions - positions.mean(axis=0)
    perpendicular = centered - np.outer(centered @ separation_axis, separation_axis)
    _, singular_values, vh = np.linalg.svd(perpendicular, full_matrices=False)
    if singular_values[0] > 1e-8:
        perpendicular_axis = vh[0]
    else:
        trial = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(trial, separation_axis)) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        perpendicular_axis = trial - np.dot(trial, separation_axis) * separation_axis
        perpendicular_axis /= np.linalg.norm(perpendicular_axis)
    return separation_axis, perpendicular_axis


def _project_positions(
    atoms,
    *,
    separation_axis: np.ndarray,
    perpendicular_axis: np.ndarray,
) -> np.ndarray:
    origin = atoms.positions[:N_FIRST_REGION].mean(axis=0)
    centered = atoms.positions - origin
    return np.column_stack((centered @ separation_axis, centered @ perpendicular_axis))


def _cross_edges_for_spacings(
    atoms_list,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, object]]:
    adapter = ForcefieldAtomsAdapter(
        radius=6.0,
        max_num_neighbors=120,
        extra_features={"graph": ["total_charge", "spin_multiplicity"]},
    )
    state = _batched_sim_state(
        atoms_list,
        device=device,
        dtype=dtype,
        regional_charges=None,
    )
    batch = adapter.from_torchsim_state(state, device=device, output_dtype=dtype)
    senders = batch.senders.detach().cpu().numpy()
    receivers = batch.receivers.detach().cpu().numpy()
    n_atoms = len(atoms_list[0])

    systems = []
    for system_index, atoms in enumerate(atoms_list):
        atom_start = system_index * n_atoms
        atom_stop = atom_start + n_atoms
        in_system = (senders >= atom_start) & (senders < atom_stop)
        local_senders = senders[in_system] - atom_start
        local_receivers = receivers[in_system] - atom_start
        is_cross = (
            (local_senders < N_FIRST_REGION) & (local_receivers >= N_FIRST_REGION)
        ) | ((local_senders >= N_FIRST_REGION) & (local_receivers < N_FIRST_REGION))
        cross_senders = local_senders[is_cross]
        cross_receivers = local_receivers[is_cross]
        unique_pairs = sorted({tuple(sorted((int(i), int(j)))) for i, j in zip(cross_senders, cross_receivers, strict=True)})
        directed_degree = np.zeros(n_atoms, dtype=int)
        for sender, receiver in zip(cross_senders, cross_receivers, strict=True):
            directed_degree[int(sender)] += 1
            directed_degree[int(receiver)] += 1
        systems.append(
            {
                "atoms": atoms,
                "directed_edges": list(zip(cross_senders.tolist(), cross_receivers.tolist(), strict=True)),
                "unique_pairs": unique_pairs,
                "directed_degree": directed_degree,
            }
        )
    return systems


def _write_edge_csv(
    systems: list[dict[str, object]],
    spacings: list[float],
    path: Path,
) -> None:
    rows = []
    for spacing, system in zip(spacings, systems, strict=True):
        atoms = system["atoms"]
        positions = atoms.positions
        for atom_i, atom_j in system["unique_pairs"]:
            rows.append(
                {
                    "spacing_angstrom": spacing,
                    "atom_i": atom_i,
                    "atom_j": atom_j,
                    "atom_i_symbol": atoms[atom_i].symbol,
                    "atom_j_symbol": atoms[atom_j].symbol,
                    "distance_angstrom": float(np.linalg.norm(positions[atom_i] - positions[atom_j])),
                }
            )
    fieldnames = [
        "spacing_angstrom",
        "atom_i",
        "atom_j",
        "atom_i_symbol",
        "atom_j_symbol",
        "distance_angstrom",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    systems: list[dict[str, object]],
    spacings: list[float],
    path: Path,
    png_path: Path | None = None,
) -> None:
    separation_axis, perpendicular_axis = _projection_basis(systems[0]["atoms"])
    projected = [
        _project_positions(
            system["atoms"],
            separation_axis=separation_axis,
            perpendicular_axis=perpendicular_axis,
        )
        for system in systems
    ]
    all_projected = np.concatenate(projected)
    x_min, y_min = all_projected.min(axis=0) - np.array([0.4, 0.4])
    x_max, y_max = all_projected.max(axis=0) + np.array([0.4, 0.4])

    n_panels = len(spacings)
    n_cols = min(4, n_panels)
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.9 * n_cols, 3.6 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    for axis, spacing, system, coords in zip(axes.ravel(), spacings, systems, projected, strict=False):
        unique_pairs = system["unique_pairs"]
        directed_edges = system["directed_edges"]
        directed_degree = system["directed_degree"]
        atoms = system["atoms"]

        for atom_i, atom_j in unique_pairs:
            xy = coords[[atom_i, atom_j]]
            axis.plot(xy[:, 0], xy[:, 1], color="tab:orange", alpha=0.42, linewidth=1.2, zorder=1)

        first = np.arange(N_FIRST_REGION)
        second = np.arange(N_FIRST_REGION, len(atoms))
        active = directed_degree > 0
        marker_sizes = 28 + 11 * directed_degree
        axis.scatter(
            coords[first, 0],
            coords[first, 1],
            s=marker_sizes[first],
            c="tab:blue",
            edgecolors=np.where(active[first], "black", "white"),
            linewidths=np.where(active[first], 1.0, 0.4),
            label="fragment 1" if spacing == spacings[0] else None,
            zorder=3,
        )
        axis.scatter(
            coords[second, 0],
            coords[second, 1],
            s=marker_sizes[second],
            c="tab:red",
            edgecolors=np.where(active[second], "black", "white"),
            linewidths=np.where(active[second], 1.0, 0.4),
            label="fragment 2" if spacing == spacings[0] else None,
            zorder=3,
        )

        for atom_index in np.flatnonzero(active):
            axis.annotate(
                str(atom_index),
                xy=coords[atom_index],
                xytext=(2, 2),
                textcoords="offset points",
                fontsize=7,
                color="black",
                zorder=4,
            )

        axis.set_title(
            f"{spacing:g} Å: {len(directed_edges)} directed / {len(unique_pairs)} unique edges"
        )
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
        axis.set_xlabel("fragment-separation projection (Å)")
        axis.set_ylabel("perpendicular projection (Å)")

    for axis in axes.ravel()[n_panels:]:
        axis.axis("off")

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(
        "OrbMol-v2 graph edge loss between ATCNE fragments near the 6 Å adapter cutoff",
        y=1.02,
    )
    fig.savefig(path, bbox_inches="tight")
    if png_path is not None:
        fig.savefig(png_path, dpi=250, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--spacings", nargs="*", default=[str(v) for v in DEFAULT_SPACINGS])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--png-output", type=Path, default=DEFAULT_PNG_OUTPUT)
    parser.add_argument("--edge-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    spacings = _parse_spacings(args.spacings)
    template = read(args.template)
    atoms_list = [_with_fragment_spacing(template, spacing) for spacing in spacings]
    systems = _cross_edges_for_spacings(
        atoms_list,
        device=torch.device(args.device),
        dtype=torch.float64,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _plot(systems, spacings, args.output, args.png_output)
    _write_edge_csv(systems, spacings, args.edge_csv)

    print(f"Wrote {args.output}")
    print(f"Wrote {args.png_output}")
    print(f"Wrote {args.edge_csv}")
    for spacing, system in zip(spacings, systems, strict=True):
        print(
            f"{spacing:g} Å: {len(system['directed_edges'])} directed edges, "
            f"{len(system['unique_pairs'])} unique cross-fragment pairs"
        )


if __name__ == "__main__":
    main()
