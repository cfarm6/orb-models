"""Plot NcQEq energy contributions as relative interaction-energy lines.

Run from the repository root:

    python examples/ncqeq/atcne-example/plot_energy_contribution_lines.py

The input CSV must contain total energies and electrostatic energies for the
three NcQEq cases written by ``plot_ncqeq_scan.py``. The short-range
contribution is calculated as ``total_energy - electrostatic_energy``. Each
energy term is shifted by its value at the largest spacing.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_INPUT_CSV = Path(__file__).parent / "ncqeq_spacing_scan.csv"
DEFAULT_OUTPUT = Path(__file__).parent / "ncqeq_energy_contribution_lines.svg"

CASES = (
    (
        "unconstrained",
        "Total-charge constraint",
        "unconstrained_energy_ev",
        "unconstrained_electrostatic_energy_ev",
    ),
    (
        "constrained",
        "Regional constraints (0, 0)",
        "constrained_energy_ev",
        "constrained_electrostatic_energy_ev",
    ),
    (
        "ionic_constrained",
        "Regional constraints (+1, -1)",
        "ionic_constrained_energy_ev",
        "ionic_constrained_electrostatic_energy_ev",
    ),
)


def _read_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as csv_file:
        return [
            {key: float(value) for key, value in row.items()} for row in csv.DictReader(csv_file)
        ]


def _relative_contributions(
    rows: list[dict[str, float]],
    *,
    total_key: str,
    electrostatic_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = np.array([row[total_key] for row in rows], dtype=float)
    electrostatic = np.array([row[electrostatic_key] for row in rows], dtype=float)
    short_range = total - electrostatic

    return (
        total - total[-1],
        electrostatic - electrostatic[-1],
        short_range - short_range[-1],
    )


def _plot_contribution_lines(
    ax: plt.Axes,
    spacing: np.ndarray,
    *,
    total: np.ndarray,
    electrostatic: np.ndarray,
    short_range: np.ndarray,
) -> None:
    ax.plot(
        spacing,
        short_range,
        "o-",
        color="tab:orange",
        linewidth=1.5,
        markersize=3,
        label="Short-range",
    )
    ax.plot(
        spacing,
        electrostatic,
        "o-",
        color="tab:blue",
        linewidth=1.5,
        markersize=3,
        label="Electrostatic",
    )
    ax.plot(
        spacing,
        total,
        "o-",
        color="black",
        linewidth=1.5,
        markersize=3,
        label="Total",
    )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)


def plot_energy_contribution_lines(rows: list[dict[str, float]], path: Path) -> None:
    spacing = np.array([row["spacing_angstrom"] for row in rows], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True, constrained_layout=True)

    for ax, (_, label, total_key, electrostatic_key) in zip(axes, CASES, strict=True):
        total, electrostatic, short_range = _relative_contributions(
            rows,
            total_key=total_key,
            electrostatic_key=electrostatic_key,
        )
        _plot_contribution_lines(
            ax,
            spacing,
            total=total,
            electrostatic=electrostatic,
            short_range=short_range,
        )
        ax.set_title(label)
        ax.set_ylabel("Relative contribution (eV)")
        ax.grid(alpha=0.3)
        np.testing.assert_allclose(electrostatic + short_range, total, atol=1e-9)

    axes[-1].set_xlabel("Structure spacing (Å)")
    axes[0].legend(loc="best")
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = _read_rows(args.input_csv)
    if not rows:
        raise ValueError(f"No rows found in {args.input_csv}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_energy_contribution_lines(rows, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
