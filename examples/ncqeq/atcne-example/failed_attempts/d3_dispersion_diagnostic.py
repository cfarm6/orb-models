# ruff: noqa: E402, I001
"""Diagnose adding a nvalchemiops D3 dispersion correction to the NcQEq scan.

This is a post-hoc additive dispersion correction for the generated spacing sweep.
D3 depends on geometry and atomic numbers, not on the NcQEq charge-constraint
case, so the same D3 energy is added to U, D0A0, D+A-, and U1 at each spacing.

Run from the repository root:

    PYTHONPATH=$PWD python examples/ncqeq/failed_attempts/d3_dispersion_diagnostic.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_dftd
from ase.io import read
from nvalchemiops.torch.interactions.dispersion import D3Parameters
from torch_dftd.dftd3_xc_params import get_dftd3_default_params
from torch_sim.models.dispersion import D3DispersionModel

sys.path.insert(0, str(Path(__file__).parents[1]))

from plot_ncqeq_scan import (
    DEFAULT_SPACING_START,
    DEFAULT_SPACING_STEP,
    DEFAULT_SPACING_STOP,
    DEFAULT_TEMPLATE,
    _batched_sim_state,
    _spacing_values,
    _with_fragment_spacing,
)

DEFAULT_BASE_CSV = Path(__file__).parents[1] / "ncqeq_spacing_scan.csv"
DEFAULT_OUTPUT_CSV = Path(__file__).with_suffix(".csv")
DEFAULT_OUTPUT_PLOT = Path(__file__).with_suffix(".svg")
DEFAULT_OUTPUT_SUMMARY = Path(__file__).with_suffix(".summary.json")

ENERGY_COLUMNS = (
    "unconstrained_energy_ev",
    "constrained_energy_ev",
    "ionic_constrained_energy_ev",
    "u1_energy_ev",
)


def _load_d3_parameters(*, device: torch.device, dtype: torch.dtype) -> D3Parameters:
    """Load Grimme D3 reference tensors shipped with torch-dftd.

    The actual D3 energy is evaluated with nvalchemiops through TorchSim's
    D3DispersionModel wrapper, which handles Angstrom/eV unit conversion.
    """
    params_path = Path(torch_dftd.__file__).parent / "nn" / "params" / "dftd3_params.npz"
    params = np.load(params_path)
    c6_pack = params["c6ab"]
    return D3Parameters(
        rcov=torch.as_tensor(params["rcov"], dtype=dtype, device=device),
        r4r2=torch.as_tensor(params["r2r4"], dtype=dtype, device=device),
        c6ab=torch.as_tensor(c6_pack[..., 0], dtype=dtype, device=device),
        cn_ref=torch.as_tensor(c6_pack[..., 1], dtype=dtype, device=device),
    )


def _read_rows(path: Path) -> list[dict[str, float]]:
    with path.open() as csv_file:
        reader = csv.DictReader(csv_file)
        return [{key: float(value) for key, value in row.items()} for row in reader]


def _write_rows(rows: list[dict[str, float]], path: Path) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _d3_energies(
    atoms_list,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    cutoff: float,
    xc: str,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    bj_params = get_dftd3_default_params(damping="bj", xc=xc)
    d3_model = D3DispersionModel(
        a1=bj_params["rs6"],
        a2=bj_params["rs18"],
        s8=bj_params["s18"],
        s6=bj_params["s6"],
        d3_params=_load_d3_parameters(device=device, dtype=dtype),
        cutoff=cutoff,
        device=device,
        dtype=dtype,
        compute_forces=False,
        compute_stress=False,
    )

    energy_chunks = []
    for start in range(0, len(atoms_list), batch_size):
        atoms_chunk = atoms_list[start : start + batch_size]
        state = _batched_sim_state(
            atoms_chunk,
            device=device,
            dtype=dtype,
            regional_charges=None,
        )
        with torch.no_grad():
            result = d3_model(state)
        energy_chunks.append(result["energy"].detach().cpu().reshape(-1).numpy())
    return np.concatenate(energy_chunks)


def _add_dispersion_columns(rows: list[dict[str, float]], d3_energy: np.ndarray) -> None:
    for row, dispersion_energy in zip(rows, d3_energy, strict=True):
        row["d3_dispersion_energy_ev"] = float(dispersion_energy)
        for column in ENERGY_COLUMNS:
            row[f"{column.removesuffix('_ev')}_plus_d3_ev"] = row[column] + float(
                dispersion_energy
            )


def _relative(values: np.ndarray) -> np.ndarray:
    return values - values[-1]


def _step_stats(spacing: np.ndarray, values: np.ndarray) -> dict[str, float]:
    window = (spacing >= 5.0) & (spacing <= 6.0)
    window_values = values[window]
    window_spacing = spacing[window]
    steps = np.abs(np.diff(window_values))
    max_step_index = int(np.argmax(steps)) if len(steps) else 0
    return {
        "range_5_to_6_ev": float(np.max(window_values) - np.min(window_values)),
        "max_abs_step_5_to_6_ev": float(steps[max_step_index]) if len(steps) else 0.0,
        "max_step_start_angstrom": float(window_spacing[max_step_index]) if len(steps) else float("nan"),
        "max_step_end_angstrom": float(window_spacing[max_step_index + 1]) if len(steps) else float("nan"),
    }


def _value_at_spacing(spacing: np.ndarray, values: np.ndarray, target: float) -> float:
    index = int(np.argmin(np.abs(spacing - target)))
    return float(values[index])


def _summary(rows: list[dict[str, float]], *, xc: str, cutoff: float) -> dict[str, Any]:
    spacing = np.array([row["spacing_angstrom"] for row in rows])
    d3 = np.array([row["d3_dispersion_energy_ev"] for row in rows])
    ionic = _relative(np.array([row["ionic_constrained_energy_ev"] for row in rows]))
    ionic_d3 = _relative(
        np.array([row["ionic_constrained_energy_plus_d3_ev"] for row in rows])
    )
    d3_relative = _relative(d3)
    return {
        "method": f"D3(BJ)/{xc} via nvalchemiops-toolkit",
        "d3_cutoff_angstrom": cutoff,
        "d3_energy_min_ev": float(np.min(d3)),
        "d3_energy_max_ev": float(np.max(d3)),
        "d3_relative_range_ev": float(np.max(d3_relative) - np.min(d3_relative)),
        "d3_relative_5_to_6_change_ev": (
            _value_at_spacing(spacing, d3_relative, 6.0)
            - _value_at_spacing(spacing, d3_relative, 5.0)
        ),
        "ionic_unmodified": _step_stats(spacing, ionic),
        "ionic_plus_d3": _step_stats(spacing, ionic_d3),
        "note": (
            "D3 is geometry-only here, so it shifts all charge-constraint cases by the "
            "same spacing-dependent amount and does not change V."
        ),
    }


def _plot(rows: list[dict[str, float]], path: Path) -> None:
    spacing = np.array([row["spacing_angstrom"] for row in rows])
    d3 = np.array([row["d3_dispersion_energy_ev"] for row in rows])
    d3_relative = _relative(d3)

    labels = {
        "unconstrained_energy_ev": "U total-charge",
        "constrained_energy_ev": "D0A0 (0, 0)",
        "ionic_constrained_energy_ev": "D+A- (+1, -1)",
        "u1_energy_ev": "U1",
    }
    fig, axes = plt.subplots(4, 1, figsize=(7, 13), sharex=True, constrained_layout=True)

    for column, label in labels.items():
        axes[0].plot(spacing, _relative(np.array([row[column] for row in rows])), "o-", label=label)
    axes[0].set_ylabel("Relative energy (eV)")
    axes[0].set_title("Original OrbMol-v2/NcQEq curves")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    for column, label in labels.items():
        d3_column = f"{column.removesuffix('_ev')}_plus_d3_ev"
        axes[1].plot(
            spacing,
            _relative(np.array([row[d3_column] for row in rows])),
            "o-",
            label=f"{label} + D3",
        )
    axes[1].set_ylabel("Relative energy (eV)")
    axes[1].set_title("Post-hoc D3-corrected curves")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(spacing, d3, "o-", label="absolute D3")
    axes[2].set_ylabel("D3 energy (eV)")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="best")

    axes[3].plot(spacing, d3_relative, "o-", label="D3 relative to 7 Å")
    axes[3].plot(
        spacing,
        _relative(np.array([row["ionic_constrained_energy_ev"] for row in rows])),
        "o-",
        label="D+A- original",
    )
    axes[3].plot(
        spacing,
        _relative(np.array([row["ionic_constrained_energy_plus_d3_ev"] for row in rows])),
        "o-",
        label="D+A- + D3",
    )
    axes[3].set_xlabel("Structure spacing (Å)")
    axes[3].set_ylabel("Relative energy (eV)")
    axes[3].set_title("D3 is smooth through the 5-6 Å region")
    axes[3].grid(alpha=0.3)
    axes[3].legend()

    fig.savefig(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-csv", type=Path, default=DEFAULT_BASE_CSV)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--spacing-start", type=float, default=DEFAULT_SPACING_START)
    parser.add_argument("--spacing-stop", type=float, default=DEFAULT_SPACING_STOP)
    parser.add_argument("--spacing-step", type=float, default=DEFAULT_SPACING_STEP)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cutoff", type=float, default=12.0)
    parser.add_argument("--xc", default="pbe", help="D3(BJ) functional parameters from torch-dftd.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-plot", type=Path, default=DEFAULT_OUTPUT_PLOT)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_OUTPUT_SUMMARY)
    args = parser.parse_args()

    rows = _read_rows(args.base_csv)
    spacing_values = _spacing_values(args.spacing_start, args.spacing_stop, args.spacing_step)
    row_spacing = [row["spacing_angstrom"] for row in rows]
    if len(row_spacing) != len(spacing_values) or not np.allclose(row_spacing, spacing_values):
        raise ValueError(
            f"Base CSV spacing values do not match generated sweep: {args.base_csv}"
        )

    device = torch.device(args.device)
    dtype = torch.float64
    template_atoms = read(args.template)
    atoms_list = [_with_fragment_spacing(template_atoms, spacing) for spacing in spacing_values]
    d3_energy = _d3_energies(
        atoms_list,
        device=device,
        dtype=dtype,
        batch_size=args.batch_size,
        cutoff=args.cutoff,
        xc=args.xc,
    )
    _add_dispersion_columns(rows, d3_energy)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, args.output_csv)
    _plot(rows, args.output_plot)
    summary = _summary(rows, xc=args.xc, cutoff=args.cutoff)
    args.output_summary.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_plot}")
    print(f"Wrote {args.output_summary}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
