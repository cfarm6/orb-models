"""Minimal Tenstorrent direct-model inference example.

Run on the TT simulator:
    python examples/TTDirectInference.py --backend simulator

Run on hardware:
    python examples/TTDirectInference.py --backend hardware
"""

from __future__ import annotations

import argparse
from typing import Literal, cast

import torch
from ase.build import bulk

TTBackendArg = Literal["auto", "simulator", "hardware"]


def build_atoms(repeats: int):
    """Build a small periodic copper crystal for direct forcefield inference."""

    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((repeats, repeats, repeats))
    atoms.set_pbc([True, True, True])
    return atoms


def format_scalar(value: torch.Tensor) -> str:
    flat = value.detach().cpu().reshape(-1)
    if flat.numel() == 0:
        return "N/A"
    return f"{float(flat[0]):.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="orb-v3-direct-20-omat",
        choices=(
            "orbmol-v1-direct",
            "orb-v3-direct-20-omat",
            "orb-v3-direct-inf-omat",
            "orb-v3-direct-20-mpa",
            "orb-v3-direct-inf-mpa",
        ),
        help="Direct Orb model to run on Tenstorrent.",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "simulator", "hardware"),
        help="Tenstorrent backend. 'auto' uses hardware when /dev/tenstorrent/0 exists, otherwise simulator.",
    )
    parser.add_argument(
        "--repeats",
        default=20,
        type=int,
        help="Crystal repeat count along each axis.",
    )
    args = parser.parse_args()

    from orb_models.extensions.tt import load_tt_direct_model

    atoms = build_atoms(args.repeats)
    if args.model == "orbmol-v1-direct":
        atoms.info["charge"] = 0.0
        atoms.info["spin"] = 1.0
    backend = cast(TTBackendArg, args.backend)
    model, atoms_adapter = load_tt_direct_model(args.model, backend=backend, compile=False)

    try:
        batch = atoms_adapter.from_ase_atoms(atoms, device="cpu").to("cpu")  # type: ignore[attr-defined]
        prediction = model.predict(batch)
    finally:
        model.close()

    print(f"model: {args.model}")
    print(f"backend: {args.backend}")
    print(f"atoms: {len(atoms)}")
    if "energy" in prediction:
        print(f"energy: {format_scalar(prediction['energy'])}")
    if "forces" in prediction:
        forces = prediction["forces"].detach().cpu()
        print(f"forces shape: {tuple(forces.shape)}")
        print(f"max |force|: {float(torch.max(torch.abs(forces))):.6f}")
    if "stress" in prediction:
        print(f"stress shape: {tuple(prediction['stress'].shape)}")


if __name__ == "__main__":
    main()
