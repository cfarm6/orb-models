"""ASE calculator example for Tenstorrent direct Orb forcefield models.

Run on the TT simulator:
    python examples/TTDirectASE.py --backend simulator

Run on hardware:
    python examples/TTDirectASE.py --backend hardware

Relax a rattled copper supercell with BFGS:
    python examples/TTDirectASE.py --backend hardware --relax --fmax 0.05
"""

from __future__ import annotations

import argparse
from typing import Literal, cast

import numpy as np
from ase.build import bulk
from ase.optimize import BFGS

TTBackendArg = Literal["auto", "simulator", "hardware"]


def build_atoms(repeats: int):
    """Build a small periodic copper crystal for direct forcefield inference."""

    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((repeats, repeats, repeats))
    atoms.set_pbc([True, True, True])
    return atoms


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
        default=2,
        type=int,
        help="Crystal repeat count along each axis.",
    )
    parser.add_argument(
        "--relax",
        action="store_true",
        help="Run a short BFGS geometry optimization after the initial evaluation.",
    )
    parser.add_argument(
        "--fmax",
        default=0.05,
        type=float,
        help="Force convergence threshold for BFGS (eV/Å).",
    )
    parser.add_argument(
        "--max-steps",
        default=20,
        type=int,
        help="Maximum BFGS steps when --relax is set.",
    )
    args = parser.parse_args()

    from orb_models.extensions.tt import TTDirectCalculator

    atoms = build_atoms(args.repeats)
    if args.model == "orbmol-v1-direct":
        atoms.info["charge"] = 0.0
        atoms.info["spin"] = 1.0
    backend = cast(TTBackendArg, args.backend)
    calc = TTDirectCalculator.from_pretrained(
        args.model,
        backend=backend,
        compile=False,
    )

    try:
        atoms.calc = calc
        print(f"model: {args.model}")
        print(f"backend: {args.backend}")
        print(f"atoms: {len(atoms)}")

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        print(f"energy: {energy:.6f}")
        print(f"forces shape: {forces.shape}")
        print(f"max |force|: {float(np.max(np.abs(forces))):.6f}")

        if "stress" in calc.implemented_properties:
            stress = atoms.get_stress(voigt=True)
            print(f"stress shape: {stress.shape}")

        if args.relax:
            atoms.rattle(0.1)
            print(f"rattled energy: {atoms.get_potential_energy():.6f}")
            dyn = BFGS(atoms)
            dyn.run(fmax=args.fmax, steps=args.max_steps)
            print(f"relaxed energy: {atoms.get_potential_energy():.6f}")
    finally:
        calc.close()


if __name__ == "__main__":
    main()
