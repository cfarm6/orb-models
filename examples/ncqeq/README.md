# NcQEq constrained charge calculations

This example shows how to run OrbMol-v2 with regional charge constraints. The
goal is to constrain the net latent charge of user-defined atom regions instead
of constraining only the total system charge.

The included spacing scan evaluates three states at every generated fragment
spacing:

1. **Total-charge constraint only**: the original OrbMol-v2 behavior.
2. **Neutral regional constraints**: first 24 atoms constrained to `0`, all
   remaining atoms constrained to `0`.
3. **Ionic regional constraints**: first 24 atoms constrained to `+1`, all
   remaining atoms constrained to `-1`.

The scan packs the generated geometries into non-periodic TorchSim `SimState`
objects and evaluates them with `torch_sim.static(..., pbar=True)` through the
`OrbTorchSimModel` calculator. The script passes an explicit TorchSim
`BinningAutoBatcher`; raw `autobatcher=True` was avoided because its CUDA memory
estimation pre-pass can dominate runtime for this conservative OrbMol-v2 model. It writes a
CSV table and SVG plots:

```bash
python examples/ncqeq/atcne-example/plot_ncqeq_scan.py
python examples/ncqeq/atcne-example/plot_energy_contribution_lines.py
```

Outputs:

- `examples/ncqeq/atcne-example/ncqeq_spacing_scan.csv`
- `examples/ncqeq/atcne-example/ncqeq_spacing_scan.svg`
- `examples/ncqeq/atcne-example/ncqeq_energy_contribution_lines.svg`

Use `--device cpu` if the GPU is unavailable:

```bash
python examples/ncqeq/atcne-example/plot_ncqeq_scan.py --device cpu
```

## Generated spacing sweep

The example uses one template structure:

```text
examples/ncqeq/atcne-example/structures/spacing=2.5.xyz
```

The first 24 atoms are treated as fragment 1. All remaining atoms are treated
as fragment 2. The script preserves the internal geometry of both fragments and
translates fragment 2 along the template centroid-to-centroid direction.

By default, the centroid-to-centroid spacing is swept from `2.5 Å` to `10.0 Å`
in `0.1 Å` increments. This produces 76 geometries and replaces the older
workflow of reading one structure file per spacing.

You can override the sweep and template:

```bash
python examples/ncqeq/atcne-example/plot_ncqeq_scan.py \
  --template examples/ncqeq/atcne-example/structures/spacing=2.5.xyz \
  --spacing-start 2.5 \
  --spacing-stop 10.0 \
  --spacing-step 0.1 \
  --device cpu
```

## ASE usage

OrbMol-v2 still needs the system-level charge and spin multiplicity:

```python
atoms.info["charge"] = 0
atoms.info["spin"] = 1
```

To add regional charge constraints, provide:

- `atoms.arrays["region_mask"]`: one integer region label per atom.
- `atoms.info["region_charges"]`: one target net charge per sorted
  unique mask label.

Example for the neutral state:

```python
import numpy as np

n_first_region = 24
atoms.set_array(
    "region_mask",
    np.array([0] * n_first_region + [1] * (len(atoms) - n_first_region)),
)
atoms.info["region_charges"] = [0.0, 0.0]
```

Example for the ionic state:

```python
atoms.set_array(
    "region_mask",
    np.array([0] * n_first_region + [1] * (len(atoms) - n_first_region)),
)
atoms.info["region_charges"] = [1.0, -1.0]
```

The target charges must sum to `atoms.info["charge"]`.

## TorchSim static/autobatched usage

For TorchSim, pass regional data through `SimState` extras. The example uses
this path instead of looping over ASE calculator calls.

Use an atom-level mask:

```python
region_mask = torch.tensor([0, 0, 1, 0, 1])
```

If every system in the static runner input has the same number of regions, pass a system-level
target tensor:

```python
state = ts.SimState(
    ...,
    charge=torch.tensor([0.0, 0.0]),
    spin=torch.tensor([1.0, 1.0]),
    region_mask=torch.tensor([0, 0, 1, 0, 1]),
    region_charges=torch.tensor([
        [0.5, -0.5],    # system 0
        [0.25, -0.25],  # system 1
    ]),
)
```

For systems with different numbers of regions, pass per-atom expanded targets
instead:

```python
state = ts.SimState(
    ...,
    charge=torch.tensor([0.0, 0.0]),
    spin=torch.tensor([1.0, 1.0]),
    region_mask=torch.tensor([0, 0, 1, 0, 1, 2]),
    region_charges=torch.tensor([0.5, 0.5, -0.5, 0.2, -0.1, -0.1]),
)
```

For each system, the unique region target values must sum to that system's
`charge`.

In the spacing scan, the 76 generated structures are packed into one TorchSim
state for each charge case and evaluated with `torch_sim.static` using a TorchSim
autobatcher and `pbar=True`:

1. total-charge-only, with no regional mask;
2. neutral regional constraints, with `region_charges[:, :] = [0, 0]`;
3. ionic regional constraints, with `region_charges[:, :] = [1, -1]`.

Each static state has concatenated atom positions, a `system_idx` assigning
atoms to spacing points, `pbc=False` for vacuum boundary conditions, one
`charge` and `spin` value per spacing point, and one two-region
`region_charges` row per spacing point for the regional cases. TorchSim
chooses the execution batches through its autobatcher.

## Reading constrained latent charges

The OrbMol-v2 predictor returns `latent_charges`. The example sums the first
region and second region charges directly:

```python
props = torch_sim.static(state, calculator, autobatcher=autobatcher, pbar=True)
charges = props[0]["latent_charges"].detach().cpu().squeeze(-1)
first_region_charge = charges[:24].sum().item()
second_region_charge = charges[24:].sum().item()
```

`OrbTorchSimModel` also returns `latent_charges` when the underlying model
predicts them. For each dictionary returned by `torch_sim.static`, sum the returned per-atom
`latent_charges` by region.

## Energies and derived quantities

The CSV contains absolute energies in eV:

- `unconstrained_energy_ev`: total-charge-only energy, `U`.
- `constrained_energy_ev`: neutral regional energy, `D0A0`.
- `ionic_constrained_energy_ev`: ionic regional energy, `D+A-`.
- `unconstrained_electrostatic_energy_ev`: electrostatic energy for the
  total-charge-only case.
- `constrained_electrostatic_energy_ev`: electrostatic energy for the neutral
  regional case.
- `ionic_constrained_electrostatic_energy_ev`: electrostatic energy for the
  ionic regional case.
- `short-range energy`: calculated by the energy-contribution line-plot helper as
  `total energy - electrostatic energy` for each case.
- `cross_fragment_edges`: number of directed graph edges connecting the first
  24 atoms to the remaining atoms for the generated spacing.

The relative-energy subplot uses one shared reference: the total-charge-only
energy at the largest spacing in the scan. The same shift is applied to every
energy curve.

The electronic coupling is calculated as:

```text
V = sqrt((U - D0A0) * (U - D+A-))
```

If the product under the square root is negative for a spacing point, the CSV
stores `nan` for `electronic_coupling_ev` and the derived `u1_energy_ev`.

The `U1` curve is calculated as:

```text
U1 = 0.5 * (D0A0 + D+A- + sqrt((D0A0 - D+A-)^2 + 4 * V^2))
```

Both `electronic_coupling_ev` and `u1_energy_ev` are written to the CSV. The
main SVG contains four panels: relative energies, first-region latent charges,
electronic coupling, and the number of directed graph edges between the two
fragments.

## Visualizing where cross-fragment graph edges are lost

For the ATCNE spacing scan, generate a panel plot of the cross-fragment graph
edges from `5.4 Å` through `6.0 Å`:

```bash
PYTHONPATH=$PWD \
  python examples/ncqeq/atcne-example/edge_loss_visualization.py --device cpu
```

Outputs:

- `examples/ncqeq/atcne-example/edge_loss_visualization.svg`
- `examples/ncqeq/atcne-example/edge_loss_visualization.png`
- `examples/ncqeq/atcne-example/edge_loss_visualization_edges.csv`

The plot projects each structure onto the fragment-separation axis and a
perpendicular principal axis. Orange lines are unique cross-fragment edges.
Marker size scales with the number of directed cross-fragment edges touching an
atom, and active atoms are labeled by atom index.

## Reproducing the included scan

Run:

```bash
python examples/ncqeq/atcne-example/plot_ncqeq_scan.py --device cpu
```

or, if CUDA memory is available:

```bash
python examples/ncqeq/atcne-example/plot_ncqeq_scan.py --device cuda
```

The script loads `pretrained.orbmol_v2(..., compile=False)`, wraps it in
`OrbTorchSimModel`, generates the fragment-spacing sweep from the template,
evaluates the sweep with `torch_sim.static(..., pbar=True)` and a TorchSim autobatcher, writes the CSV, and updates the
SVG.

## Failed cutoff/edge-preservation attempts

See `examples/ncqeq/atcne-example/failed_attempts/` for scripts, plots, CSVs, and a write-up
documenting failed no-retrain attempts to preserve inter-fragment edges by
increasing the `ForcefieldAtomsAdapter` radius or monkey-patching the internal
GNS cutoff.
