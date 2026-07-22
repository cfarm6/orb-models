# Failed attempts: preserving NcQEq inter-fragment edges without retraining

This folder documents inference-only attempts to remove or reduce the unexpected
change in the NcQEq spacing scan between roughly `5–6 Å`. The goal was to test
whether preserving graph edges between the two fragments could smooth the
behavior without retraining OrbMol-v2.

## Summary

The key failed attempt was increasing `ForcefieldAtomsAdapter.radius`. This does
preserve more cross-fragment edges in the constructed graph, but it does **not**
change the OrbMol-v2 predictions. The model has an internal learned 6 Å cutoff
that still gates those added edges.

A second, intentionally unsafe diagnostic monkey-patches the internal GNS cutoff
to 10 Å. That changes the predictions, but it produces large absolute energy
shifts and is not a valid no-retrain fix.

A third diagnostic adds a post-hoc D3(BJ) dispersion correction from
`nvalchemiops-toolkit`. The correction is smooth and small through the `5–6 Å`
region, but it does not remove the ionic-constraint feature.

## Artifacts

- `adapter_radius_diagnostic.py` — tests adapter radii `6.0`, `6.5`, `8.0`, and
  `10.0 Å` while leaving the model internals unchanged.
- `adapter_radius_diagnostic.csv` — energies, electrostatic energies, first-region
  charges, and cross-fragment directed-edge counts for the adapter-radius test.
- `adapter_radius_diagnostic.svg` — shows that cross-fragment edges are restored
  while ionic energies remain unchanged.
- `internal_cutoff_diagnostic.py` — compares baseline inference, adapter-only
  radius increase, and an unsafe internal-cutoff monkey patch.
- `internal_cutoff_diagnostic.csv` — energies and charge summaries for the
  internal-cutoff diagnostic.
- `internal_cutoff_diagnostic.svg` — shows that the unsafe cutoff patch changes
  energies by hundreds of eV.
- `d3_dispersion_diagnostic.py` — adds a post-hoc D3(BJ) dispersion correction
  using the `nvalchemiops-toolkit` implementation through TorchSim.
- `d3_dispersion_diagnostic.csv` — original and D3-corrected energy columns for
  the full `2.5–7.0 Å` spacing sweep.
- `d3_dispersion_diagnostic.svg` — compares original and D3-corrected relative
  energy curves and the smooth D3 correction.
- `d3_dispersion_diagnostic.summary.json` — numerical summary of the D3 impact
  in the `5–6 Å` region.

## Reproducing

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=$PWD \
  .venv/bin/python examples/ncqeq/failed_attempts/adapter_radius_diagnostic.py \
  --device cpu

CUDA_VISIBLE_DEVICES='' PYTHONPATH=$PWD \
  .venv/bin/python examples/ncqeq/failed_attempts/internal_cutoff_diagnostic.py \
  --device cpu

CUDA_VISIBLE_DEVICES='' PYTHONPATH=$PWD \
  .venv/bin/python examples/ncqeq/failed_attempts/d3_dispersion_diagnostic.py \
  --device cpu
```

The scripts use the same template fragment geometry as the main NcQEq example:

```text
examples/ncqeq/structures/spacing=2.5.xyz
```

The diagnostic spacing points are:

```text
5.0, 5.2, 5.4, 5.5, 5.6, 5.8, 6.0, 6.2 Å
```

The D3 diagnostic uses the full main-example spacing sweep:

```text
2.5–7.0 Å in 0.1 Å increments
```

## Attempt 1: increase `ForcefieldAtomsAdapter.radius`

### What was tested

`adapter_radius_diagnostic.py` loads `pretrained.orbmol_v2(...)`, then mutates
only:

```python
atoms_adapter.radius = radius
```

The tested radii are:

```text
6.0, 6.5, 8.0, 10.0 Å
```

The diagnostic counts directed cross-fragment graph edges for each spacing and
charge case, then evaluates the same TorchSim batched states.

### Result

For the ionic regional case at `6.0 Å` spacing:

| Adapter radius | Cross-fragment directed edges at 6.0 Å | Max ionic energy difference vs 6 Å adapter |
|---:|---:|---:|
| 6.0 Å | 0 | 0 eV |
| 6.5 Å | 124 | 0 eV |
| 8.0 Å | 400 | 0 eV |
| 10.0 Å | 480 | 0 eV |

This means the adapter successfully preserves cross-fragment edges, but those
edges do not influence the final prediction.

### Interpretation

This failed because the adapter radius only controls graph construction. The
OrbMol-v2 architecture still applies internal 6 Å cutoff weights to edge
embeddings and message passing. Edges beyond that internal cutoff are present in
the graph but have no effective learned contribution.

Relevant settings in the model code:

- `orb_models/forcefield/pretrained.py`: `ForcefieldAtomsAdapter(radius=6.0)`.
- `orb_models/forcefield/pretrained.py`: `BesselBasis(r_max=6.0)`.
- `orb_models/forcefield/pretrained.py`: `outer_product_with_cutoff=True`.
- `orb_models/forcefield/pretrained.py`: `interaction_params={"distance_cutoff": True, ...}`.
- `orb_models/common/models/nn_util.py`: `get_cutoff(..., r_max=6.0)` masks with
  `(r < r_max)`.

## Attempt 2: monkey-patch internal cutoff to 10 Å

### What was tested

`internal_cutoff_diagnostic.py` compares three inference scenarios:

1. `baseline_adapter6_internal6`
2. `adapter10_internal6`
3. `adapter10_internal10`

The third scenario monkey-patches the GNS cutoff function:

```python
gns.get_cutoff = lambda distances: nn_util.get_cutoff(distances, r_max=10.0)
```

This is intentionally recorded as a failed/unsafe diagnostic, not a fix.

### Result

For the ionic regional case at `6.0 Å` spacing:

| Scenario | Ionic energy at 6.0 Å | Max ionic energy difference vs baseline |
|---|---:|---:|
| `baseline_adapter6_internal6` | `-26859.059345008696 eV` | `0 eV` |
| `adapter10_internal6` | `-26859.059344952107 eV` | `5.66e-08 eV` |
| `adapter10_internal10` | `-26656.533807636504 eV` | `210.63 eV` |

The adapter-only scenario is effectively identical to baseline. The internal
cutoff monkey patch changes predictions, but the shift is much too large to be a
safe inference-only correction.

### Interpretation

Changing the internal cutoff after loading a checkpoint changes the model's
feature distribution and message-passing weights outside the trained/configured
regime. It also does not update all other radius-dependent model assumptions in
a principled way. This can create edges with out-of-distribution radial features
and changes the absolute-energy baseline by hundreds of eV.

## Attempt 3: add post-hoc D3 dispersion from `nvalchemiops-toolkit`

### What was tested

`d3_dispersion_diagnostic.py` evaluates a D3(BJ) dispersion energy for each
generated spacing using `nvalchemiops-toolkit` through TorchSim's
`D3DispersionModel`. The currently available D3(BJ) parameter table does not
include `ωB97M-V`, so this diagnostic uses the bundled `pbe` D3(BJ) parameters.

The correction is added post-hoc:

```text
E_corrected(spacing, case) = E_OrbMol-v2(spacing, case) + E_D3(spacing)
```

D3 depends on geometry and atomic numbers, not on the NcQEq charge-constraint
case. Therefore the same D3 value is added to the total-charge, neutral
regional, ionic regional, and U1 curves at each spacing.

### Result

For the full `2.5–7.0 Å` sweep:

| Quantity | Value |
|---|---:|
| D3 energy range | `-0.515` to `-0.397 eV` |
| D3 relative range vs 7 Å | `0.118 eV` |
| D3 relative change from 5 to 6 Å | `0.00694 eV` |
| Ionic curve range from 5 to 6 Å, original | `1.224 eV` |
| Ionic curve range from 5 to 6 Å, +D3 | `1.220 eV` |
| Max ionic step from 5 to 6 Å, original | `0.3957 eV` between `5.5–5.6 Å` |
| Max ionic step from 5 to 6 Å, +D3 | `0.3951 eV` between `5.5–5.6 Å` |

### Interpretation

The D3 correction is physically smooth through the problematic spacing window,
but it is too small and too charge-state-independent to correct the observed
ionic-constraint feature. Because it shifts all charge configurations by the
same geometry-dependent amount, it also leaves the electronic coupling `V`
unchanged when `V` is computed from same-spacing energy differences.

## Conclusion

Increasing `ForcefieldAtomsAdapter.radius` alone is not sufficient. It preserves
cross-fragment graph edges, but OrbMol-v2's internal 6 Å cutoff still removes
their learned contribution. Directly changing the internal cutoff is also not a
safe no-retrain fix because it substantially changes the model output.
Post-hoc D3 dispersion is a controlled long-range tail, but the tested
`nvalchemiops-toolkit` D3(BJ) correction does not improve the `5–6 Å` behavior.

A safer future direction would need an explicitly designed inference-time tail
or coupling correction that preserves the trained 6 Å local model while adding a
controlled long-range term, rather than simply increasing the adapter radius or
monkey-patching the internal GNS cutoff.
