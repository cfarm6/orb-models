[README.md#8F24]

<p align="center">
  <img src="./assets/logo_color_text.png" alt="Orbital Materials" width="600"/>
</p>
<br/>

# Pretrained models for atomic simulations

![example workflow](https://github.com/orbital-materials/orb-models/actions/workflows/test.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/orb-models.svg)](https://badge.fury.io/py/orb-models)

### Install

```bash
pip install orb-models
```

Orb models are expected to work on MacOS and Linux. Windows support is not guaranteed.

Alternatively, you can use Docker to run orb-models; [see instructions below](#docker).

#### Tenstorrent environment

Tenstorrent support is optional, currently targets direct forcefield models, and is only available by installing this branch from source. Install the Orb optional dependency from this checkout first:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[tenstorrent]"
```

For hardware runs you also need Tenstorrent's driver/runtime stack so that `/dev/tenstorrent/0` exists and `ttnn` imports. The recommended setup path is Tenstorrent's installer:

```bash
curl -fsSL https://github.com/tenstorrent/tt-installer/releases/download/v2.1.0/install.sh -O
chmod +x install.sh
./install.sh --install-container-runtime=no
```

See Tenstorrent's [TT-Metalium install guide](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/installing.html) for device-specific driver, firmware, TT-SMI, and Docker options.

Verify the simulator and hardware pieces independently:

```bash
python -c "from ttl.sim import ttnn; print('tt-lang simulator ok')"
ls /dev/tenstorrent/0
python -c "import ttnn; print('ttnn hardware runtime ok')"
```

Run the Tenstorrent direct-model examples on the simulator, hardware, or auto backend:

```bash
python examples/TTDirectInference.py --backend simulator
python examples/TTDirectInference.py --backend hardware
python examples/TTDirectInference.py --backend auto
python examples/TTDirectInference.py --model orbmol-v1-direct --backend simulator --repeats 1

python examples/TTDirectASE.py --backend simulator
python examples/TTDirectASE.py --backend hardware --relax --fmax 0.05
python examples/TTDirectASE.py --model orbmol-v1-direct --backend simulator
```

`auto` uses hardware when `/dev/tenstorrent/0` exists; otherwise it uses the simulator. To run the TT parity checks:

```bash
pytest -q tests/forcefield/tt
```

### Updates

**June 2026**: Added experimental Tenstorrent support for direct forcefield models on this branch. The TT extension provides simulator and hardware backends for supported direct Orb/OrbMol models through `load_tt_direct_model(...)` and `TTDirectCalculator`.

**May 2026**: Release of OrbMol-v2 — adds a `CoulombModule` for long-range electrostatics on top of the OrbMol architecture, using direct Coulomb summation for non-periodic systems and Particle Mesh Ewald (via `nvalchemiops`) for periodic. Trained on OMol25 and OPoly26 (ωB97M-V/def2-TZVPD); load with `pretrained.orbmol_v2(device="cuda")`. See [MODELS.md](MODELS.md) for the full architecture description.

* **Long-range electrostatics and learnable charges.** GSCDB138 Normalized Error Ratio drops from **6.05 → 1.62** (3.7× lower, comparable to a good DFT functional).
* **Full-model compilation.** `model.compile(...)` now wraps the full regressor for all models, giving ~1.7× speedup at 10k atoms on a single 80 GB GPU.

`model.predict(...)["energy"]` now returns **fp64** by default to preserve kJ/mol resolution against OMol-scale references (~1e4–1e5 eV). Pass `fp64_energy=False` to opt out.

**February 2026**: Improved GPU-accelerated graph construction with [ALCHEMI Toolkit-Ops](https://github.com/NVIDIA/nvalchemi-toolkit-ops) and batched simulation with [TorchSim](https://github.com/TorchSim/torch-sim):

* Alchemi-based graph construction (GPU-accelerated, up to 12x faster for large single systems, and sub-linear batch scaling delivering >100x graph construction speed-up for large batches of small systems)
* TorchSim wrapper for batched optimisation and simulation, see [usage with TorchSim](#usage-with-torchsim)
* Alchemi-based D3 dispersion correction module, see [D3 correction](#d3-correction)


**August 2025**: Release of the [OrbMol potentials](https://www.orbitalindustries.com/posts/orbmol-extending-orb-to-molecular-systems):

* Trained on the [Open Molecules 2025 (OMol25)](https://arxiv.org/pdf/2505.08762) dataset—over 100M high-accuracy DFT calculations (ωB97M-V/def2-TZVPD) on diverse molecular systems including metal complexes, biomolecules, and electrolytes.
* Architecturally similar to the highly-performant Orb-v3 models, but now explicit total charges and spin multiplicities can be passed as input.  
* To get started with these models, see: [How to specify total charge and spin multiplicity for OrbMol](#how-to-specify-total-charge-and-spin-multiplicity-for-orbmol).

**April 2025**: Release of the [Orb-v3 set of potentials](https://arxiv.org/abs/2504.06231).

**October 2024**: Release of the [Orb-v2 set of potentials](https://arxiv.org/abs/2410.22570). 

**September 2024**: Release of v1 models - state of the art performance on the matbench discovery dataset.


### Available models
See [MODELS.md](MODELS.md) for a full list of available models along with usage guidance.


### Usage

Note: These examples are designed to run on the `main` branch of orb-models. If you are using a pip installed version of `orb-models`, you may want to look at the corresponding [README.md from that tag](https://github.com/orbital-materials/orb-models/tags).

#### Direct usage

```python
import ase
from ase.build import bulk

from orb_models.forcefield import pretrained

device = "cpu"  # or device="cuda"
orbff, atoms_adapter = pretrained.orb_v3_conservative_inf_omat(
  device=device,
  precision="float32-high",   # or "float32-highest" / "float64
)
atoms = bulk('Cu', 'fcc', a=3.58, cubic=True)
graph = atoms_adapter.from_ase_atoms(atoms, device=device)

# If you have several graphs, batch them like so:
# graph = atoms_adapter.batch([graph1, graph2])
# or 
# graph = atoms_adapter.from_ase_atoms_list([atoms1, atoms2])

result = orbff.predict(graph, split=False)

# Convert to ASE atoms (unbatches the results and transfers to cpu if necessary)
atoms = graph.to_ase_atoms(
    energy=result["energy"],
    forces=result["grad_forces"],
    stress=result["grad_stress"]
)
```

#### Usage with ASE calculator

```python
import ase
from ase.build import bulk

from orb_models.forcefield import pretrained
from orb_models.forcefield.inference.calculator import ORBCalculator

device="cpu" # or device="cuda"
# or choose another model using ORB_PRETRAINED_MODELS[model_name]()
orbff, atoms_adapter = pretrained.orb_v3_conservative_inf_omat(
  device=device,
  precision="float32-high",   # or "float32-highest" / "float64
)
calc = ORBCalculator(orbff, atoms_adapter=atoms_adapter, device=device)
atoms = bulk('Cu', 'fcc', a=3.58, cubic=True)

atoms.calc = calc
atoms.get_potential_energy()
```

You can use this calculator with any ASE calculator-compatible code. For example, you can use it to perform a geometry optimization:

```python
from ase.optimize import BFGS

# Rattle the atoms to get them out of the minimum energy configuration
atoms.rattle(0.5)
print("Rattled Energy:", atoms.get_potential_energy())

calc = ORBCalculator(orbff, atoms_adapter=atoms_adapter, device="cpu") # or device="cuda"
dyn = BFGS(atoms)
dyn.run(fmax=0.01)
print("Optimized Energy:", atoms.get_potential_energy())
```

Or you can use it to run MD simulations. The script, an example input xyz file and a Colab notebook demonstration are available in the [examples directory](./examples). This should work with any input, simply modify the input_file and cell_size parameters. We recommend using constant volume simulations.

#### Usage with Tenstorrent-supported Orb models

The Tenstorrent extension wraps supported Orb forcefield models and runs eligible linear layers through TT-Lang/TTNN while keeping graph featurization on CPU. See [`examples/TTDirectInference.py`](./examples/TTDirectInference.py) for a minimal energy/forces inference example and [`examples/TTDirectASE.py`](./examples/TTDirectASE.py) for the ASE calculator workflow.

Supported model names for `load_tt_direct_model(...)` and `TTDirectCalculator.from_pretrained(...)`:

- `orb-v3-direct-20-omat`
- `orb-v3-direct-inf-omat`
- `orb-v3-direct-20-mpa`
- `orb-v3-direct-inf-mpa`
- `orb-v3-direct-omol`
- `orbmol-v1-direct`
- `separate-d3-3layer`
- `separate-d3-5layer`
- `separate-d4-3layer`
- `separate-d4-5layer`
- `orb-v2`
- `orb-mptraj-only-v2`
- `orb-d3-v2`
- `orb-d3-sm-v2`
- `orb-d3-xs-v2`

For OrbMol/OMol models (`orb-v3-direct-omol`, `orbmol-v1-direct`), set `atoms.info["charge"]` and `atoms.info["spin"]` before featurization. The TT examples set a neutral singlet by default (`charge = 0.0`, `spin = 1.0`).

Use `TTDirectCalculator` for the standard ASE workflow (relaxations, MD, optimizers):

```python
from ase.build import bulk
from ase.optimize import BFGS

from orb_models.extensions.tt import TTDirectCalculator

atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
calc = TTDirectCalculator.from_pretrained(
    "orb-v3-direct-20-omat",
    backend="simulator",  # or "hardware" / "auto"
    compile=False,
)
try:
    atoms.calc = calc
    print(atoms.get_potential_energy())
    print(atoms.get_forces())
    BFGS(atoms).run(fmax=0.05)
finally:
    calc.close()
```

Lower-level access without ASE:

```python
from ase.build import bulk

from orb_models.extensions.tt import load_tt_direct_model

atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
orbff, atoms_adapter = load_tt_direct_model(
    "orb-v3-direct-20-omat",
    # "orbmol-v1-direct" is also supported; set atoms.info["charge"] and
    # atoms.info["spin"] before featurization for charged/spin-polarized systems.
    backend="simulator",  # or "hardware" / "auto"
    compile=False,
)
try:
    graph = atoms_adapter.from_ase_atoms(atoms, device="cpu").to("cpu")
    result = orbff.predict(graph)
finally:
    orbff.close()

result["energy"], result["forces"]
```

#### Usage with TorchSim

For batched optimisation, we recommend using [TorchSim](https://github.com/TorchSim/torch-sim). It's an optional dependency that can be installed with `pip install torch-sim-atomistic`. 

```python
import ase
import torch
import torch_sim as ts
from ase.build import bulk

from orb_models.forcefield import pretrained
from orb_models.forcefield.inference.orb_torchsim import OrbTorchSimModel


device = "cpu"  # or device="cuda"
# or choose another model using ORB_PRETRAINED_MODELS[model_name]()
orbff, atoms_adapter = pretrained.orb_v3_conservative_inf_omat(
  device=device,
  precision="float32-high",   # or "float32-highest" / "float64
)

atoms1 = bulk('Cu', 'fcc', a=3.58, cubic=True)
atoms2 = bulk('Si', 'diamond', a=5.43, cubic=True)
atoms_list = [atoms1, atoms2]
ts_state = ts.io.atoms_to_state(atoms_list, device, dtype=torch.get_default_dtype())

ts_model = OrbTorchSimModel(orbff, atoms_adapter)
results = ts_model(ts_state)
results["energy"]
```

You can use this module for geometry optimisation and MD simulation:

```python
# Rattle the atoms to get them out of the minimum energy configuration
atoms1.rattle(0.5)
atoms2.rattle(0.5)
atoms_list = [atoms1, atoms2]
ts_state = ts.io.atoms_to_state(atoms_list, device, dtype=torch.get_default_dtype())

ts_model = OrbTorchSimModel(orbff, atoms_adapter)
results = ts_model(ts_state)
print("Rattled energies:", results["energy"])

# Optimise with TorchSim
relaxed_state = ts.optimize(
    system=ts_state,
    convergence_fn=ts.generate_force_convergence_fn(
        force_tol=0.01,
        include_cell_forces=False,
    ),
    model=ts_model,
    optimizer=ts.Optimizer["fire"],
    max_steps=100,
    steps_between_swaps=10,
)
results = ts_model(relaxed_state)
print("Rattled energies:", results["energy"])
```

#### How to specify total charge and spin multiplicity for OrbMol

The OrbMol models *require* total charge and spin multiplicity to be specified. This can be done by setting them in `atoms.info` dictionary.

```python
import ase
from ase.build import molecule

from orb_models.forcefield import pretrained

device = "cpu"  # or device="cuda"
orbff, atoms_adapter = pretrained.orbmol_v2(
  device=device,
  precision="float32-high",   # or "float32-highest" / "float64
)
atoms = molecule("C6H6")

atoms.info["charge"] = 0  # total charge
atoms.info["spin"] = 1  #  spin multiplicity
graph = atoms_adapter.from_ase_atoms(atoms, device=device)

result = orbff.predict(graph, split=False)
```

…
Please join the discussion on Discord by following [this](https://discord.gg/SyD6vWSSTB) link.

[Showing lines 1-300 of 509. Use :301 to continue]
