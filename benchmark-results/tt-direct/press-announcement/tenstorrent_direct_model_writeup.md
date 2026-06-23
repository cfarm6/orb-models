# Tenstorrent acceleration for Orb direct forcefield models

Experimental Tenstorrent support in this branch runs supported Orb direct forcefield models with CPU graph featurization and Tenstorrent hardware execution for eligible linear/MLP layers. The public entrypoints are `load_tt_direct_model(...)` for lower-level inference and `TTDirectCalculator` for ASE workflows.

## Headline result

On 2048-atom synthetic crystal benchmarks, Tenstorrent hardware delivered up to **9.6× faster** direct forcefield inference than CPU, with a **6.4× geometric-mean speedup** across four Orb-v3 direct models. Mean latency reduction across the same runs was **83.9%**.

| Model | CPU wall time | TT hardware wall time | Speedup | Latency reduction |
|---|---:|---:|---:|---:|
| `orb_v3_direct_20_omat` | 6.46 s | 0.67 s | 9.6× | 89.6% |
| `orb_v3_direct_20_mpa` | 3.86 s | 0.70 s | 5.5× | 81.8% |
| `orb_v3_direct_inf_omat` | 9.44 s | 1.69 s | 5.6× | 82.1% |
| `orb_v3_direct_inf_mpa` | 9.34 s | 1.67 s | 5.6× | 82.1% |

## Plot package

Generated press-ready assets are available as PNG and SVG:

1. `01_tt_hardware_speedup_hero` — headline speedup by model.
2. `02_cpu_vs_tt_wall_time` — CPU vs Tenstorrent hardware wall-time bars.
3. `03_latency_reduction` — latency reduction percentage by model.
4. `04_predictions_per_minute` — derived prediction throughput per minute.
5. `05_social_card` — compact announcement card.

The reusable data table is `tt_hardware_benchmark_summary.csv`.

## Benchmark setup

- Command family: `scripts/speed.py --method orb-forward-compare-tt`.
- System size: 2048 atoms per synthetic crystal.
- CPU and TT measurements are taken from paired rows in the same benchmark output file.
- TT backend: hardware (`tt_backend=hardware`).
- Source files: `benchmark-results/tt-direct/*_hardware_after_mlp_fusion.txt`.
- The benchmark output includes CPU-vs-TT parity columns (`parity_atol=0.1`, `parity_rtol=0.1`) for each run.

## Model/integration summary

Tenstorrent support is exposed through the `orb_models.extensions.tt` extension and currently requires installing this branch from source. The implementation wraps supported pretrained direct models, leaves graph construction on CPU, and routes eligible neural-network linear/MLP operations to Tenstorrent hardware or the TT simulator. This keeps the standard Orb model API intact while adding TT-backed inference paths for benchmark and ASE calculator workflows.
