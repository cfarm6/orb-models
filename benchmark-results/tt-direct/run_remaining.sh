#!/usr/bin/env bash
set -euo pipefail
cd /home/carson/orb-models
mkdir -p benchmark-results/tt-direct

COMMON_ARGS=(
  --method orb-forward-compare-tt
  --device cpu
  --num_threads 4
  --num_evals 20
  --num_atoms 32 128 512 2048
)

run_model() {
  local name="$1"
  local outfile="benchmark-results/tt-direct/${name}_hardware.txt"
  echo "=== Starting ${name} at $(date) ===" | tee -a benchmark-results/tt-direct/run.log
  .venv/bin/python scripts/speed.py "${COMMON_ARGS[@]}" \
    --extra_kwargs "name=${name} precision=float32-high compile=False tt_backend=hardware pbc=True" \
    > "${outfile}" 2>&1
  echo "=== Finished ${name} at $(date) ===" | tee -a benchmark-results/tt-direct/run.log
}

for model in orb_v3_direct_20_mpa orb_v3_direct_inf_mpa; do
  run_model "${model}"
done

echo "=== Starting simulator smoke at $(date) ===" | tee -a benchmark-results/tt-direct/run.log
.venv/bin/python scripts/speed.py \
  --method orb-forward-compare-tt \
  --device cpu \
  --num_threads 4 \
  --num_evals 3 \
  --num_atoms 32 \
  --extra_kwargs "name=orb_v3_direct_20_omat precision=float32-high compile=False tt_backend=simulator pbc=True" \
  > benchmark-results/tt-direct/orb_v3_direct_20_omat_simulator.txt 2>&1
echo "=== All remaining benchmarks done at $(date) ===" | tee -a benchmark-results/tt-direct/run.log
