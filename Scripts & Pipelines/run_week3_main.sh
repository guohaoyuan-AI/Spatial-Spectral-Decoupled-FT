#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1

STAGE_DIR="two"
DATASET="CIFAR100"
EXPERIMENT="Stage2_MainBenchmark_VPT"
EPOCHS=15
BATCH_SIZE=32
SEED=42
LORA_R=16

COMMON_ARGS=(
  --stage-dir "${STAGE_DIR}"
  --dataset "${DATASET}"
  --experiment "${EXPERIMENT}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --seed "${SEED}"
  --train-interpolation bicubic
  --test-interpolation bicubic
  --eval-interpolations bicubic,bilinear,nearest
  --pretrained
)

echo "========================================================"
echo "Phase 2: Main Benchmark for SS-Adapter"
echo "========================================================"

echo "[1/6] Lower Bound: Linear Probing"
python -m yanzheng.run_validation \
  "${COMMON_ARGS[@]}" \
  --variant head_only \
  --init-scale 0.0 \
  --lora-r 0

echo "[2/6] Spatial Baseline: VPT"
python -m yanzheng.run_validation \
  "${COMMON_ARGS[@]}" \
  --variant vpt \
  --init-scale 0.0 \
  --lora-r 0

echo "[3/6] Channel Baseline: LoRA"
python -m yanzheng.run_validation \
  "${COMMON_ARGS[@]}" \
  --variant lora_only \
  --init-scale 0.0 \
  --lora-r "${LORA_R}"

echo "[4/6] Channel Baseline: DoRA"
python -m yanzheng.run_validation \
  "${COMMON_ARGS[@]}" \
  --variant dora_only \
  --init-scale 0.0 \
  --lora-r "${LORA_R}" \
  --use-dora

echo "[5/6] Composite: LoRA + SS-Adapter (Ours)"
python -m yanzheng.run_validation \
  "${COMMON_ARGS[@]}" \
  --variant lora_composite \
  --init-scale 1e-4 \
  --lora-r "${LORA_R}"

echo "[6/6] Composite: DoRA + SS-Adapter (Ours)"
python -m yanzheng.run_validation \
  "${COMMON_ARGS[@]}" \
  --variant dora_composite \
  --init-scale 1e-4 \
  --lora-r "${LORA_R}" \
  --use-dora

echo "========================================================"
echo "[Done] All CIFAR-100 benchmarks completed."
echo "Results saved to: ./two/outputs/summary.csv"
echo "========================================================"