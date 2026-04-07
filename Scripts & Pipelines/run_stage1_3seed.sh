#!/bin/bash
set -e

PROJECT_ROOT=$(pwd)
OUTPUT_DIR="${PROJECT_ROOT}/outputs"

cd "$PROJECT_ROOT"

variants=("baseline" "spi" "afm" "spi_afm")
seeds=(42 43 44)

for variant in "${variants[@]}"; do
  for seed in "${seeds[@]}"; do
    echo "=================================================="
    echo "Running variant=${variant}, seed=${seed}"
    echo "=================================================="

    python -m yanzheng.run_validation \
      --experiment C \
      --variant "$variant" \
      --dataset CIFAR10 \
      --batch-size 96 \
      --seed "$seed" \
      --train-interpolation bicubic \
      --test-interpolation bicubic \
      --eval-interpolations bicubic,bilinear,nearest \
      --output-dir "$OUTPUT_DIR"
  done
done

echo "All 3-seed runs finished."