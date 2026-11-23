#!/usr/bin/env bash
# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#
# CI, gpu tier: the machine with the GPU and the converted models.
#
#   bash tools/ci/gpu.sh            # MODELS=~/models by default
#
# CUDA build (dev preset) and every suite (kernel tests; golden tests on
# CUDA, fused and unfused), the real-model check against transformers, the
# HTTP API check, then the benchmark.  Output in logs/ci/.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs/ci
exec > >(tee logs/ci/gpu.log) 2>&1
MODELS=${MODELS:-$HOME/models}
BUILD=${BUILD:-$HOME/flyx-build}
MODEL=${MODEL:-$MODELS/qwen35-4b-nvfp4.flyx}
MODEL_BF16=${MODEL_BF16:-$MODELS/qwen35-4b.flyx}
echo "== flyx CI, gpu tier, $(git rev-parse --short HEAD), $(date -u +%Y-%m-%dT%H:%MZ)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
cmake --preset dev -B "$BUILD"
cmake --build "$BUILD"
ctest --preset dev --test-dir "$BUILD"
export BUILD MODELS MODEL MODEL_BF16
STEP=check bash tools/quickstart.sh
STEP=serve bash tools/quickstart.sh
{
  echo "# $(git rev-parse --short HEAD), $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
  for m in "$MODEL" "$MODEL_BF16"; do
    for mode in "--no-spec" "" "--drafter dflash2"; do
      echo "## $m ${mode:-model default}"
      # shellcheck disable=SC2086
      "$BUILD/bin/flyx" bench "$m" $mode -n 256 2>&1 | grep -E "^(prompt |decode:|cpu:)"
    done
  done
} | tee logs/ci/bench.txt
echo "== gpu tier: ok"
