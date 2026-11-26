#!/usr/bin/env bash
# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#
# The GPU suite, by hand on a machine with an NVIDIA GPU and the converted
# models (tools/quickstart.sh provisions both): no CI workflow runs it, there is
# no GPU runner.
#
#   bash tools/ci/gpu.sh            # MODELS=~/models by default
#   FULL=1 bash tools/ci/gpu.sh     # DV's tests too (AGENTS.md gate 3)
#
# The dev preset's CUDA build and ctest's gpu preset: every test but DV's
# (kernel tests, golden tests on CUDA fused and unfused, the CPU tests), after
# gpu_preflight (the GPU not already in use); FULL=1 configures dev-full and
# runs the full preset instead.  Then the real-model check against
# transformers, the HTTP API check and the benchmark.  Output in logs/ci/:
# gpu.log, ctest.xml (JUnit), bench.txt.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs/ci
exec > >(tee logs/ci/gpu.log) 2>&1
MODELS=${MODELS:-$HOME/models}
BUILD=${BUILD:-$HOME/flyx-build}
MODEL=${MODEL:-$MODELS/qwen35-4b-nvfp4.flyx}
MODEL_BF16=${MODEL_BF16:-$MODELS/qwen35-4b.flyx}
CONFIGURE=dev TESTS=gpu
if [[ -n "${FULL:-}" ]]; then CONFIGURE=dev-full TESTS=full; fi
echo "== flyx CI, gpu tier, $(git rev-parse --short HEAD), $(date -u +%Y-%m-%dT%H:%MZ)"
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader
cmake --preset "$CONFIGURE" -B "$BUILD"
cmake --build "$BUILD"
SECONDS=0
# ctest reads a relative --output-junit against the build directory: name it from here.
ctest --preset "$TESTS" --test-dir "$BUILD" --output-junit "$PWD/logs/ci/ctest.xml"
echo "== gpu tier: ctest --preset $TESTS ${SECONDS}s"
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
