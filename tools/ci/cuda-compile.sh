#!/usr/bin/env bash
# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#
# CI, cuda-compile tier: the dev preset's build with the CUDA backend (its kernels, their op tests, the
# microbenchmarks) compiled, and nothing run: the hosted runners have no GPU.
#
#   bash tools/ci/cuda-compile.sh            # configure preset dev
#   bash tools/ci/cuda-compile.sh dev-full   # dev's build tree with DV's tests registered (nightly)
#
# nvcc is $CUDACXX, else the one on PATH (the hosted workflow installs CUDA 13.0's from PyPI, as
# tools/quickstart.sh does).  One architecture, FLYX_CUDA_ARCHS (default 120-real): device code under another
# architecture's #if is compiled only by a build for that one (the release package's: 80, 89, 90, 100, 120).  The
# tree is $BUILD (default ~/flyx-build-cuda-compile, never the dev preset's own).  Output in logs/ci/cuda-compile.log.
set -euo pipefail
cd "$(dirname "$0")/../.."
PRESET=${1:-dev}
case "$PRESET" in
  dev | dev-full) ;;
  *) echo "usage: bash tools/ci/cuda-compile.sh [dev|dev-full]" >&2; exit 2 ;;
esac
BUILD=${BUILD:-$HOME/flyx-build-cuda-compile}
NVCC=${CUDACXX:-$(command -v nvcc || true)}
[[ -n "$NVCC" ]] || { echo "no nvcc: set CUDACXX or put CUDA 13's nvcc on PATH" >&2; exit 1; }
mkdir -p logs/ci
exec > >(tee logs/ci/cuda-compile.log) 2>&1
echo "== flyx CI, cuda-compile ($PRESET), $(git rev-parse --short HEAD), $(date -u +%Y-%m-%dT%H:%MZ)"
"$NVCC" --version | tail -n 2
SECONDS=0
# TODO(FLYX_ENABLE_DV): when DV's code gets a build switch of its own (off in the release package), dev-full turns it
# on here, so that the nightly compile keeps building DV's kernels that pull requests no longer do.
cmake --preset "$PRESET" -B "$BUILD" -DFLYX_CUDA_ARCHS="${FLYX_CUDA_ARCHS:-120-real}" -DCMAKE_CUDA_COMPILER="$NVCC"
echo "== cuda-compile: configure ${SECONDS}s"
SECONDS=0
cmake --build "$BUILD"
echo "== cuda-compile: build ${SECONDS}s"
# Without a CUDA compiler the configure only warns and builds the rest: a pass would then say nothing of the kernels.
for built in libflyx-cuda.so cuda_op_test cuda_bench; do
  [[ -f "$BUILD/bin/$built" ]] || { echo "== cuda-compile: $built was not built (no CUDA backend?)" >&2; exit 1; }
done
echo "== cuda-compile: ok"
