#!/usr/bin/env bash
# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#
# CI, cpu tier: what every change must pass on a machine without a GPU.
#
#   bash tools/ci/cpu.sh                      # every test that needs no GPU but DV's
#   bash tools/ci/cpu.sh cpu-full             # DV's tests too (FLYX_TEST_DV): nightly
#   bash tools/ci/cpu.sh cpu-full -L '^dv$'   # DV's tests alone: a pull request that changes DV's code
#
# CPU-only build (the float64 reference backend), then the tests of the preset named, cpu or cpu-full (the arguments
# after it go to ctest): the reference against transformers, the tokenizer, the golden tests on cpu_ref, the
# layering and numerics checks.  cpu-full is cpu's build tree configured with DV's tests registered, so the two share
# one build.  Output in logs/ci/: the log (<preset>.log) and ctest's JUnit report (ctest.xml).
set -euo pipefail
cd "$(dirname "$0")/../.."
PRESET=${1:-cpu}
case "$PRESET" in
  cpu | cpu-full) ;;
  *) echo "usage: bash tools/ci/cpu.sh [cpu|cpu-full] [ctest arguments...]" >&2; exit 2 ;;
esac
if (( $# )); then shift; fi
mkdir -p logs/ci
exec > >(tee "logs/ci/$PRESET.log") 2>&1
echo "== flyx CI, $PRESET, $(git rev-parse --short HEAD), $(date -u +%Y-%m-%dT%H:%MZ)"
SECONDS=0
cmake --preset "$PRESET"
cmake --build --preset cpu  # cpu-full's tree is cpu's
echo "== $PRESET: configure and build ${SECONDS}s"
SECONDS=0
# ctest reads a relative --output-junit against the build directory: name it from here.
ctest --preset "$PRESET" --output-junit "$PWD/logs/ci/ctest.xml" "$@"
echo "== $PRESET: ctest ${SECONDS}s"
echo "== $PRESET: ok"
