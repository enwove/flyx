#!/usr/bin/env bash
# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#
# CI, cpu tier: what every change must pass on a machine without a GPU.
#
#   bash tools/ci/cpu.sh
#
# CPU-only build (the float64 reference backend), then every test that needs
# no GPU but DV's (FLYX_TEST_DV: the cpu-full presets): the reference against
# transformers, the tokenizer, the golden tests on cpu_ref, the layering and
# numerics checks.  Output in logs/ci/: the log (cpu.log) and ctest's JUnit
# report (ctest.xml).
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs/ci
exec > >(tee logs/ci/cpu.log) 2>&1
echo "== flyx CI, cpu tier, $(git rev-parse --short HEAD), $(date -u +%Y-%m-%dT%H:%MZ)"
SECONDS=0
cmake --preset cpu
cmake --build --preset cpu
echo "== cpu tier: configure and build ${SECONDS}s"
SECONDS=0
# ctest reads a relative --output-junit against the build directory: name it from here.
ctest --preset cpu --output-junit "$PWD/logs/ci/ctest.xml"
echo "== cpu tier: ctest ${SECONDS}s"
echo "== cpu tier: ok"
