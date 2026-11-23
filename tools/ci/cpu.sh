#!/usr/bin/env bash
# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#
# CI, cpu tier: what every change must pass on a machine without a GPU.
#
#   bash tools/ci/cpu.sh
#
# CPU-only build (the float64 reference backend), then every test that needs
# no GPU: the reference against transformers, the tokenizer, the golden tests
# on cpu_ref, the layering and numerics checks.  Output in logs/ci/.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs/ci
exec > >(tee logs/ci/cpu.log) 2>&1
echo "== flyx CI, cpu tier, $(git rev-parse --short HEAD), $(date -u +%Y-%m-%dT%H:%MZ)"
cmake --preset cpu
cmake --build --preset cpu
ctest --preset cpu
echo "== cpu tier: ok"
