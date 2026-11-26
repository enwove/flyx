#!/usr/bin/env bash
# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
# Run on Linux release hardware with Docker's NVIDIA Container Toolkit.
set -euo pipefail
cd "$(dirname "$0")/.."
if (( $# > 1 )); then echo "usage: $0 [CUDA_DEVEL_IMAGE]" >&2; exit 2; fi
image_args=()
if (( $# )); then image_args=(--build-arg "CUDA_IMAGE=$1"); fi
iid=$(mktemp)
trap 'rm -f "$iid"' EXIT
docker build --platform linux/amd64 --iidfile "$iid" "${image_args[@]}" -f packaging/Dockerfile.linux .

# The checkout is read-only. Only the verified dist/ outputs survive the container.
mkdir -p dist
docker run --rm --platform linux/amd64 --gpus all --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e JOBS -e MAX_GLIBC -v "$PWD:/src:ro" -v "$PWD/dist:/src/dist" \
  -w /src "$(cat "$iid")" bash packaging/package.sh
