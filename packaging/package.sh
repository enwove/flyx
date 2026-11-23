#!/usr/bin/env bash
# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#
# Build the release packages and check them on this machine:
#
#   bash packaging/package.sh            # tarball + Python wheel into dist/
#
#   dist/flyx-<version>-linux-x86_64-cuda13.tar.gz
#       bin/ (flyx; flyx-server, a link to it), lib/ (libflyx.so, flyx/ backends),
#       include/, python/; needs only libc and the NVIDIA driver (>= 580)
#   dist/flyx-<version>-py3-none-manylinux_<glibc>_x86_64.whl
#       the ctypes binding with the same libraries inside (no torch)
#
# The smoke test unpacks the tarball into an empty directory, checks the
# dynamic dependencies (nothing but libc / libm / libdl / libpthread / the
# loader), and runs the CLI on the tiny test model with both backends and the
# server with the float64 one.  For a true clean-system test run packaging/smoke.sh on a machine
# with only the driver installed.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-$HOME/.venvs/flyx}
[[ -f "$VENV/bin/activate" ]] && . "$VENV/bin/activate"
python -m auditwheel --version
BUILD=${BUILD:-$HOME/flyx-build-release}
VERSION=$(sed -n 's/^project(flyx VERSION \([0-9.]*\).*/\1/p' CMakeLists.txt)
NAME=flyx-$VERSION-linux-x86_64-cuda13
mkdir -p dist

NVCC=${CUDACXX:-$(command -v nvcc || true)}
if [[ -z "$NVCC" ]]; then
  for p in /usr/local/cuda-13*/bin/nvcc /usr/local/cuda/bin/nvcc "$VENV"/lib/python3*/site-packages/nvidia/cu13/bin/nvcc; do
    [[ -x "$p" ]] && NVCC=$p && break
  done
fi
if [[ -z "$NVCC" || ! -x "$NVCC" ]]; then
  echo "error: CUDA 13 nvcc not found (set CUDACXX or install CUDA 13)" >&2
  exit 1
fi
echo "== configure ($BUILD)"
cmake --preset release-linux -B "$BUILD" -DCMAKE_CUDA_COMPILER="$(readlink -f "$NVCC")" > /dev/null
echo "== build"
cmake --build "$BUILD" ${JOBS:+-j "$JOBS"}  # JOBS: fewer compiles at once where memory is short
echo "== test"
(cd "$BUILD" && ctest --output-on-failure -j1)
# Only checked artifacts reach dist/. Each candidate is on that filesystem for the final rename.
WORK=$(mktemp -d "$PWD/dist/.package.XXXXXXXX")
trap 'rm -rf "$WORK"' EXIT
STAGE=$WORK/$NAME
echo "== install -> $STAGE"
cmake --install "$BUILD" --prefix "$STAGE" --strip > /dev/null  # no symbol tables: about 1 MB a file

echo "== tarball"
TARBALL=$WORK/$NAME.tar.gz
tar -C "$WORK" -czf "$TARBALL" "$NAME"
bash packaging/smoke.sh "$TARBALL" "$BUILD/assets"

echo "== wheel"
W=$WORK/wheel
mkdir -p "$W/flyx/_lib/flyx"
cp python/flyx/*.py "$W/flyx/"
cp "$STAGE/lib/libflyx.so" "$W/flyx/_lib/"
cp "$STAGE"/lib/flyx/*.so "$W/flyx/_lib/flyx/"
cp "$STAGE/LICENSE" "$STAGE/NOTICE" "$STAGE/README.md" "$W/"
cp -r "$STAGE/third_party" "$W/"
cat > "$W/pyproject.toml" <<EOF
[build-system]
requires = ["setuptools>=77", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "flyx"
version = "$VERSION"
description = "Local LLM inference engine: lowest single-user latency on one consumer GPU"
readme = "README.md"
license = "Apache-2.0"
license-files = ["LICENSE", "NOTICE", "third_party/**/LICENSE*", "third_party/**/NOTICE",
                 "third_party/**/README.*.md", "third_party/**/picojson.h"]
requires-python = ">=3.9"
keywords = ["llm", "inference", "cuda", "gguf", "speculative-decoding"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

[project.urls]
Homepage = "https://github.com/enwove/flyx"
Repository = "https://github.com/enwove/flyx"
Issues = "https://github.com/enwove/flyx/issues"

[tool.setuptools]
packages = ["flyx"]
include-package-data = true

[tool.setuptools.package-data]
flyx = ["_lib/*.so", "_lib/flyx/*.so"]
EOF
python -m pip wheel --no-deps --no-build-isolation -q -w "$W/out" "$W"
# Let the linked ELF libraries determine the minimum compatible platform, in both the filename and WHEEL metadata.
python -m auditwheel repair --plat auto -w "$W/checked" "$W"/out/*.whl
python -m pip install --no-index --no-deps -q --target "$W/installed" "$W"/checked/*.whl
python -I packaging/smoke_binding.py "$W/installed" "$BUILD/assets/tiny.flyx"
mv "$W"/checked/*.whl dist/
mv "$TARBALL" dist/
echo "   dist/$NAME.tar.gz ($(du -h "dist/$NAME.tar.gz" | cut -f1))"
