#!/usr/bin/env bash
# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#
# Smoke test of a release tarball, in an empty directory:
#
#   bash packaging/smoke.sh dist/flyx-<v>-linux-x86_64-cuda13.tar.gz ASSET_DIR
#
# ASSET_DIR holds tiny.flyx (built by the test suite: <build>/assets).
# Checks: only system libraries are needed; the CLI generates with the
# float64 backend and (when a GPU is present) the CUDA backend; the server
# (flyx serve) starts, answers /health and a completion, and stops; the
# flyx-server link is the server.
set -euo pipefail
TARBALL=$(readlink -f "$1")
ASSETS=$(readlink -f "$2")
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
tar -C "$T" -xzf "$TARBALL"
P=$(ls -d "$T"/flyx-*)
echo "== smoke test in $P"

for required in bin/flyx bin/flyx-server lib/libflyx.so lib/flyx/libflyx-cpu_ref.so lib/flyx/libflyx-cuda.so \
                include/flyx/flyx.h python/flyx/__init__.py LICENSE NOTICE; do
  [[ -f "$P/$required" ]] || { echo "  FAIL package is missing $required"; exit 1; }
done
bad=0
for f in "$P"/bin/* "$P"/lib/*.so "$P"/lib/flyx/*.so; do
  if ! linked=$(ldd "$f"); then echo "  FAIL could not inspect $(basename "$f")"; exit 1; fi
  deps=$(awk '$1 !~ /^(linux-vdso\.so|\/lib64\/ld-linux-|lib(c|m|dl|pthread|rt)\.so\.)/ {print $1}' <<< "$linked")
  if [[ -n "$deps" ]]; then echo "  FAIL $(basename "$f") needs: $deps"; bad=1; fi
done
[[ $bad == 0 ]] && echo "  ok   only libc / libm / the loader are needed"

port=()
if [[ -n ${PORT:-} ]]; then port=(--port "$PORT"); fi
env -i PATH=/usr/bin:/bin python3 "$(dirname "$0")/smoke_model.py" "$P/bin/flyx" "$ASSETS/tiny.flyx" "${port[@]}"
# flyx-server is a link to flyx; called by that name it is `flyx serve`.
u=$("$P/bin/flyx-server" --help 2>&1 || true)
if [[ "$u" == *"usage: flyx serve"* ]]; then echo "  ok   flyx-server is flyx serve"
else echo "  FAIL flyx-server does not start the server"; bad=1; fi
exit $bad
