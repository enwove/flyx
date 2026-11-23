#!/usr/bin/env bash
# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#
# A GGUF's decoding in every engine at hand, in one sitting: untraced speed (llama-bench pp512 / tg128,
# flyx bench), a kernel trace of each decode (bench/trace/kernel_trace.cpp), and SGLang's GGUF mat-vec over
# the model's matrices (sgl_gguf_matrices.py), traced the same way; then the matrices matched across them
# (bench/trace/per_matrix.py). was made so.
#
#   OUT=~/trace27 bash bench/external/trace_27b_baselines.sh MODEL.gguf
#
# Engines: llama.cpp in $LLAMA, ik_llama.cpp in $IK (each built with CUDA, llama-bench), a Python with torch and
# sgl-kernel in $SGL_PY (empty: skipped); flyx in $BIN.  LD_LIBRARY_PATH must reach the CUDA runtime they
# were built with.
set -uo pipefail
cd "$(dirname "$0")/../.."
F=$1
OUT=${OUT:-$HOME/trace27}
BIN=${BIN:-$HOME/flyx-build/bin}
LLAMA=${LLAMA:-$HOME/llama.cpp}
IK=${IK:-$HOME/ik_llama.cpp}
SGL_PY=${SGL_PY:-}
T="CUDA_INJECTION64_PATH=$BIN/libflyx-trace.so"
mkdir -p "$OUT"
{
  echo "# $(date -u +%Y-%m-%dT%H:%MZ) flyx $(git rev-parse --short HEAD)"
  echo "## GPU: $(nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader)"
  echo "## clock: $("$BIN/cuda_bench" mma 2>&1 | tail -1)"
  for e in "llama:$LLAMA" "ik:$IK"; do
    name=${e%%:*} dir=${e#*:}
    [ -x "$dir/build/bin/llama-bench" ] || continue
    echo "## $name ($(git -C "$dir" log -1 --format='%h %cd' --date=short))"
    "$dir/build/bin/llama-bench" -m "$F" -ngl 99 -fa 1 -p 512 -n 128 -r 2 2>&1 | grep -E "pp512|tg128|rror"
    env "$T" FLYX_TRACE="$OUT/${name}_tg.tsv" "$dir/build/bin/llama-bench" -m "$F" -ngl 99 -fa 1 -r 1 -p 0 -n 128 \
      2>&1 | grep -E "tg128|rror"
  done
  echo "## flyx"
  "$BIN/flyx" bench "$F" -n 128 --ctx 4096 --no-spec --no-lookup 2>&1 | grep "^decode"
  env "$T" FLYX_TRACE="$OUT/flyx_tg.tsv" "$BIN/flyx" bench "$F" -n 128 --ctx 4096 --no-spec --no-lookup \
    2>&1 | grep "^decode"
  if [ -n "$SGL_PY" ]; then
    echo "## SGLang's ggml_mul_mat_vec_a8 over the model's matrices"
    env "$T" FLYX_TRACE="$OUT/sgl.tsv" "$SGL_PY" bench/external/sgl_gguf_matrices.py run "$F" "$OUT/sgl_ran.txt" 12 \
      2>&1 | tail -2
    "$SGL_PY" bench/external/sgl_gguf_matrices.py analyze "$F" "$OUT/sgl_ran.txt" "$OUT/sgl.tsv" > "$OUT/sgl_matrix.tsv"
  fi
  echo "## clock: $("$BIN/cuda_bench" mma 2>&1 | tail -1)"
} 2>&1 | tee "$OUT/baselines.txt"
args=("flyx=$OUT/flyx_tg.tsv:flyx")
for name in llama ik; do [ -f "$OUT/${name}_tg.tsv" ] && args+=("$name=$OUT/${name}_tg.tsv:ggml"); done
[ -f "$OUT/sgl_matrix.tsv" ] && args+=("sglang=$OUT/sgl_matrix.tsv:matrix")
python3 bench/trace/per_matrix.py "$F" "${args[@]}" --roles --best | tee "$OUT/per_matrix.txt"
