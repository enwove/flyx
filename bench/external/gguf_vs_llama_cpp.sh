#!/usr/bin/env bash
# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#
# flyx and llama.cpp on the same quantized GGUF files, one line a file:
#   flyx bench (three prompts, N new tokens): plain, then with the drafters the file allows
#   prefill of the ~500-token prompt in a warm process (bench/ab/warm_prefill.py, median of 5)
#   llama-bench pp512 and tgN
#
#   bash bench/external/gguf_vs_llama_cpp.sh FILE.gguf...
#   CTX=4096 N=128 bash bench/external/gguf_vs_llama_cpp.sh BIG.gguf    # a model that leaves little memory
#
# Expects llama.cpp built with CUDA in $LLAMA and the ~500-token prompt file
# (vs_llama_cpp.sh says how the files are made).
set -euo pipefail
cd "$(dirname "$0")/../.."
LLAMA=${LLAMA:-$HOME/llama.cpp}
BIN=${BIN:-$HOME/flyx-build/bin}
SHORT=${SHORT:-$HOME/p512.txt}
N=${N:-256}
CTX=${CTX:-8192}
if (( $# == 0 )); then echo 'usage: gguf_vs_llama_cpp.sh FILE.gguf...' >&2; exit 2; fi

report() {
  local text status
  text=$("$@" 2>&1) || { status=$?; printf '%s\n' "$text" >&2; return "$status"; }
  printf '%s\n' "$text"
}

missing() { echo "$name: missing or invalid $1 in benchmark output" >&2; exit 1; }

echo "# $(date -u +%Y-%m-%dT%H:%MZ), flyx $(git rev-parse --short HEAD), llama.cpp $(git -C "$LLAMA" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader
for f in "$@"; do
  name=$(basename "$f" .gguf)
  number='[0-9]+([.][0-9]+)?'
  text=$(report "$BIN/flyx" bench "$f" -n "$N" --ctx "$CTX" --no-spec --no-lookup)
  plain=$(sed -nE "s/^decode: ($number) tok\/s,.*/\\1 tok\/s/p" <<< "$text")
  [[ -n $plain ]] || missing 'flyx plain decode rate'
  text=$(report "$BIN/flyx" bench "$f" -n "$N" --ctx "$CTX")
  spec=$(sed -nE "s/^decode: ($number) tok\/s, ($number) tokens\/round.*/\\1 tok\/s at \\3 a round/p" <<< "$text")
  [[ -n $spec ]] || missing 'flyx speculative decode rate'
  text=$(report python3 bench/ab/warm_prefill.py "$BIN/libflyx.so" "$f" "$SHORT" 5 "$CTX")
  pre=$(sed -nE "s/^([0-9]+) tokens;.*median ($number) .*/\\1 \\2/p" <<< "$text" |
        awk '$1>0 && $2>0 {printf "%.0f tok/s (%s tokens, %s ms)", $1 * 1000 / $2, $1, $2}')
  [[ -n $pre ]] || missing 'warm prefill median'
  text=$(report "$LLAMA/build/bin/llama-bench" -m "$f" -ngl 99 -fa 1 -p 512 -n "$N" -r 2)
  ll=$(sed -nE "s/^\\|.*\\| *(pp512|tg[0-9]+) *\\| *($number) .*/\\1 \\2/p" <<< "$text" | tr '\n' ' ')
  [[ " $ll" == *' pp512 '* && " $ll" == *" tg$N "* ]] || missing "llama-bench pp512 or tg$N"
  echo "$name | flyx plain $plain | drafted $spec | prefill $pre | llama.cpp $ll"
done
