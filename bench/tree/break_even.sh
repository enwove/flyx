#!/usr/bin/env bash
# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#
# The break-even line of tree verification: what a round verifying
# R rows costs against plain decoding's round, in one sitting.
#
#   bash bench/tree/break_even.sh OUT NVFP4.gguf GGUF.gguf TEXT [DFLASH2_FILE]
#
# Writes into OUT: tree_{N,M}_{1024,8192,24576}.txt (tree_bench: tree rounds of 1..15 drafts, four shapes, plain round
# alongside), head.txt (the Q6_K head by rows: cuda_bench round / pass), append_{N,Nx,M}_*.txt (append_prefill.py:
# passes of 16..256 rows appended at the context; Nx with bf16 inputs), rounds_{N,M}.txt (flyx bench --json: plain,
# MTP, DFlash2), power.txt (the GPU's power limit before and after each part).  TEXT: a long text for the contexts
# (30K tokens or more).  bench/tree/break_even_tables.py prints the record's tables from OUT.
set -uo pipefail
OUT=${1:?OUT} N=${2:?NVFP4 file} M=${3:?GGUF file} TEXT=${4:?text} DF=${5:-}
B=${BIN:-$HOME/flyx-build/bin}
mkdir -p "$OUT"
power() { nvidia-smi --query-gpu=power.limit --format=csv,noheader >> "$OUT/power.txt"; }
: > "$OUT/power.txt"
power
for f in N M; do
  model=$([ $f = N ] && echo "$N" || echo "$M")
  for at in 1024 8192 24576; do
    "$B/tree_bench" "$model" --ctx-at $at --rounds 16 > "$OUT/tree_${f}_${at}.txt" 2>&1
  done
done
power
for r in 1 16; do
  echo "$r $("$B/cuda_bench" round Q6_K 248320x5120 $r 2>&1 | sed -n 's/.*: \([0-9.]*\) us,.*/\1/p')"
done > "$OUT/head.txt"
for r in 24 32 48 64 96 128 192 256; do
  echo "$r $("$B/cuda_bench" pass Q6_K 248320x5120 $r 2>&1 | sed -n 's/.*: \([0-9.]*\) ms,.*/\1/p' | awk '{print $1 * 1000}')"
done >> "$OUT/head.txt"
ROWS=1,16,24,32,48,64,96,128,192,256
for at in 1024 8192 24576; do
  for v in N Nx M; do
    model=$([ $v = M ] && echo "$M" || echo "$N")
    act=$([ $v = Nx ] && echo "--activations exact" || echo "")
    # shellcheck disable=SC2086
    python3 -I bench/ab/append_prefill.py "$B/libflyx.so" "$model" "$TEXT" --base $at --runs 3 --rows $ROWS \
      --max-ctx 32768 $act > "$OUT/append_${v}_${at}.txt" 2>&1
  done
done
power
for f in N M; do
  model=$([ $f = N ] && echo "$N" || echo "$M")
  for d in "--no-spec" "--drafter mtp" ${DF:+"--drafter dflash2 --drafter-file $DF"}; do
    # shellcheck disable=SC2086
    "$B/flyx" bench "$model" --json $d 2>/dev/null | python3 -c 'import json, sys
j = json.load(sys.stdin)
P = j["prompts"]
r = {k: j[k] for k in ("rounds", "round_ms", "decode_tok_s", "tokens_per_round") if k in j}
r["drafts_per_round"] = sum(p["drafted"] for p in P) / max(1, sum(p["rounds"] for p in P))
print(json.dumps(r))'
  done > "$OUT/rounds_${f}.txt"
done
power
