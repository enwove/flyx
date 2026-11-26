#!/usr/bin/env bash
# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#
# Qwen3.8-27B's two drafters on one GPU: plain decoding, the MTP layer (inside unsloth's UD-Q4_K_XL
# GGUF; beside the 2-bit UD-IQ2_S in a GGUF of its own), z-lab's DFlash2 drafter (a drafter file, converted with
# `python -m flyx_convert --draft DIR -o FILE`), and both, the drafter chosen every round; each as the engine
# ships (lookup on, the width chosen every round).  Three prompts without thinking (512 new tokens at most)
# and a reasoning prompt with thinking (2048).  Two repetitions; the first checks that every drafted reply is
# plain decoding's text. Each invocation creates a run directory under $OUT and prints its path;
# bench/drafters_table.py reads that directory for the medians.
#
#   MODELS=~/models DRAFTER=~/models/qwen38-27b-dflash2.flyx bash bench/drafters_27b.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source bench/drafters_27b_common.sh
OUT=${OUT:-/tmp/drafters_27b}
mkdir -p -- "$OUT"
OUT=$(mktemp -d -- "$OUT/run.XXXXXXXX")
printf 'Results in %s\n' "$OUT"
echo "# $(date -u +%Y-%m-%dT%H:%MZ), flyx $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used --format=csv,noheader

run() {  # name workload rep command...
  local name=$1 wl=$2 rep=$3
  shift 3
  "$@" > "$OUT/$name.$wl.$rep.out" 2> "$OUT/$name.$wl.$rep.err" || {
    local status=$?
    echo "$name $wl $rep: exit $status (see $OUT/$name.$wl.$rep.err)" >&2
    return "$status"
  }
  echo "$name $wl $rep: ok"
}

for rep in 1 2; do
  for wl in code cache list reason; do
    set_workload_args "$wl"
    run flyx_q4-plain $wl $rep "$F" run "$Q4" "${workload_args[@]}" --no-spec
    run flyx_q4-mtp $wl $rep "$F" run "$Q4" "${workload_args[@]}" --drafter mtp
    run flyx_q4-dflash2 $wl $rep "$F" run "$Q4" "${workload_args[@]}" --drafter dflash2 --drafter-file "$DF"
    run flyx_q4-both $wl $rep "$F" run "$Q4" "${workload_args[@]}" --drafter-file "$DF"
    run flyx_iq2-plain $wl $rep "$F" run "$IQ2" "${workload_args[@]}" --no-spec
    run flyx_iq2-mtp $wl $rep "$F" run "$IQ2" "${workload_args[@]}" --drafter mtp --drafter-file "$MTP" --drafter-file "$DF"
    run flyx_iq2-dflash2 $wl $rep "$F" run "$IQ2" "${workload_args[@]}" --drafter dflash2 --drafter-file "$MTP" \
      --drafter-file "$DF"
    run flyx_iq2-both $wl $rep "$F" run "$IQ2" "${workload_args[@]}" --drafter-file "$MTP" --drafter-file "$DF"
    if [ $rep = 1 ]; then
      for m in q4 iq2; do
        for d in mtp dflash2 both; do
          if ! cmp -s "$OUT/flyx_$m-plain.$wl.1.out" "$OUT/flyx_$m-$d.$wl.1.out"; then
            echo "  $m $d $wl: DIFFERENT from plain" >&2
            exit 1
          fi
          echo "  $m $d $wl: the plain text"
        done
      done
    fi
  done
done
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv,noheader
python3 bench/drafters_table.py "$OUT"
