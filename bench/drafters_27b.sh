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
F=${BIN:-$HOME/flyx-build/bin}/flyx
M=${MODELS:-$HOME/models}
DF=${DRAFTER:-$M/qwen38-27b-dflash2.flyx}
OUT=${OUT:-/tmp/drafters_27b}
Q4=$M/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf
IQ2=$M/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ2_S.gguf
MTP=$M/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf
mkdir -p -- "$OUT"
OUT=$(mktemp -d -- "$OUT/run.XXXXXXXX")
printf 'Results in %s\n' "$OUT"
P_code="Write a Python function that parses an ISO 8601 date string without using datetime, with tests."
P_cache="Explain how a CPU cache works to a new engineer, with a concrete example of a cache miss."
P_list="Copy this list exactly, one item per line: alpha, bravo, charlie, delta, echo, foxtrot, golf, hotel, india, juliet, kilo, lima, mike, november, oscar, papa, quebec, romeo, sierra, tango, uniform, victor, whiskey."
P_reason="A train leaves a station at 9:40 and travels at 79 km/h. A second train leaves the same station at 10:05 on a parallel track at 95 km/h. At what time and how far from the station does the second train catch up with the first? Explain your reasoning."
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
    case $wl in
      code) P=$P_code n=512 th=no ;;
      cache) P=$P_cache n=512 th=no ;;
      list) P=$P_list n=512 th=no ;;
      reason) P=$P_reason n=2048 th=yes ;;
    esac
    ft=()
    [ $th = no ] && ft=(--no-think)
    common=(-p "$P" --chat "${ft[@]}" -n $n --ctx 8192)
    run flyx_q4-plain $wl $rep "$F" run "$Q4" "${common[@]}" --no-spec
    run flyx_q4-mtp $wl $rep "$F" run "$Q4" "${common[@]}" --drafter mtp
    run flyx_q4-dflash2 $wl $rep "$F" run "$Q4" "${common[@]}" --drafter dflash2 --drafter-file "$DF"
    run flyx_q4-both $wl $rep "$F" run "$Q4" "${common[@]}" --drafter-file "$DF"
    run flyx_iq2-plain $wl $rep "$F" run "$IQ2" "${common[@]}" --no-spec
    run flyx_iq2-mtp $wl $rep "$F" run "$IQ2" "${common[@]}" --drafter mtp --drafter-file "$MTP" --drafter-file "$DF"
    run flyx_iq2-dflash2 $wl $rep "$F" run "$IQ2" "${common[@]}" --drafter dflash2 --drafter-file "$MTP" \
      --drafter-file "$DF"
    run flyx_iq2-both $wl $rep "$F" run "$IQ2" "${common[@]}" --drafter-file "$MTP" --drafter-file "$DF"
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
