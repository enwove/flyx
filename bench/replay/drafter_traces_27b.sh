#!/usr/bin/env bash
# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#
# The rounds bench/replay/drafter_choice.py replays: Qwen3.8-27B's two drafters, each alone at 7
# drafts a round with lookup off, on bench/drafters_27b.sh's prompts and both files, every round a JSON line
# (`flyx run --rounds`); and the cost of every round kind, from a verbose load with both drafters.  Then
#   python bench/replay/drafter_choice.py RUN_DIR     # the directory printed at startup, under $OUT
#
#   MODELS=~/models DRAFTER=~/models/qwen38-27b-dflash2.flyx OUT=/tmp/traces27 bash bench/replay/drafter_traces_27b.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
F=${BIN:-$HOME/flyx-build/bin}/flyx
M=${MODELS:-$HOME/models}
DF=${DRAFTER:-$M/qwen38-27b-dflash2.flyx}
O=${OUT:-/tmp/traces27}
Q4=$M/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf
IQ2=$M/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ2_S.gguf
MTP=$M/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf
mkdir -p -- "$O"
O=$(mktemp -d -- "$O/run.XXXXXXXX")
printf 'Results in %s\n' "$O"
P_code="Write a Python function that parses an ISO 8601 date string without using datetime, with tests."
P_cache="Explain how a CPU cache works to a new engineer, with a concrete example of a cache miss."
P_list="Copy this list exactly, one item per line: alpha, bravo, charlie, delta, echo, foxtrot, golf, hotel, india, juliet, kilo, lima, mike, november, oscar, papa, quebec, romeo, sierra, tango, uniform, victor, whiskey."
P_reason="A train leaves a station at 9:40 and travels at 79 km/h. A second train leaves the same station at 10:05 on a parallel track at 95 km/h. At what time and how far from the station does the second train catch up with the first? Explain your reasoning."
for m in q4 iq2; do
  if [ $m = q4 ]; then files=(--drafter-file "$DF"); G=$Q4; else files=(--drafter-file "$MTP" --drafter-file "$DF"); G=$IQ2; fi
  "$F" run "$G" -p "hi" --chat --no-think -n 4 --ctx 8192 "${files[@]}" -v > /dev/null 2> "$O/$m.costs.err"
  grep -h "round cost" "$O/$m.costs.err"
  for wl in code cache list reason; do
    case $wl in
      code) P=$P_code n=512 th=no ;;
      cache) P=$P_cache n=512 th=no ;;
      list) P=$P_list n=512 th=no ;;
      reason) P=$P_reason n=2048 th=yes ;;
    esac
    ft=()
    [ $th = no ] && ft=(--no-think)
    for d in mtp dflash2; do
      FLYX_LOOKUP=0 "$F" run "$G" -p "$P" --chat "${ft[@]}" -n $n --ctx 8192 "${files[@]}" --drafter $d --width 7 \
        --rounds "$O/$m.$d.$wl.jsonl" > "$O/$m.$d.$wl.out" 2> "$O/$m.$d.$wl.err"
      echo "$m $d $wl: ok"
    done
    if ! cmp -s "$O/$m.mtp.$wl.out" "$O/$m.dflash2.$wl.out"; then
      echo "$m $wl: TEXT DIFFERS" >&2
      exit 1
    fi
    echo "  the same text"
  done
done
