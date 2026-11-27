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
source bench/drafters_27b_common.sh
O=${OUT:-/tmp/traces27}
mkdir -p -- "$O"
O=$(mktemp -d -- "$O/run.XXXXXXXX")
printf 'Results in %s\n' "$O"
for m in q4 iq2; do
  if [ $m = q4 ]; then files=(--drafter-file "$DF"); G=$Q4; else files=(--drafter-file "$MTP" --drafter-file "$DF"); G=$IQ2; fi
  "$F" run "$G" -p "hi" --chat --no-think -n 4 --ctx 8192 "${files[@]}" -v > /dev/null 2> "$O/$m.costs.err"
  grep -h "round cost" "$O/$m.costs.err"
  for wl in code cache list reason; do
    set_workload_args "$wl"
    for d in mtp dflash2; do
      FLYX_LOOKUP=0 "$F" run "$G" "${workload_args[@]}" "${files[@]}" --drafter $d --width 7 \
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
