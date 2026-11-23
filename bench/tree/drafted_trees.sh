#!/usr/bin/env bash
# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#
# What DFlash2's own trees (FLYX_DRAFT_TREE, --draft-trees) give on a model, in one sitting:
#
#   bash bench/tree/drafted_trees.sh OUT MODEL DFLASH2_FILE [MTP_FILE] [REPEATS]
#
# Writes into OUT:
#   kinds.txt     tree_bench --drafted 256: DFlash2's chains and trees taking turns on flyx bench's prompts; per
#                 round, the drafts accepted, tokens a round and tokens a ms at its cost measured at load
#   ab.jsonl      flyx bench --json, chains against chains + trees, REPEATS times (default 2) in alternating order:
#                 DFlash2 alone and both drafters (the MTP layer from the model file or MTP_FILE), lookup off (the
#                 drafters alone) and on (the default); one JSON object a run with the configuration
#   power.txt     the GPU's power limit and clocks before and after
# The greedy text is the same in every run (speculation never changes it), so tok/s compares the rounds alone.
set -uo pipefail
OUT=${1:?OUT} M=${2:?model} DF=${3:?DFlash2 file} MTP=${4:-} REP=${5:-2}
B=${BIN:-$HOME/flyx-build/bin}
mkdir -p "$OUT"
power() { nvidia-smi --query-gpu=power.limit,clocks.max.sm,temperature.gpu --format=csv,noheader >> "$OUT/power.txt"; }
: > "$OUT/power.txt"
power
"$B/tree_bench" "$M" --drafted 256 --drafter-file "$DF" > "$OUT/kinds.txt" 2>"$OUT/kinds.err"
power
: > "$OUT/ab.jsonl"
run() {  # label, flags...
  local label=$1
  shift
  # shellcheck disable=SC2068
  "$B/flyx" bench "$M" --json $@ 2>/dev/null | python3 -c 'import json, sys
j = json.load(sys.stdin)
P = j["prompts"]
names = j["rounds"]
choices = [0] * len(names)
accepted = [0] * len(names)
for p in P:
    for i, n in enumerate(p.get("round_choices", [])):
        choices[i] += n
    for i, n in enumerate(p.get("round_accepted", [])):
        accepted[i] += n
print(json.dumps({"config": sys.argv[1], "decode_tok_s": j["decode_tok_s"], "tokens_per_round": j["tokens_per_round"],
                  "rounds": names, "round_ms": j["round_ms"], "choices": choices, "accepted": accepted,
                  "per_prompt_tok_s": [(p["new_tokens"] - 1) * 1000 / p["decode_ms"] for p in P]}))' "$label" >> "$OUT/ab.jsonl"
}
mtp=${MTP:+--drafter-file $MTP}
for r in $(seq 1 "$REP"); do
  for lookup in "--no-lookup" ""; do
    l=${lookup:+nolookup}
    l=${l:-lookup}
    if [ $((r % 2)) -eq 1 ]; then order="chains trees"; else order="trees chains"; fi
    for kind in $order; do
      t=$([ $kind = trees ] && echo "--draft-trees" || echo "")
      run "dflash2 $kind $l" --drafter dflash2 --drafter-file "$DF" $lookup $t
      run "all $kind $l" --drafter all --drafter-file "$DF" $mtp $lookup $t
    done
  done
done
power
