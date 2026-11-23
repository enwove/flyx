#!/usr/bin/env bash
# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#
# A/B: two sides run alternately, so slow drift (GPU clocks, other users of the GPU, which on WSL include the
# Windows desktop) hits both alike, and in alternating order (AB, BA, ...), so the second run of a pair is each
# side's equally often.  Prints every pair, then each side's median and the median of the per-pair differences
# (B - A); use an even number of pairs.
#
#   bash bench/ab/ab.sh decode MODEL A B [pairs=6] [flyx bench options...]
#       decode tok/s (`flyx bench --json`)
#   PROMPT_FILE=F bash bench/ab/ab.sh prefill MODEL A B [pairs=4] [flyx run options...]
#       prefill ms of one long prompt (`flyx run -v --no-spec -n 1`)
#
# A side is a bin directory (a build: ~/bin-before, ~/flyx-build/bin) or environment settings for the build in
# BIN (default ~/flyx-build/bin): "" or "FLYX_BACKEND_OPTIONS=pdl=1".
set -euo pipefail
MODE=$1 MODEL=$2 A=$3 B=$4
[[ $MODE == decode || $MODE == prefill ]] || { echo "ab.sh: the mode is decode or prefill" >&2; exit 2; }
PAIRS=${5:-$([[ $MODE == prefill ]] && echo 4 || echo 6)}
shift 5 || shift $#
BIN=${BIN:-$HOME/flyx-build/bin}

side() {  # run flyx on side $1: its build, or the default build with its environment
  local s=$1
  shift
  # shellcheck disable=SC2086
  if [[ -d $s ]]; then "$s/flyx" "$@"; else env $s "$BIN/flyx" "$@"; fi
}
measure() {  # one number for side $1, the other arguments passed on to flyx
  local s=$1
  shift
  if [[ $MODE == decode ]]; then
    side "$s" bench "$MODEL" --json "$@" 2>/dev/null |
      python3 -c 'import json, sys; print("%.1f" % json.load(sys.stdin)["decode_tok_s"])'
  else
    side "$s" run "$MODEL" -p "$(cat "$PROMPT_FILE")" -n 1 --no-spec -v "$@" 2>&1 |
      sed -n 's/.*| prefill \([0-9.]*\) ms.*/\1/p'
  fi
}

# Odd pairs run B first: the second run of a pair finds a warmer GPU (on an RTX 5090 at its power cap, MTP decoding:
# A then B every pair read B 1.5% slower, 1 of 8 pairs faster; the same builds swapped, the same median).
a=() b=()
for ((i = 0; i < PAIRS; ++i)); do
  if ((i % 2)); then
    y=$(measure "$B" "$@") x=$(measure "$A" "$@")
  else
    x=$(measure "$A" "$@") y=$(measure "$B" "$@")
  fi
  a+=("$x") b+=("$y")
  echo "pair $i: A $x  B $y$( ((i % 2)) && echo '  (B first)')"
done
python3 - "$MODE" "${a[@]}" -- "${b[@]}" <<'EOF'
import statistics, sys
mode, args = sys.argv[1], sys.argv[2:]
i = args.index("--")
a, b = [float(x) for x in args[:i]], [float(x) for x in args[i + 1:]]
d = [y - x for x, y in zip(a, b)]
unit = "tok/s" if mode == "decode" else "ms"
b_faster = sum(x > 0 for x in d) if mode == "decode" else sum(x < 0 for x in d)
ma, md = statistics.median(a), statistics.median(d)
print(f"A median {ma:.1f} {unit}  B median {statistics.median(b):.1f} {unit}  "
      f"B - A per pair: median {md:+.1f} {unit} ({md / ma * 100:+.1f}%), {b_faster}/{len(d)} pairs B faster")
EOF
