# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
# Workload shared by the 27B drafter timing and trace drivers.
F=${BIN:-$HOME/flyx-build/bin}/flyx
M=${MODELS:-$HOME/models}
DF=${DRAFTER:-$M/qwen38-27b-dflash2.flyx}
Q4=$M/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf
IQ2=$M/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ2_S.gguf
MTP=$M/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf

set_workload_args() {
  local prompt tokens=512
  local thinking=(--no-think)
  case $1 in
    code) prompt="Write a Python function that parses an ISO 8601 date string without using datetime, with tests." ;;
    cache) prompt="Explain how a CPU cache works to a new engineer, with a concrete example of a cache miss." ;;
    list) prompt="Copy this list exactly, one item per line: alpha, bravo, charlie, delta, echo, foxtrot, golf, hotel, india, juliet, kilo, lima, mike, november, oscar, papa, quebec, romeo, sierra, tango, uniform, victor, whiskey." ;;
    reason) prompt="A train leaves a station at 9:40 and travels at 79 km/h. A second train leaves the same station at 10:05 on a parallel track at 95 km/h. At what time and how far from the station does the second train catch up with the first? Explain your reasoning."
            tokens=2048
            thinking=() ;;
    *) echo "unknown drafter workload: $1" >&2; return 1 ;;
  esac
  workload_args=(-p "$prompt" --chat "${thinking[@]}" -n "$tokens" --ctx 8192)
}
