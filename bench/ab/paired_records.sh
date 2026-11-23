#!/usr/bin/env bash
# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#
# Two builds' `flyx bench --json` records of the same commands, the builds taking turns (A then B in odd pairs, B
# then A in even ones), every run's record kept: what bench/ab/summarize_records.py reads (decode tok/s, a round's
# ms, tokens per forward, the same answers on both sides).  On a GPU other programs use as well: nvidia-smi's line
# before every run; with QUIET_MIB set, each run first waits (up to 30 minutes) until the GPU's memory in use is
# below that many MiB; and beside the runs a sample a minute of the GPU's memory and load (sampler.log, nvidia-smi's
# own timestamps), by which the summary tells which runs another program's GPU work came back during.
#
#   [QUIET_MIB=2200] bash bench/ab/paired_records.sh OUT PAIRS A_BIN B_BIN CONFIG... -- BENCH_ARGS...
#
# A_BIN, B_BIN: the builds' bin directories (flyx or flyx.exe in each).  CONFIG: NAME=MODEL[,ARG...], the model file
# and that configuration's own flyx bench arguments, comma-separated (res2=minicpm5-1b-dv-res2.flyx,--drafter,dv).
# BENCH_ARGS: the arguments every run takes (--prompts, --suffix, --kv, --ctx, --limit ...).  Records:
# OUT/<a|b>_<NAME>_<pair>.json; OUT/runs.log the runs' start and end (and their last lines of output).
set -uo pipefail
[[ $# -ge 6 ]] || { sed -n '3,16p' "$0" >&2; exit 2; }
OUT=$1 PAIRS=$2 A_BIN=$3 B_BIN=$4
shift 4
CONFIGS=()
while [[ $# -gt 0 && $1 != -- ]]; do CONFIGS+=("$1"); shift; done
[[ ${1:-} == -- ]] && shift
COMMON=("$@")
mkdir -p "$OUT"
LOG=$OUT/runs.log

exe() { if [[ -x $1/flyx.exe ]]; then echo "$1/flyx.exe"; else echo "$1/flyx"; fi; }
now() { date '+%Y/%m/%d %H:%M:%S'; }
GPU_FIELDS=name,driver_version,memory.used,memory.free,memory.total,clocks.sm,clocks.mem,power.draw,temperature.gpu
gpu() { nvidia-smi --query-gpu=$GPU_FIELDS,utilization.gpu --format=csv,noheader; }

run() {  # side, bin directory, configuration, pair
  local side=$1 bin=$2 pair=$4 name=${3%%=*} args=()
  IFS=, read -r -a args <<< "${3#*=}"
  local model=${args[0]} extra=("${args[@]:1}")
  if [[ -n ${QUIET_MIB:-} ]]; then
    local waited=0
    until [[ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) -lt $QUIET_MIB ]]; do
      [[ $waited -ge 1800 ]] && { echo "[$(now)] waited 30 minutes for the GPU, running anyway"; break; }
      sleep 10
      waited=$((waited + 10))
    done
  fi
  echo "[$(now)] gpu: $(gpu)"
  echo "[$(now)] start $side $name $pair: flyx bench $model ${extra[*]} ${COMMON[*]}"
  "$(exe "$bin")" bench "$model" "${extra[@]}" "${COMMON[@]}" --json "$OUT/${side}_${name}_$pair.json" 2>&1 | tail -3
  echo "[$(now)] end $side $name $pair: exit ${PIPESTATUS[0]}"
}

nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu --format=csv,noheader -l 60 \
  >> "$OUT/sampler.log" &
SAMPLER=$!
trap 'kill $SAMPLER 2> /dev/null' EXIT
{
  echo "[$(now)] pairs $PAIRS; a: $A_BIN; b: $B_BIN; ${CONFIGS[*]} -- ${COMMON[*]}"
  for pair in $(seq 1 "$PAIRS"); do
    for cfg in "${CONFIGS[@]}"; do
      if (( pair % 2 )); then run a "$A_BIN" "$cfg" "$pair"; run b "$B_BIN" "$cfg" "$pair"
      else run b "$B_BIN" "$cfg" "$pair"; run a "$A_BIN" "$cfg" "$pair"; fi
    done
  done
  echo "[$(now)] gpu: $(gpu)"
  echo "[$(now)] all done"
} >> "$LOG" 2>&1
