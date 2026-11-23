#!/usr/bin/env bash
# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#
# flyx vs llama.cpp on the same GPU: the same BF16 weights first, then
# 4-bit weights (not the same ones: llama.cpp's quantizations of its BF16
# GGUF, our NVFP4 file from NVIDIA ModelOpt's checkpoint), each part skipped
# when its files are missing.
#
#   bash bench/external/vs_llama_cpp.sh
#
# Expects: llama.cpp built with CUDA in $LLAMA (llama-bench, llama-server),
# a BF16 GGUF of Qwen3.5-4B made by llama.cpp's own converter (MTP head
# included), our model files, and ~6000- and ~500-token prompt files.
#   python $LLAMA/convert_hf_to_gguf.py ~/models/Qwen3.5-4B --outtype bf16 \
#       --outfile ~/models/gguf/qwen35-4b-bf16.gguf
#   $LLAMA/build/bin/llama-quantize ~/models/gguf/qwen35-4b-bf16.gguf ~/models/gguf/qwen35-4b-nvfp4.gguf NVFP4
#   $LLAMA/build/bin/llama-quantize ~/models/gguf/qwen35-4b-bf16.gguf ~/models/gguf/qwen35-4b-q4_k_m.gguf Q4_K_M
set -euo pipefail
cd "$(dirname "$0")/../.."
LLAMA=${LLAMA:-$HOME/llama.cpp}
GGUF=${GGUF:-$HOME/models/gguf/qwen35-4b-bf16.gguf}
MODEL=${MODEL:-$HOME/models/qwen35-4b.flyx}
MODEL_NVFP4=${MODEL_NVFP4:-$HOME/models/qwen35-4b-nvfp4.flyx}
GGUF_DIR=$(dirname "$GGUF")
BIN=${BIN:-$HOME/flyx-build/bin}
LONG=${LONG:-$HOME/long.txt}      # ~6000 tokens
SHORT=${SHORT:-$HOME/p512.txt}    # ~500 tokens
PORT=${PORT:-18090}
OUT=${OUT:-${TMPDIR:-/tmp}/flyx-vs-llama}

echo "# $(date -u +%Y-%m-%dT%H:%MZ), flyx $(git rev-parse --short HEAD), llama.cpp $(git -C "$LLAMA" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "## llama.cpp: llama-bench (prefill 512 / 6000, decode 256)"
"$LLAMA/build/bin/llama-bench" -m "$GGUF" -ngl 99 -fa 1 -p 512,6000 -n 256 -r 3

mkdir -p -- "$OUT"
OUT=$(mktemp -d -- "$OUT/run.XXXXXXXX")
printf 'Server logs in %s\n' "$OUT"
server_pid=
stop_server() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=
  fi
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

serve() {  # name, then llama-server options
  local name=$1; shift
  local log="$OUT/llama_server_$name.log" url="http://127.0.0.1:$PORT" ready=false i
  local alias="flyx-${OUT##*/}"
  "$LLAMA/build/bin/llama-server" -m "$GGUF" -ngl 99 -fa on -c 4096 -np 1 --port "$PORT" --alias "$alias" "$@" \
      > "$log" 2>&1 &
  server_pid=$!
  for ((i=0; i<300; i++)); do
    if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
    if curl -fs --max-time 2 -o /dev/null "$url/health"; then ready=true; break; fi
    sleep 1
  done
  if [[ "$ready" != true ]] || ! kill -0 "$server_pid" 2>/dev/null; then
    echo "llama-server did not become ready. See $log" >&2
    return 1
  fi
  # Another process could have taken the port while this model was loading.
  if ! curl -fs --max-time 3 "$url/v1/models" | python3 -c \
      'import json, sys; sys.exit(not any(m.get("id") == sys.argv[1] for m in json.load(sys.stdin)["data"]))' "$alias"; then
    echo "$url is not serving this benchmark's model. See $log" >&2
    return 1
  fi
  echo "## llama.cpp: llama-server, $name"
  python3 bench/external/llama_server_prompts.py "$url" 256
  stop_server
}
serve plain
for n in 2 3 4 6; do serve "mtp, $n drafts" --spec-type draft-mtp --spec-draft-n-max $n; done

ours() {  # a model file: decode (plain, MTP, DFlash2) on the same prompts, then prefill
  echo "## flyx $(basename "$1"): bench (same prompts)"
  "$BIN/flyx" bench "$1" -n 256 --no-spec
  FLYX_DRAFTER=mtp "$BIN/flyx" bench "$1" -n 256
  FLYX_DRAFTER=dflash2 "$BIN/flyx" bench "$1" -n 256
  echo "## flyx $(basename "$1"): prefill"
  for f in "$SHORT" "$LONG"; do
    for i in 1 2 3; do
      "$BIN/flyx" run "$1" -n 4 -v --no-spec -p "$(cat "$f")" 2>&1 | grep -o "prompt [0-9]* (reused [0-9]*) | prefill [0-9.]* ms"
    done
  done
}
ours "$MODEL"

for q in nvfp4 q4_k_m; do
  g="$GGUF_DIR/qwen35-4b-$q.gguf"
  [[ -f "$g" ]] || continue
  echo "## llama.cpp: llama-bench, $q (its own quantization of the BF16 GGUF)"
  "$LLAMA/build/bin/llama-bench" -m "$g" -ngl 99 -fa 1 -p 512,6000 -n 256 -r 3
done
if [[ -f "$MODEL_NVFP4" ]]; then ours "$MODEL_NVFP4"; fi
