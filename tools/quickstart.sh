#!/usr/bin/env bash
# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#
# Build, test, fetch Qwen3.5-4B + DFlash2, convert, check against
# transformers and benchmark.  Run from anywhere in WSL / Linux:
#
#   bash tools/quickstart.sh             # everything
#   STEP=build bash tools/quickstart.sh  # one step: deps | build | test | fetch | convert | check | serve | bench
#
# No sudo needed.  Python packages go to a venv in the WSL home ($VENV), the
# build tree to $BUILD and models to $MODELS, all on the Linux file system:
# WSL reads /mnt/<drive> through a slow network file system, so a model there
# takes minutes to load instead of seconds.  If no CUDA 13 toolkit is
# installed, the CUDA 13.0 compiler is installed into the venv from PyPI.
# Output of every step also goes to logs/<step>.log.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOGS=$(pwd)/logs
STEP=${STEP:-all}
BUILD=${BUILD:-$HOME/flyx-build}
MODELS=${MODELS:-$HOME/models}
MODEL=${MODEL:-$MODELS/qwen35-4b-nvfp4.flyx}
MODEL_BF16=${MODEL_BF16:-$MODELS/qwen35-4b.flyx}
VENV=${VENV:-$HOME/.venvs/flyx}
mkdir -p "$MODELS"

run() { echo "== $1"; shift; "$@" 2>&1 | tee -a "$LOGS/$CUR.log"; }
want() { [[ "$STEP" == all || "$STEP" == "$1" ]]; }

# nvcc from $CUDACXX, PATH, /usr/local/cuda*, or the venv's PyPI toolkit.
find_nvcc() {
  local c
  for c in "${CUDACXX:-}" "$(command -v nvcc || true)" /usr/local/cuda/bin/nvcc /usr/local/cuda-13*/bin/nvcc \
           "$VENV"/lib/python3*/site-packages/nvidia/cu13/bin/nvcc; do
    if [[ -n "$c" && -x "$c" ]]; then NVCC=$(readlink -f "$c"); return 0; fi
  done
  return 1
}

if want deps; then CUR=deps
  if [[ -e "$VENV" || -L "$VENV" ]] && [[ ! -f "$VENV/pyvenv.cfg" ]]; then
    if [[ ! -d "$VENV" ]]; then echo "VENV is not a directory: $VENV" >&2; exit 1; fi
    first_entry=$(find "$VENV/." -mindepth 1 -maxdepth 1 -print -quit)
    if [[ -n "$first_entry" ]]; then
      echo "VENV is already in use and is not a Python virtual environment: $VENV" >&2
      exit 1
    fi
  fi
  for tool in g++ python3 curl; do
    command -v $tool >/dev/null || { echo "$tool not found: run 'sudo apt install build-essential python3 curl' once, then retry"; exit 1; }
  done
  SMI=$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)
  IS_WSL=0
  grep -qi microsoft /proc/version 2>/dev/null && IS_WSL=1
  if [[ -x "$SMI" ]]; then
    run "GPU" "$SMI" --query-gpu=name,driver_version,memory.total --format=csv
    DRV=$("$SMI" --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
    if (( DRV < 580 )); then
      if (( IS_WSL )); then echo "driver $DRV is too old for CUDA 13.0 (need 580 or newer): update the Windows NVIDIA driver"
      else echo "driver $DRV is too old for CUDA 13.0 (need 580 or newer): update the NVIDIA driver"; fi
      exit 1
    fi
  else
    if (( IS_WSL )); then echo "nvidia-smi not found: is the NVIDIA driver installed on Windows, and is this WSL 2?"
    else echo "nvidia-smi not found: install the NVIDIA driver (580 or newer)"; fi
    exit 1
  fi
  if ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
    if [[ ! -x "$VENV/bin/python" ]]; then
      if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
        echo "python3 is $(python3 --version): need Python 3.11 or newer" >&2
        exit 1
      fi
      python3 -m venv --without-pip "$VENV"
    fi
    if ! "$VENV/bin/python" -m ensurepip --upgrade >/dev/null 2>&1; then
      # Some distributions ship Python without ensurepip. Bootstrap this environment's interpreter directly.
      (
        bootstrap=$(mktemp)
        trap 'rm -f -- "$bootstrap"' EXIT
        curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$bootstrap"
        "$VENV/bin/python" "$bootstrap" -q
      )
    fi
  fi
  . "$VENV/bin/activate"
  run "pip" python -m pip install -q -U pip
  run "python packages" python -m pip install -q "cmake>=3.24" ninja sccache "torch==2.15.0" transformers tokenizers \
      gguf safetensors "huggingface_hub[cli]"
  if ! find_nvcc; then
    echo "no CUDA toolkit found: installing the CUDA 13.0 compiler into $VENV"
    run "CUDA 13.0 compiler" python -m pip install -q "nvidia-cuda-nvcc==13.0.88" "nvidia-nvvm==13.0.88" \
        "nvidia-cuda-runtime==13.0.96" "nvidia-cuda-cccl==13.0.85" "nvidia-cuda-crt==13.0.88"
    CU=$(ls -d "$VENV"/lib/python3*/site-packages/nvidia/cu13 | head -1)
    ln -sfn lib "$CU/lib64"                          # nvcc looks for lib64
    ln -sfn libcudart.so.13 "$CU/lib/libcudart.so"   # CMake looks for the unversioned name
    find_nvcc
  fi
  run "nvcc" "$NVCC" --version
fi

[[ -f "$VENV/bin/activate" ]] && . "$VENV/bin/activate"
find_nvcc || NVCC=""

if want build; then CUR=build
  [[ -n "$NVCC" ]] || { echo "no nvcc: run 'STEP=deps bash tools/quickstart.sh' first"; exit 1; }
  run "configure" cmake -S . -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$NVCC"
  run "build" ninja -C "$BUILD"
fi
if want test; then CUR=test
  (cd "$BUILD" && run "all tests" ctest --output-on-failure)
fi
if want fetch; then CUR=fetch
  run "Qwen3.5-4B" hf download Qwen/Qwen3.5-4B --local-dir "$MODELS/Qwen3.5-4B"
  run "DFlash2" hf download vllm-project/Qwen3.5-4B-DFlash2 --local-dir "$MODELS/Qwen3.5-4B-DFlash2"
  run "Qwen3.5-4B NVFP4" hf download AxionML/Qwen3.5-4B-NVFP4 --local-dir "$MODELS/Qwen3.5-4B-NVFP4"
fi
if want convert; then CUR=convert
  run "convert" env PYTHONPATH=python/convert python -m flyx_convert \
      --model "$MODELS/Qwen3.5-4B" --draft "$MODELS/Qwen3.5-4B-DFlash2" -o "$MODEL_BF16"
  run "convert NVFP4" env PYTHONPATH=python/convert python -m flyx_convert \
      --model "$MODELS/Qwen3.5-4B-NVFP4" --draft "$MODELS/Qwen3.5-4B-DFlash2" -o "$MODEL"
fi
if want check; then CUR=check
  run "real model checks (BF16)" python tests/e2e/check_real_model.py --hf "$MODELS/Qwen3.5-4B" \
      --model "$MODEL_BF16" --lib "$BUILD/bin/libflyx.so"
  run "real model checks (NVFP4)" python tests/e2e/check_real_model.py --hf "$MODELS/Qwen3.5-4B" --model "$MODEL" \
      --lib "$BUILD/bin/libflyx.so"
fi
if want serve; then CUR=serve
  PORT=${PORT:-18080}
  config=$(mktemp -d)
  XDG_CONFIG_HOME="$config" "$BUILD/bin/flyx" serve "$MODEL" --port "$PORT" --no-browser --no-monitor > "$LOGS/server.log" 2>&1 &
  SRV=$!
  finish_server() {
    kill "$SRV" 2>/dev/null || true
    wait "$SRV" 2>/dev/null || true
    rm -rf -- "$config"
  }
  trap finish_server EXIT
  url="http://127.0.0.1:$PORT"
  ready=false
  for ((i=0; i<300; i++)); do
    if ! kill -0 "$SRV" 2>/dev/null; then cat "$LOGS/server.log" >&2; exit 1; fi
    if state=$(curl -fsS --max-time 2 "$url/v1/telemetry?recent=0" 2>/dev/null); then
      # The checks unload and reload the model. Only test the process started above.
      python -c 'import json, sys; sys.exit(json.load(sys.stdin)["server"]["pid"] != int(sys.argv[1]))' "$SRV" <<< "$state" ||
        { echo "another server answered on $url" >&2; exit 1; }
      ready=true
      break
    fi
    sleep 1
  done
  if [[ $ready != true ]]; then echo "timed out starting the test server on $url" >&2; exit 1; fi
  run "HTTP API checks" python tests/apps/flyx-server/check_server.py --url "$url"
  finish_server
  trap - EXIT
fi
if want bench; then CUR=bench
  run "hello" "$BUILD/bin/flyx" run "$MODEL" --chat --no-think -n 128 -v \
      -p "Explain speculative decoding in three sentences."
  run "bench DFlash2" "$BUILD/bin/flyx" bench "$MODEL" -n 256 --drafter dflash2
  run "bench plain" "$BUILD/bin/flyx" bench "$MODEL" -n 256 --no-spec
fi
echo "done; logs in logs/"
