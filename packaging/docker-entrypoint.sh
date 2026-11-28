#!/usr/bin/env bash
# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#
# Container entrypoint: `flyx serve` from the environment the image and compose.yaml set,
# so a container needs no shell arguments (docs/building.md's Docker section):
#
#   docker run --gpus all -p 8080:8080 -v flyx-data:/data flyx:local                  # dashboard, load a model in it
#   docker run --gpus all -p 8080:8080 \
#     -e FLYX_MODEL=unsloth/Inkling-2-Mini-GGUF:UD-IQ3_XXS flyx:local                  # serve this model at once
#   docker run --gpus all -p 8080:8080 \
#     -e FLYX_ARGS='--sessions 4 --kv 16' flyx:local                                   # more server flags
#   docker run --gpus all -p 8080:8080 flyx:local --no-spec                            # arguments replace the flags,
#                                                                                      # the binding stays as published
#
# Every container listens on every interface: a port mapping reaches nothing bound to 127.0.0.1, and a request that
# arrives through a published port is not localhost to the server either (docs/building.md).  Set FLYX_API_KEY (or
# pass --api-key) before serving a port others can reach.
set -euo pipefail

# The binding, from FLYX_HOST and FLYX_PORT, unless the caller named one: a port mapping published by Docker is
# useless with the server on its own loopback, so arguments take the same binding as the environment does.
net=()
host_named=0
port_named=0
for arg in "$@"; do
  case "$arg" in
    --host | --host=*) host_named=1 ;;
    --port | --port=*) port_named=1 ;;
  esac
done
[[ $host_named -eq 0 ]] && net+=(--host "${FLYX_HOST:-0.0.0.0}")
[[ $port_named -eq 0 ]] && net+=(--port "${FLYX_PORT:-8080}")

if (($#)); then
  exec flyx serve "${net[@]}" "$@"
fi

args=("${net[@]}")
if [[ -n ${FLYX_MODEL:-} ]]; then args=("$FLYX_MODEL" "${args[@]}"); fi
if [[ -n ${FLYX_ARGS:-} ]]; then
  # One flag per word; a flag's own value with a space in it needs the argument form (`flyx:local --stop "a b"`).
  read -ra extra <<<"$FLYX_ARGS"
  args+=("${extra[@]}")
fi
exec flyx serve "${args[@]}"
