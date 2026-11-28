#!/usr/bin/env bash
# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#
# Container entrypoint: `flyx serve` from the environment the image and packaging/docker-compose.yml set,
# so a container needs no shell arguments (docs/building.md's Docker section):
#
#   docker run --gpus all -p 127.0.0.1:8080:8080 -v flyx-data:/data flyx:local             # dashboard, load a model in it
#   docker run --gpus all -p 127.0.0.1:8080:8080 -e FLYX_API_KEY=secret \
#     -e FLYX_MODEL=unsloth/Inkling-2-Mini-GGUF:UD-IQ3_XXS flyx:local                      # serve this model at once
#   docker run --gpus all -p 127.0.0.1:8080:8080 \
#     -e FLYX_ARGS='--sessions 4 --kv 16' flyx:local                                       # more server flags
#   docker run --gpus all -p 127.0.0.1:8080:8080 flyx:local --no-spec                      # arguments replace the flags,
#                                                                                          # the binding stays as published
#
# Every container listens on every interface: a port published by Docker reaches nothing bound to the container's own
# loopback, and a request that arrives through a published port is not localhost to the server either — it carries the
# bridge's gateway address, and the browser's `Host` names the machine, not the container (docs/building.md).  Set
# FLYX_API_KEY (or pass --api-key) before serving a port others can reach; publishing is loopback-only by default in
# the compose file, which is what keeps the keyless case a local one.
set -euo pipefail

# The binding, from FLYX_HOST and FLYX_PORT, unless the caller named one.  `flyx serve` spells these two words apart
# (`--host X`, not `--host=X`), so an argument in the `=` form is split here rather than passed on as a model name.
args=()
host_named=0
port_named=0
for arg in "$@"; do
  case "$arg" in
    --host=*) args+=(--host "${arg#--host=}"); host_named=1 ;;
    --port=*) args+=(--port "${arg#--port=}"); port_named=1 ;;
    *) args+=("$arg") ;;
  esac
done
[[ $host_named -eq 0 ]] && args+=(--host "${FLYX_HOST:-0.0.0.0}")
[[ $port_named -eq 0 ]] && args+=(--port "${FLYX_PORT:-8080}")

if (($#)); then
  exec flyx serve "${args[@]}"
fi

# No arguments: the model of FLYX_MODEL, then the flags of FLYX_ARGS.  FLYX_ARGS is split on whitespace, newlines
# included, with no shell evaluation of it; a flag whose own value contains a space belongs in the arguments above,
# where quoting survives (`flyx:local --name "Inkling dev"`).
serve=(flyx serve "${args[@]}")
if [[ -n ${FLYX_MODEL:-} ]]; then
  serve+=("$FLYX_MODEL")
fi
if [[ -n ${FLYX_ARGS:-} ]]; then
  extra=()
  read -ra extra <<< "${FLYX_ARGS//$'\n'/ }"
  serve+=("${extra[@]}")
fi
exec "${serve[@]}"
