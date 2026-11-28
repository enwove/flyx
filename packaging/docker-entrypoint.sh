#!/usr/bin/env bash
# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#
# Container entrypoint: `flyx serve` from the environment the image and packaging/docker-compose.yml set,
# so a container needs no shell arguments (docs/building.md's Docker section):
#
#   docker run --gpus all --init -p 127.0.0.1:8080:8080 -v flyx-data:/data flyx:local    # the log prints the URL to open
#   docker run --gpus all --init -p 127.0.0.1:8080:8080 \
#     -e FLYX_MODEL=unsloth/Inkling-2-Mini-GGUF:UD-IQ3_XXS flyx:local                    # serve this model at once
#   docker run --gpus all --init -p 127.0.0.1:8080:8080 \
#     -e FLYX_ARGS='--sessions 4 --kv 16' flyx:local                                     # more server flags
#   docker run --gpus all --init -p 127.0.0.1:8080:8080 flyx:local --no-spec             # arguments replace the flags,
#                                                                                        # the binding stays as published
#
# Every container listens on every interface: a port published by Docker reaches nothing bound to the container's own
# loopback.  A key is not optional in practice, and this script supplies one (see below): through a published port a
# request carries the bridge's gateway address and the machine's name in `Host`, which is neither localhost nor an
# address the container trusts, so without a key the endpoints the dashboard reads answer 403.
set -euo pipefail

# FLYX_ARGS: more flags for the server, split on whitespace, newlines included, with no shell evaluation of it.  A
# flag whose own value contains a space belongs in the container's arguments below, where quoting survives
# (`flyx:local --name "Inkling dev"`).
extra=()
if [[ -n ${FLYX_ARGS:-} ]]; then
  read -ra extra <<< "${FLYX_ARGS//$'\n'/ }"
fi

# The caller's arguments, with `--host=X` and `--port=X` split into the two words `flyx serve` takes (it reads the
# `=` form as a model name).  Naming the host or the port in either spelling counts, so the default stays out.
args=()
host_named=0
port_named=0
for arg in "$@"; do
  case "$arg" in
    --host=*) args+=(--host "${arg#--host=}"); host_named=1 ;;
    --port=*) args+=(--port "${arg#--port=}"); port_named=1 ;;
    --host) args+=("$arg"); host_named=1 ;;
    --port) args+=("$arg"); port_named=1 ;;
    *) args+=("$arg") ;;
  esac
done

# The binding, from FLYX_HOST and FLYX_PORT unless the caller named one: a port mapping published by Docker reaches
# nothing bound to the container's own loopback, so it applies to the arguments case too.
binding=()
if [[ $host_named -eq 0 ]]; then
  binding+=(--host "${FLYX_HOST:-0.0.0.0}")
fi
if [[ $port_named -eq 0 ]]; then
  binding+=(--port "${FLYX_PORT:-8080}")
fi

# The port the address below prints: the last `--port` of the arguments and FLYX_ARGS, since `serve` takes the last,
# and FLYX_PORT when none of them names one.
port="${FLYX_PORT:-8080}"
previous=""
for word in "${args[@]}" "${extra[@]}"; do
  if [[ $previous == --port ]]; then
    port="$word"
  fi
  previous="$word"
done

# A key, unless the caller supplied one: FLYX_API_KEY, --api-key, or --no-auth (the keyless answer for
# network_mode: host, where the container has the machine's addresses).  None of those: a random key, kept in the
# volume so a restart keeps the address a browser has bookmarked, and the address is printed because the dashboard
# needs the key in its own address to load a model (docs/building.md's Docker section).
words="${args[*]} ${extra[*]:-}"
api=()
if [[ -n ${FLYX_API_KEY:-} ]]; then
  api=(--api-key "$FLYX_API_KEY")
elif [[ $words == *--api-key* || $words == *--no-auth* ]]; then
  :
else
  keyfile="${HOME:-/data}/docker-api-key"
  key=""
  if [[ -r $keyfile ]]; then
    key="$(head -c 128 "$keyfile" 2>/dev/null | tr -dc 'a-fA-F0-9')"
  fi
  if [[ ${#key} -lt 32 ]]; then
    key="$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')"
    if ! (umask 077 && printf '%s\n' "$key" > "$keyfile" 2>/dev/null); then
      echo "flyx-server: $keyfile is not writable, so this key lasts for the run only" >&2
    fi
  fi
  api=(--api-key "$key")
  echo "flyx-server: the dashboard at http://127.0.0.1:${port}/#key=$key"
  echo "flyx-server: the same key as Authorization: Bearer $key"
fi

if (($#)); then
  exec flyx serve "${binding[@]}" "${args[@]}" "${api[@]}"
fi

serve=(flyx serve "${binding[@]}")
if [[ -n ${FLYX_MODEL:-} ]]; then
  serve+=("$FLYX_MODEL")
fi
if ((${#extra[@]})); then
  serve+=("${extra[@]}")
fi
exec "${serve[@]}" "${api[@]}"
