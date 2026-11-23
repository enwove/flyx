#!/usr/bin/env bash
# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#
# Prefill time of two builds (bin directories) on one long prompt, alternating: bench/ab/ab.sh prefill (this name
# stays for the commands in older records).
#
#   bash bench/ab/ab_prefill.sh MODEL PROMPT_FILE ~/bin-before ~/flyx-build/bin [pairs=4] [run options...]
PROMPT_FILE=$2 exec bash "$(dirname "$0")/ab.sh" prefill "$1" "${@:3}"
