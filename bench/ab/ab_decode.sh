#!/usr/bin/env bash
# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#
# Decode speed of two environment settings of one build, alternating: bench/ab/ab.sh decode (this name stays for
# the commands in older records).
#
#   bash bench/ab/ab_decode.sh MODEL "" "FLYX_BACKEND_OPTIONS=pdl=1" [pairs=6] [bench options...]
exec bash "$(dirname "$0")/ab.sh" decode "$@"
