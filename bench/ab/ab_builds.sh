#!/usr/bin/env bash
# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#
# Decode speed of two builds (bin directories), alternating: bench/ab/ab.sh decode (this name stays for the
# commands in older records).
#
#   bash bench/ab/ab_builds.sh MODEL ~/bin-before ~/flyx-build/bin [pairs=6] [bench options...]
exec bash "$(dirname "$0")/ab.sh" decode "$@"
