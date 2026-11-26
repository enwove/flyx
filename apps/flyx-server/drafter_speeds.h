// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// How fast this machine's own runs decoded with a drafter, by model: every request a model served drafting with the
// drafter ran to its end, its output tokens over its decode time, kept in <config>/flyx/drafter_speeds.json.  The
// verified drafters a model offers are ordered by it (docs/models.md); none is measured anywhere else.
//
// A request only adds to sums in memory (no file is read or written on the scheduler's thread; the file is read at the
// first query); a thread of its own writes them every 30 seconds and at exit, adding them to what the file has then
// under a lock the servers sharing the file all take, so each keeps the others' runs; a write that fails keeps its
// runs for the next.
#pragma once

#include <string>
#include <vector>

#include "apps/common/model_source.h"
#include "nlohmann/json.hpp"

namespace flyx::server {

// A finished request's tokens and decode time, for the model (a Hub name "org/name[:quant]", or an absolute path) and
// each drafter (a Hub repository) it drafted with.
void add_drafter_speed(const std::string& model, const std::vector<std::string>& drafters, int tokens, double decode_ms);
// The mean decode tokens a second of the drafter's runs with the model, or with any of a Hub repository's models
// ("org/name": its quantizations, "org/name:quant"); 0: none measured.
double drafter_speed(const std::string& model, const std::string& drafter);
// Verified drafters (apps::known_drafters) as the API gives them: those made for the model first, each tier the fastest
// on this machine's own runs with `model` first (the list's order among those not run here): [{repo, kind, base, exact,
// relation, size, tok_s}] (relation: null for one made for the model, and where the Hub was not asked; size: null where
// unknown; tok_s: null before a run).
nlohmann::ordered_json known_drafters_json(const std::string& model, const std::vector<apps::KnownDrafter>& known);
// Writes what has not been written yet (tests; the writer thread does it every 30 seconds and at exit); false when the
// write failed (the lock not had within 2 seconds, or the file not written), the runs then kept for the next.
bool flush_drafter_speeds();
// The file ("" when there is no configuration directory).
std::string drafter_speeds_path();

}  // namespace flyx::server
