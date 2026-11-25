// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Architecture-agnostic DSpark drafter parsing and block builder (GGUF arch "dflash" with
// dflash.has_markov_context = true).
#pragma once

#include <vector>

#include <nlohmann/json.hpp>

#include "core/modelfile/modelfile.h"
#include "flyx/backend_abi.h"
#include "models/drafters/target.h"
#include "models/model.h"

namespace flyx::models::drafters {

struct DSparkSpec {
  flyx_dspark_block block{};
  int companion = -1;
  int layers = 0;
  int swa_window = 0;
  double eps = 1e-6;
  std::vector<bool> swa_pattern;
  std::vector<int> taps;  // 0..target.layers (target.layers = post output_norm when target.has_post_norm_tap)
};

// Returns true if a companion header describes a DSpark GGUF drafter (arch "dflash" + has_markov_context = true).
bool is_dspark_companion(const nlohmann::json& header);

// Throws unless the DSpark companion was made for this target (width and optional vocabulary).
// Checked at scan() whether or not the drafter is chosen; internal config is read when chosen (parse_dspark_gguf).
void check_dspark_made_for(const nlohmann::json& header, const TargetSpec& target);

// Parse and validate a DSpark companion GGUF header + tensor shapes against `target`.
DSparkSpec parse_dspark_gguf(const ModelFile& file, const nlohmann::json& header, const TargetSpec& target,
                             int companion);

// Materialize FLYX_BLOCK_DSPARK_LAYER blocks, FLYX_BLOCK_DSPARK, and its hidden ring state in `T`.
// Returns the FLYX_BLOCK_DSPARK block index.
int build_dspark(Tables& T, const ModelFile& file, const DSparkSpec& spec, const TargetSpec& target);

}  // namespace flyx::models::drafters
