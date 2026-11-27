// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// OpenAI's gpt-oss (arch "gpt-oss": gpt-oss-20b, gpt-oss-120b) from its GGUF file: every layer attention + a
// sparse mixture of experts.  Its attention has biases on q, k, v and o, a learned sink logit per head, YaRN rotary,
// and every other layer (the first among them) a sliding window of 128 positions; its experts an activation of
// their own (flyx_moe_act FLYX_MOE_SWIGLU_OAI) with biases, gate and up stored apart.  One drafter: prompt lookup
// (the engine's).
#pragma once

#include <memory>
#include <string>

#include "models/model.h"

namespace flyx::models::gpt_oss {

std::unique_ptr<Model> open(const ModelFile& file, const ModelOptions& o, std::string* err);

}  // namespace flyx::models::gpt_oss
