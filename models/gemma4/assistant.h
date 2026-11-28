// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Gemma 4's assistant (google/gemma-4-*-it-assistant: transformers' Gemma4AssistantForCausalLM and
// Gemma4UnifiedAssistantForCausalLM) beside its target, as the target's MTP drafter (backend_abi.h
// flyx_gemma4_mtp_block): a drafter directory, its config.json and safetensors (core/modelfile/drafter_dir.cpp).
#pragma once
#include <nlohmann/json.hpp>

#include "core/modelfile/modelfile.h"
#include "models/model.h"

namespace flyx::models::gemma4 {

// What the assistant reads of its target: its embedding and head, and the K/V of its last layer of each kind.
struct AssistantTarget {
  int hidden = 0, vocab = 0;
  int embed = -1, head = -1;                      // blocks
  int kv_sliding = -1, kv_full = -1;              // states
  int n_kv_sliding = 0, n_kv_full = 0, hd_sliding = 0, hd_full = 0;
  int window = 0;                                 // the target's sliding window
};

// Whether a companion file is a Gemma 4 assistant; throws on one made for a target of another width or vocabulary.
bool is_assistant(const nlohmann::json& companion_header, int target_hidden, int target_vocab);

// The drafter block, its layers and its ring's state in T from the companion's config and tensors; throws on an
// assistant that does not read this target's K/V.
int build_assistant(Tables& T, const ModelFile& file, int companion, const AssistantTarget& t);

}  // namespace flyx::models::gemma4
