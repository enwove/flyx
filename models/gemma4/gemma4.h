// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Google's Gemma 4 (arch "gemma4": gemma-4-31B, gemma-4-12B) from its GGUF file: every layer attention + a gated
// MLP, both between RMSNorms on either side ("sandwich" norms), the layer's output times a learned scalar.  Most
// layers slide over a window (theta 10k, the whole head rotated); the others attend to the whole context with
// wider heads, fewer K/V heads, k's projection for v (attention_k_eq_v) and proportional rotary (theta 1M, a quarter
// of the head turned at the whole head's frequencies).  q and k RMS-normed per head with weights, v without, q.k
// unscaled; GELU (tanh) MLPs; the embedding times sqrt(hidden), tied to the head, whose logits are softcapped.
// Its drafters: its assistant beside it (models/gemma4/assistant.h, --drafter mtp), and prompt lookup (the
// engine's).
#pragma once

#include <memory>
#include <string>

#include "models/model.h"

namespace flyx::models::gemma4 {

std::unique_ptr<Model> open(const ModelFile& file, const ModelOptions& o, std::string* err);

}  // namespace flyx::models::gemma4
