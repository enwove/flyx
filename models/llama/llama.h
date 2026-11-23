// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The Llama family's dense decoders (attention layers, SwiGLU MLPs) as blocks
// and programs: Llama itself (arch "llama": MiniCPM5) and Qwen3 (arch "qwen3"),
// which adds per-head q/k norms.  Written only from block kinds the backends
// already run: Qwen3 changed no backend (stage 3); Llama's
// attention names no q/k norm, which the backends take as none.  One drafter:
// DV, in a Llama .flyx file (models/dv.h).
#pragma once

#include <memory>
#include <string>

#include "models/model.h"

namespace flyx::models::llama {

std::unique_ptr<Model> open(const ModelFile& file, const ModelOptions& o, std::string* err);

}  // namespace flyx::models::llama
