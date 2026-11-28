// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "models/model.h"

namespace flyx::models::qwen4exp {
std::unique_ptr<Model> open(const ModelFile& file, const ModelOptions& options, std::string* err);
}  // namespace flyx::models::qwen4exp
