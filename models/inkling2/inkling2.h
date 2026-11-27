// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <string>

#include "models/model.h"

namespace flyx::models::inkling2 {

std::unique_ptr<Model> open(const ModelFile& file, const ModelOptions& o, std::string* err);

}  // namespace flyx::models::inkling2
