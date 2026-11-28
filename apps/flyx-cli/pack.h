// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// flyx pack (pack.cpp): a GGUF model, its drafters and the settings measured best for it, in one .flyx file.
#pragma once

namespace flyx::pack {

// flyx pack MODEL -o FILE [options]: argv[1] is "pack".
int run(int argc, char** argv);

}  // namespace flyx::pack
