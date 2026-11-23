// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Registers statically linked backends into core's registry for single-file standalone builds
// (FLYX_STANDALONE_EXE=ON). Only linked into flyx-standalone; dynamic module builds never compile this file.
#include "core/exec/engine.h"
#include "include/flyx/backend_abi.h"

extern "C" {
const flyx_backend_api* flyx_backend_get_api_cpu_ref(void);
#ifdef FLYX_HAVE_BUILTIN_CUDA
const flyx_backend_api* flyx_backend_get_api_cuda(void);
#endif
}

namespace flyx::apps {

void register_builtin_backends() {
#ifdef FLYX_HAVE_BUILTIN_CUDA
  flyx::register_builtin_backend("cuda", &flyx_backend_get_api_cuda);
#endif
  flyx::register_builtin_backend("cpu_ref", &flyx_backend_get_api_cpu_ref);
}

}  // namespace flyx::apps
