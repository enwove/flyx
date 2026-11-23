// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The backend entry point for the session budget test. The test supplies the CPU backend's table with its measured
// session size and allocation replaced, then the engine loads this module like any other backend.
#include "flyx/backend_abi.h"

namespace {
const flyx_backend_api* api = nullptr;
}

extern "C" FLYX_BACKEND_EXPORT void flyx_test_delegate(const flyx_backend_api* delegated) { api = delegated; }
extern "C" FLYX_BACKEND_EXPORT const flyx_backend_api* flyx_backend_get_api() { return api; }
