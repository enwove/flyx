// Shim for fastest: json.cpp (llama.cpp) includes ggml.h only for GGML_ASSERT.
// Not part of llama.cpp; see README.flyx.md.
#pragma once

#include <cstdio>
#include <cstdlib>

#define GGML_ASSERT(x)                                                              \
    do {                                                                            \
        if (!(x)) {                                                                 \
            std::fprintf(stderr, "%s:%d: assertion failed: %s\n", __FILE__, __LINE__, #x); \
            std::abort();                                                           \
        }                                                                           \
    } while (0)
