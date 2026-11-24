// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Reusable image preprocessing utilities and VisionAdapter implementations
// shared across multimodal architectures under models/.  Plain C++, zero CUDA.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "core/interfaces/vision.h"

namespace flyx::models::vision {

// Separable bicubic resize (Keys cubic a = -0.5 with antialiasing filter support
// when downsampling, matching PIL/PyTorch BICUBIC resampling) on RGB888 images.
// Returns float32 RGB in [0, 255] of size [dst_h * dst_w * 3].
std::vector<float> resize_bicubic_rgb(const uint8_t* src, int sw, int sh, int dw, int dh);

// Qwen-VL / Qwen3.5 / Qwen3.8 aspect-preserving smart resize: rounds dimensions
// to multiples of `factor` (= patch_size * merge_size) while keeping total pixels
// within [min_pixels, max_pixels] and aspect ratio <= 200.
bool smart_resize_dims(int h, int w, int factor, int64_t min_pixels, int64_t max_pixels,
                       int* out_h, int* out_w, std::string* err);

// Configuration for Qwen3-VL / Qwen3.5 / Qwen3.8 VisionAdapter.
struct Qwen35VisionParams {
  int patch_size = 16;
  int temporal_patch_size = 2;
  int merge_size = 2;
  int out_hidden = 0;
  int32_t placeholder_token = -1;
  int64_t min_pixels = 65536;     // 256 x 256 (shortest_edge in preprocessor_config.json)
  int64_t max_pixels = 16777216;  // 4096 x 4096 (longest_edge in preprocessor_config.json)
  float image_mean[3] = {0.5f, 0.5f, 0.5f};
  float image_std[3] = {0.5f, 0.5f, 0.5f};
};

// VisionAdapter for the Qwen3-VL / Qwen3.5 / Qwen3.8 family:
//   - Smart resize (multiple of patch_size * merge_size = 32) + bicubic resample
//   - Spatial-merge block patch packing [t=0: c,ph,pw | t=1: c,ph,pw]
//   - 3D MRoPE coordinate construction (T, H, W) + decode scalar rope_delta
class Qwen35VisionAdapter final : public interfaces::VisionAdapter {
 public:
  explicit Qwen35VisionAdapter(const Qwen35VisionParams& params) : p_(params) {}
  const char* name() const override { return "qwen35_vit"; }
  int32_t placeholder_token() const override { return p_.placeholder_token; }
  int32_t out_hidden() const override { return p_.out_hidden; }
  bool preprocess(const interfaces::RgbImage& img, int max_tokens_budget,
                  interfaces::PreparedMedia* out, std::string* err) const override;
  interfaces::PromptPositions build_positions(std::span<const int32_t> tokens,
                                              std::span<const interfaces::MediaSpan> spans) const override;

 private:
  Qwen35VisionParams p_;
};

}  // namespace flyx::models::vision
