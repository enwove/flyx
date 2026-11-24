// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "models/vision/preprocess.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace flyx::models::vision {
namespace {

// Catmull-Rom / Keys cubic kernel with a = -0.5 (PIL Image.BICUBIC / PyTorch bicubic).
double cubic_weight(double x) {
  x = std::abs(x);
  if (x < 1.0) return ((1.5 * x - 2.5) * x) * x + 1.0;
  if (x < 2.0) return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
  return 0.0;
}

struct Tap {
  int start = 0;
  int count = 0;
  std::vector<double> w;
};

std::vector<Tap> build_bicubic_taps(int src_len, int dst_len) {
  std::vector<Tap> taps(static_cast<size_t>(dst_len));
  const double scale = double(src_len) / double(dst_len);
  const double filter_scale = scale > 1.0 ? scale : 1.0;  // antialias when downsampling
  const double support = 2.0 * filter_scale;
  for (int o = 0; o < dst_len; ++o) {
    const double center = (double(o) + 0.5) * scale - 0.5;
    const int left = std::max(0, int(std::ceil(center - support - 1e-9)));
    const int right = std::min(src_len - 1, int(std::floor(center + support + 1e-9)));
    Tap& t = taps[size_t(o)];
    t.start = left;
    t.count = std::max(1, right - left + 1);
    t.w.resize(size_t(t.count));
    double sum = 0.0;
    for (int i = 0; i < t.count; ++i) {
      const int idx = std::min(src_len - 1, left + i);
      const double w = cubic_weight((double(idx) - center) / filter_scale);
      t.w[size_t(i)] = w;
      sum += w;
    }
    if (sum != 0.0) {
      for (double& w : t.w) w /= sum;
    } else {
      t.w[0] = 1.0;
    }
  }
  return taps;
}

}  // namespace

std::vector<float> resize_bicubic_rgb(const uint8_t* src, int sw, int sh, int dw, int dh) {
  if (sw == dw && sh == dh) {
    std::vector<float> out(size_t(dw) * size_t(dh) * 3);
    for (size_t i = 0; i < out.size(); ++i) out[i] = float(src[i]);
    return out;
  }
  // Horizontal pass: [sh, sw, 3] -> [sh, dw, 3]
  const auto x_taps = build_bicubic_taps(sw, dw);
  std::vector<float> tmp(size_t(sh) * size_t(dw) * 3, 0.0f);
  for (int y = 0; y < sh; ++y) {
    const uint8_t* srow = src + size_t(y) * size_t(sw) * 3;
    float* drow = tmp.data() + size_t(y) * size_t(dw) * 3;
    for (int x = 0; x < dw; ++x) {
      const Tap& t = x_taps[size_t(x)];
      double c0 = 0.0, c1 = 0.0, c2 = 0.0;
      for (int i = 0; i < t.count; ++i) {
        const double w = t.w[size_t(i)];
        const uint8_t* p = srow + size_t(t.start + i) * 3;
        c0 += w * double(p[0]);
        c1 += w * double(p[1]);
        c2 += w * double(p[2]);
      }
      drow[size_t(x) * 3 + 0] = float(c0);
      drow[size_t(x) * 3 + 1] = float(c1);
      drow[size_t(x) * 3 + 2] = float(c2);
    }
  }
  // Vertical pass: [sh, dw, 3] -> [dh, dw, 3]
  const auto y_taps = build_bicubic_taps(sh, dh);
  std::vector<float> out(size_t(dh) * size_t(dw) * 3, 0.0f);
  for (int y = 0; y < dh; ++y) {
    const Tap& t = y_taps[size_t(y)];
    float* drow = out.data() + size_t(y) * size_t(dw) * 3;
    for (int x = 0; x < dw; ++x) {
      double c0 = 0.0, c1 = 0.0, c2 = 0.0;
      for (int i = 0; i < t.count; ++i) {
        const double w = t.w[size_t(i)];
        const float* p = tmp.data() + (size_t(t.start + i) * size_t(dw) + size_t(x)) * 3;
        c0 += w * double(p[0]);
        c1 += w * double(p[1]);
        c2 += w * double(p[2]);
      }
      drow[size_t(x) * 3 + 0] = float(std::clamp(c0, 0.0, 255.0));
      drow[size_t(x) * 3 + 1] = float(std::clamp(c1, 0.0, 255.0));
      drow[size_t(x) * 3 + 2] = float(std::clamp(c2, 0.0, 255.0));
    }
  }
  return out;
}

bool smart_resize_dims(int h, int w, int factor, int64_t min_pixels, int64_t max_pixels,
                       int* out_h, int* out_w, std::string* err) {
  if (h <= 0 || w <= 0 || factor <= 0) {
    if (err) *err = "invalid image dimensions";
    return false;
  }
  if (double(std::max(h, w)) / double(std::min(h, w)) > 200.0) {
    if (err) *err = "image aspect ratio exceeds 200:1";
    return false;
  }
  const int64_t min_tile = int64_t(factor) * factor;
  if (max_pixels < min_tile) max_pixels = min_tile;
  if (min_pixels > max_pixels) min_pixels = max_pixels;
  int64_t h_bar = std::max<int64_t>(factor, int64_t(std::llround(double(h) / factor)) * factor);
  int64_t w_bar = std::max<int64_t>(factor, int64_t(std::llround(double(w) / factor)) * factor);
  if (h_bar * w_bar > max_pixels) {
    const double beta = std::sqrt(double(h) * double(w) / double(max_pixels));
    h_bar = std::max<int64_t>(factor, int64_t(std::floor(double(h) / beta / factor)) * factor);
    w_bar = std::max<int64_t>(factor, int64_t(std::floor(double(w) / beta / factor)) * factor);
  } else if (h_bar * w_bar < min_pixels) {
    const double beta = std::sqrt(double(min_pixels) / (double(h) * double(w)));
    h_bar = std::max<int64_t>(factor, int64_t(std::ceil(double(h) * beta / factor)) * factor);
    w_bar = std::max<int64_t>(factor, int64_t(std::ceil(double(w) * beta / factor)) * factor);
  }
  // Enforce max_pixels strictly (extreme aspect ratio floor-clamp or upscale ceil overshoot).
  if (h_bar * w_bar > max_pixels) {
    if (h_bar >= w_bar) {
      h_bar = std::max<int64_t>(factor, (max_pixels / w_bar / factor) * factor);
      if (h_bar * w_bar > max_pixels) w_bar = std::max<int64_t>(factor, (max_pixels / h_bar / factor) * factor);
    } else {
      w_bar = std::max<int64_t>(factor, (max_pixels / h_bar / factor) * factor);
      if (h_bar * w_bar > max_pixels) h_bar = std::max<int64_t>(factor, (max_pixels / w_bar / factor) * factor);
    }
  }
  if (h_bar > INT32_MAX || w_bar > INT32_MAX) {
    if (err) *err = "resized image dimensions overflow";
    return false;
  }
  *out_h = int(h_bar);
  *out_w = int(w_bar);
  return true;
}

bool Qwen35VisionAdapter::preprocess(const interfaces::RgbImage& img, int max_tokens_budget,
                                     interfaces::PreparedMedia* out, std::string* err) const {
  *out = {};
  if (img.width <= 0 || img.height <= 0 ||
      img.rgb.size() < size_t(img.width) * size_t(img.height) * 3) {
    if (err) *err = "invalid RGB image buffer";
    return false;
  }
  const int P = p_.patch_size;
  const int M = p_.merge_size;
  const int T_p = p_.temporal_patch_size;
  const int factor = P * M;
  int64_t max_px = p_.max_pixels;
  if (max_tokens_budget > 0) {
    const int64_t budget_px = int64_t(max_tokens_budget) * int64_t(factor) * int64_t(factor);
    if (budget_px < max_px) max_px = budget_px;
  }
  int rh = 0, rw = 0;
  if (!smart_resize_dims(img.height, img.width, factor, p_.min_pixels, max_px, &rh, &rw, err)) return false;

  const std::vector<float> resized = resize_bicubic_rgb(img.rgb.data(), img.width, img.height, rw, rh);
  const int grid_h = rh / P;
  const int grid_w = rw / P;
  const int n_patches = grid_h * grid_w;
  const int slice_dim = 3 * P * P;
  const int patch_dim = T_p * slice_dim;

  // Pack patches in spatial-merge order: (bh, bw, mh, mw).
  // Inside each patch: temporal slice t=0 [3, P, P], then temporal slice t=1 [3, P, P]
  // (for static images, t=1 duplicates t=0, matching Conv3d with v.patch_embd.weight + v.patch_embd.weight.1).
  out->patches.resize(size_t(n_patches) * size_t(patch_dim));
  const float inv255 = 1.0f / 255.0f;
  const int bh_n = grid_h / M, bw_n = grid_w / M;
  int p_idx = 0;
  for (int bh = 0; bh < bh_n; ++bh) {
    for (int bw = 0; bw < bw_n; ++bw) {
      for (int mh = 0; mh < M; ++mh) {
        for (int mw = 0; mw < M; ++mw) {
          const int y0 = (bh * M + mh) * P;
          const int x0 = (bw * M + mw) * P;
          float* dst = out->patches.data() + size_t(p_idx) * size_t(patch_dim);
          for (int c = 0; c < 3; ++c) {
            const float mean = p_.image_mean[c];
            const float inv_std = 1.0f / p_.image_std[c];
            for (int py = 0; py < P; ++py) {
              const float* srow = resized.data() + (size_t(y0 + py) * size_t(rw) + size_t(x0)) * 3;
              float* d0 = dst + (size_t(c) * P + size_t(py)) * P;
              for (int px = 0; px < P; ++px) {
                const float v = (srow[size_t(px) * 3 + c] * inv255 - mean) * inv_std;
                d0[px] = v;
                if (T_p == 2) d0[slice_dim + px] = v;
              }
            }
          }
          for (int t = 2; t < T_p; ++t)
            std::copy_n(dst, slice_dim, dst + size_t(t) * slice_dim);
          ++p_idx;
        }
      }
    }
  }

  const int n_embeds = bh_n * bw_n;
  out->n_patches = n_patches;
  out->patch_dim = patch_dim;
  out->grid_thw = {1, grid_h, grid_w};
  out->cu_seqlens = {0, n_patches};
  out->n_embeds = n_embeds;
  out->tokens.assign(size_t(n_embeds), p_.placeholder_token);
  out->embed_offsets.clear();  // empty = every token in [0, n_embeds) receives a vision embedding
  return true;
}

interfaces::PromptPositions Qwen35VisionAdapter::build_positions(
    std::span<const int32_t> tokens, std::span<const interfaces::MediaSpan> spans) const {
  interfaces::PromptPositions res{};
  const int32_t L = int32_t(tokens.size());
  if (spans.empty()) {
    res.is_3d = false;
    res.rope_delta = 0;
    return res;
  }
  res.is_3d = true;
  res.pos_3d.resize(size_t(L) * 3, 0);
  res.pos_rope_deltas.resize(size_t(L) + 1, 0);
  const int M = std::max(1, p_.merge_size);
  int32_t cur = 0;
  int32_t tok_i = 0;
  size_t sp_i = 0;
  res.pos_rope_deltas[0] = 0;
  while (tok_i < L) {
    if (sp_i < spans.size() && spans[sp_i].token_pos == tok_i) {
      const auto& sp = spans[sp_i++];
      const int32_t llm_t = std::max<int32_t>(1, sp.grid_thw[0]);
      const int32_t llm_h = std::max<int32_t>(1, sp.grid_thw[1] / M);
      const int32_t llm_w = std::max<int32_t>(1, sp.grid_thw[2] / M);
      int32_t k = 0;
      for (int32_t t = 0; t < llm_t && k < sp.n_tokens; ++t)
        for (int32_t h = 0; h < llm_h && k < sp.n_tokens; ++h)
          for (int32_t w = 0; w < llm_w && k < sp.n_tokens; ++w, ++k) {
            const size_t idx = size_t(tok_i + k) * 3;
            res.pos_3d[idx + 0] = cur + t;
            res.pos_3d[idx + 1] = cur + h;
            res.pos_3d[idx + 2] = cur + w;
            res.pos_rope_deltas[size_t(tok_i + k) + 1] = cur - (tok_i + k + 1);
          }
      tok_i += sp.n_tokens;
      cur += std::max({llm_t, llm_h, llm_w});
      if (tok_i <= L) res.pos_rope_deltas[size_t(tok_i)] = cur - tok_i;
    } else {
      const size_t idx = size_t(tok_i) * 3;
      res.pos_3d[idx + 0] = cur;
      res.pos_3d[idx + 1] = cur;
      res.pos_3d[idx + 2] = cur;
      ++cur;
      ++tok_i;
      res.pos_rope_deltas[size_t(tok_i)] = cur - tok_i;
    }
  }
  res.rope_delta = cur - L;
  return res;
}

}  // namespace flyx::models::vision
