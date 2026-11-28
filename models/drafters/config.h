// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <climits>
#include <cmath>
#include "models/drafters/target.h"
#include "models/model.h"

namespace flyx::models::drafters {

struct Qwen3BlockConfig {
  int hidden = 0, layers = 0, n_heads = 0, n_kv = 0, head_dim = 0, inter = 0, block = 0, mask_token = 0;
  double theta = 0.0, eps = 0.0;
  std::vector<int> taps;  // 1-based (1..target.layers; target.layers: after the final norm)
};
inline std::string mismatch_msg(const TargetSpec& target) {
  return "a drafter made for another model (this one: width " + std::to_string(target.hidden) + ", vocabulary " +
         std::to_string(target.vocab) + ", " + std::to_string(target.layers) + " layers)";
}

inline Drafter block_drafter_kind(const nlohmann::json& draft) {
  const Drafter k = draft.is_object() ? drafter_named(draft.value("kind", "")) : Drafter::kNone;
  return is_tapped_drafter(k) ? k : Drafter::kNone;
}

inline int config_int(const nlohmann::json& j, const char* key, const std::string& drafter) {
  const auto& v = j.at(key);
  if (!v.is_number_integer()) throw std::runtime_error("invalid " + drafter + " drafter config");
  const int64_t n = v.get<int64_t>();
  if (n < INT_MIN || n > INT_MAX) throw std::runtime_error("invalid " + drafter + " drafter config");
  return int(n);
}

inline Qwen3BlockConfig parse_qwen3_block(const nlohmann::json& j, const TargetSpec& target, const std::string& title) {
  Qwen3BlockConfig c;
  c.hidden = config_int(j, "hidden", title);
  c.layers = config_int(j, "layers", title);
  c.n_heads = config_int(j, "n_heads", title);
  c.n_kv = config_int(j, "n_kv", title);
  c.head_dim = config_int(j, "head_dim", title);
  c.inter = config_int(j, "inter", title);
  c.block = config_int(j, "block", title);
  c.mask_token = config_int(j, "mask_token", title);
  for (int64_t t : j.at("taps").get<std::vector<int64_t>>()) {
    if (t < 1 || t > target.layers)
      throw std::runtime_error(title + " tap after layer " + std::to_string(t) + " of " +
                               std::to_string(target.layers));
    c.taps.push_back(int(t));
  }
  c.theta = j.at("rope_theta").get<double>();
  c.eps = j.at("eps").get<double>();
  if (c.hidden <= 0 || c.hidden > kMaxWidth || c.layers <= 0 || c.layers > kMaxLayers || c.n_heads <= 0 ||
      c.n_heads > kMaxHeads || c.n_kv <= 0 || c.n_heads % c.n_kv != 0 || c.head_dim <= 0 || c.head_dim % 2 != 0 ||
      c.head_dim > kMaxHeadDim || c.inter <= 0 || c.inter > kMaxWidth || c.mask_token < 0 ||
      c.mask_token >= target.padded_vocab() || !(c.eps > 0) || !std::isfinite(c.eps) || !(c.theta > 0) ||
      !std::isfinite(c.theta) || c.taps.empty() || int64_t(c.taps.size()) * target.hidden > INT_MAX ||
      int64_t(2) * c.layers * c.n_kv * c.head_dim > INT_MAX || int64_t(c.n_heads + 2 * c.n_kv) * c.head_dim > INT_MAX)
    throw std::runtime_error("invalid " + title + " drafter config");
  if (c.hidden != target.hidden)
    throw std::runtime_error(title + " hidden size " + std::to_string(c.hidden) + " does not match the target's " +
                             std::to_string(target.hidden));
  return c;
}

inline void check_config_target(const nlohmann::json& j, const TargetSpec& target) {
  if (block_drafter_kind(j) == Drafter::kNone) throw std::runtime_error(mismatch_msg(target));
  if (!j.contains("target") || !j["target"].is_object()) return;
  const nlohmann::json& t = j["target"];
  const int tv = t.value("vocab", target.vocab);
  const bool vocab_ok = tv == target.vocab || tv == target.padded_vocab();
  const bool layers_ok = !t.contains("layers") || t["layers"].is_null() || t["layers"].get<int>() == target.layers;
  if (t.value("hidden", target.hidden) != target.hidden || !vocab_ok || !layers_ok)
    throw std::runtime_error(mismatch_msg(target));
}

}  // namespace flyx::models::drafters
