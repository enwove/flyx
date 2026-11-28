// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// EAGLE-4 from a "draft" config of kind "eagle4" and its draft.* tensors, the names python/convert writes
// (python/convert/flyx_convert/eagle4.py): fc, fc_norm.<i> (one a tap), L0.{in_norm, hidden_norm, post_norm, attn.q,
// attn.k, attn.v, attn.o, mlp.gate, mlp.up, mlp.down}, norm, head, cand_down, cand_up.
#include "models/drafters/eagle4.h"

#include <algorithm>
#include <cstring>
#include <regex>
#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "core/modelfile/modelfile.h"
#include "models/drafters/config.h"

#include "models/drafters/directory.h"

namespace flyx::models::drafters {
namespace {

using nlohmann::json;

void expect(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("EAGLE-4 drafter: " + what);
}

// python/convert/flyx_convert/eagle4.py's canonical_eagle4_config(config, target=None): the variant the released
// drafter is (one layer, per-tap norms, its normalised output carried on, the low-rank candidate head, the target's
// vocabulary), anything else refused.
json eagle4_config(const json& sc) {
  auto is_int = [&](const char* k) { return sc.contains(k) && sc[k].is_number_integer(); };
  auto positive = [&](const char* k) { return is_int(k) && sc[k].get<int64_t>() > 0; };
  expect(sc.value("architectures", json()) == json::array({"LlamaForCausalLMEagle4"}), "architectures");
  expect(sc.value("model_type", json()) == "llama", "model_type (llama)");
  expect(is_int("num_hidden_layers") && sc["num_hidden_layers"].get<int64_t>() == 1, "one layer");
  expect(sc.value("hidden_act", json()) == "silu", "hidden_act (silu)");
  expect(falsy(sc.value("attention_bias", json())) && falsy(sc.value("mlp_bias", json())),
                 "attention or MLP biases");
  expect(sc.value("tie_word_embeddings", json()) == json(false), "tied embeddings (it has a head of its own)");
  expect(sc.value("pretraining_tp", json(1)) == json(1), "pretraining_tp");
  for (const char* k : {"hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "vocab_size"})
    expect(positive(k), k);
  const int64_t H = sc["hidden_size"].get<int64_t>(), vocab = sc["vocab_size"].get<int64_t>();
  const int64_t heads = sc["num_attention_heads"].get<int64_t>(), kv = sc["num_key_value_heads"].get<int64_t>();
  const json hdj = sc.value("head_dim", json(H / heads));
  expect(hdj.is_number_integer() && hdj.get<int64_t>() > 0 && hdj.get<int64_t>() % 2 == 0 && heads % kv == 0,
                 "heads and head_dim");
  expect(sc.value("rms_norm_eps", json()).is_number() && sc["rms_norm_eps"].get<double>() > 0, "rms_norm_eps");
  expect(sc.value("target_hidden_size", json(H)) == json(H), "a target width other than its own");
  expect(sc.value("draft_vocab_size", json(vocab)) == json(vocab), "a draft vocabulary of its own (d2t / t2d)");
  expect(sc.value("fc_norm", json()) == json(true), "fc_norm must be true");
  expect(sc.value("norm_output", json()) == json(true), "norm_output must be true");
  const json ec = sc.value("eagle_config", json());
  expect(ec.is_object() && ec.value("use_aux_hidden_state", json()) == json(true),
                 "eagle_config.use_aux_hidden_state");
  const json ids = ec.value("eagle_aux_hidden_state_layer_ids", json());
  expect(ids.is_array() && !ids.empty(), "eagle_aux_hidden_state_layer_ids");
  std::vector<int64_t> taps;
  for (const auto& i : ids) {
    expect(i.is_number_integer() && i.get<int64_t>() >= 0 &&
                       (taps.empty() || i.get<int64_t>() + 1 > taps.back()),
                   "eagle_aux_hidden_state_layer_ids (increasing, non-negative)");
    taps.push_back(i.get<int64_t>() + 1);
  }
  expect(sc.value("num_aux_hidden_states", json(taps.size())) == json(taps.size()), "num_aux_hidden_states");
  const json nt = sc.value("num_target_layers", json());
  expect(nt.is_null() || (nt.is_number_integer() && nt.get<int64_t>() > 0), "num_target_layers");
  const json rp = sc.value("rope_parameters", json());
  bool rope_ok = rp.is_object() && rp.value("rope_type", json("default")) == "default" &&
                 rp.value("rope_theta", json()).is_number() && sc.value("rope_scaling", json()).is_null();
  for (auto it = rope_ok ? rp.begin() : json::const_iterator(); rope_ok && it != rp.end(); ++it)
    rope_ok = it.key() == "rope_type" || it.key() == "rope_theta";
  expect(rope_ok, "rope_parameters (default rotary, its theta)");
  expect(sc.value("rope_is_neox_style", json(true)) == json(true), "rotary other than NeoX's halves");
  const json types = sc.value("layer_types", json());
  expect(types.is_array() && types.size() == 1, "layer_types (one)");
  int64_t window = 0;
  if (types[0] == "sliding_attention") {
    expect(sc.value("use_sliding_window", json()) == json(true) && positive("sliding_window"),
                   "a sliding layer without its window");
    window = sc["sliding_window"].get<int64_t>();
  } else {
    expect(types[0] == "full_attention" && falsy(sc.value("use_sliding_window", json())), "layer type");
  }
  const json head = sc.value("efficient_lm_head_config", json());
  expect(head.is_object() && head.size() == 3 && head.value("type", json()) == "low_rank" &&
                     head.value("rank", json()).is_number_integer() &&
                     head.value("num_candidates", json()).is_number_integer() && head["rank"].get<int64_t>() > 0 &&
                     head["rank"].get<int64_t>() <= H && head["num_candidates"].get<int64_t>() > 0 &&
                     head["num_candidates"].get<int64_t>() <= vocab,
                 "efficient_lm_head_config (low_rank, its rank and candidates)");
  json d;
  d["kind"] = "eagle4";
  d["hidden"] = H;
  d["layers"] = 1;
  d["n_heads"] = heads;
  d["n_kv"] = kv;
  d["head_dim"] = hdj.get<int64_t>();
  d["inter"] = sc["intermediate_size"];
  d["eps"] = sc["rms_norm_eps"].get<double>();
  d["rope_theta"] = rp["rope_theta"].get<double>();
  d["window"] = window;
  d["taps"] = taps;
  d["cand_rank"] = head["rank"];
  d["n_cands"] = head["num_candidates"];
  d["target"] = {{"hidden", H}, {"vocab", vocab}, {"layers", nt}};
  return d;
}

// flyx_convert/eagle4.py's _RULES (the released checkpoint's names); "" for its copy of the target's embedding.
bool canonical_eagle4(const std::string& name, std::string* out) {
  static const std::vector<std::pair<std::regex, std::string>> rules = {
      {std::regex(R"(fc\.weight)"), "fc"},
      {std::regex(R"(fc_norm\.(\d+)\.weight)"), "fc_norm.$1"},
      {std::regex(R"(layers\.0\.input_layernorm\.weight)"), "L0.in_norm"},
      {std::regex(R"(layers\.0\.hidden_norm\.weight)"), "L0.hidden_norm"},
      {std::regex(R"(layers\.0\.post_attention_layernorm\.weight)"), "L0.post_norm"},
      {std::regex(R"(layers\.0\.self_attn\.(q|k|v|o)_proj\.weight)"), "L0.attn.$1"},
      {std::regex(R"(layers\.0\.mlp\.(gate|up|down)_proj\.weight)"), "L0.mlp.$1"},
      {std::regex(R"(norm\.weight)"), "norm"},
      {std::regex(R"(lm_head\.weight)"), "head"},
      {std::regex(R"(efficient_lm_head_down_proj\.weight)"), "cand_down"},
      {std::regex(R"(efficient_lm_head\.weight)"), "cand_up"},
  };
  const std::string inner = inner_name(name);
  if (inner.rfind("embed_tokens.", 0) == 0) return out->clear(), true;
  for (const auto& [re, fmt] : rules) {
    std::smatch m;
    if (std::regex_match(inner, m, re)) return *out = m.format(fmt), true;
  }
  return false;
}

}  // namespace

DirectoryDraft eagle4_directory(const ModelFile::Contents& raw) {
  DirectoryDraft out;
  out.config = eagle4_config(raw.directory_config);
  for (const auto& v : raw.tensors) {
    std::string name;
    if (!canonical_eagle4(v.name, &name)) throw std::runtime_error("unknown draft tensor " + std::string(v.name));
    if (!name.empty()) out.add(name, v);
  }
  std::vector<std::string> need = {"fc", "norm", "head", "cand_down", "cand_up"};
  for (const char* n : {"in_norm", "hidden_norm", "post_norm", "attn.q", "attn.k", "attn.v", "attn.o", "mlp.gate",
                        "mlp.up", "mlp.down"})
    need.push_back(std::string("L0.") + n);
  for (size_t i = 0; i < out.config.at("taps").size(); ++i) need.push_back("fc_norm." + std::to_string(i));
  for (const std::string& n : need) out.require(n);
  if (out.tensors.size() != need.size()) throw std::runtime_error("an EAGLE-4 tensor its config does not have");
  return out;
}

Eagle4Spec parse_eagle4_json(const json& d, const TargetSpec& t, int companion) {
  check_config_target(d, t);
  Eagle4Spec spec{};
  spec.companion = companion;
  flyx_eagle4_block& b = spec.block;
  const std::string title = "EAGLE-4";
  b.hidden = config_int(d, "hidden", title);
  const int layers = config_int(d, "layers", title);
  b.n_heads = config_int(d, "n_heads", title);
  b.n_kv = config_int(d, "n_kv", title);
  b.head_dim = config_int(d, "head_dim", title);
  b.inter = config_int(d, "inter", title);
  b.window = config_int(d, "window", title);
  b.cand_rank = config_int(d, "cand_rank", title);
  b.n_cands = config_int(d, "n_cands", title);
  b.theta = d.at("rope_theta").get<double>();
  spec.eps = d.at("eps").get<double>();
  for (int64_t tap : d.at("taps").get<std::vector<int64_t>>()) {
    // Its taps are residual rows (the outputs of target layers): none after the last layer, which flyx's taps take
    // after the final norm.
    expect(tap >= 1 && tap < t.layers,
           "tap after layer " + std::to_string(tap) + " of " + std::to_string(t.layers) + " (1.." +
               std::to_string(t.layers - 1) + ")");
    spec.taps.push_back(int(tap));
  }
  b.n_taps = int32_t(spec.taps.size());
  b.target_hidden = t.hidden;
  b.vocab = t.vocab;
  // Its head's rows as its file has them (python/convert writes "target" for every EAGLE-4): the target's
  // vocabulary, or the target's padded one.
  spec.rows = d.at("target").at("vocab").get<int>();
  expect(spec.rows == t.vocab || spec.rows == t.padded_vocab(), "a vocabulary of its own");
  expect(layers == 1, "one layer (it has " + std::to_string(layers) + ")");
  expect(b.n_taps >= 1 && b.n_taps <= FLYX_EAGLE4_MAX_TAPS,
         "1.." + std::to_string(FLYX_EAGLE4_MAX_TAPS) + " taps");
  expect(b.hidden == t.hidden, "hidden size " + std::to_string(b.hidden) + " is not the target's " +
                                   std::to_string(t.hidden));
  expect(b.hidden > 0 && b.hidden <= kMaxWidth && b.n_heads > 0 && b.n_heads <= kMaxHeads && b.n_kv > 0 &&
             b.n_heads % b.n_kv == 0 && b.head_dim > 0 && b.head_dim % 2 == 0 && b.head_dim <= kMaxHeadDim &&
             b.inter > 0 && b.inter <= kMaxWidth && b.window >= 0 && spec.eps > 0 && std::isfinite(spec.eps) &&
             b.theta > 0 && std::isfinite(b.theta) &&
             int64_t(b.n_taps) * t.hidden <= INT_MAX && int64_t(b.n_heads + 2 * b.n_kv) * b.head_dim <= INT_MAX,
         "invalid dimensions");
  expect(b.cand_rank > 0 && b.cand_rank <= kMaxWidth && b.n_cands >= 1 && b.n_cands <= t.vocab,
         "candidate head outside rank 1.." + std::to_string(kMaxWidth) + ", 1..vocab candidates");
  return spec;
}

int build_eagle4(Tables& T, const Eagle4Spec& spec, const TargetSpec& t) {
  flyx_eagle4_block b = spec.block;
  const int H = b.hidden, q = b.n_heads * b.head_dim, kv = b.n_kv * b.head_dim, I = b.inter;
  const double eps = spec.eps;
  // The file's name for a drafter tensor ("draft.fc"): in the model file, or in its drafter file.
  auto dn = [&](const std::string& name) {
    return spec.companion >= 0 ? ModelFile::companion_name(spec.companion, name) : name;
  };
  b.embed = t.embed;
  b.fc = T.weight(dn("draft.fc"), {H, int64_t(b.n_taps) * t.hidden});
  for (int i = 0; i < FLYX_EAGLE4_MAX_TAPS; ++i)
    b.fc_norm[i] = i < b.n_taps ? T.norm(dn("draft.fc_norm." + std::to_string(i)), t.hidden, 0.0, eps)
                                : flyx_norm{-1, 0, 0.0, 0.0};
  b.in_norm = T.norm(dn("draft.L0.in_norm"), H, 0.0, eps);
  b.hidden_norm = T.norm(dn("draft.L0.hidden_norm"), H, 0.0, eps);
  b.post_norm = T.norm(dn("draft.L0.post_norm"), H, 0.0, eps);
  b.q = T.weight(dn("draft.L0.attn.q"), {q, 2 * H});
  b.k = T.weight(dn("draft.L0.attn.k"), {kv, 2 * H});
  b.v = T.weight(dn("draft.L0.attn.v"), {kv, 2 * H});
  b.o = T.weight(dn("draft.L0.attn.o"), {H, q});
  b.gate = T.weight(dn("draft.L0.mlp.gate"), {I, H});
  b.up = T.weight(dn("draft.L0.mlp.up"), {I, H});
  b.down = T.weight(dn("draft.L0.mlp.down"), {H, I});
  b.norm = T.norm(dn("draft.norm"), H, 0.0, eps);
  b.head = T.weight(dn("draft.head"), {spec.rows, H});
  b.cand_down = T.weight(dn("draft.cand_down"), {b.cand_rank, H});
  b.cand_up = T.weight(dn("draft.cand_up"), {spec.rows, b.cand_rank});
  flyx_state st{};
  st.kind = FLYX_STATE_KV;
  st.draft = 1;
  st.layers = 1;
  st.n_kv = b.n_kv;
  st.head_dim = b.head_dim;
  b.kv = T.state(st);  // every position's: its window is the attention's (flyx_eagle4_block.window)
  flyx_state ring{};
  ring.kind = FLYX_STATE_RING;
  ring.draft = 1;
  ring.width = H;
  b.ring = T.state(ring);
  flyx_block blk{};
  blk.kind = FLYX_BLOCK_EAGLE4;
  blk.eagle4 = b;
  return T.block(blk);
}

}  // namespace flyx::models::drafters
