// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// A LoRA adapter as Hugging Face PEFT saves it: adapter_config.json and adapter_model.safetensors in a directory.
// Read as a companion of the model it adapts (ModelFile::add): a "lora" description (rank, scale, the linears it
// sits beside) and its tensors under the names a .flyx file gives a target's own adapters
// (target.L{l}.{attn,mlp,gdn}.<linear>.lora_{a,b}, python/convert/flyx_convert/lora.py), their bytes read in place
// in the file's own dtype (F32, BF16 or F16) and in the Hugging Face model's row order ("layout": "hf"): models/<arch>
// knows how its weights' rows sit against those.
//
// Only what the engine can run as y = W x + scale b (a x) is read: plain LoRA on the linears of the decoder layers.
// Anything else in the file (DoRA magnitudes, embedding adapters, modules saved whole, a head's or a vision tower's
// adapters, ranks or alphas per module) is refused by name, never left out: a model without it would decode another
// model.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <set>

#include "core/interfaces/json_depth.h"
#include "core/modelfile/readers.h"
#include "core/os/path.h"

namespace flyx::modelfile {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

using os::from_path;
using os::to_path;

constexpr const char* kConfig = "adapter_config.json";
constexpr const char* kWeights = "adapter_model.safetensors";

// A string of a file as an error message or a listing shows it: at most its first 117 bytes and "...", cut between
// two UTF-8 characters (JSON's strings are UTF-8: a character cut in two would not dump).
std::string shown(const std::string& s) {
  if (s.size() <= 120) return s;
  size_t n = 117;
  while (n > 0 && (static_cast<unsigned char>(s[n]) & 0xC0) == 0x80) --n;
  return s.substr(0, n) + "...";
}

int dtype_of(const std::string& s) {
  static const std::map<std::string, int> m = {{"BF16", FLYX_DT_BF16}, {"F16", FLYX_DT_F16}, {"F32", FLYX_DT_F32}};
  auto it = m.find(s);
  return it == m.end() ? -1 : it->second;
}

json read_json(const fs::path& p, const std::string& what) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  const std::streamoff size = f.tellg();
  if (!f || size < 0) throw std::runtime_error("cannot read " + what);
  if (uint64_t(size) > kPeftConfigMaxBytes)
    throw std::runtime_error(what + ": " + std::to_string(size) + " bytes, more than an adapter's config holds (16 MiB)");
  f.seekg(0);
  std::string text(size_t(size), '\0');
  if (size > 0 && !f.read(text.data(), size)) throw std::runtime_error("cannot read " + what);
  if (!interfaces::json_depth_within(text)) throw std::runtime_error(what + " nested too deeply");
  return json::parse(text);
}

bool given(const json& c, const char* key) {
  if (!c.contains(key) || c[key].is_null()) return false;
  const json& v = c[key];
  if (v.is_boolean()) return v.get<bool>();
  if (v.is_array() || v.is_object() || v.is_string()) return !v.empty();
  return true;
}

// The description of the adapter: what PEFT's LoraConfig says, checked for what the engine can run.
json lora_config(const json& c) {
  const std::string type = c.value("peft_type", json()).is_string() ? c["peft_type"].get<std::string>() : "";
  if (type != "LORA") throw std::runtime_error("peft_type " + c.value("peft_type", json()).dump() + ": LORA is supported");
  if (given(c, "use_dora")) throw std::runtime_error("use_dora: DoRA adapters are not supported");
  if (given(c, "rank_pattern") || given(c, "alpha_pattern"))
    throw std::runtime_error("rank_pattern / alpha_pattern: ranks or alphas per module are not supported");
  if (given(c, "modules_to_save"))
    throw std::runtime_error("modules_to_save " + c["modules_to_save"].dump() + ": modules saved whole are not supported");
  if (given(c, "layer_replication")) throw std::runtime_error("layer_replication is not supported");
  if (given(c, "lora_bias")) throw std::runtime_error("lora_bias: adapters with a bias are not supported");
  if (given(c, "fan_in_fan_out")) throw std::runtime_error("fan_in_fan_out: Conv1D-style weights are not supported");
  // Plain LoRA tensors that mean something else: run on every token as plain LoRA, they would give other outputs.
  if (given(c, "alora_invocation_tokens"))
    throw std::runtime_error("alora_invocation_tokens: an activated LoRA (from its invocation tokens on) is not supported");
  if (given(c, "trainable_token_indices"))
    throw std::runtime_error("trainable_token_indices: adapters that retrain token embeddings are not supported");
  // Initializations that change the base's weights too (PiSSA, OLoRA, CorDA: the base less the adapter's start; LoftQ:
  // a quantized base): the adapter runs only beside that base, unless it was converted on saving (PEFT then writes
  // init_lora_weights true, with twice the rank).
  if (c.contains("init_lora_weights") && c["init_lora_weights"].is_string()) {
    const std::string init = c["init_lora_weights"].get<std::string>();
    for (const char* changed : {"pissa", "olora", "corda", "loftq"})
      if (init.starts_with(changed))
        throw std::runtime_error("init_lora_weights " + init + ": made for a base its initialization changed; " +
                                 "save it converted (path_initial_model_for_weight_conversion) to run it on the model");
  }
  if (!c.contains("r") || !c["r"].is_number_integer() || c["r"].get<int64_t>() <= 0 || c["r"].get<int64_t>() > 4096)
    throw std::runtime_error("r " + c.value("r", json()).dump() + ": a rank of 1 to 4096");
  const int64_t r = c["r"].get<int64_t>();
  // PEFT's default lora_alpha is 8.
  if (c.contains("lora_alpha") && !c["lora_alpha"].is_number())
    throw std::runtime_error("lora_alpha must be a number");
  const json use_rslora = c.value("use_rslora", json());
  if (!use_rslora.is_null() && !use_rslora.is_boolean())
    throw std::runtime_error("use_rslora must be a boolean or null");
  const double alpha = c.value("lora_alpha", 8.0);
  const bool rslora = use_rslora == true;
  const double scale = rslora ? alpha / std::sqrt(double(r)) : alpha / double(r);
  if (!std::isfinite(float(scale)) || float(scale) == 0)
    throw std::runtime_error("lora_alpha must give a finite nonzero float scale");
  return {{"rank", r}, {"scale", scale}, {"alpha", alpha}, {"rslora", rslora}, {"layout", "hf"}};
}

// A PEFT tensor name's layer, linear (as a .flyx file names it, "attn.q") and side ('A' or 'B'); false with `why` for
// a name the engine cannot run.
//   base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight                      (Qwen3, Llama)
//   base_model.model.model.language_model.layers.3.linear_attn.in_proj_qkv.lora_B.weight (Qwen3.5 / 3.8)
//   model.language_model.layers.3.linear_attn.in_proj_qkv.weight.lora_B.default          (the same, as some tools
//                                                                                        that extract a LoRA by SVD
//                                                                                        name it)
bool parse_name(const std::string& name, int* layer, std::string* linear, char* side, std::string* why) {
  // Longer than any model's (the longest above is 85 bytes): refused before the regex, whose backtracking takes a
  // frame of the stack a character.
  if (name.size() > 256) return *why = "a name of " + std::to_string(name.size()) + " bytes, not a decoder layer's LoRA", false;
  static const std::regex re(
      R"(^(?:base_model\.model\.)?(?:model\.)?(?:language_model\.)?(?:model\.)?layers\.(\d+)\.)"
      R"((self_attn|mlp|linear_attn)\.(\w+)\.(?:lora_(A|B)(?:\.\w+)?\.weight|weight\.lora_(A|B)(?:\.\w+)?)$)");
  static const std::map<std::string, std::string> linears = {
      {"self_attn.q_proj", "attn.q"},        {"self_attn.k_proj", "attn.k"},       {"self_attn.v_proj", "attn.v"},
      {"self_attn.o_proj", "attn.o"},        {"mlp.gate_proj", "mlp.gate"},        {"mlp.up_proj", "mlp.up"},
      {"mlp.down_proj", "mlp.down"},         {"linear_attn.in_proj_qkv", "gdn.qkv"}, {"linear_attn.in_proj_z", "gdn.z"},
      {"linear_attn.in_proj_b", "gdn.b"},    {"linear_attn.in_proj_a", "gdn.a"},   {"linear_attn.out_proj", "gdn.out"}};
  std::smatch m;
  if (!std::regex_match(name, m, re)) {
    if (name.find("lora_magnitude_vector") != std::string::npos) *why = "a DoRA magnitude";
    else if (name.find("lora_embedding_") != std::string::npos) *why = "an embedding's adapter";
    else if (name.find("visual") != std::string::npos || name.find("vision") != std::string::npos)
      *why = "a vision tower's adapter";
    else if (name.find("lm_head") != std::string::npos) *why = "the head's adapter";
    else if (name.find("embed_tokens") != std::string::npos) *why = "the embedding's adapter";
    else *why = "not a decoder layer's LoRA";
    return false;
  }
  const auto it = linears.find(m.str(2) + "." + m.str(3));
  if (it == linears.end()) return *why = "an adapter on " + m.str(2) + "." + m.str(3) + ", a linear flyx does not adapt", false;
  const std::string digits = m.str(1);
  if (digits.size() > 5) return *why = "a layer past any model's", false;
  *layer = std::stoi(digits);
  *linear = it->second;
  *side = (m[4].matched ? m.str(4) : m.str(5))[0];
  return true;
}

// One tensor of the safetensors header as read_peft_dir reads it: a decoder layer's linear, F32, BF16 or F16, lora_A
// [r, in] or lora_B [out, r].  false with `why` (its name first).
struct Tensor {
  int layer = 0, dtype = -1;
  std::string linear;
  char side = 0;
  std::vector<int64_t> shape;
};
bool check_tensor(const std::string& key, const json& r, int64_t rank, Tensor* t, std::string* why) {
  const std::string name = shown(key);
  std::string no;
  if (!parse_name(key, &t->layer, &t->linear, &t->side, &no)) return *why = name + ": " + no, false;
  const std::string dtype = shown(r.at("dtype").get<std::string>());
  t->dtype = dtype_of(dtype);
  if (t->dtype < 0) return *why = name + ": dtype " + dtype + " (F32, BF16 or F16)", false;
  t->shape = r.at("shape").get<std::vector<int64_t>>();
  if (t->shape.size() != 2 || t->shape[0] <= 0 || t->shape[1] <= 0) return *why = name + ": not a matrix", false;
  if ((t->side == 'A' ? t->shape[0] : t->shape[1]) != rank)
    return *why = name + ": shape [" + std::to_string(t->shape[0]) + ", " + std::to_string(t->shape[1]) +
                  "] is not of rank r = " + std::to_string(rank) + " (lora_A [r, in], lora_B [out, r])",
           false;
  return true;
}

// A safetensors header's tensors as the adapter's: each one checked (check_tensor), and every decoder layer's linear
// with one lora_A and one lora_B, never a second of either (two spellings of one name) nor one alone.  The listing of
// a repository's adapters (peft_header_ok) and the load (read_peft_dir) both take this, so that one a listing calls
// loadable loads.  false with `why`.
struct Paired {
  std::vector<std::pair<std::string, Tensor>> tensors;  // the header's key and its tensor, in the header's order
  std::set<std::string> linears;                         // those it sits beside ("attn.q", "gdn.qkv", ...)
  int layers = 0;                                        // one past its last layer
};
bool pair_tensors(const json& h, int64_t rank, Paired* out, std::string* why) {
  if (!h.is_object()) return *why = "not a JSON object", false;
  std::map<std::string, std::pair<bool, bool>> sides;  // "L3.attn.q" -> whether its lora_A and its lora_B were seen
  for (auto it = h.begin(); it != h.end(); ++it) {
    if (it.key() == "__metadata__") continue;
    Tensor t;
    if (!check_tensor(it.key(), it.value(), rank, &t, why)) return false;
    const std::string key = "L" + std::to_string(t.layer) + "." + t.linear;
    auto& [a, b] = sides.try_emplace(key, false, false).first->second;
    bool& seen = t.side == 'A' ? a : b;
    if (seen) return *why = shown(it.key()) + ": a second lora_" + t.side + " on " + key, false;
    seen = true;
    out->linears.insert(t.linear);
    out->layers = std::max(out->layers, t.layer + 1);
    out->tensors.emplace_back(it.key(), std::move(t));
  }
  if (sides.empty()) return *why = std::string("no LoRA tensors in ") + kWeights, false;
  for (const auto& [key, s] : sides)
    if (!s.first || !s.second) return *why = key + " has lora_" + (s.first ? "A" : "B") + " alone", false;
  return true;
}

}  // namespace

bool peft_config_ok(const std::string& text, PeftConfig* out, std::string* why) {
  try {
    if (!interfaces::json_depth_within(text)) throw std::runtime_error("nested too deeply");
    const json c = json::parse(text);
    if (!c.is_object()) throw std::runtime_error("not a JSON object");
    if (c.contains("base_model_name_or_path") && c["base_model_name_or_path"].is_string())
      out->base = shown(c["base_model_name_or_path"].get<std::string>());
    if (c.contains("target_modules") && c["target_modules"].is_string())
      out->targets = {shown(c["target_modules"].get<std::string>())};
    else if (c.contains("target_modules") && c["target_modules"].is_array())
      for (const json& t : c["target_modules"])
        if (t.is_string() && out->targets.size() < 64) out->targets.push_back(shown(t.get<std::string>()));
    const json lora = lora_config(c);
    out->rank = int(lora["rank"].get<int64_t>());
    out->scale = lora["scale"].get<double>();
    return true;
  } catch (const std::exception& e) {
    *why = e.what();
    return false;
  }
}

bool peft_header_ok(const std::string& header, int rank, std::string* why, std::vector<std::string>* linears,
                    int* layers) {
  try {
    if (header.size() > kSafetensorsHeaderMaxBytes) throw std::runtime_error("a header of more than 16 MiB");
    if (!interfaces::json_depth_within(header)) throw std::runtime_error("nested too deeply");
    Paired p;
    if (!pair_tensors(json::parse(header), rank, &p, why)) return false;
    if (linears) linears->assign(p.linears.begin(), p.linears.end());
    if (layers) *layers = p.layers;
    return true;
  } catch (const std::exception& e) {
    *why = e.what();
    return false;
  }
}

bool is_peft_dir(const std::string& path, const PathAllowFn& allow_path) {
  const fs::path root = to_path(path);
  const fs::path cfg = root / kConfig;
  if (allow_path && (!allow_path(path) || !allow_path(from_path(cfg)))) return false;
  std::error_code ec;
  return fs::is_directory(root, ec) && fs::is_regular_file(cfg, ec);
}

bool read_peft_dir(const std::string& dir, std::vector<std::unique_ptr<os::MappedFile>>* maps,
                   ModelFile::Contents* out, std::string* err, const PathAllowFn& allow_path) {
  const fs::path root = to_path(dir);
  const fs::path weights = root / kWeights;
  const fs::path cfg_path = root / kConfig;
  const fs::path bin_path = root / "adapter_model.bin";
  if (allow_path && !allow_path(dir))
    return *err = dir + ": path is not allowed by remote path policy", false;
  if (allow_path && !allow_path(from_path(weights)))
    return *err = from_path(weights) + ": path is not allowed by remote path policy", false;
  std::error_code ec;
  if (!fs::is_regular_file(weights, ec)) {
    if ((!allow_path || allow_path(from_path(bin_path))) && fs::is_regular_file(bin_path, ec))
      return *err = dir + ": adapter_model.bin (a pickle) is not read; save the adapter as safetensors", false;
    return *err = dir + ": no " + kWeights + " beside its " + kConfig, false;
  }
  if (allow_path && !allow_path(from_path(cfg_path)))
    return *err = from_path(cfg_path) + ": path is not allowed by remote path policy", false;
  try {
    const json cfg = read_json(cfg_path, kConfig);
    json lora = lora_config(cfg);
    const int64_t rank = lora["rank"].get<int64_t>();

    auto map = std::make_unique<os::MappedFile>();
    const std::string file = from_path(weights);
    if (!map->open(file, err)) return false;
    const uint8_t* p = map->data();
    const size_t size = map->size();
    uint64_t hlen = 0;
    if (size < 8) return *err = file + ": not a safetensors file", false;
    std::memcpy(&hlen, p, 8);
    // Bounded before it is parsed, as the Hub's read of it is (hub::safetensors_header).
    if (hlen > kSafetensorsHeaderMaxBytes)
      return *err = file + ": a header of " + std::to_string(hlen) + " bytes, more than a safetensors header holds (16 MiB)",
             false;
    if (hlen > size - 8) return *err = file + ": truncated header", false;
    const std::string_view hdr(reinterpret_cast<const char*>(p + 8), size_t(hlen));
    if (!interfaces::json_depth_within(hdr)) return *err = file + ": header JSON nested too deeply", false;
    const json h = json::parse(hdr);
    Paired paired;
    std::string why;
    if (!pair_tensors(h, rank, &paired, &why)) return *err = dir + ": " + why, false;
    const size_t data_start = 8 + size_t(hlen), data_size = size - data_start;

    *out = {};
    for (const auto& [key, t] : paired.tensors) {
      const auto offsets = h.at(key).at("data_offsets").get<std::vector<uint64_t>>();
      flyx_tensor_view v{};
      v.dtype = t.dtype;
      if (offsets.size() != 2 || offsets[0] > offsets[1] || offsets[1] > data_size)
        return *err = dir + ": " + shown(key) + ": outside the file (truncated download?)", false;
      uint64_t want = 0;
      if (!plain_tensor_nbytes(v.dtype, t.shape, &want) || offsets[1] - offsets[0] != want)
        return *err = dir + ": " + shown(key) + ": byte size does not match shape and dtype", false;
      v.ndim = 2;
      v.shape[0] = t.shape[0];
      v.shape[1] = t.shape[1];
      v.data = p + data_start + offsets[0];
      v.nbytes = offsets[1] - offsets[0];
      const std::string name = "target.L" + std::to_string(t.layer) + "." + t.linear + (t.side == 'A' ? ".lora_a" : ".lora_b");
      out->index[name] = out->tensors.size();
      out->names.push_back(name);
      out->tensors.push_back(v);
    }
    for (size_t i = 0; i < out->tensors.size(); ++i) out->tensors[i].name = out->names[i].c_str();
    lora["layers"] = paired.layers;
    lora["linears"] = std::vector<std::string>(paired.linears.begin(), paired.linears.end());
    out->arch = "lora";
    json source = {{"lora", from_path(root.filename().empty() ? root.parent_path().filename() : root.filename())}};
    if (cfg.contains("base_model_name_or_path") && cfg["base_model_name_or_path"].is_string())
      source["base_model"] = cfg["base_model_name_or_path"];
    out->header = {{"arch", "lora"}, {"kind", "lora"}, {"lora", lora}, {"source", source}};
    maps->push_back(std::move(map));
  } catch (const std::exception& e) {
    *err = dir + ": " + e.what();
    return false;
  }
  return true;
}

}  // namespace flyx::modelfile
