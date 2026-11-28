// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "models/model.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "core/interfaces/options.h"
#include "models/inkling2/inkling2.h"
#include "models/llama/llama.h"
#include "models/qwen35/qwen35.h"

namespace flyx::models {

namespace {
// Structural properties of a drafter kind; policy functions (draft_widths, adaptive_checkpoints, block_drafts)
// decide from these traits plus target properties (verify_scales_with_width) rather than per-drafter branches.
struct DrafterTraits {
  bool sequential;        // drafts one token per step; stopping early saves both draft and verify work
  bool tree_only;         // drafts trees over a persistent canvas (DV)
  bool block;             // a parallel block drafter (models/drafters/registry.h)
  bool tapped;            // drafts from the target's taps in the registry's slot without being a block (EAGLE-4)
  bool trees_by_default;  // its trees beside its chains pay on attention-only targets (drafts_trees_by_default)
  bool trees_beside_recurrence;  // ... on a gated delta net's too, whose tree recurrence a tree's rows run
  bool widest_trees;      // its trees are of FLYX_MAX_ROWS - 1 drafts only (EAGLE-4's)
  // The controller rates its trees by the tokens their rounds give, as DV's (Model::trees_best_first), and learns its
  // chains' acceptance from its chain rounds alone.
  bool trees_best_first;
  // A block drafted at once whose drafts a small head then refines one at a time: a step costs little beside the
  // verify row its draft adds, so a round that stops may verify fewer drafts than it has (draft_steps_cheap).  Not a
  // sequential drafter's, whose step is a pass of a layer: with earlier checkpoints allowed Qwen3.8-27B's MTP rounds
  // gave up drafts that paid (RTX 5070 Ti, 24 chat prompts of 256 tokens: 3.06 -> 2.99 tokens a round, 144.4 ->
  // 142.4 tok/s), the checkpoints' costs counting as saved the draft steps a stop has paid.
  bool refines_by_step;
};

// Every drafter kind: the --drafter mode naming it (its word: interfaces::kDrafterModes), its name in messages, its
// traits.
struct DrafterInfo {
  Drafter kind;
  interfaces::DrafterMode mode;
  const char* title;
  DrafterTraits traits;
};
constexpr DrafterInfo kDrafters[] = {
    {Drafter::kMtp, interfaces::DrafterMode::kMtp, "MTP", {.sequential = true}},
    {Drafter::kDFlash2, interfaces::DrafterMode::kDFlash2, "DFlash2", {.block = true}},
    {Drafter::kDFlash, interfaces::DrafterMode::kDFlash, "DFlash",
     {.block = true}},
    {Drafter::kDSpark, interfaces::DrafterMode::kDSpark, "DSpark",
     {.block = true, .trees_by_default = true, .refines_by_step = true}},
    {Drafter::kDSpine, interfaces::DrafterMode::kDSpine, "DSpine",
     {.block = true}},
    {Drafter::kEagle4, interfaces::DrafterMode::kEagle4, "EAGLE-4",
     {.sequential = true,
      .tapped = true,
      .trees_by_default = true,
      .trees_beside_recurrence = true,
      .widest_trees = true,
      .trees_best_first = true}},
    {Drafter::kDv, interfaces::DrafterMode::kDv, "DV", {.tree_only = true}},
};

const DrafterInfo* info(Drafter d) {
  for (const DrafterInfo& i : kDrafters)
    if (i.kind == d) return &i;
  return nullptr;
}

DrafterTraits drafter_traits(Drafter d) {
  const DrafterInfo* i = info(d);
  return i ? i->traits : DrafterTraits{};
}

// The kind a --drafter mode names (kNone: auto, all, none).
Drafter drafter_of(interfaces::DrafterMode m) {
  for (const DrafterInfo& i : kDrafters)
    if (i.mode == m) return i.kind;
  return Drafter::kNone;
}
}  // namespace

const char* drafter_name(Drafter d) {
  const DrafterInfo* i = info(d);
  return i ? interfaces::drafter_mode_word(i->mode) : "none";
}

const char* drafter_title(Drafter d) {
  const DrafterInfo* i = info(d);
  return i ? i->title : "no";
}

Drafter drafter_named(std::string_view name) {
  interfaces::DrafterMode m = interfaces::DrafterMode::kAuto;
  return interfaces::parse_drafter_mode(name, &m) ? drafter_of(m) : Drafter::kNone;
}

bool is_block_drafter(Drafter d) { return drafter_traits(d).block; }
bool is_tapped_drafter(Drafter d) { return drafter_traits(d).block || drafter_traits(d).tapped; }
bool drafts_trees_by_default(Drafter d, bool recurrent) {
  const DrafterTraits& t = drafter_traits(d);
  return t.trees_by_default && (!recurrent || t.trees_beside_recurrence);
}

// Width ladder and intra-round adaptive stopping, driven by drafter traits + target structure:
// - Sequential drafters (MTP, EAGLE-4): tensor-core aligned 1, 3, 5, 7, ..., max_drafts (<= 15).
// - Parallel block drafters: {1, 2, 3, 5, 7} when target verify cost scales steeply with width (e.g. sparse MoE, where
//   each draft row reads its own experts: one or two drafts often pay where three do not), {3, 7} on dense targets,
//   plus FLYX_MAX_ROWS - 1 when its block stretches past max_drafts (DrafterReach::stretch).
// - Tree-only drafters (DV): up to FLYX_MAX_WIDE_ROWS - 1 (plus 15 when wider).
std::vector<int> Model::draft_widths(int d) const {
  const DrafterBlock& b = drafters_.at(size_t(d));
  if (b.max_drafts <= 0) return {};
  const DrafterTraits tr = drafter_traits(b.kind);
  if (tr.tree_only) {
    const int widest = std::min(FLYX_MAX_WIDE_ROWS - 1, b.max_drafts);
    if (widest > FLYX_MAX_ROWS - 1) return {FLYX_MAX_ROWS - 1, widest};
    return {widest};
  }
  const int widest = b.stretch ? b.max_drafts : std::min(FLYX_MAX_ROWS - 1, b.max_drafts);
  const std::vector<int> ladder = tr.sequential                ? std::vector<int>{1, 3, 5, 7}
                                  : verify_scales_with_width_ ? std::vector<int>{1, 2, 3, 5, 7}
                                                              : std::vector<int>{3, 7};
  std::vector<int> widths;
  for (int k : ladder)
    if (k < widest) widths.push_back(k);
  widths.push_back(widest);
  if (b.stretch && widest < FLYX_MAX_ROWS - 1) widths.push_back(FLYX_MAX_ROWS - 1);
  return widths;
}

std::vector<int> Model::adaptive_checkpoints(int d) const {
  const DrafterBlock& b = drafters_.at(size_t(d));
  const DrafterTraits tr = drafter_traits(b.kind);
  if (b.max_drafts <= 0 || tr.tree_only) return {};  // tree-only drafters (DV): no intra-round chain stop
  // Intra-round GPU confidence stopping (stop_decide_step / stop_decide_block) pays whenever cutting verify width
  // saves substantial work:
  // - sequential drafters save both remaining draft steps and verify rows;
  // - targets with steep verify-cost scaling (e.g. sparse MoE active-expert fan-out) benefit on any block drafter.
  // Dense targets paired with block drafters have nearly flat verify cost across widths (RTX PRO 6000, QUASAR NVFP4
  // 27B + DFlash2: adaptive 425-439 vs fixed 443-494 tok/s on code) and avoid extra verify subgraphs in VRAM.
  if (!tr.sequential && !verify_scales_with_width_) return {};
  // Keep a drafter's adaptive graph within its trained block (b.max_drafts); stretching every adaptive round past the
  // trained block degrades bidirectional draft quality and adds draft/verify work (stretched widths stay in
  // draft_widths for explicit --width or widened high-acceptance runs).
  std::vector<int> cps;
  for (int w : draft_widths(d))
    if (w <= b.max_drafts) cps.push_back(w);
  return cps.size() > 1 ? cps : std::vector<int>{};
}

bool Model::drafts_sequentially(int d) const { return drafter_traits(drafters_.at(size_t(d)).kind).sequential; }
bool Model::draft_steps_cheap(int d) const { return drafter_traits(drafters_.at(size_t(d)).kind).refines_by_step; }

int Model::block_drafts(int d) const {
  const DrafterBlock& b = drafters_.at(size_t(d));
  return drafter_traits(b.kind).sequential ? std::min(FLYX_SEQUENTIAL_TRUNK, b.max_drafts) : b.max_drafts;
}

bool Model::drafts_trees(int d) const { return drafter_traits(drafters_.at(size_t(d)).kind).tree_only; }

bool Model::widest_trees(int d) const { return drafter_traits(drafters_.at(size_t(d)).kind).widest_trees; }

bool Model::trees_best_first(int d) const {
  return drafter_traits(drafters_.at(size_t(d)).kind).trees_best_first;
}

bool Model::restarts(int d) const { return drafter_traits(drafters_.at(size_t(d)).kind).tree_only; }

std::vector<std::string> Model::drafter_fallbacks() const {
  std::vector<std::string> modes;
  bool tapped = false;
  for (const auto& d : drafters_) {
    if (tables_.blocks[size_t(d.block)].kind == FLYX_BLOCK_DSPARK) {
      if (drafters_.size() > 1) modes.emplace_back(models::drafter_name(d.kind));
    } else if (is_tapped_drafter(d.kind)) {
      tapped = true;
    }
  }
  if (tapped || drafters_.size() > 1) modes.emplace_back(models::drafter_name(Drafter::kMtp));
  return modes;
}

void Model::add_drafter(Drafter kind, int block, DrafterReach reach) {
  drafters_.push_back({kind, block, reach.max_drafts, reach.stretch});
  drafter_names_.clear();
  for (const DrafterBlock& d : drafters_)
    drafter_names_ += (drafter_names_.empty() ? "" : "+") + std::string(models::drafter_name(d.kind));
}

bool choose_drafters(const ModelOptions& o, const DrafterSources& has, std::vector<Drafter>* out, std::string* err) {
  out->clear();
  interfaces::DrafterMode mode = interfaces::DrafterMode::kAuto;
  if (!interfaces::parse_drafter_mode(o.drafter, &mode))
    return *err = "unknown drafter '" + o.drafter + "' (" + interfaces::drafter_mode_words() + ")", false;
  if (!o.speculative || mode == interfaces::DrafterMode::kNone) return true;
  bool mtp = false, block = false, dspark_gguf = false, dv = false;
  const bool has_block = has.block != Drafter::kNone;
  switch (mode) {
    case interfaces::DrafterMode::kMtp:
      if (!has.mtp) return *err = "the model file has no MTP layer", false;
      mtp = true;
      break;
    case interfaces::DrafterMode::kDv:
      if (!has.dv) return *err = "the model file has no DV drafter", false;
      dv = true;
      break;
    case interfaces::DrafterMode::kAll:
      mtp = has.mtp;
      block = has_block;
      dspark_gguf = has.dspark_gguf;
      dv = has.dv;
      break;
    case interfaces::DrafterMode::kAuto: {
      // Every drafter given in a file of its own; of the model file's own, the MTP layer if it has one, else its block
      // drafter (MTP was measured faster on the 4B than DFlash2, 462 against 341 tok/s: a file with both keeps it until
      // choosing between them every round is measured there).  DV only when named (dv, all).
      const bool own_mtp = has.mtp && !has.mtp_beside, own_block = has_block && !has.block_beside;
      mtp = has.mtp_beside || own_mtp;
      block = has.block_beside || (own_block && !own_mtp);
      dspark_gguf = has.dspark_gguf;  // always a file of its own
      break;
    }
    case interfaces::DrafterMode::kNone: break;
    default: {  // a drafter of the target's taps of the kind named (DFlash2's block, DSpine's, EAGLE-4), or
                // llama.cpp's DSpark GGUF
      const Drafter want = drafter_of(mode);
      block = has.block == want;
      dspark_gguf = want == Drafter::kDSpark && has.dspark_gguf;
      if (!block && !dspark_gguf)
        return *err = std::string("the model file has no ") + drafter_title(want) + " drafter", false;
      break;
    }
  }
  // EAGLE-4 keeps its h per position in the one ring a sequence has, where an MTP layer keeps the target's hidden: a
  // model drafts with one of them, EAGLE-4 when both are there (auto, all).
  if (block && has.block == Drafter::kEagle4) mtp = false;
  // DV's canvas follows its own rounds only, and the controller weighs its warm-up against plain decoding's and
  // lookup's rounds alone: no drafter beside it.
  if (dv && (mtp || block || dspark_gguf)) return *err = "DV drafts alone: not beside another drafter", false;
  if (mtp) out->push_back(Drafter::kMtp);
  if (block) out->push_back(has.block);  // its drafter of the target's taps
  if (dspark_gguf) out->push_back(Drafter::kDSpark);
  if (dv) out->push_back(Drafter::kDv);
  return true;
}

std::string declared_activations(const ModelFile& file) {
  const auto& h = file.header();
  if (!h.is_object() || !h.contains("quant") || !h.at("quant").is_object()) return "";
  const auto& q = h.at("quant");
  return q.contains("activations") && q.at("activations").is_string() ? q.at("activations").get<std::string>() : "";
}

bool is_lora_companion(const ModelFile& file, int i) {
  return file.companion_role(i) == ModelFile::CompanionRole::Lora;
}

CompanionLora find_lora(const ModelFile& file, const std::set<std::string>& linears, int layers) {
  CompanionLora l;
  for (int i = 0; i < file.companions(); ++i) {
    if (!is_lora_companion(file, i)) continue;
    if (l) throw std::runtime_error("more than one LoRA adapter (--lora): one is run");
    const nlohmann::json& d = file.header().at("companions").at(size_t(i)).at("lora");
    l.companion = i;
    l.rank = d.at("rank").get<int>();
    l.scale = d.at("scale").get<float>();
    l.layers = d.at("layers").get<int>();
    if (d.value("layout", "") != "hf") throw std::runtime_error("lora: an adapter in a row order other than hf's");
    if (l.rank < 1 || l.rank > kMaxHeadDim || !std::isfinite(l.scale) || l.scale == 0.f)
      throw std::runtime_error("lora: rank " + std::to_string(l.rank) + ", scale " + std::to_string(l.scale) +
                               " (a rank of 1 to " + std::to_string(kMaxHeadDim) + ", a finite scale other than 0)");
    if (l.layers > layers)
      throw std::runtime_error("lora: adapters on " + std::to_string(l.layers) + " layers, the model has " +
                               std::to_string(layers));
    for (const auto& n : d.at("linears")) {
      const std::string s = n.get<std::string>();
      if (!linears.count(s)) throw std::runtime_error("lora: an adapter on " + s + ", a linear this model does not have");
    }
  }
  return l;
}

flyx_lora Tables::lora(const CompanionLora& l, const std::string& name, int64_t out, int64_t in) {
  const std::string c = ModelFile::companion_name(l.companion, name);
  if (!l || !has(c + ".lora_a")) return flyx_lora{-1, -1, 0, l.scale};
  const flyx_lora a{weight(c + ".lora_a", {l.rank, in}), weight(c + ".lora_b", {out, l.rank}), l.rank, l.scale};
  for (int w : {a.a, a.b}) {
    const int dt = weights[size_t(w)].t->dtype;
    if (dt != FLYX_DT_BF16 && dt != FLYX_DT_F16 && dt != FLYX_DT_F32)
      throw std::runtime_error(std::string(weights[size_t(w)].t->name) + ": an adapter that is not bf16, f16 or f32");
  }
  return a;
}

int Tables::weight(const std::string& name, std::vector<int64_t> shape, const std::string& ggml_scale_name) {
  auto it = index_.find(name);
  if (it != index_.end()) return it->second;
  const flyx_tensor_view* t = tensor(name);
  if (!t) throw std::runtime_error("missing tensor " + name);
  bool same = t->ndim == int32_t(shape.size());
  for (size_t i = 0; same && i < shape.size(); ++i) same = t->shape[i] == shape[i];
  if (!same) {
    std::string have, want;
    for (int i = 0; i < t->ndim; ++i) have += (i ? "x" : "") + std::to_string(t->shape[i]);
    for (size_t i = 0; i < shape.size(); ++i) want += (i ? "x" : "") + std::to_string(shape[i]);
    throw std::runtime_error(name + ": shape " + have + ", the config says " + want);
  }
  flyx_weight w{t, nullptr, nullptr, FLYX_ACT_FORMAT, 0};
  // In passes of more rows GGML block weights take 8-bit inputs and NVFP4 weights (ModelOpt's or GGML's) NVFP4 ones,
  // unless the file says otherwise: what their quantizations were made and measured with.
  // GGML's NVFP4 as llama.cpp reads the same file: NVFP4 inputs in prefill (rounded once from float32, as its FP4
  // ones), bf16 in rounds (finer than its 8 bits).
  const bool fp4 = t->dtype == FLYX_DT_NVFP4 || t->dtype == FLYX_DT_GGML_NVFP4;
  if (flyx_block_values(t->dtype) && !fp4 && !exact_) w.activations = FLYX_ACT_Q8;
  // 3D NVFP4 expert tensors keep bf16 activations (finer than FP4 and row-independent across sparse MoE dispatches).
  if (fp4 && t->ndim == 2 && !exact_ && nvfp4_inputs_) w.activations = FLYX_ACT_NVFP4;
  if (t->dtype == FLYX_DT_NVFP4) {
    w.scale = file_->tensor(name + ".scale");
    w.scale2 = file_->tensor(name + ".scale2");
    const bool ok_scale = w.scale && w.scale->dtype == FLYX_DT_F8E4M3 && t->ndim == 2 && t->shape[1] % 16 == 0 &&
                          w.scale->ndim == 2 && w.scale->shape[0] == t->shape[0] &&
                          w.scale->shape[1] == t->shape[1] / 16 &&
                          w.scale->nbytes == uint64_t(t->shape[0]) * uint64_t(t->shape[1] / 16);
    const bool ok_scale2 = w.scale2 && w.scale2->dtype == FLYX_DT_F32 && w.scale2->nbytes >= sizeof(float);
    if (!ok_scale || !ok_scale2)
      throw std::runtime_error(name + ": NVFP4 without valid scales");
  } else if (!ggml_scale_name.empty()) {
    if (const flyx_tensor_view* s2 = file_->tensor(ggml_scale_name)) {
      const bool ok_s2 = s2->dtype == FLYX_DT_F32 && s2->ndim == 1 &&
                         (s2->shape[0] == 1 || (t->ndim == 3 && s2->shape[0] == t->shape[0]));
      if (!ok_s2) throw std::runtime_error(ggml_scale_name + ": invalid GGML scale tensor");
      w.scale2 = s2;
    }
  }
  weights.push_back(w);
  return index_[name] = int(weights.size()) - 1;
}

bool Tables::set_activations(const std::string& how, const std::string& declared, bool fp4_inputs,
                              std::string* err) {
  if (!how.empty() && how != "exact" && how != "nvfp4")
    return *err = "activations: '" + how + "' (exact, nvfp4, or as the file has it)", false;
  if (!declared.empty() && declared != "bf16" && declared != "nvfp4")
    return *err = "the file's activations: '" + declared + "' (bf16 or nvfp4)", false;
  exact_ = how == "exact";
  // A GGUF file may name tensors whose inputs are not to be rounded to 4 bits (llama.cpp's
  // general.tensor_extra.prec_a4, false for them; by default NVFP4 weights take FP4 inputs on Blackwell). The
  // backend rounds one model's NVFP4 inputs one way, so then none are: finer than llama.cpp for the others. A file
  // whose flags do not name one tensor each is refused whatever the word.
  bool flagged = false;
  const auto& h = file_->header();
  if (h.contains("gguf") && h.at("gguf").contains("general.tensor_extra.prec_a4")) {
    const auto& g = h.at("gguf");
    const auto& flags = g.at("general.tensor_extra.prec_a4");
    if (!g.contains("general.tensor_extra.name") || !flags.is_array() ||
        g.at("general.tensor_extra.name").size() != flags.size())
      return *err = "general.tensor_extra.prec_a4: not one flag a named tensor", false;
    for (const auto& f : flags) flagged = flagged || (f.is_boolean() && !f.get<bool>());
  }
  // NVFP4 inputs go through the FP4 tensor cores in tiles of 128 values of K: a GGUF file with a GGML NVFP4 matrix
  // of another K (a tiny model's 64-wide layers) keeps bf16 inputs for all of them (finer).
  bool other_k = false;
  for (const auto& name : file_->tensor_names()) {
    const flyx_tensor_view* t = file_->tensor(name);
    other_k = other_k || (t && t->dtype == FLYX_DT_GGML_NVFP4 && (t->ndim == 2 || t->ndim == 3) &&
                          t->shape[t->ndim - 1] % 128);
  }
  // Asked for by name, NVFP4 weights take NVFP4 inputs whatever the file says: a file converted with bf16 inputs
  // takes them without a second conversion, and neither the device nor the GGUF rules above
  // take them back. A backend that cannot compute them (no FP4 tensor cores, a K its kernels do not take) refuses the
  // model at open rather than giving bf16 ones (no silent change of contract). A model without NVFP4
  // weights is what it is without the word, as with "exact" on a bf16 model: a server that switches models keeps it.
  // Otherwise a GPU without FP4 tensor cores gives NVFP4 weights bf16 inputs: finer than the 8 bits llama.cpp gives
  // them there.
  nvfp4_inputs_ = how == "nvfp4" || (declared != "bf16" && fp4_inputs && !flagged && !other_k);
  return true;
}

int Tables::state(const flyx_state& s) {
  states.push_back(s);
  return int(states.size()) - 1;
}

int Tables::block(const flyx_block& b) {
  blocks.push_back(b);
  return int(blocks.size()) - 1;
}

int Tables::embed_block(int hidden, int vocab, const NameFn& n) {
  flyx_block e{};
  e.kind = FLYX_BLOCK_EMBED;
  e.embed = {hidden, vocab, weight(n("target.embed"), {vocab, hidden})};
  return block(e);
}

int Tables::head_block(int hidden, int vocab, int embed_block_idx, double norm_offset, double eps, const NameFn& n) {
  flyx_block hb{};
  hb.kind = FLYX_BLOCK_HEAD;
  hb.head.hidden = hidden;
  hb.head.vocab = vocab;
  hb.head.norm = norm(n("target.norm"), hidden, norm_offset, eps);
  // No output matrix: the head is the embedding table.
  hb.head.w = has(n("target.lm_head")) ? weight(n("target.lm_head"), {vocab, hidden}) : at(embed_block_idx).embed.table;
  return block(hb);
}

int Tables::attn_block(const std::string& prefix, const AttnBlockSpec& s, const NameFn& n) {
  const int H = s.hidden, q = s.n_heads * s.head_dim, kv = s.n_kv * s.head_dim;
  const int q_rows = (s.gate == FLYX_GATE_NONE ? 1 : 2) * q;
  flyx_state st{};
  st.kind = FLYX_STATE_KV;
  st.draft = s.draft;
  st.layers = 1;
  st.n_kv = s.n_kv;
  st.head_dim = s.head_dim;
  flyx_block b{};
  b.kind = FLYX_BLOCK_ATTN;
  auto& a = b.attn;
  a.hidden = H;
  a.n_heads = s.n_heads;
  a.n_kv = s.n_kv;
  a.head_dim = s.head_dim;
  a.rot_dim = s.rot_dim;
  a.gate = s.gate;
  a.theta = s.theta;
  a.norm = norm(n(prefix + "in_norm"), H, s.norm_offset, s.eps);
  a.q = weight(n(prefix + "attn.q"), {q_rows, H});
  a.k = weight(n(prefix + "attn.k"), {kv, H});
  a.v = weight(n(prefix + "attn.v"), {kv, H});
  a.o = weight(n(prefix + "attn.o"), {H, q});
  if (s.qk_norm) {
    a.q_norm = norm(n(prefix + "attn.q_norm"), s.head_dim, s.norm_offset, s.eps);
    a.k_norm = norm(n(prefix + "attn.k_norm"), s.head_dim, s.norm_offset, s.eps);
  } else {
    a.q_norm = a.k_norm = flyx_norm{-1, 0, 0.0, 0.0};  // the heads go to the rotary as projected (backend_abi.h)
  }
  a.kv = state(st);
  for (int i = 0; i < 3; ++i) a.mrope_section[i] = s.mrope_section[i];
  a.mrope_interleaved = s.mrope_interleaved ? 1 : 0;
  return block(b);
}

int Tables::mlp_block(const std::string& prefix, int hidden, int inter, double norm_offset, double eps, const NameFn& n) {
  flyx_block b{};
  b.kind = FLYX_BLOCK_MLP;
  auto& f = b.mlp;
  f.hidden = hidden;
  f.inter = inter;
  f.norm = norm(n(prefix + "post_norm"), hidden, norm_offset, eps);
  f.gate = weight(n(prefix + "mlp.gate"), {inter, hidden});
  f.up = weight(n(prefix + "mlp.up"), {inter, hidden});
  f.down = weight(n(prefix + "mlp.down"), {hidden, inter});
  return block(b);
}

void Model::finish(int hidden, int vocab) {
  desc_ = {};
  desc_.hidden = hidden;
  desc_.vocab = vocab;
  desc_.n_weights = int32_t(tables_.weights.size());
  desc_.n_states = int32_t(tables_.states.size());
  desc_.n_blocks = int32_t(tables_.blocks.size());
  desc_.weights = tables_.weights.data();
  desc_.states = tables_.states.data();
  desc_.blocks = tables_.blocks.data();
}

namespace {

// The parts a file says a reader must not ignore (a .flyx header's "requires", format 2: core/modelfile), each one
// the model of its architecture reads (`reads`); a companion's never, as no model reads them from one.
bool reads_required(const ModelFile& file, const std::vector<std::string>& reads, std::string* err) {
  const nlohmann::json& h = file.header();
  if (!h.is_object()) return true;
  if (h.contains("requires"))
    for (const auto& r : h.at("requires"))
      if (std::find(reads.begin(), reads.end(), r.get<std::string>()) == reads.end())
        return *err = "the file requires its part \"" + r.get<std::string>() + "\", which this build's " + file.arch() +
                      " model does not read (a later build may)",
               false;
  if (h.contains("companions"))
    for (const auto& c : h.at("companions"))
      if (c.is_object() && c.contains("requires"))
        return *err = "a file beside the model requires its parts " + c.at("requires").dump() +
                      ", which no model reads from a companion",
               false;
  return true;
}

struct ArchEntry {
  const char* name;
  std::unique_ptr<Model> (*open)(const ModelFile&, const ModelOptions&, std::string*);
  bool gguf;    // its GGUF files read too (a Llama's are not: llama.cpp's converter permutes its q and k rows)
  bool vision;  // its build opens a vision projector (--mmproj)
  const char* reads;  // a part a file may require (format 2's "requires") that the model reads, or null
};

constexpr ArchEntry kArchitectures[] = {
    {"qwen35", qwen35::open, true, true, nullptr},
    {"qwen3", llama::open, true, false, "lora"},  // Llama's layer with q/k norms; their own adapters (a V-LoRA)
    {"llama", llama::open, false, false, "lora"},
    {"inkling2_loop", inkling2::open, true, false, nullptr},
};

}  // namespace

int32_t find_added_token(const ModelFile& file, std::string_view content) {
  const std::string_view raw = file.blob("tokenizer.added");
  if (raw.empty()) return -1;
  const nlohmann::json j = nlohmann::json::parse(raw.begin(), raw.end(), nullptr, false);
  if (!j.is_array()) return -1;
  for (const auto& item : j) {
    if (!item.is_object()) continue;
    if (item.contains("content") && item["content"].is_string() && item["content"].get<std::string>() == content &&
        item.contains("id") && item["id"].is_number_integer()) {
      return item["id"].get<int32_t>();
    }
  }
  return -1;
}

std::unique_ptr<Model> open(const ModelFile& file, const ModelOptions& o, std::string* err) {
  for (const ArchEntry& entry : kArchitectures) {
    if (file.arch() != entry.name) continue;
    std::vector<std::string> reads;
    if (entry.reads) reads.emplace_back(entry.reads);
    if (!reads_required(file, reads, err)) return nullptr;
    try {
      return entry.open(file, o, err);
    } catch (const std::exception& e) {
      *err = std::string("bad model file: ") + e.what();
      return nullptr;
    }
  }
  std::string known;
  for (const ArchEntry& entry : kArchitectures) known += (known.empty() ? "" : ", ") + std::string(entry.name);
  *err = "unsupported architecture '" + file.arch() + "' (this build knows " + known + ")";
  return nullptr;
}

std::vector<std::string> architectures() {
  std::vector<std::string> names;
  for (const ArchEntry& entry : kArchitectures)
    if (entry.gguf) names.emplace_back(entry.name);
  return names;
}

std::vector<std::string> vision_architectures() {
  std::vector<std::string> names;
  for (const ArchEntry& entry : kArchitectures)
    if (entry.vision) names.emplace_back(entry.name);
  return names;
}

}  // namespace flyx::models

namespace flyx::interfaces {

std::unique_ptr<Model> open_model(const ModelFile& file, const ModelOptions& o, std::string* err) {
  return models::open(file, o, err);
}

std::vector<std::string> model_architectures() { return models::architectures(); }
std::vector<std::string> vision_architectures() { return models::vision_architectures(); }

}  // namespace flyx::interfaces
