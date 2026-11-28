// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "models/drafters/registry.h"

#include <climits>
#include <stdexcept>
#include <string>

#include "models/gguf.h"

namespace flyx::models::drafters {

using nlohmann::json;

namespace {

std::string companion_label(int i) { return "companion file " + std::to_string(i + 1) + ": "; }

// What fn() reads of companion i, its refusals naming the companion.
template <class F>
auto in_companion(int i, F fn) {
  try {
    return fn();
  } catch (const std::exception& e) {
    throw std::runtime_error(companion_label(i) + e.what());
  }
}

}  // namespace

DirectoryDraft directory_draft(const ModelFile::Contents& raw) {
  const auto& kind = raw.header.at("draft").at("kind");
  if (kind == "eagle4") return eagle4_directory(raw);
  if (kind == "dspine") return dspine_directory(raw);
  return dflash_directory(raw);
}

void DrafterRegistry::scan(const ModelFile& file, const TargetSpec& target, const MtpHooks* mtp, bool has_dv,
                           std::vector<int>* unclaimed) {
  *this = DrafterRegistry();
  if (unclaimed) unclaimed->clear();

  const json& hdr = file.header();
  sources_.dv = has_dv;
  if (mtp && mtp->embedded) sources_.mtp = true;
  if (hdr.is_object() && hdr.contains("draft") && hdr["draft"].is_object()) {
    sources_.block = block_drafter_kind(hdr["draft"]);
    if (sources_.block == Drafter::kNone)
      throw std::runtime_error("the file's drafter (\"draft\": kind '" + hdr["draft"].value("kind", "") +
                               "') is no drafter of the target's taps this build runs");
  }

  for (int i = 0; i < file.companions(); ++i) {
    const json& h = hdr.at("companions").at(size_t(i));
    const std::string what = companion_label(i);
    // the architecture's own MTP drafter (Gemma 4's assistant)
    if (mtp && mtp->claims && in_companion(i, [&] { return mtp->claims(h); })) {
      if (sources_.mtp) throw std::runtime_error(what + "the model has its MTP drafter already");
      sources_.mtp = true;
      sources_.mtp_beside = true;
      mtp_companion_ = i;
      continue;
    }
    if (h.is_object() && h.contains("draft") && h["draft"].is_object() &&
        h["draft"].value("kind", "") == "gemma4_mtp")
      throw std::runtime_error(what + "Gemma 4's assistant, which a " + target.arch + " model does not run");
    // llama.cpp's DSpark GGUF (its own block, beside a drafter of DFlash2's block), or a drafter of DFlash2's block
    // (a drafter file's "draft", or an arch "dflash" GGUF).  One DSpark a model, of either.
    if (is_dspark_companion(h)) {
      if (!target.dspark_taps)
        throw std::runtime_error(what + "a DSpark drafter, which a " + target.arch + " model does not run");
      if (sources_.dspark_gguf || sources_.block == Drafter::kDSpark)
        throw std::runtime_error(what + "the model has a DSpark drafter already");
      try {
        check_made_for(h, target);
      } catch (const std::exception& e) {
        throw std::runtime_error(what + e.what());
      }
      sources_.dspark_gguf = true;
      dspark_companion_ = i;
    } else if (is_block_companion(h)) {
      if (sources_.block != Drafter::kNone)
        throw std::runtime_error(what + "the model has a block drafter already (DFlash2's, DFlash's, DSpine's or "
                                        "EAGLE-4's: one drafter of its taps)");
      try {
        check_made_for(h, target);
      } catch (const std::exception& e) {
        throw std::runtime_error(what + e.what());
      }
      sources_.block = GgufMeta::is_header(h) ? Drafter::kDFlash2 : block_drafter_kind(h.at("draft"));
      if (sources_.block == Drafter::kDSpark && sources_.dspark_gguf)
        throw std::runtime_error(what + "the model has a DSpark drafter already");
      sources_.block_beside = true;
      block_companion_ = i;
    } else if (mtp && GgufMeta::is_header(h) && h.value("arch", "") == target.arch) {
      if (!target.gguf) throw std::runtime_error(what + "an MTP layer from a GGUF file goes with a GGUF model");
      if (sources_.mtp) throw std::runtime_error(what + "the model has its MTP layer already");
      const GgufMeta cm(h);
      if (cm.count("embedding_length") != target.hidden || cm.count_or("nextn_predict_layers", 0, 0) != 1 ||
          cm.count("block_count") != target.layers + 1)
        throw std::runtime_error(what + "no MTP layer for this model (" + target.arch + ", width " +
                                 std::to_string(target.hidden) + ", " + std::to_string(target.layers) + " layers)");
      sources_.mtp = true;
      sources_.mtp_beside = true;
      mtp_companion_ = i;
    } else {
      if (unclaimed) unclaimed->push_back(i);
    }
  }
}

const DFlash2Spec& DrafterRegistry::ensure_dflash2_spec(const ModelFile& file, const TargetSpec& target) {
  if (dflash2_spec_) return *dflash2_spec_;
  const int i = block_companion_;
  if (i < 0) return dflash2_spec_.emplace(parse_dflash2_json(file, file.header().at("draft"), target, -1));
  const json& h = file.header().at("companions").at(size_t(i));
  return dflash2_spec_.emplace(in_companion(i, [&] {
    if (directory_) {
      const auto pred = directory_->tensors.find("draft.sel.pred");
      return parse_dflash2_json(file, directory_->config, target, i,
                                pred == directory_->tensors.end() ? nullptr : &pred->second);
    }
    return GgufMeta::is_header(h) ? parse_dflash2_gguf(file, h, target, i)
                                  : parse_dflash2_json(file, h.at("draft"), target, i);
  }));
}

const DSparkSpec& DrafterRegistry::ensure_dspark_spec(const ModelFile& file, const TargetSpec& target) {
  if (dspark_spec_) return *dspark_spec_;
  const int i = dspark_companion_;
  const json& h = file.header().at("companions").at(size_t(i));
  return dspark_spec_.emplace(in_companion(i, [&] { return parse_dspark_gguf(file, h, target, i); }));
}

const DSpineSpec& DrafterRegistry::ensure_dspine_spec(const ModelFile& file, const TargetSpec& target) {
  if (dspine_spec_) return *dspine_spec_;
  const int i = block_companion_;
  if (i < 0) return dspine_spec_.emplace(parse_dspine_json(file.header().at("draft"), target, -1));
  return dspine_spec_.emplace(in_companion(i, [&] {
    return parse_dspine_json(directory_ ? directory_->config : file.header().at("companions").at(size_t(i)).at("draft"), target, i);
  }));
}

const Eagle4Spec& DrafterRegistry::ensure_eagle4_spec(const ModelFile& file, const TargetSpec& target) {
  if (eagle4_spec_) return *eagle4_spec_;
  const int i = block_companion_;
  if (i < 0) return eagle4_spec_.emplace(parse_eagle4_json(file.header().at("draft"), target, -1));
  return eagle4_spec_.emplace(in_companion(i, [&] {
    return parse_eagle4_json(directory_ ? directory_->config : file.header().at("companions").at(size_t(i)).at("draft"), target, i);
  }));
}

void DrafterRegistry::validate_selected(const ModelFile& file, const std::vector<Drafter>& selected,
                                        const TargetSpec& target) {
  int64_t stride = 0;
  for (Drafter kind : selected) {
    if (!is_tapped_drafter(kind)) continue;
    if (kind == sources_.block && block_companion_ >= 0 && !file.companion(block_companion_).directory_config.is_null())
      directory_ = in_companion(block_companion_, [&] { return directory_draft(file.companion(block_companion_)); });
    const int taps = kind == Drafter::kDSpark && sources_.dspark_gguf ? ensure_dspark_spec(file, target).block.n_taps
                     : kind == Drafter::kDSpine                     ? ensure_dspine_spec(file, target).block.n_taps
                     : kind == Drafter::kEagle4                     ? ensure_eagle4_spec(file, target).block.n_taps
                                                                     : ensure_dflash2_spec(file, target).block.n_taps;
    stride += int64_t(taps) * target.hidden;
    if (stride > INT_MAX) throw std::runtime_error("combined drafter feature stride exceeds int32");
  }
}

int DrafterRegistry::attach(Drafter kind, const ModelFile& file, Tables& T, const TargetSpec& target,
                            const MtpHooks* mtp, DrafterReach* reach) {
  target_layers_ = target.layers;
  explicit_post_norm_ = target.has_post_norm_tap;
  if (kind == Drafter::kMtp && mtp && mtp->build) {  // a step a draft, as MTP's
    const int blk = mtp->build(mtp_companion_);
    context_blocks_.push_back(blk);
    *reach = {FLYX_MAX_ROWS - 1, false};
    return blk;
  }
  if (kind == Drafter::kMtp) {
    if (!mtp || !mtp->build_layer || !mtp->tensor_name)
      throw std::runtime_error("MTP drafter is not supported by this model architecture");
    const auto [attn_blk, mlp_blk] = mtp->build_layer(mtp_companion_);
    flyx_block b{};
    b.kind = FLYX_BLOCK_MTP;
    auto& m = b.mtp;
    m.embed = target.embed;
    m.head = target.head;
    m.attn = attn_blk;
    m.mlp = mlp_blk;
    m.norm_emb = T.norm(mtp->tensor_name("mtp.norm_emb", mtp_companion_), target.hidden, mtp->norm_offset, mtp->eps);
    m.norm_hidden =
        T.norm(mtp->tensor_name("mtp.norm_hidden", mtp_companion_), target.hidden, mtp->norm_offset, mtp->eps);
    m.norm = T.norm(mtp->tensor_name("mtp.norm", mtp_companion_), target.hidden, mtp->norm_offset, mtp->eps);
    m.fc = T.weight(mtp->tensor_name("mtp.fc", mtp_companion_), {target.hidden, 2 * target.hidden});
    flyx_state ring{};
    ring.kind = FLYX_STATE_RING;
    ring.draft = 1;
    ring.width = target.hidden;
    m.ring = T.state(ring);
    const int blk = T.block(b);
    context_blocks_.push_back(blk);
    *reach = {FLYX_MAX_ROWS - 1, false};
    return blk;
  }
  if (!is_tapped_drafter(kind)) throw std::runtime_error("unsupported drafter kind in DrafterRegistry::attach");
  auto record = [&](BlockTaps& attached, int block, const std::vector<int>& taps, DrafterReach r) {
    attached = {block, taps};
    context_blocks_.push_back(block);
    *reach = r;
    return block;
  };
  if (kind == Drafter::kDSpark && sources_.dspark_gguf) {  // its own block, per-row mask embeddings: its block at most
    const DSparkSpec& spec = ensure_dspark_spec(file, target);
    return record(dspark_, build_dspark(T, file, spec, target), spec.taps, {spec.block.block - 1, false});
  }
  if (directory_ && kind == sources_.block)
    for (const auto& [name, view] : directory_->tensors)
      T.add_view(ModelFile::companion_name(block_companion_, name), view);
  if (kind == Drafter::kEagle4) {  // a step a draft: a round's rows, as MTP's
    const Eagle4Spec& spec = ensure_eagle4_spec(file, target);
    return record(block_, build_eagle4(T, spec, target), spec.taps, {FLYX_MAX_ROWS - 1, false});
  }
  if (kind == Drafter::kDSpine) {  // causal, its block at most
    const DSpineSpec& spec = ensure_dspine_spec(file, target);
    return record(block_, build_dspine(T, spec, target), spec.taps, {spec.block.block - 1, false});
  }
  // DFlash2's block: one shared mask token every row, so its block stretches to a round's rows
  const DFlash2Spec& spec = ensure_dflash2_spec(file, target);
  return record(block_, build_dflash2(T, spec, target), spec.taps, {spec.block_drafts(), true});
}

void DrafterRegistry::BlockTaps::emit(std::vector<flyx_node>* p, int position, int flags) const {
  for (size_t slot = 0; slot < taps.size(); ++slot)
    if (taps[slot] == position) p->push_back({FLYX_NODE_TAP, block, int32_t(slot), flags});
}

void DrafterRegistry::emit_pre_visit_taps(std::vector<flyx_node>* p, int visit_0based) const {
  dspark_.emit(p, visit_0based, 0);
}

void DrafterRegistry::emit_taps(std::vector<flyx_node>* p, int layer_1based) const {
  const bool is_post_norm = layer_1based == target_layers_;
  if (is_post_norm && explicit_post_norm_) return;
  block_.emit(p, layer_1based, is_post_norm ? FLYX_TAP_NORMED : 0);
}

void DrafterRegistry::emit_post_norm_taps(std::vector<flyx_node>* p, int n_layers) const {
  dspark_.emit(p, n_layers, FLYX_TAP_NORMED);
  block_.emit(p, n_layers, FLYX_TAP_NORMED);
}

void DrafterRegistry::emit_contexts(std::vector<flyx_node>* p) const {
  for (int blk : context_blocks_) p->push_back({FLYX_NODE_CONTEXT, blk, 0, 0});
}

}  // namespace flyx::models::drafters
