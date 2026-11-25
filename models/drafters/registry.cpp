// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "models/drafters/registry.h"

#include <stdexcept>
#include <string>

#include "models/gguf.h"

namespace flyx::models::drafters {

using nlohmann::json;

void DrafterRegistry::scan(const ModelFile& file, const TargetSpec& target, const MtpHooks* mtp, bool has_dv,
                           std::vector<int>* unclaimed) {
  sources_ = {};
  mtp_companion_ = -1;
  dflash2_gguf_ = false;
  dflash2_companion_ = -1;
  dflash2_json_ = json();
  dflash2_spec_.reset();
  dflash2_block_ = -1;
  dflash2_taps_.clear();
  context_blocks_.clear();
  if (unclaimed) unclaimed->clear();

  const json& hdr = file.header();
  sources_.dv = has_dv;
  if (mtp && mtp->embedded) sources_.mtp = true;
  if (hdr.is_object() && hdr.contains("draft") && hdr["draft"].is_object()) {
    sources_.dflash2 = true;
    dflash2_gguf_ = false;
    dflash2_companion_ = -1;
    dflash2_json_ = hdr["draft"];
  }

  for (int i = 0; i < file.companions(); ++i) {
    const json& h = hdr.at("companions").at(size_t(i));
    const std::string what = "companion file " + std::to_string(i + 1) + ": ";
    if (is_dflash_companion(h)) {
      if (sources_.dflash2) throw std::runtime_error(what + "the model has a DFlash2 drafter already");
      if (GgufMeta::is_header(h)) {
        try {
          dflash2_spec_ = parse_dflash2_gguf(file, h, target, i);
        } catch (const std::exception& e) {
          throw std::runtime_error(what + e.what());
        }
        dflash2_gguf_ = true;
      } else {
        const json& d = h.at("draft");
        const json t = d.value("target", json::object());
        const int th = t.value("hidden", target.hidden);
        const int tv = t.value("vocab", target.vocab);
        const bool vocab_ok = tv == target.vocab || tv == target.padded_vocab();
        const bool layers_ok = !t.contains("layers") || t["layers"].is_null() || t["layers"].get<int>() == target.layers;
        if (d.value("kind", "") != "dflash2" || th != target.hidden || !vocab_ok || !layers_ok)
          throw std::runtime_error(what + "a drafter made for another model (this one: width " +
                                   std::to_string(target.hidden) + ", vocabulary " + std::to_string(target.vocab) +
                                   ", " + std::to_string(target.layers) + " layers)");
        dflash2_gguf_ = false;
        dflash2_json_ = d;
      }
      sources_.dflash2 = true;
      sources_.dflash2_beside = true;
      dflash2_companion_ = i;
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
  if (!dflash2_spec_) {
    if (dflash2_gguf_)
      dflash2_spec_ = parse_dflash2_gguf(file, file.header().at("companions").at(size_t(dflash2_companion_)), target,
                                         dflash2_companion_);
    else
      dflash2_spec_ = parse_dflash2_json(dflash2_json_, target, dflash2_companion_, &file);
  }
  return *dflash2_spec_;
}

void DrafterRegistry::validate_selected(const ModelFile& file, const std::vector<Drafter>& selected,
                                        const TargetSpec& target) {
  for (Drafter kind : selected)
    if (kind == Drafter::kDFlash2) (void)ensure_dflash2_spec(file, target);
}

int DrafterRegistry::attach(Drafter kind, const ModelFile& file, Tables& T, const TargetSpec& target,
                            const MtpHooks* mtp, int* max_drafts_out) {
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
    if (max_drafts_out) *max_drafts_out = FLYX_MAX_ROWS - 1;
    return blk;
  }
  if (kind == Drafter::kDFlash2) {
    const DFlash2Spec& spec = ensure_dflash2_spec(file, target);
    dflash2_block_ = build_dflash2(T, spec, target);
    dflash2_taps_ = spec.taps;
    context_blocks_.push_back(dflash2_block_);
    if (max_drafts_out) *max_drafts_out = spec.block.block - 1;
    return dflash2_block_;
  }
  throw std::runtime_error("unsupported drafter kind in DrafterRegistry::attach");
}

void DrafterRegistry::emit_taps(std::vector<flyx_node>* p, int layer_1based) const {
  if (dflash2_block_ < 0) return;
  for (size_t s = 0; s < dflash2_taps_.size(); ++s)
    if (dflash2_taps_[s] == layer_1based) p->push_back({FLYX_NODE_TAP, dflash2_block_, int32_t(s), 0});
}

void DrafterRegistry::emit_contexts(std::vector<flyx_node>* p) const {
  for (int blk : context_blocks_) p->push_back({FLYX_NODE_CONTEXT, blk, 0, 0});
}

}  // namespace flyx::models::drafters
