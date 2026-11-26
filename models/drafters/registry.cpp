// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "models/drafters/registry.h"

#include <stdexcept>
#include <string>

#include "models/gguf.h"

namespace flyx::models::drafters {

using nlohmann::json;

namespace {

std::string companion_label(int i) { return "companion file " + std::to_string(i + 1) + ": "; }

}  // namespace

void DrafterRegistry::scan(const ModelFile& file, const TargetSpec& target, const MtpHooks* mtp, bool has_dv,
                           std::vector<int>* unclaimed) {
  *this = DrafterRegistry();
  if (unclaimed) unclaimed->clear();

  const json& hdr = file.header();
  sources_.dv = has_dv;
  if (mtp && mtp->embedded) sources_.mtp = true;
  if (hdr.is_object() && hdr.contains("draft") && hdr["draft"].is_object()) sources_.dflash2 = true;

  for (int i = 0; i < file.companions(); ++i) {
    const json& h = hdr.at("companions").at(size_t(i));
    const std::string what = companion_label(i);
    if (is_dspark_companion(h)) {
      if (!target.dspark_taps)
        throw std::runtime_error(what + "a DSpark drafter, which a " + target.arch + " model does not run");
      if (sources_.dspark) throw std::runtime_error(what + "the model has a DSpark drafter already");
      try {
        check_made_for(h, target);
      } catch (const std::exception& e) {
        throw std::runtime_error(what + e.what());
      }
      sources_.dspark = true;
      sources_.dspark_beside = true;
      dspark_companion_ = i;
    } else if (is_dflash_companion(h)) {
      if (sources_.dflash2) throw std::runtime_error(what + "the model has a DFlash2 drafter already");
      try {
        check_made_for(h, target);
      } catch (const std::exception& e) {
        throw std::runtime_error(what + e.what());
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
  if (dflash2_spec_) return *dflash2_spec_;
  const int i = dflash2_companion_;
  if (i < 0) return dflash2_spec_.emplace(parse_dflash2_json(file, file.header().at("draft"), target, -1));
  const json& h = file.header().at("companions").at(size_t(i));
  try {
    dflash2_spec_ = GgufMeta::is_header(h) ? parse_dflash2_gguf(file, h, target, i)
                                           : parse_dflash2_json(file, h.at("draft"), target, i);
  } catch (const std::exception& e) {
    throw std::runtime_error(companion_label(i) + e.what());
  }
  return *dflash2_spec_;
}

const DSparkSpec& DrafterRegistry::ensure_dspark_spec(const ModelFile& file, const TargetSpec& target) {
  if (dspark_spec_) return *dspark_spec_;
  const int i = dspark_companion_;
  if (i < 0) throw std::runtime_error("no DSpark companion file available");
  const json& h = file.header().at("companions").at(size_t(i));
  try {
    dspark_spec_ = parse_dspark_gguf(file, h, target, i);
  } catch (const std::exception& e) {
    throw std::runtime_error(companion_label(i) + e.what());
  }
  return *dspark_spec_;
}

void DrafterRegistry::validate_selected(const ModelFile& file, const std::vector<Drafter>& selected,
                                        const TargetSpec& target) {
  int64_t stride = 0;
  for (Drafter kind : selected) {
    int taps = 0;
    if (kind == Drafter::kDFlash2) taps = ensure_dflash2_spec(file, target).block.n_taps;
    else if (kind == Drafter::kDSpark) taps = ensure_dspark_spec(file, target).block.n_taps;
    stride += int64_t(taps) * target.hidden;
    if (stride > INT_MAX) throw std::runtime_error("combined drafter feature stride exceeds int32");
  }
}

int DrafterRegistry::attach(Drafter kind, const ModelFile& file, Tables& T, const TargetSpec& target,
                            const MtpHooks* mtp, int* max_drafts_out) {
  target_layers_ = target.layers;
  explicit_post_norm_ = target.has_post_norm_tap;
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
  auto record = [&](BlockTaps& attached, int block, const auto& spec) {
    attached = {block, spec.taps};
    context_blocks_.push_back(block);
    if (max_drafts_out) *max_drafts_out = spec.block.block - 1;
    return block;
  };
  if (kind == Drafter::kDFlash2) {
    const DFlash2Spec& spec = ensure_dflash2_spec(file, target);
    return record(dflash2_, build_dflash2(T, spec, target), spec);
  }
  if (kind == Drafter::kDSpark) {
    const DSparkSpec& spec = ensure_dspark_spec(file, target);
    return record(dspark_, build_dspark(T, file, spec, target), spec);
  }
  throw std::runtime_error("unsupported drafter kind in DrafterRegistry::attach");
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
  dflash2_.emit(p, layer_1based, is_post_norm ? FLYX_TAP_NORMED : 0);
}

void DrafterRegistry::emit_post_norm_taps(std::vector<flyx_node>* p, int n_layers) const {
  dspark_.emit(p, n_layers, FLYX_TAP_NORMED);
  dflash2_.emit(p, n_layers, FLYX_TAP_NORMED);
}

void DrafterRegistry::emit_contexts(std::vector<flyx_node>* p) const {
  for (int blk : context_blocks_) p->push_back({FLYX_NODE_CONTEXT, blk, 0, 0});
}

}  // namespace flyx::models::drafters
