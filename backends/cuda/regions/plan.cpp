// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Plans: which block a node runs, which blocks the kernels take, and the
// implementation of each node (the generic one or a fused override that fits);
// lower() resolves a program into a plan, forward() executes its nodes on a sequence (a prefill pass or a round).
#include "backends/cuda/regions/layers.h"

#include <algorithm>

namespace flyx::cuda {

std::string node_name(const flyx_node& n, const Block* b) {
  switch (n.kind) {
    case FLYX_NODE_RUN:
      if (std::holds_alternative<EmbedBlock>(*b)) return "embed";
      if (std::holds_alternative<GdnBlock>(*b)) return "gdn mixer";
      if (std::holds_alternative<AttnBlock>(*b)) return "attention mixer";
      if (std::holds_alternative<RelAttnBlock>(*b)) return "rel attention mixer";
      if (std::holds_alternative<MlpBlock>(*b)) return "mlp";
      if (std::holds_alternative<VisionStemBlock>(*b)) return "vision stem";
      if (std::holds_alternative<VitAttnBlock>(*b)) return "vit attention";
      if (std::holds_alternative<VitMlpBlock>(*b)) return "vit mlp";
      if (std::holds_alternative<VisionMergeBlock>(*b)) return "vision merge";
      if (std::holds_alternative<MoeBlock>(*b)) return "moe";
      return "head + choose";
    case FLYX_NODE_TAP: return "tap";
    case FLYX_NODE_DRAFT:
      if (std::holds_alternative<DvBlock>(*b)) return "dv tree";
      if (std::holds_alternative<DSparkBlock>(*b)) return "dspark block";
      return std::holds_alternative<MtpBlock>(*b) ? "mtp draft" : "dflash2 block";
    case FLYX_NODE_CONTEXT:
      if (std::holds_alternative<DvBlock>(*b)) return "dv canvas";
      if (std::holds_alternative<DSparkBlock>(*b)) return "dspark context";
      return std::holds_alternative<MtpBlock>(*b) ? "mtp context" : "dflash2 context";
    case FLYX_NODE_ACCEPT: return "accept";
    case FLYX_NODE_COMMIT: return "commit (fold)";
    case FLYX_NODE_GIVEN: return "given drafts";
    case FLYX_NODE_RES_RESET: return "res reset";
    case FLYX_NODE_LOOP_PRELUDE: return "loop prelude";
    case FLYX_NODE_LOOP_NEXT: return "loop next";
    case FLYX_NODE_RES_MIX: return "res mix";
    default: return "?";
  }
}

// ---- which blocks the kernels take ----------------------------------------------------------
bool supported(const flyx_bmodel* m, const Block& blk, std::string* why) {
  auto no = [&](const std::string& w) { return *why = w, false; };
  if (const auto* a = std::get_if<AttnBlock>(&blk)) {
    if (!attention_head_dim(a->hd) || a->rot > a->hd || a->rot % 2 || a->nkv < 1 || a->nh % a->nkv)
      return no("attention dimensions outside what the CUDA kernels take (head dim 32, 64, 128 or 256)");
    if (!a->q_norm.w != !a->k_norm.w) return no("the attention prep takes q and k norms both or neither");
    if (a->q_norm.w && (a->q_norm.offset != a->k_norm.offset || a->q_norm.eps != a->k_norm.eps))
      return no("the attention prep takes one offset and eps for the q and k norms");
    if (m->states.kv[size_t(a->kv)].fp8 && !kv_fp8_supported(m->opt.kernels, a->nh, a->nkv, a->hd))
      return no("an FP8 KV cache needs 1, 2 or 4 query heads per KV head");
  } else if (const auto* r = std::get_if<RelAttnBlock>(&blk)) {
    if (!attention_head_dim(r->hd) || r->nkv < 1 || r->nh % r->nkv || r->d_rel < 1 || r->conv < 2 || r->conv > 64)
      return no("rel_attn dimensions outside what the CUDA kernels take (head dim 32, 64, 128 or 256, conv 2..64)");
  } else if (const auto* e = std::get_if<MoeBlock>(&blk)) {
    if (e->n_experts < 1 || e->top_k < 1 || e->top_k > e->n_experts || (e->post_conv && (e->conv < 2 || e->conv > 64)))
      return no("MoE dimensions outside what the CUDA kernels support");
  } else if (const auto* g = std::get_if<GdnBlock>(&blk)) {
    if (g->gd.dk > 128 || g->gd.ck > 8 || g->gd.nk < 1 || g->gd.nv % g->gd.nk)
      return no("gated delta net dimensions outside what the CUDA kernels support (key head dim up to 128, conv up to 8)");
    if (g->gated.offset != 0.f) return no("the gated delta net's output norm takes w, not offset + w");
  } else if (const auto* t = std::get_if<MtpBlock>(&blk)) {
    if (t->norm_emb.offset != t->norm_hidden.offset || t->norm_emb.eps != t->norm_hidden.eps)
      return no("the MTP gather takes one offset and eps for its two norms");
    if (m->states.ring_width != block<AttnBlock>(m, t->attn).hidden) return no("the MTP ring keeps one hidden vector per position");
  } else if (const auto* d = std::get_if<DFlashBlock>(&blk)) {
    if (!attention_head_dim(d->hd) || d->sel_topk > 16 || d->block < 2 || d->block > FLYX_MAX_ROWS ||
        d->conv_group < 1 || d->hidden % d->conv_group || d->nkv < 1 || d->nh % d->nkv)
      return no("drafter dimensions outside what the CUDA kernels support");
    if (d->hidden != m->hidden || d->target_hidden != m->hidden) return no("drafter hidden size differs from the target's");
  } else if (const auto* ds = std::get_if<DSparkBlock>(&blk)) {
    if (!attention_head_dim(ds->hd) || ds->block < 2 || ds->block > FLYX_MAX_ROWS || ds->nkv < 1 || ds->nh % ds->nkv ||
        ds->hidden != m->hidden || ds->markov_rank < 1)
      return no("DSpark drafter dimensions outside what the CUDA kernels support");
    if (m->states.ring_width != ds->hidden) return no("DSpark ring width differs from hidden size");
  } else if (const auto* v = std::get_if<DvBlock>(&blk)) {
    if (v->hidden != m->hidden || v->topk != 16 || v->res_k != 16)  // a slot's 16 candidates (DV's kernels: kDvTop)
      return no("DV dimensions outside what the CUDA backend takes (its hidden size, 16 candidates a slot)");
  } else if (const auto* va = std::get_if<VitAttnBlock>(&blk)) {
    if (va->nh < 1 || va->hd < 1 || va->hd > 256 || va->hidden % 8 || (va->nh * va->hd) % 8)
      return no("vit_attn dimensions outside what the CUDA kernels support (head_dim <= 256, widths multiple of 8)");
    if (va->rope_mode == FLYX_VROPE_AXIAL_2D && va->hd % 4 != 0)
      return no("vit_attn 2D axial RoPE requires head_dim divisible by 4");
  } else if (const auto* vf = std::get_if<VitMlpBlock>(&blk)) {
    if (vf->hidden % 8 || vf->inter % 8)
      return no("vit_mlp dimensions must be multiples of 8");
  } else if (const auto* vm = std::get_if<VisionMergeBlock>(&blk)) {
    const int merged = vm->merge_mode == FLYX_VMERGE_SPATIAL_CONCAT
                           ? vm->in_hidden * vm->merge_size * vm->merge_size
                           : vm->in_hidden;
    if (merged % 8 || (vm->w_fc1.present() && vm->w_fc2.present() && vm->merge_hidden % 8))
      return no("vision_merge dimensions must be multiples of 8");
  }
  return true;
}

// The block a node runs, if it is one the node can run.
const Block* block_of(const flyx_bmodel* m, const flyx_node& n, std::string* err) {
  if (n.kind == FLYX_NODE_ACCEPT || n.kind == FLYX_NODE_COMMIT || n.kind == FLYX_NODE_GIVEN ||
      n.kind == FLYX_NODE_RES_RESET || n.kind == FLYX_NODE_LOOP_PRELUDE || n.kind == FLYX_NODE_LOOP_NEXT)
    return nullptr;
  if (n.kind == FLYX_NODE_RES_MIX) {
    if (m->vec_weights.find(n.arg0) == m->vec_weights.end())
      *err = "RES_MIX references missing 1D weight " + std::to_string(n.arg0);
    return nullptr;
  }
  if (n.block < 0 || n.block >= int(m->blocks.size())) {
    *err = "node names block " + std::to_string(n.block);
    return nullptr;
  }
  const Block* b = &m->blocks[size_t(n.block)];
  const bool drafter = std::holds_alternative<MtpBlock>(*b) || std::holds_alternative<DFlashBlock>(*b) ||
                       std::holds_alternative<DvBlock>(*b) || std::holds_alternative<DSparkBlock>(*b);
  bool fits = false;
  switch (n.kind) {
    case FLYX_NODE_RUN:
      fits = std::holds_alternative<EmbedBlock>(*b) || std::holds_alternative<AttnBlock>(*b) ||
             std::holds_alternative<GdnBlock>(*b) || std::holds_alternative<MlpBlock>(*b) ||
             std::holds_alternative<HeadBlock>(*b) || std::holds_alternative<VisionStemBlock>(*b) ||
             std::holds_alternative<VitAttnBlock>(*b) || std::holds_alternative<VitMlpBlock>(*b) ||
             std::holds_alternative<VisionMergeBlock>(*b) || std::holds_alternative<RelAttnBlock>(*b) ||
             std::holds_alternative<MoeBlock>(*b);
      break;
    case FLYX_NODE_TAP: {
      if (const auto* d = std::get_if<DFlashBlock>(b)) fits = n.arg0 >= 0 && n.arg0 < d->n_taps;
      else if (const auto* ds = std::get_if<DSparkBlock>(b)) fits = n.arg0 >= 0 && n.arg0 < ds->n_taps;
      break;
    }
    case FLYX_NODE_DRAFT:
    case FLYX_NODE_CONTEXT: fits = drafter; break;
    default: *err = "unknown node kind " + std::to_string(n.kind); return nullptr;
  }
  if (!fits) *err = "node kind " + std::to_string(n.kind) + " cannot run block " + std::to_string(n.block);
  return fits ? b : nullptr;
}

// ---- implementations ------------------------------------------------------------------------
// The 1..16-row GEMMs need K % 64 (NVFP4) / % 32 (bf16) for every input the producers split.  GGML block weights
// with 8-bit inputs take them from the producers in 8 bits, rounded once from the float32 values, a
// warp a group of 32: the gated norm's heads must be whole groups.  (The attention's output for them stays float32
// and is staged: layers.cpp.)
bool fused_attn(const flyx_bmodel*, const Block& blk) {
  const auto* a = std::get_if<AttnBlock>(&blk);
  return a && a->hidden % 64 == 0 && a->nh * a->hd % 64 == 0;
}
bool fused_gdn(const flyx_bmodel*, const Block& blk) {
  const auto* g = std::get_if<GdnBlock>(&blk);
  return g && g->hidden % 64 == 0 && g->v_width() % 64 == 0 && (!g->w_out.q8_inputs() || g->gd.dv % 32 == 0);
}
bool fused_mlp(const flyx_bmodel*, const Block& blk) {
  const auto* f = std::get_if<MlpBlock>(&blk);
  return f && f->hidden % 64 == 0 && f->inter % 64 == 0;
}
bool fused_mtp_draft(const flyx_bmodel* m, const Block& blk) {
  const auto* t = std::get_if<MtpBlock>(&blk);
  if (!t) return false;
  const auto& a = block<AttnBlock>(m, t->attn);
  const auto& f = block<MlpBlock>(m, t->mlp);
  return a.hidden % 64 == 0 && f.hidden % 64 == 0 && f.inter % 64 == 0;
}
bool fused_mtp_context(const flyx_bmodel* m, const Block& blk) {
  const auto* t = std::get_if<MtpBlock>(&blk);
  return t && block<AttnBlock>(m, t->attn).hidden % 64 == 0;
}
bool fused_dflash_draft(const flyx_bmodel*, const Block& blk) {
  const auto* d = std::get_if<DFlashBlock>(&blk);
  return d && d->inter % 64 == 0;
}
bool fused_dflash_context(const flyx_bmodel*, const Block& blk) {
  const auto* d = std::get_if<DFlashBlock>(&blk);
  return d && d->hidden % 64 == 0;
}
bool fused_rel_attn(const flyx_bmodel*, const Block& blk) {
  const auto* r = std::get_if<RelAttnBlock>(&blk);
  return r && (r->nh * r->hd) % 64 == 0;
}
bool fused_moe(const flyx_bmodel*, const Block& blk) { return std::holds_alternative<MoeBlock>(blk); }
bool fused_dspark_draft(const flyx_bmodel*, const Block& blk) {
  const auto* d = std::get_if<DSparkBlock>(&blk);
  return d && d->hidden % 64 == 0 && d->inter % 64 == 0;
}

void run_node(const Pass& ps, const flyx_node& n, const Block* b);

// RegionImpl (backend.h) lists an implementation; the table below is what resolve() picks from.
const RegionImpl kGeneric = {0, "generic", 1 << 30, nullptr, run_node};
const RegionImpl kOverrides[] = {
    {FLYX_NODE_RUN, "fused producers", 1 << 30, fused_gdn,
     [](const Pass& ps, const flyx_node&, const Block* b) { gdn_layer(ps, std::get<GdnBlock>(*b), true); }},
    {FLYX_NODE_RUN, "fused producers", 1 << 30, fused_attn,
     [](const Pass& ps, const flyx_node&, const Block* b) { attn_layer(ps, std::get<AttnBlock>(*b), true); }},
    {FLYX_NODE_RUN, "fused producers", 1 << 30, fused_rel_attn,
     [](const Pass& ps, const flyx_node&, const Block* b) { rel_attn_layer(ps, std::get<RelAttnBlock>(*b), true); }},
    {FLYX_NODE_RUN, "fused producers", 1 << 30, fused_mlp,
     [](const Pass& ps, const flyx_node&, const Block* b) { mlp_layer(ps, std::get<MlpBlock>(*b), true); }},
    {FLYX_NODE_RUN, "fused SiTU", 1 << 30, fused_moe,
     [](const Pass& ps, const flyx_node&, const Block* b) { moe_layer(ps, std::get<MoeBlock>(*b), true); }},
    {FLYX_NODE_DRAFT, "fused producers", 1 << 30, fused_mtp_draft,
     [](const Pass& ps, const flyx_node& n, const Block* b) {
       mtp_draft(ps.s, std::get<MtpBlock>(*b), n.arg0, ps.v.L, true);
     }},
    {FLYX_NODE_DRAFT, "fused producers", 1 << 30, fused_dflash_draft,
     [](const Pass& ps, const flyx_node& n, const Block* b) {
       draft_block(ps.s, std::get<DFlashBlock>(*b), n.arg0, ps.v.L, n.arg1 == FLYX_DRAFT_TREE, true);
     }},
    {FLYX_NODE_DRAFT, "fused producers", 1 << 30, fused_dspark_draft,
     [](const Pass& ps, const flyx_node& n, const Block* b) {
       dspark_draft(ps.s, std::get<DSparkBlock>(*b), n.arg0, ps.v.L, true);
     }},
    {FLYX_NODE_CONTEXT, "fused producers", 1 << 30, fused_mtp_context,
     [](const Pass& ps, const flyx_node&, const Block* b) { mtp_context(ps, std::get<MtpBlock>(*b), true); }},
    {FLYX_NODE_CONTEXT, "fused producers", 1 << 30, fused_dflash_context,
     [](const Pass& ps, const flyx_node&, const Block* b) { draft_context(ps, std::get<DFlashBlock>(*b), true); }},
};

bool generic_impl(const RegionImpl* r) { return r == &kGeneric; }

const RegionImpl* resolve(const flyx_bmodel* m, const flyx_node& n, const Block* b, int max_rows) {
  const bool generic_only = !m->opt.fusion;
  const RegionImpl* best = &kGeneric;
  if (!generic_only)
    for (const auto& r : kOverrides)
      if (r.kind == n.kind && max_rows <= r.max_rows && (!r.fits || (b && r.fits(m, *b)))) best = &r;
  return best;
}

namespace {

void forward_nodes(const Pass& ps, size_t begin, size_t end, int copy_prefix) {
  auto* s = ps.s;
  cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
  cudaStreamIsCapturing(s->st, &cs);
  const bool prof = s->profile && cs == cudaStreamCaptureStatusNone && begin == 0;
  const size_t nn = end - begin;
  if (prof)
    while (s->prof_ev.size() < nn + 1) {
      cudaEvent_t e;
      CK(cudaEventCreate(&e));
      s->prof_ev.push_back(e);
    }
  if (prof) CK(cudaEventRecord(s->prof_ev[0], s->st));
  // What the pass's attentions read through stagings, in order: the caches' views and the copies in
  // host memory (0073); their copies to the stagings start ahead.
  // A prefill's windows of keys past the stagings take the two parities in turn, a small pass past them none (0074).
  std::vector<flyx_seq::StageItem> staged;
  int parity = 0;
  auto stage_kv = [&](int kv_idx) {
    const KvBuf& kb = s->kv[size_t(kv_idx)];
    if (kb.vk && ps.reads_cache(kb)) {
      if (const int windows = ps.round ? 0 : s->windows(kb, ps.rows)) {
        const int from = s->window_from(kb), len = s->window_len(kb), end_pos = s->position + ps.rows;
        for (int w = 0; w < windows; ++w) {
          const int w0 = from + w * len;
          staged.push_back({kv_idx, false, w0, std::min(end_pos, w0 + len), parity});
          parity = kb.wslot[1] >= 0 ? parity ^ 1 : 0;
        }
      } else if (!s->reads_direct(kb, ps.rows, ps.round)) {
        staged.push_back({kv_idx, false});
      }
    }
    if (kb.kq_stage && ps.reads_copy_only(kb))
      staged.push_back({kv_idx, true, copy_prefix, s->copy_scan_end < copy_prefix ? s->copy_scan_end : 0});
  };
  for (size_t i = begin; i < end; ++i) {
    if (ps.p->nodes[i].kind != FLYX_NODE_RUN || !ps.p->block[i]) continue;
    if (const auto* a = std::get_if<AttnBlock>(ps.p->block[i])) stage_kv(a->kv);
    else if (const auto* r = std::get_if<RelAttnBlock>(ps.p->block[i])) stage_kv(r->kv);
  }
  s->stage_begin(std::move(staged), ps.v.L.st);
  ResState res_state{};
  res_state.x_cur = ps.v.b.x;
  Pass cur = ps;
  cur.res = &res_state;
  for (size_t i = begin; i < end; ++i) {
    const flyx_node& nd = ps.p->nodes[i];
    const Block* blk = ps.p->block[i];
    const bool takes_pending =
        nd.kind == FLYX_NODE_RUN && blk &&
        (std::holds_alternative<RelAttnBlock>(*blk) || std::holds_alternative<MoeBlock>(*blk));
    if (!takes_pending) flush_pending_post_conv(cur, true);
    ps.p->impl[i]->fn(cur, nd, blk);
    if (prof) CK(cudaEventRecord(s->prof_ev[i - begin + 1], s->st));
  }
  flush_pending_post_conv(cur, true);
  s->stage_end(ps.v.L.st);
  if (!prof) return;
  CK(cudaEventSynchronize(s->prof_ev[nn]));
  const std::string phase = ps.round ? "round " + std::to_string(ps.width()) + ": " : "prefill: ";
  for (size_t i = begin; i < end; ++i) {
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, s->prof_ev[i - begin], s->prof_ev[i - begin + 1]));
    auto& e = s->prof[phase + node_name(ps.p->nodes[i], ps.p->block[i])];
    e.ms += ms;
    ++e.calls;
  }
  ++s->prof[phase + "(passes)"].calls;
}

}  // namespace

void forward(const Pass& ps, int copy_prefix) { forward_nodes(ps, 0, ps.p->nodes.size(), copy_prefix); }

void forward_target(const Pass& ps, int copy_prefix) {
  forward_nodes(ps, target_start(ps.p), ps.p->nodes.size(), copy_prefix);
}

bool sequential_draft(const flyx_plan* p) {
  for (size_t i = 0; i < p->nodes.size(); ++i)
    if (p->nodes[i].kind == FLYX_NODE_DRAFT && p->block[i])
      return std::holds_alternative<MtpBlock>(*p->block[i]) || std::holds_alternative<DSparkBlock>(*p->block[i]);
  return false;
}

void draft_step_range(const Pass& ps, int start, int end) {
  for (size_t i = 0; i < ps.p->nodes.size(); ++i) {
    if (ps.p->nodes[i].kind != FLYX_NODE_DRAFT || !ps.p->block[i]) continue;
    const bool fused = ps.p->impl[i] != &kGeneric;
    if (const auto* t = std::get_if<MtpBlock>(ps.p->block[i])) {
      mtp_draft(ps.s, *t, end, ps.v.L, fused, start);
    } else if (const auto* d = std::get_if<DFlashBlock>(ps.p->block[i])) {
      draft_block(ps.s, *d, end, ps.v.L, ps.p->nodes[i].arg1 == FLYX_DRAFT_TREE, fused);
    } else if (const auto* ds = std::get_if<DSparkBlock>(ps.p->block[i])) {
      dspark_draft(ps.s, *ds, end, ps.v.L, fused, start, ps.p->width);
    }
    return;
  }
}

void run_node(const Pass& ps, const flyx_node& n, const Block* b) {
  auto* s = ps.s;
  const RowView& rv = ps.v;
  cudaStream_t st = rv.L.st;
  switch (n.kind) {
    case FLYX_NODE_RUN:
      if (const auto* e = std::get_if<EmbedBlock>(b)) {
        if (ps.round) build_rows(s->P, s->drafts, -1, rv.b.tokens, ps.rows, st);
        if (residual16(ps)) {
          embed(rv.b.tokens, e->table, e->hidden, x16_of(rv), ps.rows, st);
          if (ps.n_embeds > 0)
            embed_scatter(nullptr, x16_of(rv), e->hidden, ps.embeds, ps.embed_rows, ps.n_embeds, st);
        } else if (e->has_norm) {
          embed_tokens(*e, rv.b.tokens, rv.b.h, ps.rows, st);
          rmsnorm(rv.b.h, e->hidden, e->norm.w, e->norm.offset, e->norm.eps, rv.b.x, e->hidden, ps.rows, e->hidden, st);
          if (ps.n_embeds > 0) embed_scatter(rv.b.x, nullptr, e->hidden, ps.embeds, ps.embed_rows, ps.n_embeds, st);
        } else {
          embed_tokens(*e, rv.b.tokens, rv.b.x, ps.rows, st);
          if (ps.n_embeds > 0) embed_scatter(rv.b.x, nullptr, e->hidden, ps.embeds, ps.embed_rows, ps.n_embeds, st);
        }
      } else if (const auto* g = std::get_if<GdnBlock>(b)) {
        gdn_layer(ps, *g);
      } else if (const auto* a = std::get_if<AttnBlock>(b)) {
        attn_layer(ps, *a);
      } else if (const auto* r = std::get_if<RelAttnBlock>(b)) {
        rel_attn_layer(ps, *r);
      } else if (const auto* f = std::get_if<MlpBlock>(b)) {
        mlp_layer(ps, *f);
      } else if (const auto* me = std::get_if<MoeBlock>(b)) {
        moe_layer(ps, *me);
      } else if (!n.arg0) {
        head(ps, std::get<HeadBlock>(*b), false);
      } else if (ps.sample_last) {
        head(ps, std::get<HeadBlock>(*b), true);
      }
      break;
    case FLYX_NODE_TAP: {
      int head_idx = -1, feats_off = 0;
      if (const auto* df = std::get_if<DFlashBlock>(b)) {
        head_idx = df->head;
        feats_off = df->feats_offset;
      } else {
        const auto& ds = std::get<DSparkBlock>(*b);
        head_idx = ds.head;
        feats_off = ds.feats_offset;
      }
      if (n.arg1 == FLYX_TAP_NORMED) {
        const auto& hb = block<HeadBlock>(s->m, head_idx);
        if (residual16(ps))
          rmsnorm(x16_of(rv), s->m->hidden, hb.norm.w, hb.norm.offset, hb.norm.eps, rv.b.h, s->m->hidden, ps.rows,
                  s->m->hidden, st);
        else
          rmsnorm(rv.b.x, s->m->hidden, hb.norm.w, hb.norm.offset, hb.norm.eps, rv.b.h, s->m->hidden, ps.rows,
                  s->m->hidden, st);
        tap_copy(rv.b.h, s->m->hidden, rv.b.feats + feats_off, rv.ld_feats, n.arg0, ps.rows, st);
      } else if (residual16(ps)) {
        tap_copy(x16_of(rv), s->m->hidden, rv.b.feats + feats_off, rv.ld_feats, n.arg0, ps.rows, st);
      } else {
        tap_copy(rv.b.x, s->m->hidden, rv.b.feats + feats_off, rv.ld_feats, n.arg0, ps.rows, st);
      }
      break;
    }
    case FLYX_NODE_DRAFT:
      if (const auto* t = std::get_if<MtpBlock>(b)) mtp_draft(s, *t, n.arg0, rv.L);
      else if (const auto* v = std::get_if<DvBlock>(b)) dv_draft(s, *v, n.arg0, rv.L);
      else if (const auto* ds = std::get_if<DSparkBlock>(b)) dspark_draft(s, *ds, n.arg0, rv.L);
      else draft_block(s, std::get<DFlashBlock>(*b), n.arg0, rv.L, n.arg1 == FLYX_DRAFT_TREE);
      break;
    case FLYX_NODE_CONTEXT:
      if (const auto* t = std::get_if<MtpBlock>(b)) mtp_context(ps, *t);
      else if (const auto* v = std::get_if<DvBlock>(b); v && ps.round) dv_context_unless_noop(ps, *v);
      else if (v) forget_canvas(s, st);  // a prefill's: the canvas starts over
      else if (const auto* ds = std::get_if<DSparkBlock>(b)) dspark_context(ps, *ds);
      else draft_context(ps, std::get<DFlashBlock>(*b));
      break;
    case FLYX_NODE_ACCEPT: {
      const bool conf = ps.p->has_draft && !ps.p->tree && ps.draft_conf;
      const float* dp = conf ? s->draft_prob : nullptr;
      const float* dm = conf ? s->draft_margin : nullptr;
      const int w = ps.width();
      if (ps.p->tree || ps.relaxed) {
        // A wide round's relaxed rules read each draft's logit as its head gathered it (child_logits; ld 0).
        const bool wide = ps.rows > FLYX_MAX_ROWS;
        accept_tree(s->P, s->G, s->drafts, rv.b.choice, w, s->ring_dev, s->reply,
                    ps.p->tree ? s->tree : nullptr, !ps.relaxed ? nullptr : wide ? s->child_z : rv.logits,
                    wide ? 0 : s->m->vocab, s->accept_stats, st, dp, dm);
        if (ps.p->tree) compact_path(ps);
      } else {
        accept(s->P, s->G, s->drafts, rv.b.choice, w, s->ring_dev, s->reply, st, dp, dm);
      }
      break;
    }
    case FLYX_NODE_COMMIT: commit(ps); break;
    case FLYX_NODE_GIVEN: given_drafts(s->P, s->oracle, n.arg0, s->drafts, st); break;
    case FLYX_NODE_RES_RESET:
      ps.res->res_n = 0;
      ps.res->prelude_n = 0;
      ps.res->loop_n = 0;
      break;
    case FLYX_NODE_LOOP_PRELUDE: {
      const size_t row_bytes = size_t(ps.rows) * size_t(s->m->hidden) * sizeof(float);
      CK(cudaMemcpyAsync(rv.b.prelude_cur, rv.b.x, row_bytes, cudaMemcpyDeviceToDevice, st));
      if (ps.res->res_n > 0)
        CK(cudaMemcpyAsync(rv.b.prelude_blocks, rv.b.res_bank, size_t(ps.res->res_n) * row_bytes,
                           cudaMemcpyDeviceToDevice, st));
      ps.res->prelude_n = ps.res->res_n;
      break;
    }
    case FLYX_NODE_LOOP_NEXT: {
      const size_t row_bytes = size_t(ps.rows) * size_t(s->m->hidden) * sizeof(float);
      const int skip = ps.res->prelude_n + 1;
      const int ln = std::max(0, ps.res->res_n - skip);
      CK(cudaMemcpyAsync(rv.b.loop_bank_out, rv.b.x, row_bytes, cudaMemcpyDeviceToDevice, st));
      if (ln > 0)
        CK(cudaMemcpyAsync(rv.b.loop_bank, rv.b.res_bank + size_t(skip) * size_t(ps.rows) * size_t(s->m->hidden),
                           size_t(ln) * row_bytes, cudaMemcpyDeviceToDevice, st));
      ps.res->loop_n = ln;
      CK(cudaMemcpyAsync(rv.b.x, rv.b.prelude_cur, row_bytes, cudaMemcpyDeviceToDevice, st));
      if (ps.res->prelude_n > 0)
        CK(cudaMemcpyAsync(rv.b.res_bank, rv.b.prelude_blocks, size_t(ps.res->prelude_n) * row_bytes,
                           cudaMemcpyDeviceToDevice, st));
      ps.res->res_n = ps.res->prelude_n;
      break;
    }
    case FLYX_NODE_RES_MIX: {
      const int H = s->m->hidden;
      attn_res_mix(rv.b.res_bank, ps.res->res_n, rv.b.x, s->m->vec_weights.at(n.arg0), 1, ps.rows, H, rv.b.h, st);
      CK(cudaMemcpyAsync(rv.b.x, rv.b.h, size_t(ps.rows) * size_t(H) * sizeof(float), cudaMemcpyDeviceToDevice, st));
      break;
    }
    default: throw std::runtime_error("cuda: unknown node kind " + std::to_string(n.kind));
  }
}

// Whether this backend runs a DV node: the tree a round asks for (dv_tree_takes: the kernels' limits) and, after the
// round, the canvas pass through the target's blocks (no: why).  Asked by lower, never by supported(): a round the
// kernels cannot draft is a plan the engine goes without (it decodes without DV and says why), not a model that does
// not load.
bool dv_runs(const flyx_bmodel* m, const DvBlock& v, const flyx_node& n, std::string* why) {
  if (n.kind == FLYX_NODE_DRAFT) {
    if (n.arg1 != FLYX_DRAFT_TREE) return *why = "DV drafts trees only", false;
    const int depths = std::min(v.slots - 1, n.arg0);
    return dv_tree_takes(v.tree, n.arg0, v.tree == FLYX_DV_TREE_BEAM ? v.beam : v.pool_ranks, depths, why);
  }
  for (int j = 0; j < v.n_layers; ++j)
    if (block<AttnBlock>(m, block<DvLayer>(m, v.first_layer + j).attn).gate)
      return *why = "DV's canvas pass runs attention without an output gate", false;
  return true;
}

int lower(flyx_bmodel* m, const flyx_program* prog, flyx_plan** out) {
  if (prog->phase == FLYX_PHASE_VISION) {
    if (prog->max_rows < 1 || prog->n_nodes < 1) return fail("lower: bad vision program");
    auto p = std::make_unique<flyx_plan>();
    p->m = m;
    p->phase = prog->phase;
    p->width = 0;
    p->max_rows = prog->max_rows;
    p->nodes.assign(prog->nodes, prog->nodes + prog->n_nodes);
    for (const auto& n : p->nodes) {
      std::string err;
      const Block* b = block_of(m, n, &err);
      if (!err.empty()) return fail("lower: " + err);
      const bool ok = n.kind == FLYX_NODE_RUN && b &&
                      (std::holds_alternative<VisionStemBlock>(*b) || std::holds_alternative<VitAttnBlock>(*b) ||
                       std::holds_alternative<VitMlpBlock>(*b) || std::holds_alternative<VisionMergeBlock>(*b));
      if (!ok) return fail("lower: vision plan contains non-vision node");
      p->block.push_back(b);
      p->impl.push_back(&kGeneric);
    }
    if (!std::holds_alternative<VisionStemBlock>(*p->block.front()))
      return fail("lower: vision plan must begin with a vision_stem block");
    *out = p.release();
    return 0;
  }
  if (prog->max_rows < 1 || (prog->phase == FLYX_PHASE_ROUND && prog->max_rows > m->wide_rows) ||
      prog->max_rows > kMaxPrefillRows)
    return fail("bad row count");
  // A wide round (more rows than FLYX_MAX_ROWS): given drafts in a chain or a tree, the generic regions on every row's
  // 1..16-row bits (Launch::round_rows).
  const bool wide = prog->phase == FLYX_PHASE_ROUND && prog->max_rows > FLYX_MAX_ROWS;
  auto p = std::make_unique<flyx_plan>();
  p->m = m;
  p->phase = prog->phase;
  p->width = prog->width;
  p->max_rows = prog->max_rows;
  p->nodes.assign(prog->nodes, prog->nodes + prog->n_nodes);
  std::map<std::string, int> used;
  int most = 0;  // drafts the round may verify: up to a full round of given ones, else what its drafter proposes
  for (const auto& n : p->nodes) {
    std::string err;
    const Block* b = block_of(m, n, &err);
    if (!err.empty()) return fail("lower: " + err);
    if (const auto* v = b ? std::get_if<DvBlock>(b) : nullptr) {
      if (std::string why; (n.kind == FLYX_NODE_DRAFT || prog->phase == FLYX_PHASE_ROUND) && !dv_runs(m, *v, n, &why))
        return fail("lower: " + why);
      if (n.kind == FLYX_NODE_DRAFT) {
        try {
          make_dv_tables(m, n.block);  // out of GPU memory: the engine has the backend trim, and lowers again
        } catch (const std::exception& e) {
          return fail(std::string("lower: DV's selector tables: ") + e.what());
        }
      }
    }
    if (n.kind == FLYX_NODE_RES_RESET && n.arg0 > m->need.max_bank)
      return fail("lower: RES_RESET bank depth exceeds model max_bank");
    if (n.kind == FLYX_NODE_GIVEN) most = m->wide_rows - 1;
    // A wide round's drafts are given (a chain or a tree), or DV's tree (its kernels draft up to 63).
    if (wide && n.kind == FLYX_NODE_DRAFT && !(b && std::holds_alternative<DvBlock>(*b)))
      return fail("lower: a wide round verifies given drafts or DV's tree");
    if (n.kind == FLYX_NODE_GIVEN && n.arg1 == FLYX_GIVEN_TREE) p->tree = true;
    if (n.kind == FLYX_NODE_RUN && b) {
      if (const auto* a = std::get_if<AttnBlock>(b)) {
        if (std::find(p->tree_kv.begin(), p->tree_kv.end(), a->kv) == p->tree_kv.end()) p->tree_kv.push_back(a->kv);
      } else if (const auto* r = std::get_if<RelAttnBlock>(b)) {
        if (std::find(p->tree_kv.begin(), p->tree_kv.end(), r->kv) == p->tree_kv.end()) p->tree_kv.push_back(r->kv);
      }
    }
    if (n.kind == FLYX_NODE_DRAFT && n.arg1 == FLYX_DRAFT_TREE) {
      // The DFlash2 selector's candidates make a tree (ops/draft/ref.h), DV's selector its own (its tree's T at
      // most); MTP drafts one token a step.
      const auto* v = std::get_if<DvBlock>(b);
      if (!v && !std::holds_alternative<DFlashBlock>(*b)) return fail("lower: only DFlash2 and DV draft trees");
      p->tree = p->drafted_tree = p->has_draft = true;
      most = v ? max_drafts(*b) : FLYX_MAX_ROWS - 1;
    } else if (n.kind == FLYX_NODE_DRAFT) {
      if (n.arg1 != 0) return fail("lower: unknown draft node option " + std::to_string(n.arg1));
      p->has_draft = true;
      most = max_drafts(*b);
    }
    p->block.push_back(b);
    p->impl.push_back(wide ? &kGeneric : resolve(m, n, b, p->max_rows));
    if (p->impl.back() != &kGeneric) ++used[node_name(n, b) + ": " + p->impl.back()->name];
  }
  if (prog->phase == FLYX_PHASE_ROUND && prog->width > most)
    return fail("round width exceeds what the drafter can propose");
  // A round writes one entry of the result ring (its accept: new_entry), the one ring_publish copies to the host.
  if (prog->phase == FLYX_PHASE_ROUND &&
      std::count_if(p->nodes.begin(), p->nodes.end(), [](const flyx_node& n) { return n.kind == FLYX_NODE_ACCEPT; }) != 1)
    return fail("lower: a round has exactly one accept node");
  if (prog->n_checkpoints > 0) {
#if CUDART_VERSION < 12040
    return fail("lower: adaptive stopping plans need CUDA 12.4+");
#else
    if (!m->opt.graphs) return fail("lower: adaptive stopping plans need CUDA graphs");
    if (prog->phase != FLYX_PHASE_ROUND || !p->has_draft || p->tree || !prog->checkpoints ||
        prog->n_checkpoints < 2 || prog->n_checkpoints > FLYX_MAX_STOP_BUCKETS || prog->width >= FLYX_MAX_ROWS)
      return fail("lower: invalid adaptive stopping checkpoints");
    for (int i = 0; i < prog->n_checkpoints; ++i) {
      const int c = prog->checkpoints[i];
      if (c < 1 || c > prog->width || (i > 0 && c <= prog->checkpoints[i - 1]))
        return fail("lower: adaptive checkpoints must be strictly increasing within [1, width]");
    }
    if (prog->checkpoints[prog->n_checkpoints - 1] != prog->width)
      return fail("lower: last adaptive checkpoint must equal plan width");
    p->checkpoints.assign(prog->checkpoints, prog->checkpoints + prog->n_checkpoints);
    p->earlier_checkpoints = prog->earlier_checkpoints != 0;
#endif
  }
  if (!p->tree) p->tree_kv.clear();
  if (p->tree && (p->phase != FLYX_PHASE_ROUND || p->width < 1)) return fail("lower: a tree is a round's drafts");
  if (wide && p->tree)  // a wide chain runs on a gated delta net (its replay holds the rows); a wide tree does not
    for (const Block* b : p->block)
      if (b && std::holds_alternative<GdnBlock>(*b))
        return fail("lower: tree rounds of more than " + std::to_string(FLYX_MAX_ROWS) +
                    " rows on a gated delta net (its tree recurrence stages 16 rows)");
  if (prog->phase == FLYX_PHASE_PREFILL && ensure_prefill_rows(m, prog->max_rows) != 0) return -1;
  if (m->verbose && !used.empty()) {
    std::fprintf(stderr, "flyx-cuda: %s plan (%d rows):", p->phase == FLYX_PHASE_ROUND ? "round" : "prefill",
                 p->max_rows);
    for (const auto& [k, v] : used) std::fprintf(stderr, " %s x%d;", k.c_str(), v);
    std::fprintf(stderr, "\n");
  }
  // Its graphs' size, in every session's room (graph_room): a round plan's, but for a tree of given drafts, which the
  // engine lowers and never runs (tests and tree_bench do: a session keeps those in the room the others leave).
  if (p->phase == FLYX_PHASE_ROUND && !(p->tree && !p->drafted_tree)) plan_lowered(m, p.get());
  *out = p.release();
  return 0;
}

void plan_free(flyx_plan* p) {
  if (p && p->phase == FLYX_PHASE_ROUND) forget_graphs(p->m, p);
  delete p;
}

}  // namespace flyx::cuda

void flyx_seq::print_profile() {
  std::fprintf(stderr, "flyx-cuda profile (no graphs; per pass, summed over layers):\n");
  std::map<std::string, long> passes;
  for (const auto& [k, v] : prof)
    if (k.ends_with("(passes)")) passes[k.substr(0, k.size() - 8)] = v.calls;
  for (const auto& [ph, np] : passes) {
    double total = 0;
    for (const auto& [k, v] : prof)
      if (k.starts_with(ph) && !k.ends_with("(passes)")) total += v.ms;
    std::fprintf(stderr, "  %s %ld passes, %.3f ms per pass\n", ph.c_str(), np, total / np);
    for (const auto& [k, v] : prof)
      if (k.starts_with(ph) && !k.ends_with("(passes)"))
        std::fprintf(stderr, "    %-24s %8.3f ms  %5.1f%%\n", k.substr(ph.size()).c_str(), v.ms / np, 100 * v.ms / total);
  }
}
