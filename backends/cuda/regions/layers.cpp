// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// What each program node runs on its block: the generic implementation of every
// region; fused=true is the override regions/plan.cpp may pick.
#include <bit>

#include "backends/cuda/regions/layers.h"

namespace flyx::cuda {

// Draft head top-k / argmax dispatch (1-bit rough pick on packed or bf16 head, else full draft logits).  k > the
// pick's kDraftCands (a wide walk's): that many candidates scored exactly, laid out in a row's room (2 k + 1 floats of
// max(V, kDraftPickFloats)); a vocabulary with less room reads the full draft logits.
void draft_head_pick(const HeadBlock& head, const float* x, int nr, int k, int* idx, float* val,
                     const DraftPickScratch& sc, const Launch& L, bool exact_1row, float* prob, float* margin) {
  const int H = head.hidden, V = head.vocab;
  const bool pick = head.coarse.bits && (k <= kDraftCands || 2 * k + 1 <= std::max(V, kDraftPickFloats));
  if (pick && head.wb.data) {
    draft_pick_blocks(x, H, head.coarse, head.wb, V, H, nr, k, idx, val, sc.logits, L, exact_1row, prob, margin);
  } else if (pick) {
    draft_pick(x, H, head.coarse, head.w, V, H, nr, k, idx, val, sc.logits, L, exact_1row, prob, margin);
  } else {
    draft_logits(head, x, H, sc.logits, nr, L);
    if (k == 1 && !val && !prob && !margin) argmax_rows(sc.logits, V, V, idx, nr, sc.argv, sc.argi, L.st);
    else topk_rows(sc.logits, V, V, k, idx, val, nr, L.st, sc.topv, sc.topi, sc.top_rows, prob, margin);
  }
}

bool draft_head_additive_ready(const HeadBlock& head, const Lin& extra, const SignHead& extra_coarse, int K_extra) {
  return head.coarse.bits && extra_coarse.bits && (head.wb.data || head.w) &&
         (extra.w || (extra.packed() && extra.blk.size() == 1 && K_extra % 128 == 0));
}

void draft_head_pick_additive_step(const HeadBlock& head, const float* x_base, const Lin& extra,
                                   const SignHead& extra_coarse, const float* extra_bias, const float* x_extra,
                                   int K_extra, float softcap, float* cand_row, int ld, float* scratch_add, int k,
                                   int* idx_out, float* val_out, float* prob_out, float* margin_out, const Launch& L,
                                   const DevParams* P, int pos_add, const DevGen* G) {
  const int H = head.hidden, V = head.vocab;
  cudaStream_t st = L.st;
  gemm_sign_rough_add(x_extra, K_extra, extra_coarse, extra_bias, V, K_extra, cand_row, ld, L);
  draft_select_cands(cand_row, ld, V, 1, L);
  int* cand_i = reinterpret_cast<int*>(cand_row);
  if (head.wb.data) {
    dots_blocks(head.wb, H, cand_i, cand_i + 2 * kDraftCandCap, cand_row + kDraftCandCap, size_t(ld), kDraftCandCap,
                x_base, H, 1, st);
  } else {
    draft_score_bf16(x_base, H, head.w, nullptr, H, 1, cand_row, ld, false, 0.f, st);
  }
  if (extra.w) {
    draft_score_bf16(x_extra, K_extra, extra.w, extra_bias, K_extra, 1, cand_row, ld, true, softcap, st);
  } else {
    dots_blocks(extra.blk[0], K_extra, cand_i, cand_i + 2 * kDraftCandCap, scratch_add, 0, kDraftCandCap, x_extra,
                K_extra, 1, st);
    draft_combine_cands(scratch_add, 0, extra_bias, softcap, 1, cand_row, ld, st);
  }
  draft_pick_topk(cand_row, ld, 1, k, idx_out, val_out, st, prob_out, margin_out, P, pos_add, G);
}

void norm_project(const float* x, float* h, float* out, int ld_out, int R, const Norm& n, const Lin& w, int N, int H,
                  const Launch& L, bool fused) {
  if (fused) {
    rmsnorm_split(x, H, n.w, n.offset, n.eps, R, H, L, w.q8_inputs());
    gemm_presplit(w, N, H, out, ld_out, R, false, L);
  } else {
    rmsnorm(x, H, n.w, n.offset, n.eps, h, H, R, H, L.st);
    gemm(h, H, w, N, H, out, ld_out, R, false, L);
  }
}

void norm_and_project(const RowView& v, int R, const Norm& n, const Lin& w, int N, int H, bool fused) {
  norm_project(v.b.x, v.b.h, v.b.proj, v.ld_proj, R, n, w, N, H, v.L, fused);
}

bool residual16_model(flyx_bmodel* m) {
  if (m->res16 < 0) {  // every block of the stack: the embedding a bf16 table, each residual add a W4A4 GEMM
    bool ok = m->opt.kernels.prefill_fp4 && gemm_w4a4_supported();
    std::vector<bool> drafter_parts(m->blocks.size(), false);  // the MTP layer's blocks add to no residual here
    for (const Block& b : m->blocks)
      if (const auto* t = std::get_if<MtpBlock>(&b)) {
        if (t->attn >= 0) drafter_parts[size_t(t->attn)] = true;
        if (t->mlp >= 0) drafter_parts[size_t(t->mlp)] = true;
      }
    for (size_t i = 0; i < m->blocks.size() && ok; ++i) {
      if (drafter_parts[i]) continue;
      const Block& b = m->blocks[i];
      if (const auto* e = std::get_if<EmbedBlock>(&b)) ok = e->table != nullptr && !e->has_norm;
      else if (const auto* g = std::get_if<GdnBlock>(&b))
        ok = g->w_in.q && g->w_out.q && g->hidden % 128 == 0 && g->v_width() % 128 == 0 && !g->lora;
      else if (const auto* a = std::get_if<AttnBlock>(&b))  // adapters (ABI 43) add float32 products: float32 rows
        ok = a->w_in.q && a->w_out.q && a->hidden % 128 == 0 && a->nh * a->hd % 128 == 0 && !a->lora;
      else if (const auto* f = std::get_if<MlpBlock>(&b))
        ok = f->w_gu.q && f->w_down.q && f->hidden % 128 == 0 && f->inter % 128 == 0 && !f->lora;
      else if (std::holds_alternative<RelAttnBlock>(b) || std::holds_alternative<MoeBlock>(b))
        ok = false;
    }
    m->res16 = ok ? 1 : 0;
  }
  return m->res16 == 1;
}

bool residual16_pass(const Pass& ps) {
  if (ps.round || ps.rows <= 16) return false;
  return residual16_model(ps.s->m) && ps.v.L.scratch && ps.v.L.scratch->af32;
}

bool residual16(const Pass& ps) { return ps.s->m->opt.fusion && residual16_pass(ps); }

void mlp_body(float* x, const MlpBlock& f, int R, float* h, float* proj, int ld_proj, float* act, int ld_act,
              const Launch& L, bool fused, bool add16) {
  const int H = f.hidden, I = f.inter;
  const Norm& n = f.norm;
  if (fused) {
    rmsnorm_split(x, H, n.w, n.offset, n.eps, R, H, L, f.w_gu.q8_inputs());
    // SwiGLU in the gate|up GEMM's epilogue: no float32 gate|up rows (bf16 and NVFP4 weights on both sides).
    if (R > 16 && I % 8 == 0 && !f.w_gu.packed() && !f.w_down.packed()) {
      gemm_glu_presplit(f.w_gu, I, H, R, L);
      gemm_on_glu(f.w_down, H, I, x, H, R, true, L);
      return;
    }
    // A round on block weights with 8-bit inputs on both sides: the gate|up GEMM writes the SwiGLU too, beside its own
    // input (the bits of swiglu_split), which the down GEMM takes.
    if (R <= 16 && f.w_gu.packed() && f.w_down.q8_inputs() &&
        gemm_blocks_round_glu_takes(f.w_gu.blk.data(), int(f.w_gu.blk.size()), R, I, L)) {
      const RoundGlu glu{proj, ld_proj, I};
      Launch gu = L;
      gu.glu = &glu;
      gemm_presplit(f.w_gu, 2 * I, H, proj, ld_proj, R, false, gu);
      const Scratch act_q8 = glu_inputs(*L.scratch);
      Launch down = L;
      down.scratch = &act_q8;
      gemm_presplit(f.w_down, H, I, x, H, R, true, down);
      return;
    }
    gemm_presplit(f.w_gu, 2 * I, H, proj, ld_proj, R, false, L);
    swiglu_split(proj, ld_proj, I, R, L, f.w_down.q8_inputs());
    gemm_presplit(f.w_down, H, I, x, H, R, true, L);
    return;
  }
  rmsnorm(x, H, n.w, n.offset, n.eps, h, H, R, H, L.st);
  gemm(h, H, f.w_gu, 2 * I, H, proj, ld_proj, R, false, L);
  swiglu(proj, ld_proj, I, act, ld_act, R, L.st);
  // The fused gate|up GEMM keeps the SwiGLU rows in bf16 (gemm_glu_staged); a down projection of NVFP4 inputs rounds
  // them from those rows (gemm_on_glu), whatever the gate|up weights are.
  if (R > 16 && I % 8 == 0 && !f.w_gu.packed() && !f.w_down.packed() && prefill_rows_bf16(L, f.w_down.q, I))
    round_rows_bf16(act, ld_act, I, R, L.st);
  if (add16) {  // h, H wide, is free past the gate|up GEMM
    gemm(act, ld_act, f.w_down, H, I, h, H, R, false, L);
    add_rows_bf16(x, H, h, H, H, R, L.st);
  } else {
    gemm(act, ld_act, f.w_down, H, I, x, H, R, true, L);
  }
}

void gdn_out_project(const RowView& v, const GdnBlock& g, int R, bool fused, bool add16) {
  const GdnDims gd = g.gd;
  const int H = g.hidden, C = gd.C(), vd = g.v_width();
  const float* z = v.b.proj + C;
  if (fused) {
    gdn_gated_norm_split(v.b.gdn_o, z, v.ld_proj, g.gated.w, g.gated.eps, R, gd, v.L, g.w_out.q8_inputs());
    gemm_presplit(g.w_out, H, vd, v.b.x, H, R, true, v.L);
  } else {
    gdn_gated_norm(v.b.gdn_o, z, v.ld_proj, g.gated.w, g.gated.eps, v.b.act, R, gd, v.L.st);
    if (add16) {  // h (H wide; gdn_o in a pass's buffers) is free past the gated norm
      gemm(v.b.act, vd, g.w_out, H, vd, v.b.h, H, R, false, v.L);
      add_rows_bf16(v.b.x, H, v.b.h, H, H, R, v.L.st);
    } else {
      gemm(v.b.act, vd, g.w_out, H, vd, v.b.x, H, R, !g.body_only, v.L);
    }
  }
}

void head_logits(const HeadBlock& h, const RowView& v, int r0, int n, bool cert, const CertScratch& cs,
                 const KernelOptions& ko, bool x16) {
  const int H = h.hidden, V = h.vocab;
  if (!h.norm.w) CK(cudaMemcpyAsync(v.b.h, v.b.x + size_t(r0) * H, size_t(n) * H * sizeof(float), cudaMemcpyDeviceToDevice, v.L.st));
  else if (x16) rmsnorm(x16_of(v) + size_t(r0) * H, H, h.norm.w, h.norm.offset, h.norm.eps, v.b.h, H, n, H, v.L.st);
  else rmsnorm(v.b.x + size_t(r0) * H, H, h.norm.w, h.norm.offset, h.norm.eps, v.b.h, H, n, H, v.L.st);
  if (h.w8 && n <= 16 && ko.head_fp8) {  // option head_fp8: the copy's scores
    gemm_fp8_skinny(v.b.h, H, h.w8, h.rs8, V, H, v.logits, V, n, false, v.L);
  } else if (h.cert && n <= 16 && cert && !ko.row_gemms) {  // option head_cert
    gemm_fp8_skinny(v.b.h, H, h.w8, h.rs8, V, H, v.logits, V, n, false, v.L);
    head_certify(v.b.h, H, h.w, h.cert, V, H, v.logits, V, n, cs.norms, cs.key, cs.cols, cs.count, v.L);
  } else if (h.w12.hi && n <= 16 && !ko.row_gemms) {
    gemm_w12_skinny(v.b.h, H, h.w12, V, H, v.logits, V, n, false, v.L);
  } else if (h.wb.data) {
    gemm_blocks(v.b.h, H, h.wb, H, v.logits, V, n, false, v.L);
  } else {
    gemm(v.b.h, H, h.w, V, H, v.logits, V, n, false, v.L);
  }
}

// A block drafter's rows (tok) embedded from the target's table, then its norm where it has one: out (tmp: scratch).
void embed_block_rows(const EmbedBlock& E, const int* tok, float* tmp, float* out, int H, int B, cudaStream_t st) {
  if (E.has_norm) {
    embed_tokens(E, tok, tmp, B, st);
    rmsnorm(tmp, H, E.norm.w, E.norm.offset, E.norm.eps, out, H, B, H, st);
  } else {
    embed_tokens(E, tok, out, B, st);  // packed tables stay in host memory
  }
}

namespace {

// One layer of a block drafter (DFlash2's, DSpine's) over the block's B rows in s->dx: attention to its context
// and the block (both ways; causal: each row its own and those before it) and the MLP, the token-conditioned
// convolutions around both where the block has them.
void block_layer(flyx_seq* s, const DFlashBlock& b, const DFlashLayer& l, const KvRef& kvr, int B, bool causal,
                 const Launch& L, bool fused) {
  const int H = b.hidden, nh = b.nh, nkv = b.nkv, hd = b.hd, I = b.inter;
  const int Tp = b.conv_taps, TG = b.conv() ? Tp * (H / b.conv_group) : 0, nqkv = (nh + 2 * nkv) * hd;
  cudaStream_t st = L.st;
  // Token-conditioned convolutions around attention and MLP where the block has them; plain layers add straight back.
  const bool conv = b.conv();
  rmsnorm(s->dx, H, l.norm.w, l.norm.offset, l.norm.eps, s->dh, H, B, H, st);
  if (conv) {
    gemm(s->dh, H, l.attn_proj, 2 * TG, H, s->ddelta, 2 * TG, B, false, L);
    grouped_conv(s->dh, H, s->ddelta, 2 * TG, l.attn_base, H, Tp, b.conv_group, s->dhc, H, B, false, st);
  }
  gemm(conv ? s->dhc : s->dh, H, l.w_qkv, nqkv, H, s->dproj, s->ld_dproj, B, false, L);
  attn_prep(s->dproj, s->ld_dproj, 0, nh * hd, (nh + nkv) * hd, l.q_norm.w, l.k_norm.w, l.q_norm.offset,
            l.q_norm.eps, b.theta, hd, nh, nkv, hd, B, s->P, kPosPosition, false, s->dq, kvr, st);
  const bool split_o = fused && !l.w_o.q8_inputs() && (nh * hd) % 64 == 0;
  attention(s->dq, kvr, nh, nkv, hd, B, causal, causal ? 0 : B, 1.f / std::sqrt(float(hd)), s->P, nullptr, 0,
            split_o ? nullptr : s->dact, s->ld_dact, s->own.part, kRoundSplits, L, 0, b.window);
  float* const o_out = conv ? s->dtmp : s->dx;  // without convolutions the projection adds into the residual
  if (split_o) gemm_presplit(l.w_o, H, nh * hd, o_out, H, B, !conv, L);
  else gemm(s->dact, s->ld_dact, l.w_o, H, nh * hd, o_out, H, B, !conv, L);
  if (conv)
    grouped_conv(s->dtmp, H, s->ddelta + TG, 2 * TG, l.attn_base + size_t(Tp) * H, H, Tp, b.conv_group, s->dx, H,
                 B, true, st);
  rmsnorm(s->dx, H, l.post_norm.w, l.post_norm.offset, l.post_norm.eps, s->dh, H, B, H, st);
  if (conv) {
    gemm(s->dh, H, l.mlp_proj, 2 * TG, H, s->ddelta, 2 * TG, B, false, L);
    grouped_conv(s->dh, H, s->ddelta, 2 * TG, l.mlp_base, H, Tp, b.conv_group, s->dhc, H, B, false, st);
  }
  gemm(conv ? s->dhc : s->dh, H, l.w_gu, 2 * I, H, s->dproj, s->ld_dproj, B, false, L);
  if (fused) {
    swiglu_split(s->dproj, s->ld_dproj, I, B, L, l.w_down.q8_inputs());
    gemm_presplit(l.w_down, H, I, o_out, H, B, !conv, L);
  } else {
    swiglu(s->dproj, s->ld_dproj, I, s->dact, s->ld_dact, B, st);
    gemm(s->dact, s->ld_dact, l.w_down, H, I, o_out, H, B, !conv, L);
  }
  if (conv)
    grouped_conv(s->dtmp, H, s->ddelta + TG, 2 * TG, l.mlp_base + size_t(Tp) * H, H, Tp, b.conv_group, s->dx, H,
                 B, true, st);
}

}  // namespace

// DSpine (backend_abi.h flyx_dspine_block, ops/dspine/ref.h): the rows the round's drafts need (no row reads a row
// after it: a block of fewer rows drafts the same first drafts), each layer a Qwen3 layer attending causally (within
// the window) then the bridge's write; the head's candidates for the rows after the anchor; and the walk, which redoes
// the last write with the drafted token's code: every sender's message at once (the anchor's and every candidate's but
// the last row's), its scores against the next row's candidates, the chain through the picks.
void dspine_block(flyx_seq* s, const DFlashBlock& b, int width, const Launch& L, bool fused) {
  const auto* m = s->m;
  const auto& E = block<EmbedBlock>(m, b.embed);
  const auto& head = block<HeadBlock>(m, b.head);
  const KvBuf& kv = s->kv[size_t(b.kv)];
  const DSpineBridge& br = b.dspine;
  auto& d = s->dsp;
  const int B = std::min(b.block, width + 1), nr = B - 1;
  const int H = b.hidden, C = br.code, R = br.rank, K = b.sel_topk, senders = 1 + (nr - 1) * K;
  cudaStream_t st = L.st;
  build_rows(s->P, nullptr, b.mask_token, s->dtok, B, st);
  embed_block_rows(E, s->dtok, s->dh, s->dx, H, B, st);
  for (int j = 0; j < b.layers(); ++j) {
    const auto& l = block<DFlashLayer>(m, b.first_layer + j);
    block_layer(s, b, l, kv.draft_ref(j), B, true, L, fused);  // causal within the block
    // The bridge's write; the last one's rows before it, their rms and gate projections are what the walk redoes.
    const bool last = j == b.layers() - 1;
    dspine_normalize(s->dx, H, br.eps, B, s->dh, d.rms, st);
    gemm(s->dh, H, l.read, C, H, d.codes, C, B, false, L);
    dspine_codes(d.codes, C, br.codebook, s->P, br.eps, B, st);
    gemm(d.codes, C, br.packed, 3 * R, C, d.gates, 3 * R, B, false, L);
    dspine_message(d.gates, R, br.gate_bias, B, d.msg, last ? d.g : nullptr, st);
    gemm(d.msg, R, l.write, H, R, s->dtmp, H, B, false, L);
    dspine_apply(s->dx, s->dtmp, d.rms, H, B, last ? d.before : nullptr, st);
  }
  rmsnorm(s->dx + H, H, b.norm.w, b.norm.offset, b.norm.eps, s->dh, H, nr, H, st);
  draft_head_pick(head, s->dh, nr, K, s->cand, s->unary, s, L, s->draft_prob, s->draft_margin);
  // The walk's GEMMs, up to (block - 1) topk rows: a round's kernel, every 64 rows at once in one launch, each row's
  // inputs as three bf16 parts (float32's precision: the scores' differences are a message's against another's; bf16
  // inputs moved them by 5e-4 of a score, more than a near tie).
  Launch walk = L;
  walk.round_rows = true;
  walk.draft = false;
  if (head.wb.data) embed_blocks(s->cand, head.wb, H, d.heads, nr * K, st);  // a packed head's rows as float32
  dspine_heads(s->cand, head.wb.data ? nullptr : head.w, b.norm.w, d.before, H, K, nr * K, d.heads, d.base, st);
  gemm(d.heads, H, br.write_t, R, H, d.tg, R, nr * K, false, walk);
  dspine_senders(s->P, s->cand, br.codebook, C, senders, d.codes, st);
  gemm(d.codes, C, br.pv, 2 * R, C, d.gates, 2 * R, senders, false, walk);
  dspine_scores(d.g, d.gates, br.gate_bias, d.tg, d.base, d.rms, R, K, senders, d.pick, st);
  dspine_walk(s->P, s->cand, d.pick, K, std::min(width, nr), s->oracle, s->drafts, st);
}

void draft_block(flyx_seq* s, const DFlashBlock& b, int width, const Launch& L, bool tree, bool fused) {
  if (b.dspine) {
    if (tree) throw std::logic_error("cuda: DSpine drafts chains, not trees");  // lower refuses its tree rounds
    return dspine_block(s, b, width, L, fused);
  }
  const auto* m = s->m;
  const auto& E = block<EmbedBlock>(m, b.embed);
  const auto& head = block<HeadBlock>(m, b.head);
  const KvBuf& kv = s->kv[size_t(b.kv)];
  // A chain wider than the block drafts from a wider block: width + 1 mask rows in one pass (basalt's wide block;
  // the drafter attends within its block both ways, at any count); from the anchor's row (draft_from 0) every row
  // drafts, so width rows.  Trees keep the block they were made for.
  const int off = b.draft_from;
  const int B = tree ? b.block : std::min(FLYX_MAX_ROWS, std::max(b.block, width + off));
  const int H = b.hidden;
  cudaStream_t st = L.st;
  build_rows(s->P, nullptr, b.mask_token, s->dtok, B, st);
  embed_block_rows(E, s->dtok, s->dh, s->dx, H, B, st);
  for (int j = 0; j < b.layers(); ++j)
    block_layer(s, b, block<DFlashLayer>(m, b.first_layer + j), kv.draft_ref(j), B, false, L, fused);
  // Rows draft_from.. predict tokens (width of them): unary top-k from the target head, then the
  // selector walk (or tree).  The block's layers above need all B rows (attention inside
  // the block is not causal), the head only the rows whose drafts are used.
  const int nr = std::min(width, B - off);
  rmsnorm(s->dx + size_t(off) * H, H, b.norm.w, b.norm.offset, b.norm.eps, s->dh, H, nr, H, st);
  draft_head_pick(head, s->dh, nr, b.sel_topk, s->cand, s->unary, s, L, s->draft_prob, s->draft_margin);
  // no sel_hid: the selector's score with a unit gate; no selector (rank 0): the head's logits alone
  const float* hid = b.unit_hid ? b.unit_hid : s->hid;
  if (b.selector() && !b.unit_hid) gemm(s->dh, H, b.sel_hid, b.sel_rank, H, s->hid, b.sel_rank, nr, false, L);
  if (tree)
    selector_tree(s->P, s->cand, s->unary, b.sel_topk, hid, b.sel_rank, b.sel_pred, b.sel_succ, width, nr,
                  s->oracle, s->drafts, s->tree, st, b.softcap, b.softcap_mode, s->G);
  else
    selector_walk(s->P, s->cand, s->unary, b.sel_topk, hid, b.sel_rank, b.sel_pred, b.sel_succ, width, s->oracle,
                  s->drafts, st, b.softcap, b.softcap_mode, s->G);
}

// One MTP draft step's attention for one sequence, the step's row at P->position + pos_offset: its key and value
// written, then attention over what the sequence keeps of the cache (its low-bit copy, its gathered window, or all
// of it).  out null: the result stays in the split scratch for a fused output projection (gemm_presplit).
void mtp_draft_attention(flyx_seq* s, const AttnBlock& a, const KvBuf& kv, int pos_offset, const float* proj,
                         int ld_proj, float* q, float* out, float* part, const Launch& L) {
  const int nh = a.nh, nkv = a.nkv, hd = a.hd;
  const float* gate = a.gate ? proj + a.nh * a.hd : nullptr;
  attn_prep(proj, ld_proj, 0, a.q_width(), a.q_width() + nkv * hd, a.q_norm.w, a.k_norm.w, a.q_norm.offset,
            a.q_norm.eps, a.theta, a.rot, nh, nkv, hd, 1, s->P, kPosPosition, false, q, kv.ref(), L.st, pos_offset,
            nullptr, 0, CanvasKeys{}, a.mrope());
  s->stage_before(kv, pos_offset, 1, L.st);
  if (kv.kq) {  // its copy of every position, the last ones exact
    params_at(s->P, s->P_window, pos_offset, L.st);
    attention_strata(q, kv.ref(), s->mtp_strata, nh, nkv, hd, 1, 1.f / std::sqrt(float(hd)), s->P_window, gate,
                     ld_proj, out, nh * hd, part, kRoundSplits, L, 0, nullptr, s->P);
  } else if (kv.window > 0) {  // its last positions, gathered
    kv_window(kv.ref(), s->mtp_window, nkv, hd, kv.window, s->P, s->P_window, pos_offset, L.st);
    // Not a programmatic dependent: the attention kernels read a row's position before they wait for the kernel
    // before, and that kernel (kv_window) wrote it.
    KernelOptions in_order = *L.opt;
    in_order.pdl = false;
    Launch after = L;
    after.opt = &in_order;
    attention(q, s->mtp_window, nh, nkv, hd, 1, true, 0, 1.f / std::sqrt(float(hd)), s->P_window, gate, ld_proj,
              out, nh * hd, part, kRoundSplits, after, pos_offset);
  } else {
    attention(q, kv.staged(), nh, nkv, hd, 1, true, 0, 1.f / std::sqrt(float(hd)), s->P, gate, ld_proj, out, nh * hd,
              part, kRoundSplits, L, pos_offset);
  }
}

// MTP: one layer from (target hidden, next token) pairs, see python/reference/flyx_ref/mtp.py.
// Draft step i runs at position P->position - 1 + i.
void mtp_draft(flyx_seq* s, const MtpBlock& b, int width, const Launch& L, bool fused, int start) {
  const auto* m = s->m;
  const auto& E = block<EmbedBlock>(m, b.embed);
  const auto& head = block<HeadBlock>(m, b.head);
  const auto& a = block<AttnBlock>(m, b.attn);
  const auto& f = block<MlpBlock>(m, b.mlp);
  const KvBuf& kv = s->kv[size_t(a.kv)];
  const RowBufs& own = s->own;
  const int H = a.hidden, nh = a.nh, hd = a.hd;
  const bool split_o = fused && !a.w_out.q8_inputs() && (nh * hd) % 64 == 0;
  cudaStream_t st = L.st;
  if (start == 0) build_rows(s->P, nullptr, 0, s->mtok, 1, st);  // mtok[0] = anchor
  for (int i = start; i < width; ++i) {
    if (E.blk.data) embed_blocks(s->mtok + i, E.blk, H, own.h, 1, st);  // own.h is free until the layer's norm
    mtp_gather(s->P, kPosPosition, false, i == 0 ? nullptr : s->mg, s->ring, nullptr, 0, s->mtok + i, E.table, own.h,
               b.norm_emb.w, b.norm_hidden.w, b.norm_emb.offset, b.norm_emb.eps, H, 1, own.mcat, st);
    gemm(own.mcat, 2 * H, b.fc, H, 2 * H, own.mx, H, 1, false, L);
    norm_project(own.mx, own.h, own.proj, s->ld_proj, 1, a.norm, a.w_in, a.n_in(), H, L, fused);
    mtp_draft_attention(s, a, kv, i - 1, own.proj, s->ld_proj, own.qbuf, split_o ? nullptr : own.act, own.part, L);
    if (split_o) gemm_presplit(a.w_out, H, nh * hd, own.mx, H, 1, true, L);
    else gemm(own.act, nh * hd, a.w_out, H, nh * hd, own.mx, H, 1, true, L);
    mlp_body(own.mx, f, 1, own.h, own.proj, s->ld_proj, own.act, s->ld_act, L, fused);
    rmsnorm(own.mx, H, b.norm.w, b.norm.offset, b.norm.eps, s->mg, H, 1, H, st);
    // the head's best kMtpCands (at most its vocabulary), then the draft: the best, or sampling the one coupled to the
    // target's noise
    const int K = std::min(ops::kMtpCands, head.vocab);
    draft_head_pick(head, s->mg, 1, K, s->mcand, s->mval, s, L, s->draft_prob + i, s->draft_margin + i);
    mtp_choose(s->P, s->G, s->reply.count, s->mcand, s->mval, K, head.vocab, i, s->mtok, st);
    mtp_oracle(s->P, s->oracle, i, s->mtok, st);
  }
  copy_ints(s->mtok + 1 + start, s->drafts + start, width - start, st);
}

// EAGLE-4: one layer a draft step from (h, the token before), see python/reference/flyx_ref/eagle4.py.  Draft step i
// runs at position P->position - 1 + i: h is the ring's at P->position - 1 for the first step, the step before's
// normed output s->mg after it.
void eagle4_draft(flyx_seq* s, const Eagle4Block& b, int width, const Launch& L, bool fused, int start) {
  const auto* m = s->m;
  const auto& E = block<EmbedBlock>(m, b.embed);
  const AttnBlock& a = b.attn;
  const KvBuf& kv = s->kv[size_t(a.kv)];
  const RowBufs& own = s->own;
  const int H = b.hidden, nh = a.nh, nkv = a.nkv, hd = a.hd, V = b.vocab;
  const int ld = std::max(V, kDraftPickFloats);
  const int K = std::min(ops::kMtpCands, V);
  cudaStream_t st = L.st;
  if (start == 0) build_rows(s->P, nullptr, 0, s->mtok, 1, st);  // mtok[0] = anchor
  for (int i = start; i < width; ++i) {
    if (E.blk.data) embed_blocks(s->mtok + i, E.blk, H, own.h, 1, st);  // own.h is free until the step's norm
    // [in_norm(embed[token]) | hidden_norm(h)], and h itself: the residual the attention and MLP add to
    mtp_gather(s->P, kPosPosition, false, i == 0 ? nullptr : s->mg, s->ring, nullptr, 0, s->mtok + i, E.table, own.h,
               b.in_norm.w, b.hidden_norm.w, b.in_norm.offset, b.in_norm.eps, H, 1, own.mcat, st);
    pair_hidden(s->P, kPosPosition, i == 0 ? nullptr : s->mg, s->ring, H, own.mx, st);
    gemm(own.mcat, 2 * H, a.w_in, a.n_in(), 2 * H, own.proj, s->ld_proj, 1, false, L);
    attn_prep(own.proj, s->ld_proj, 0, a.q_width(), a.q_width() + nkv * hd, nullptr, nullptr, 0.f, 0.f, a.theta,
              a.rot, nh, nkv, hd, 1, s->P, kPosPosition, false, own.qbuf, kv.ref(), st, i - 1, nullptr, 0,
              CanvasKeys{}, a.mrope());
    s->stage_before(kv, i - 1, 1, st);
    attention(own.qbuf, kv.staged(), nh, nkv, hd, 1, true, 0, 1.f / std::sqrt(float(hd)), s->P, nullptr, 0, own.act,
              nh * hd, own.part, kRoundSplits, L, i - 1, b.window);
    gemm(own.act, nh * hd, a.w_out, H, nh * hd, own.mx, H, 1, true, L);
    mlp_body(own.mx, b.mlp, 1, own.h, own.proj, s->ld_proj, own.act, s->ld_act, L, fused);
    rmsnorm(own.mx, H, b.norm.w, b.norm.offset, b.norm.eps, s->mg, H, 1, H, st);
    // The candidate head: its low-rank scores over the vocabulary, their n_cands best scored exactly by the head, the
    // best kMtpCands of those; then the draft as MTP's (mtp_choose).
    gemm(s->mg, H, b.cand_down, b.cand_rank, H, s->ez, b.cand_rank, 1, false, L);
    gemm(s->ez, b.cand_rank, b.cand_up, V, b.cand_rank, s->logits, ld, 1, false, L);
    draft_pick_rough(s->mg, H, b.head, V, H, 1, b.n_cands, K, s->mcand, s->mval, s->logits, L,
                     s->draft_prob + i, s->draft_margin + i);
    mtp_choose(s->P, s->G, s->reply.count, s->mcand, s->mval, K, V, i, s->mtok, st);
    mtp_oracle(s->P, s->oracle, i, s->mtok, st);
  }
  copy_ints(s->mtok + 1 + start, s->drafts + start, width - start, st);
}

int eagle4_tree_steps(const flyx_bmodel* m, int width) {
  const int beam = m->opt.eagle4_beam;
  // by default two steps for every three drafts (15: 10 steps, deeper than its trunk of kEagleTrunk; Qwen3.8-27B's
  // EAGLE-4 at beam 4 verified as many tokens a round as at 12 steps, the steps cheaper)
  const int steps = m->opt.eagle4_steps > 0 ? m->opt.eagle4_steps : 2 * width / 3;
  // the nodes: the root and beam a step; enough of them for the width
  return std::max(std::min(steps, (kEagleSlots - 1) / beam), (width + beam - 1) / beam);
}

// One step of the layer on `rows` rows (tokens tok, hidden inputs h [rows][H]) at depth d: their K/V to the slots of
// their nodes f0 .. f0 + rows - 1 (P->position - 1 + node), their attention by the tree (nullptr: the root's row, a
// chain's), their normed outputs to g [rows][H], their `beam` best candidates to cand / val, the best's probability to
// prob.
static void eagle4_tree_step(flyx_seq* s, const Eagle4Block& b, const int* tok, const float* h, int rows, int f0,
                             const TreeRows* prep, const TreeRows* att, float* g, int kc, const Launch& L,
                             bool fused) {
  const auto* m = s->m;
  const auto& E = block<EmbedBlock>(m, b.embed);
  const AttnBlock& a = b.attn;
  const KvBuf& kv = s->kv[size_t(a.kv)];
  const RowBufs& own = s->own;
  const int H = b.hidden, nh = a.nh, nkv = a.nkv, hd = a.hd, V = b.vocab;
  const int ld = std::max(V, kDraftPickFloats);
  cudaStream_t st = L.st;
  if (E.blk.data) embed_blocks(tok, E.blk, H, own.h, rows, st);
  mtp_gather(s->P, kPosPosition, false, h, s->ring, h + H, H, tok, E.table, own.h, b.in_norm.w, b.hidden_norm.w,
             b.in_norm.offset, b.in_norm.eps, H, rows, own.mcat, st);
  CK(cudaMemcpyAsync(own.mx, h, size_t(rows) * H * sizeof(float), cudaMemcpyDeviceToDevice, st));
  gemm(own.mcat, 2 * H, a.w_in, a.n_in(), 2 * H, own.proj, s->ld_proj, rows, false, L);
  attn_prep(own.proj, s->ld_proj, 0, a.q_width(), a.q_width() + nkv * hd, nullptr, nullptr, 0.f, 0.f, a.theta, a.rot,
            nh, nkv, hd, rows, s->P, kPosPosition, false, own.qbuf, kv.ref(), st, f0 - 1, prep, 0, CanvasKeys{},
            a.mrope());
  attention(own.qbuf, kv.ref(), nh, nkv, hd, rows, true, 0, 1.f / std::sqrt(float(hd)), s->P, nullptr, 0, own.act,
            nh * hd, own.part, kRoundSplits, L, att ? -1 : f0 - 1, b.window, att);
  gemm(own.act, nh * hd, a.w_out, H, nh * hd, own.mx, H, rows, true, L);
  mlp_body(own.mx, b.mlp, rows, own.h, own.proj, s->ld_proj, own.act, s->ld_act, L, fused);
  rmsnorm(own.mx, H, b.norm.w, b.norm.offset, b.norm.eps, g, H, rows, H, st);
  gemm(g, H, b.cand_down, b.cand_rank, H, s->ez, b.cand_rank, rows, false, L);
  gemm(s->ez, b.cand_rank, b.cand_up, V, b.cand_rank, s->logits, ld, rows, false, L);
  draft_pick_rough(g, H, b.head, V, H, rows, b.n_cands, kc, s->ecand, s->eval, s->logits, L, s->eprob);
}

void eagle4_draft_tree(flyx_seq* s, const Eagle4Block& b, int width, const Launch& L, bool fused) {
  const auto* m = s->m;
  const int H = b.hidden, V = b.vocab, beam = m->opt.eagle4_beam, steps = eagle4_tree_steps(m, width);
  const int kc = eagle4_tree_cands(beam, V);  // a node's beam children, and its chain's candidates where a trunk's
  cudaStream_t st = L.st;
  build_rows(s->P, nullptr, 0, s->mtok, 1, st);  // mtok[0] = anchor
  // the root: (the ring's h at P->position - 1, the anchor) at P->position - 1, its K/V at that slot (node 0's)
  pair_hidden(s->P, kPosPosition, nullptr, s->ring, H, s->ehbuf, st);
  eagle4_tree_step(s, b, s->mtok, s->ehbuf, 1, 0, nullptr, nullptr, s->egnode, kc, L, fused);
  eagle_tree_start(s->etree, s->P, s->G, s->reply.count, s->oracle, s->mtok, s->ecand, s->eval, s->eprob, kc, beam, V,
                   s->egnode, s->ehbuf, H, st);
  for (int d = 1; d < steps; ++d) {  // the frontier: nodes 1 + beam (d - 1) .. at depth d
    const int f0 = 1 + beam * (d - 1);
    eagle4_tree_step(s, b, s->etree->tokbuf, s->ehbuf, beam, f0, &s->etree->prep, &s->etree->att,
                     s->egnode + size_t(f0) * H, kc, L, fused);
    eagle_tree_grow(s->etree, s->P, s->G, s->reply.count, s->oracle, s->ecand, s->eval, s->eprob, kc, beam, V,
                    s->egnode, s->ehbuf, H, st);
  }
  eagle_tree_finish(s->etree, width, kEagleTrunk, s->drafts, s->tree, st);
}

// After a pass: each row's h = fc([fc_norm_i(tap_i)]) into the ring, and the K/V of the positions base-1 ..
// base+rows-2 from (h at j, token at j+1), as mtp_context.  In a round the entries past the accepted rows hold
// rejected rows; they are rewritten (by the next draft steps or context pass) before anything attends to them.
void eagle4_context(const Pass& ps, const Eagle4Block& b, bool) {
  const RowView& rv = ps.v;
  auto* s = ps.s;
  const auto* m = s->m;
  const auto& E = block<EmbedBlock>(m, b.embed);
  const AttnBlock& a = b.attn;
  const KvBuf& kv = s->kv[size_t(a.kv)];
  const int H = b.hidden, Ht = b.target_hidden, R = ps.rows, nh = a.nh, nkv = a.nkv, hd = a.hd;
  const int sel = ps.round ? kPosStart : kPosPosition;
  cudaStream_t st = rv.L.st;
  const Launch& L = rv.L;
  float* taps = rv.b.feats + b.feats_offset;
  for (int t = 0; t < b.n_taps; ++t)  // each tap normed in place: nothing reads the taps after this pass's context
    rmsnorm(taps + size_t(t) * Ht, rv.ld_feats, b.fc_norm[t].w, b.fc_norm[t].offset, b.fc_norm[t].eps,
            taps + size_t(t) * Ht, rv.ld_feats, R, Ht, st);
  gemm(taps, rv.ld_feats, b.fc, H, b.n_taps * Ht, rv.b.mhn, H, R, false, L);
  if (E.blk.data) {
    embed_blocks(rv.b.tokens, E.blk, H, rv.b.h, R, st);  // rv.b.h is free here
    if (ps.n_embeds > 0) embed_scatter(rv.b.h, nullptr, H, ps.embeds, ps.embed_rows, ps.n_embeds, st);
  } else if (ps.n_embeds > 0) {
    embed(rv.b.tokens, E.table, H, rv.b.h, R, st);
    embed_scatter(rv.b.h, nullptr, H, ps.embeds, ps.embed_rows, ps.n_embeds, st);
  }
  const bf16* table = (E.blk.data || ps.n_embeds > 0) ? nullptr : E.table;
  mtp_gather(s->P, sel, ps.round, nullptr, s->ring, rv.b.mhn, H, rv.b.tokens, table, rv.b.h, b.in_norm.w,
             b.hidden_norm.w, b.in_norm.offset, b.in_norm.eps, H, R, rv.b.mcat, st);
  if (!ps.round && s->pass_snap > 0 && s->ring_snap) {
    // Capture h[pass_snap - 1] before hring_put overwrites the kHring-slot ring when end - pass_snap >= kHring.
    const int snap_row = s->pass_snap - s->position;
    const float* src = snap_row == 0 ? s->ring + size_t((s->pass_snap - 1) % kHring) * size_t(H)
                                     : rv.b.mhn + size_t(snap_row - 1) * size_t(H);
    CK(cudaMemcpyAsync(s->ring_snap, src, size_t(H) * sizeof(float), cudaMemcpyDeviceToDevice, st));
  }
  hring_put(s->P, sel, ps.round, rv.b.mhn, H, R, s->ring, H, st);
  // q|k|v into act: in a prefill pass mcat and mhn sit in proj's slot (alloc_rows)
  const int n_in = a.n_in();
  gemm(rv.b.mcat, 2 * H, a.w_in, n_in, 2 * H, rv.b.act, n_in, R, false, L);
  attn_prep(rv.b.act, n_in, 0, a.q_width(), a.q_width() + nkv * hd, nullptr, nullptr, 0.f, 0.f, a.theta, a.rot, nh,
            nkv, hd, R, s->P, sel, ps.round, rv.b.qbuf, kv.ref(), st, -1, nullptr, 0, CanvasKeys{}, a.mrope());
}

// After a pass: keep the rows' normalised target hidden and write the MTP K/V
// entries base-1 .. base+rows-2 from (hidden at j, token at j+1).  In a round
// the entries past the accepted rows hold rejected rows; they are rewritten
// (by the next draft steps or context pass) before anything attends to them.
void mtp_context(const Pass& ps, const MtpBlock& b, bool fused) {
  const RowView& rv = ps.v;
  auto* s = ps.s;
  const auto* m = s->m;
  const auto& E = block<EmbedBlock>(m, b.embed);
  const auto& head = block<HeadBlock>(m, b.head);
  const auto& a = block<AttnBlock>(m, b.attn);
  const KvBuf& kv = s->kv[size_t(a.kv)];
  const int H = a.hidden, R = ps.rows, nh = a.nh, nkv = a.nkv, hd = a.hd;
  const int sel = ps.round ? kPosStart : kPosPosition;
  cudaStream_t st = rv.L.st;
  const Launch& L = rv.L;
  if (residual16(ps)) rmsnorm(x16_of(rv), H, head.norm.w, head.norm.offset, head.norm.eps, rv.b.mhn, H, R, H, st);
  else rmsnorm(rv.b.x, H, head.norm.w, head.norm.offset, head.norm.eps, rv.b.mhn, H, R, H, st);
  if (E.blk.data) {
    embed_blocks(rv.b.tokens, E.blk, H, rv.b.h, R, st);  // rv.b.h is free until the layer's norm
    if (ps.n_embeds > 0) embed_scatter(rv.b.h, nullptr, H, ps.embeds, ps.embed_rows, ps.n_embeds, st);
  } else if (ps.n_embeds > 0) {
    embed(rv.b.tokens, E.table, H, rv.b.h, R, st);
    embed_scatter(rv.b.h, nullptr, H, ps.embeds, ps.embed_rows, ps.n_embeds, st);
  }
  const bf16* mtp_table = (E.blk.data || ps.n_embeds > 0) ? nullptr : E.table;
  mtp_gather(s->P, sel, ps.round, nullptr, s->ring, rv.b.mhn, H, rv.b.tokens, mtp_table, rv.b.h, b.norm_emb.w,
             b.norm_hidden.w, b.norm_emb.offset, b.norm_emb.eps, H, R, rv.b.mcat, st);
  if (!ps.round && s->pass_snap > 0 && s->ring_snap) {
    // Capture hidden[pass_snap - 1] before hring_put overwrites the kHring-slot ring when end - pass_snap >= kHring.
    const int snap_row = s->pass_snap - s->position;
    const float* src = snap_row == 0 ? s->ring + size_t((s->pass_snap - 1) % kHring) * size_t(H)
                                     : rv.b.mhn + size_t(snap_row - 1) * size_t(H);
    CK(cudaMemcpyAsync(s->ring_snap, src, size_t(H) * sizeof(float), cudaMemcpyDeviceToDevice, st));
  }
  hring_put(s->P, sel, ps.round, rv.b.mhn, H, R, s->ring, H, st);
  gemm(rv.b.mcat, 2 * H, b.fc, H, 2 * H, rv.b.mx, H, R, false, L);
  norm_project(rv.b.mx, rv.b.h, rv.b.proj, rv.ld_proj, R, a.norm, a.w_in, a.n_in(), H, L, fused);
  attn_prep(rv.b.proj, rv.ld_proj, 0, a.q_width(), a.q_width() + nkv * hd, a.q_norm.w, a.k_norm.w, a.q_norm.offset,
            a.q_norm.eps, a.theta, a.rot, nh, nkv, hd, R, s->P, sel, ps.round, rv.b.qbuf, kv.ref(), st, -1, nullptr, 0,
            CanvasKeys{}, a.mrope(ps.mtp_pos_3d));
}

// fused: the producers write the next GEMM's input already rounded (split for 1..16 rows,
// bf16 for more), so the GEMM reads it as is.
void gdn_layer(const Pass& ps, const GdnBlock& g, bool fused) {
  // Adapters (ABI 50) read float32 rows: the normed input (the fused producer writes them beside its staged ones,
  // norm_project_adapted) and the gated norm's output (gdn_round writes them beside its staged ones in a round, else
  // the generic producer); their products go to the in-projection's rows before the convolution and the gates read
  // them, then to the residual after the out projection.
  if (g.body_only || g.gd.sigmoid_output) fused = false;
  const bool fused_mixer = fused;
  if (g.lora) fused = false;  // outside gdn_round: the generic producers
  const RowView& rv = ps.v;
  auto* s = ps.s;
  GdnBuf& sb = s->gdn[size_t(g.state)];
  const GdnDims gd = g.gd;
  const int H = g.hidden, R = ps.rows, C = gd.C(), vd = g.v_width();
  cudaStream_t st = rv.L.st;
  const Launch& L = rv.L;
  // A prefill pass with NVFP4 inputs: the in-projection's rows in bf16 (as vLLM and SGLang hold them), read by the
  // convolution, the gates and the gated norm; the proj buffer holds them (half its bytes).
  const bool x16 = residual16(ps);
  if (x16 && !fused) throw std::logic_error("cuda: a bf16 residual's pass runs the fused producers");
  if (g.body_only) {
    gemm(rv.b.x, H, g.w_in, g.n_in(), H, rv.b.proj, rv.ld_proj, R, false, L);
  } else if (!ps.round && fused && R > 16 && !g.w_in.packed() && g.w_in.q) {
    if (x16) rmsnorm_split(x16_of(rv), H, g.norm.w, g.norm.offset, g.norm.eps, R, H, L, g.w_in.q8_inputs());
    else rmsnorm_split(rv.b.x, H, g.norm.w, g.norm.offset, g.norm.eps, R, H, L, g.w_in.q8_inputs());
    bf16* p16 = reinterpret_cast<bf16*>(rv.b.proj);
    if (gemm_staged_bf16(g.w_in.q, g.w_in.s, g.w_in.rs, g.n_in(), H, p16, rv.ld_proj, R, L)) {
      const GdnBufs bf{rv.b.pre.base, 0, rv.b.pre.rows_cap, gd};
      const int snap_row = s->pass_snap >= 0 ? s->pass_snap - s->position : -1;
      const int snap_sel = snap_row >= 0 ? kSlotExplicit + s->snap_slot : -1;
      gdn_conv(p16, rv.ld_proj, g.conv_w, sb.win, sb.w_stride, kSlotCur, kSlotCur, bf, kSlotCur, s->P, R, st, snap_sel,
               snap_row);
      gdn_prep(bf, kSlotCur, p16 + C + vd, p16 + C + vd + gd.nv, rv.ld_proj, g.A_log, g.dt_bias, s->P, R, st);
      if (!gdn_chunk(sb.S, sb.s_stride, kSlotCur, bf, s->P, s->position, rv.b.gdn_o, R, st, snap_sel, snap_row))
        gdn_recur(sb.S, sb.s_stride, kSlotCur, kSlotCur, bf, kSlotCur, s->P, rv.b.gdn_o, R, nullptr, false, st);
      gdn_gated_norm_split(rv.b.gdn_o, p16 + C, rv.ld_proj, g.gated.w, g.gated.eps, R, gd, L, g.w_out.q8_inputs());
      if (!x16) gemm_presplit(g.w_out, H, vd, rv.b.x, H, R, true, L);
      else if (!gemm_staged_bf16(g.w_out.q, g.w_out.s, g.w_out.rs, H, vd, x16_of(rv), H, R, L, true))
        throw std::logic_error("cuda: a bf16 residual's out projection is a W4A4 GEMM");
      return;
    }
    if (x16) throw std::logic_error("cuda: a bf16 residual's in-projection is a W4A4 GEMM");
    gemm_presplit(g.w_in, g.n_in(), H, rv.b.proj, rv.ld_proj, R, false, L);
  } else if (g.lora) {  // the producer's float32 rows (beside a fused one's staged ones) for the adapters
    norm_project_adapted(rv, R, g.norm, g.w_in, g.n_in(), H, fused_mixer, g.lora, 4, s->m->need.lora);
    const int out[4] = {C, vd, gd.nv, gd.nv};
    lora_in_end(rv, g.lora, 4, out, rv.b.h, H, H, R, s->m->need.lora, L);
  } else {
    norm_and_project(rv, R, g.norm, g.w_in, g.n_in(), H, fused);
    if (!ps.round && R > 16 && !g.w_in.packed() && prefill_rows_bf16(L, g.w_in.q, H))  // the fused path's bf16 rows
      round_rows_bf16(rv.b.proj, rv.ld_proj, g.n_in(), R, st);
  }
  const float* b = rv.b.proj + C + vd;
  const float* a = b + gd.nv;
  // Plain decoding (no drafts) always keeps its one row: the pass writes the
  // next window and state instead of the commit folding them (same bits, one
  // pass fewer).
  const bool direct = ps.round && ps.p->width == 0;
  const int next = direct ? kSlotNext : -1;
  const TreeRows* tree = tree_of(ps);
  if (fused_mixer && ps.round && !tree &&
      gdn_round(rv.b.proj, rv.ld_proj, g.conv_w, sb.win, sb.w_stride, sb.S, sb.s_stride, next, g.A_log, g.dt_bias,
                g.gated.w, g.gated.eps, sb.replay, s->P, R, nullptr, L, g.w_out.q8_inputs(), 0, 0,
                g.lora ? rv.b.act : nullptr)) {
    if (g.lora) lora_begin(rv, g.lora.a_out, g.lora.rank, rv.b.act, vd, vd, R, s->m->need.lora, L);
    gemm_presplit(g.w_out, H, vd, rv.b.x, H, R, true, L);
    if (g.lora) lora_out_end(rv, g.lora, 4, rv.b.act, vd, vd, H, R, s->m->need.lora, L);  // the gated norm's rows
    return;
  }
  if (ps.round) {
    const GdnBufs& bf = sb.replay;
    gdn_conv(rv.b.proj, rv.ld_proj, g.conv_w, sb.win, sb.w_stride, kSlotCur, next, kSkipIfStopped, bf, kSlotCur, s->P,
             R, st, -1, -1, tree);
    gdn_prep(bf, kSlotCur, b, a, rv.ld_proj, g.A_log, g.dt_bias, s->P, R, st);
    if (!tree) {
      gdn_recur(sb.S, sb.s_stride, kSlotCur, next, bf, kSlotCur, s->P, rv.b.gdn_o, R, nullptr,
                direct ? kSkipIfStopped : kSkipNever, st);
    } else if (!gdn_recur_tree(sb.S, sb.s_stride, kSlotCur, bf, kSlotCur, s->P, rv.b.gdn_o, R, tree, st)) {
      throw std::runtime_error("tree rounds: a gated delta net of key dim " + std::to_string(gd.dk) +
                               " (16 to 128 with an aligned state are taken)");
    }
  } else {
    const GdnBufs bf{rv.b.pre.base, 0, rv.b.pre.rows_cap, gd};  // the shared prefill scratch, this block's dims
    // The pass's chunk boundary for checkpoint() (runner.cpp): window and state there to the snapshot slot.
    const int snap_row = s->pass_snap >= 0 ? s->pass_snap - s->position : -1;
    const int snap_sel = snap_row >= 0 ? kSlotExplicit + s->snap_slot : -1;
    gdn_conv(rv.b.proj, rv.ld_proj, g.conv_w, sb.win, sb.w_stride, kSlotCur, kSlotCur, kSkipNever, bf, kSlotCur, s->P,
             R, st, snap_sel, snap_row);
    gdn_prep(bf, kSlotCur, b, a, rv.ld_proj, g.A_log, g.dt_bias, s->P, R, st);
    // Chunks on tensor cores where the shapes allow (numerics.md: P, K); s->position is the pass's first row.
    if (!gdn_chunk(sb.S, sb.s_stride, kSlotCur, bf, s->P, s->position, rv.b.gdn_o, R, st, snap_sel, snap_row))
      gdn_recur(sb.S, sb.s_stride, kSlotCur, kSlotCur, bf, kSlotCur, s->P, rv.b.gdn_o, R, nullptr, false, st);
  }
  gdn_out_project(rv, g, R, fused, residual16_pass(ps));
  if (g.lora) lora_out(rv, g.lora, 4, rv.b.act, vd, vd, H, R, s->m->need.lora, L);  // the gated norm's rows
}

// A prefill's attention over a cache whose keys go past its stagings: the device part's whole 64s
// where they are, then a window of the stagings' positions at a time, each copied to a staging (ahead, beside the
// layers and the windows before it), the rows' softmax state carried between them.  The bits of one pass.
namespace {

void attn_windows(const Pass& ps, const AttnBlock& a, const KvBuf& kb, int windows, float* out, int ld_out) {
  const RowView& rv = ps.v;
  auto* s = ps.s;
  const int R = ps.rows, nh = a.nh, nkv = a.nkv, hd = a.hd;
  cudaStream_t st = rv.L.st;
  const float scale = 1.f / std::sqrt(float(hd));
  const float* gate = a.gate ? rv.b.proj + nh * hd : nullptr;
  const int from = s->window_from(kb), len = s->window_len(kb), end = s->position + R;
  // A pass of a round's rows or fewer (pageable caches only: windows() gives the others none) takes a round's
  // kernels, as it does without windows; attention=split has no windows at all (alloc_state).
  if (R <= FLYX_MAX_ROWS) {
    if (!rv.b.split_state) throw std::logic_error("cuda: prefill split window state buffer missing");
    // Two buffers of the rows' state, read and written in turn: a launch reads what the one before left
    // (SplitWindow::carry) while another of its blocks writes what the next will read.
    float* const planes[2] = {rv.b.split_state, rv.b.split_state + size_t(R) * s->m->need.split_state};
    SplitWindow sw;
    int launches = 0;
    auto run = [&](const KvRef& kv, int k_lo, int k_hi, bool last) {
      sw.k_lo = k_lo;
      sw.k_hi = k_hi;
      sw.last = last;
      sw.state = planes[launches & 1];
      sw.carry = planes[(launches + 1) & 1];
      attention_split(rv.b.qbuf, kv, nh, nkv, hd, R, true, 0, scale, s->P, gate, rv.ld_proj, out, ld_out, rv.b.part,
                      kPrefillSplits, rv.L, 0, 0, nullptr, {}, sw);
      sw.first = false;
      ++launches;
    };
    if (from > 0) run(kb.ref(), 0, from, false);
    for (int i = 0; i < windows; ++i) {
      const int w0 = from + i * len, w1 = std::min(end, w0 + len);
      run(s->window_before(kb, w0, w1, R, st), w0, w1, i + 1 == windows);
      s->stage_after(kb, st);
    }
    return;
  }
  if (!rv.b.attn_state) throw std::logic_error("cuda: prefill window state buffer missing");
  FlashWindow w;
  w.state = rv.b.attn_state;
  auto run = [&](const KvRef& kv, int k_lo, int k_hi, bool last) {
    w.k_lo = k_lo;
    w.k_hi = k_hi;
    w.last = last;
    w.history = k_hi <= s->position;
    if (!attention_flash(rv.b.qbuf, kv, nh, nkv, hd, R, scale, s->P, gate, rv.ld_proj, out, ld_out, rv.L, 0, w))
      throw std::logic_error("cuda: windows of keys on a head dim the tensor-core prefill does not take");
    w.first = false;
  };
  if (from > 0) run(kb.ref(), 0, from, false);
  for (int i = 0; i < windows; ++i) {
    const int w0 = from + i * len, w1 = std::min(end, w0 + len);
    run(s->window_before(kb, w0, w1, R, st), w0, w1, i + 1 == windows);
    s->stage_after(kb, st);
  }
}

}  // namespace

// A block's own adapters (ABI 43, Adapters) beside its linears, on the float32 rows the linears read: u = scale (a x),
// then each linear's y += b u.  The a's of the linears that read one input are one GEMM (q|k|v, gate|up).  A round's
// rows, and a prefill pass's up to kLoraEarlyRows: u before the linear's own GEMM, by lora_down_round (a round) or
// lora_down_rows (a row's bits alone) on the side stream beside that GEMM where there is one (it needs no scratch).
// More rows: u after the linear's GEMM, by gemm (tensor cores; gemm() stages its input in the scratch the linear's
// staged input is in).  Then every b: in one launch for a round (lora_up), a launch a linear for a prefill pass
// (lora_up_rows), either taking the scale step; an MLP's gate and up in its SwiGLU where the rank allows (mlp_layer).
namespace {
bool round_adapters(int R, const Launch& L) { return R <= 16 || L.round_rows; }
// u before the linear's GEMM (lora_begin), else after it (lora_*_end): a round's and up to kLoraEarlyRows
bool down_early(int R, const Launch& L) { return round_adapters(R, L) || R <= kLoraEarlyRows; }
bool adapters_beside(int R, const Launch& L) {
  return down_early(R, L) && L.side && L.side->st && (!L.opt || L.opt->side_stream);
}
// u ready for a consumer of its own after the linear's GEMM (lora_*_end's first step): a prefill pass's u (unscaled:
// lora_up takes the scale) where lora_begin left it, else the join with lora_begin's side stream.
void lora_ready(const RowView& rv, const bf16* a, int n, const float* x, int ldx, int K, int R, int ldu,
                const Launch& L) {
  if (!down_early(R, L)) lora_down(x, ldx, a, n, K, 1.f, rv.b.lora, ldu, R, L);
  if (adapters_beside(R, L)) CK(cudaStreamWaitEvent(L.st, L.side->join, 0));
}
// y += b u: lora_up for a round, lora_up_rows for a prefill pass.
void lora_up_any(const RowView& rv, const Adapters& l, const LoraUp& up, float* y, int ldy, int R, int ldu,
                 const Launch& L) {
  if (round_adapters(R, L)) lora_up(rv.b.lora, ldu, l.rank, l.scale, up, y, ldy, R, L);
  else lora_up_rows(rv.b.lora, ldu, l.rank, l.scale, up, y, ldy, R, L);
}
}  // namespace

void lora_begin(const RowView& rv, const bf16* a, int n, const float* x, int ldx, int K, int R, int ldu,
                const Launch& L) {
  if (!down_early(R, L)) return;  // lora_*_end's
  // a round's one launch (lora_down_round, decoding's latency), a prefill pass's split over K (its tiles fill the GPU)
  auto down = [&](cudaStream_t st) {
    if (round_adapters(R, L)) lora_down_round(x, ldx, a, n, K, rv.b.lora, ldu, R, st);
    else lora_down_rows(x, ldx, a, n, K, rv.b.lora, ldu, R, rv.b.lora_part, kLoraEarlyRows, st);
  };
  if (adapters_beside(R, L)) {
    CK(cudaEventRecord(L.side->fork, L.st));
    CK(cudaStreamWaitEvent(L.side->st, L.side->fork, 0));
    down(L.side->st);
    CK(cudaEventRecord(L.side->join, L.side->st));
  } else {
    down(L.st);
  }
}

void lora_in_end(const RowView& rv, const Adapters& l, int linears, const int* out, const float* x, int ldx, int K,
                 int R, int ldu, const Launch& L) {
  const int r = l.rank;
  lora_ready(rv, l.a_in, linears * r, x, ldx, K, R, ldu, L);
  LoraUp up;
  up.linears = linears;
  for (int i = 0, col = 0; i < linears; col += out[i], ++i) {
    up.b[i] = l.b[i];
    up.n[i] = out[i];
    up.u_col[i] = i * r;
    up.y_col[i] = col;
  }
  lora_up_any(rv, l, up, rv.b.proj, rv.ld_proj, R, ldu, L);
}

void lora_out_end(const RowView& rv, const Adapters& l, int linear, const float* x, int ldx, int K, int H, int R,
                  int ldu, const Launch& L) {
  lora_ready(rv, l.a_out, l.rank, x, ldx, K, R, ldu, L);
  LoraUp up;
  up.linears = 1;
  up.b[0] = l.b[linear];
  up.n[0] = H;
  lora_up_any(rv, l, up, rv.b.x, H, R, ldu, L);
}

// The last linear's (o, down), whose product goes to the residual rows.
void lora_out(const RowView& rv, const Adapters& l, int linear, const float* x, int ldx, int K, int H, int R, int ldu,
              const Launch& L) {
  lora_begin(rv, l.a_out, l.rank, x, ldx, K, R, ldu, L);
  lora_out_end(rv, l, linear, x, ldx, K, H, R, ldu, L);
}

// A block's norm and in-projection (norm_project), its adapters' first product beside the projection, from the
// producer's float32 rows (rv.b.h).
void norm_project_adapted(const RowView& rv, int R, const Norm& n, const Lin& w, int N, int H, bool fused,
                          const Adapters& l, int linears, int ldu) {
  const Launch& L = rv.L;
  if (fused) rmsnorm_split(rv.b.x, H, n.w, n.offset, n.eps, R, H, L, w.q8_inputs(), rv.b.h);
  else rmsnorm(rv.b.x, H, n.w, n.offset, n.eps, rv.b.h, H, R, H, L.st);
  lora_begin(rv, l.a_in, linears * l.rank, rv.b.h, H, H, R, ldu, L);
  if (fused) gemm_presplit(w, N, H, rv.b.proj, rv.ld_proj, R, false, L);
  else gemm(rv.b.h, H, w, N, H, rv.b.proj, rv.ld_proj, R, false, L);
}

void attn_layer(const Pass& ps, const AttnBlock& a, bool fused) {
  const RowView& rv = ps.v;
  auto* s = ps.s;
  const int H = a.hidden, R = ps.rows, nh = a.nh, nkv = a.nkv, hd = a.hd;
  cudaStream_t st = rv.L.st;
  const Launch& L = rv.L;
  const bool x16 = residual16(ps);
  // Adapters read float32 rows (lora_down_rows: a row's bits whatever the rows, and a wide round's attention writes
  // float32 rows, not the staged ones): the fused producer writes them beside its staged ones (rv.b.h), the attention
  // its output in float32 (rv.b.act), which the out projection's GEMM stages.
  const int ldu = s->m->need.lora;
  if (a.body_only) {
    gemm(rv.b.x, H, a.w_in, a.n_in(), H, rv.b.proj, rv.ld_proj, R, false, L);
  } else if (x16) {
    if (!fused) throw std::logic_error("cuda: a bf16 residual's pass runs the fused producers");
    rmsnorm_split(x16_of(rv), H, a.norm.w, a.norm.offset, a.norm.eps, R, H, L, a.w_in.q8_inputs());
    gemm_presplit(a.w_in, a.n_in(), H, rv.b.proj, rv.ld_proj, R, false, L);
  } else if (a.lora) {
    norm_project_adapted(rv, R, a.norm, a.w_in, a.n_in(), H, fused, a.lora, 3, ldu);
    const int out[3] = {a.q_width(), nkv * hd, nkv * hd};  // q's covers the gate's rows (with one)
    lora_in_end(rv, a.lora, 3, out, rv.b.h, H, H, R, ldu, L);
  } else {
    norm_and_project(rv, R, a.norm, a.w_in, a.n_in(), H, fused);
  }
  const KvBuf& kb = s->kv[size_t(a.kv)];
  KvRef kv = kb.ref();
  const TreeRows* tree = tree_of(ps);
  // Stage aligned 16-row prefill Strata writes in dead device memory (rv.b.act) and flush contiguously
  // via cudaMemcpyAsync instead of scattering 4-byte writes across PCIe into host-pinned Strata buffers.
  const bool stage_strata =
      !ps.round && !tree && rv.b.act && kb.kq && kb.kq_host && kb.kq == kb.kq_host && kb.ring == 0 &&
      R >= 16 && (R & 15) == 0 && (s->position & 15) == 0;
  const bool stage_grids = kb.copy_slot >= 0;  // copy_slot < 0 keeps kqs/vqs directly on device
  const size_t q_words = size_t(R) * size_t(nkv) * size_t(kb.words);
  const size_t grid_cnt = size_t(R) * size_t(nkv);
  const size_t need_bytes =
      2 * q_words * sizeof(uint32_t) + (stage_grids ? 2 * grid_cnt * sizeof(__half2) : 0);
  const bool use_stage = stage_strata && need_bytes <= size_t(R) * size_t(rv.ld_act) * sizeof(float);
  uint32_t *stg_kq = nullptr, *stg_vq = nullptr;
  __half2 *stg_kqs = nullptr, *stg_vqs = nullptr;
  const size_t goff = size_t(s->position) * size_t(nkv);
  if (use_stage) {
    stg_kq = reinterpret_cast<uint32_t*>(rv.b.act);
    stg_vq = stg_kq + q_words;
    kv.kq = stg_kq;
    kv.vq = stg_vq;
    if (stage_grids) {
      stg_kqs = reinterpret_cast<__half2*>(stg_vq + q_words);
      stg_vqs = stg_kqs + grid_cnt;
      kv.kqs = stg_kqs;
      kv.vqs = stg_vqs;
    } else {
      kv.kqs = kb.kqs + goff;
      kv.vqs = kb.vqs + goff;
    }
  }
  attn_prep(rv.b.proj, rv.ld_proj, 0, a.q_width(), a.q_width() + nkv * hd, a.q_norm.w, a.k_norm.w, a.q_norm.offset,
            a.q_norm.eps, a.theta, a.rot, nh, nkv, hd, R, s->P, kPosPosition, false, rv.b.qbuf, kv, st, 0, tree,
            use_stage ? s->position : 0, CanvasKeys{}, a.mrope(ps.pos_3d));
  if (use_stage) {
    const size_t woff = strata_words_at(s->position, 0, nkv, kb.words);
    CK(cudaMemcpyAsync(kb.kq + woff, stg_kq, q_words * sizeof(uint32_t), cudaMemcpyDefault, st));
    CK(cudaMemcpyAsync(kb.vq + woff, stg_vq, q_words * sizeof(uint32_t), cudaMemcpyDefault, st));
    if (stage_grids) {
      CK(cudaMemcpyAsync(kb.kqs + goff, stg_kqs, grid_cnt * sizeof(__half2), cudaMemcpyDefault, st));
      CK(cudaMemcpyAsync(kb.vqs + goff, stg_vqs, grid_cnt * sizeof(__half2), cudaMemcpyDefault, st));
    }
  }
  // The attention writes the out projection's input itself, but not in 8 bits: for such weights it stays float32.
  const bool split = fused && !a.w_out.q8_inputs() && !a.lora;
  if (ps.reads_cache(kb)) {
    if (const int windows = ps.round ? 0 : s->windows(kb, R)) {
      attn_windows(ps, a, kb, windows, split ? nullptr : rv.b.act, nh * hd);
    } else {
      // The positions in host memory through the stagings, the rows just written among them; a small
      // pass past them, where they are (0074).
      const bool direct = s->reads_direct(kb, R, ps.round);
      if (!direct) s->stage_before(kb, 0, R, st);
      attention(rv.b.qbuf, direct ? kb.ref() : kb.staged(), nh, nkv, hd, R, true, 0, 1.f / std::sqrt(float(hd)), s->P,
                a.gate ? rv.b.proj + nh * hd : nullptr, rv.ld_proj, split ? nullptr : rv.b.act, nh * hd, rv.b.part,
                ps.round ? kRoundSplits : kPrefillSplits, L, 0, 0, tree);
      if (!direct) s->stage_after(kb, st);
    }
  }
  // A round reads the low-bit copy, a few positions the cache; about the switch, its rows replace
  // exact attention's at the switch and past it.
  if (ps.reads_copy(kb)) {
    const bool only = ps.reads_copy_only(kb);
    const cudaEvent_t full = only ? s->copy_before(kb, R, st, true) : nullptr;
    const float scale = 1.f / std::sqrt(float(hd));
    const int from = ps.strata == kStrataBoth ? s->strata_from : 0;
    if (R <= FLYX_MAX_ROWS) {
      attention_strata(rv.b.qbuf, kb.copy_read(only), s->strata, nh, nkv, hd, R, scale, s->P,
                       a.gate ? rv.b.proj + nh * hd : nullptr, rv.ld_proj, split ? nullptr : rv.b.act, nh * hd,
                       rv.b.part, kRoundSplits, L, from, full);
    } else {  // a wide round (generic: float32 out): 16 rows at a time, each as a round's rows at its own position
      const size_t q_row = size_t(nh) * hd;
      for (int a0 = 0; a0 < R; a0 += FLYX_MAX_ROWS) {
        const int k = std::min(FLYX_MAX_ROWS, R - a0);
        shift_params(s->P, s->P_shift, a0, st);
        attention_strata(rv.b.qbuf + a0 * q_row, kb.copy_read(only), s->strata, nh, nkv, hd, k, scale, s->P_shift,
                         a.gate ? rv.b.proj + size_t(a0) * rv.ld_proj + nh * hd : nullptr, rv.ld_proj,
                         rv.b.act + a0 * q_row, nh * hd, rv.b.part, kRoundSplits, L, from, a0 == 0 ? full : nullptr,
                         s->P);
      }
    }
    if (only) s->stage_after(kb, st);
  }
  if (a.lora) lora_begin(rv, a.lora.a_out, a.lora.rank, rv.b.act, nh * hd, nh * hd, R, ldu, L);
  if (x16) {
    if (!split || !gemm_staged_bf16(a.w_out.q, a.w_out.s, a.w_out.rs, H, nh * hd, x16_of(rv), H, R, L, true))
      throw std::logic_error("cuda: a bf16 residual's out projection is a W4A4 GEMM");
  } else if (split) {
    gemm_presplit(a.w_out, H, nh * hd, rv.b.x, H, R, !a.body_only, L);
  } else if (residual16_pass(ps)) {  // fusion=0: as gemm_staged_bf16 adds it; h (H wide; qbuf in a pass) is free here
    gemm(rv.b.act, nh * hd, a.w_out, H, nh * hd, rv.b.h, H, R, false, L);
    add_rows_bf16(rv.b.x, H, rv.b.h, H, H, R, st);
  } else {
    gemm(rv.b.act, nh * hd, a.w_out, H, nh * hd, rv.b.x, H, R, !a.body_only, L);
  }
  if (a.lora) lora_out_end(rv, a.lora, FLYX_LIN_O, rv.b.act, nh * hd, nh * hd, H, R, ldu, L);
}

void mlp_layer(const Pass& ps, const MlpBlock& f, bool fused) {
  const RowView& rv = ps.v;
  if (residual16(ps)) {  // norm from the bf16 rows, SwiGLU in the gate|up GEMM, the down GEMM added to them
    const int H = f.hidden, I = f.inter, R = ps.rows;
    if (!fused) throw std::logic_error("cuda: a bf16 residual's pass runs the fused producers");
    rmsnorm_split(x16_of(rv), H, f.norm.w, f.norm.offset, f.norm.eps, R, H, rv.L, f.w_gu.q8_inputs());
    gemm_glu_presplit(f.w_gu, I, H, R, rv.L);
    if (!gemm_on_glu_bf16(f.w_down.q, f.w_down.s, f.w_down.rs, H, I, x16_of(rv), H, R, true, rv.L))
      throw std::logic_error("cuda: a bf16 residual's down projection is a W4A4 GEMM");
    return;
  }
  if (!f.lora) {
    mlp_body(rv.b.x, f, ps.rows, rv.b.h, rv.b.proj, rv.ld_proj, rv.b.act, rv.ld_act, rv.L, fused,
             residual16_pass(ps));
    return;
  }
  // The block's own adapters (ABI 43): the gate and up rows take their products before the SwiGLU, so the gate|up GEMM
  // and then the SwiGLU, never the SwiGLU in the GEMM's epilogue.
  const int H = f.hidden, R = ps.rows, I = f.inter, ldu = ps.s->m->need.lora;
  const Norm& n = f.norm;
  cudaStream_t st = rv.L.st;
  const Launch& L = rv.L;
  const int gu[2] = {I, I};
  norm_project_adapted(rv, R, n, f.w_gu, 2 * I, H, fused, f.lora, 2, ldu);
  // the SwiGLU rows: I apart beside staged ones, rv.ld_act apart from the generic producer
  const int lda = fused ? I : rv.ld_act;
  const float* act = rv.b.act;  // the SwiGLU's float32 rows (beside the staged ones), the adapters' (attn_layer)
  // The SwiGLU takes gate's and up's second products (lora_up's bits) as it reads them, for an adapter of one lane an
  // output (lora_lanes: rank 16, 24, 40, ...): a round's one launch fewer, a prefill pass's (16 rows a block) no pass
  // over gate|up's rows of its own.  More lanes: their b rows (64 bytes an output and more) outrun the L1 a block's
  // 16 rows share, and their work the launch the SwiGLU saves.  Measured as lora_up's lanes are (option lora_glu=0:
  // gate's and up's in lora_up): rank 16, plain decoding 85.7 tok/s against 85.1, MTP 57.5 rounds/s against 57.0.
  const bool glu = !L.opt || L.opt->lora_glu;
  if (fused && glu && lora_lanes(f.lora.rank, L.opt ? L.opt->lora_lanes : 0) == 1) {
    lora_ready(rv, f.lora.a_in, 2 * f.lora.rank, rv.b.h, H, H, R, ldu, L);
    LoraGlu lg;
    lg.u = rv.b.lora;
    lg.ldu = ldu;
    lg.gate = f.lora.b[0];
    lg.up = f.lora.b[1];
    lg.rank = f.lora.rank;
    lg.scale = f.lora.scale;
    swiglu_split(rv.b.proj, rv.ld_proj, I, R, L, f.w_down.q8_inputs(), rv.b.act, lg);
  } else {
    lora_in_end(rv, f.lora, 2, gu, rv.b.h, H, H, R, ldu, L);
    if (fused) swiglu_split(rv.b.proj, rv.ld_proj, I, R, L, f.w_down.q8_inputs(), rv.b.act);
    else swiglu(rv.b.proj, rv.ld_proj, I, rv.b.act, rv.ld_act, R, st);
  }
  lora_begin(rv, f.lora.a_out, f.lora.rank, act, lda, I, R, ldu, L);
  if (fused) gemm_presplit(f.w_down, H, I, rv.b.x, H, R, true, L);
  else gemm(rv.b.act, rv.ld_act, f.w_down, H, I, rv.b.x, H, R, true, L);
  lora_out_end(rv, f.lora, 2, act, lda, I, H, R, ldu, L);
}

namespace {

// A block's loop gate controllers run in two ways (ops/attnres/ref.h loop_control).  Rounds, and the passes of
// FLYX_MAX_ROWS rows or fewer that compute as rounds, read the exact weights, all of a block's
// controllers in two launches (loop_control).  A prefill pass of more rows runs each as linears on tensor cores from
// bf16 inputs (apply_loop_ctrl), as its other bf16 linears run there: the two agree within tolerance
// (docs/design/numerics.md, "Not promised"), and in float32 such a pass took 15% longer (Inkling-2-Mini, 6,280
// tokens in passes of 1,024, RTX 5070 Ti).
bool loop_as_round(const Pass& ps) { return ps.round || ps.rows <= FLYX_MAX_ROWS; }

// The same weights, offset and epsilon bit for bit: the same rows of the same input.
bool same_norm(const Norm& a, const Norm& b) {
  return a.w == b.w && std::bit_cast<uint32_t>(a.offset) == std::bit_cast<uint32_t>(b.offset) &&
         std::bit_cast<uint32_t>(a.eps) == std::bit_cast<uint32_t>(b.eps);
}

LoopGate loop_gate(const LoopWeights& w, const float* cur, const float* msg, float* out) {
  return {cur, msg, w.down.round, w.gain.round, w.forget.round, out};
}

void apply_loop_ctrl(const float* cur, const float* msg, int rows, int H, int rank, const LoopWeights& w,
                     float* ctrl_cat, float* ctrl_gate, float* ctrl_gf, float* out, const Launch& L) {
  cudaStream_t st = L.st;
  loop_ctrl_prep(cur, msg, rows, H, ctrl_cat, st);
  gemm(ctrl_cat, 2 * H, w.down, rank, 2 * H, ctrl_gate, rank, rows, false, L);
  if (w.stacked) {
    gemm(ctrl_gate, rank, w.gain_forget, 2 * H, rank, ctrl_gf, 2 * H, rows, false, L);
  } else {
    gemm(ctrl_gate, rank, w.gain, H, rank, ctrl_gf, 2 * H, rows, false, L);
    gemm(ctrl_gate, rank, w.forget, H, rank, ctrl_gf + H, 2 * H, rows, false, L);
  }
  loop_ctrl_apply(cur, msg, ctrl_gf, 2 * H, ctrl_gf + H, 2 * H, rows, H, out, st);
}

void apply_loop_ctrl_norm(const float* cur, const float* msg, int rows, int H, int rank, const LoopWeights& w,
                          const Norm& n, float* ctrl_cat, float* ctrl_gate, float* ctrl_gf, float* y, int ldy,
                          const Launch& L) {
  cudaStream_t st = L.st;
  loop_ctrl_prep(cur, msg, rows, H, ctrl_cat, st);
  gemm(ctrl_cat, 2 * H, w.down, rank, 2 * H, ctrl_gate, rank, rows, false, L);
  if (w.stacked) {
    gemm(ctrl_gate, rank, w.gain_forget, 2 * H, rank, ctrl_gf, 2 * H, rows, false, L);
  } else {
    gemm(ctrl_gate, rank, w.gain, H, rank, ctrl_gf, 2 * H, rows, false, L);
    gemm(ctrl_gate, rank, w.forget, H, rank, ctrl_gf + H, 2 * H, rows, false, L);
  }
  loop_ctrl_apply_norm(cur, msg, ctrl_gf, 2 * H, ctrl_gf + H, 2 * H, n.w, n.offset, n.eps, rows, H, y, ldy, st);
}

SublayerPostConv make_sublayer_post_conv(const Pass& ps, const PendingPostConv& pend) {
  auto* s = ps.s;
  ShortConvBuf& sc = s->sconv[size_t(pend.sconv)];
  const bool direct = ps.round && ps.p->width == 0;
  const int next_sel = ps.round ? (direct ? kSlotNext : -1) : kSlotCur;
  const int skip = ps.round && direct ? kSkipIfStopped : kSkipNever;
  float* rep = ps.round ? sc.replay : nullptr;
  const size_t rep_stride = ps.round ? sc.r_stride : 0;
  return SublayerPostConv{pend.post_raw, pend.conv_w, sc.win, sc.w_stride, kSlotCur, next_sel, skip, rep, rep_stride,
                          kSlotCur, pend.ch_off, sc.channels, pend.conv, pend.res_push};
}

void rel_attn_windows(const Pass& ps, const RelAttnBlock& a, const KvBuf& kb, int windows, float* out, int ld_out) {
  const RowView& rv = ps.v;
  auto* s = ps.s;
  const int R = ps.rows, nh = a.nh, nkv = a.nkv, hd = a.hd, dr = a.d_rel;
  cudaStream_t st = rv.L.st;
  if (!rv.b.attn_state) throw std::logic_error("cuda: prefill window state buffer missing");
  const int from = s->window_from(kb), len = s->window_len(kb), end = s->position + R;
  RelWindow w;
  w.state = rv.b.attn_state;
  auto run = [&](const KvRef& kv, int k_lo, int k_hi, bool last) {
    w.k_lo = k_lo;
    w.k_hi = k_hi;
    w.last = last;
    rel_attention(rv.b.qbuf, rv.b.rel_q, kv, a.rel_bias, rv.b.sink, nh, rv.b.proj + nh * hd, rv.ld_proj, nh, nkv, hd,
                  dr, a.rel_extent, a.window, R, s->P, out, ld_out, rv.b.part, kPrefillSplits, rv.L, 0, w);
    w.first = false;
  };
  if (from > 0) run(kb.ref(), 0, from, false);
  for (int i = 0; i < windows; ++i) {
    const int w0 = from + i * len, w1 = std::min(end, w0 + len);
    run(s->window_before(kb, w0, w1, R, st), w0, w1, i + 1 == windows);
    s->stage_after(kb, st);
  }
}

}  // namespace

void flush_pending_post_conv(const Pass& ps, bool require_in_x) {
  if (!ps.res) return;
  const RowView& rv = ps.v;
  auto* s = ps.s;
  const int H = s->m->hidden, R = ps.rows;
  cudaStream_t st = rv.L.st;
  float* x_cur = ps.res->x_cur ? ps.res->x_cur : rv.b.x;
  const size_t row_bytes = size_t(R) * size_t(H) * sizeof(float);
  if (ps.res->pending.post_raw) {
    const PendingPostConv pend = ps.res->pending;
    const SublayerPostConv spc = make_sublayer_post_conv(ps, pend);
    float* dst = require_in_x ? rv.b.x : x_cur;
    if (pend.res_push) {
      CK(cudaMemcpyAsync(rv.b.res_bank + size_t(ps.res->res_n) * R * H, x_cur, row_bytes, cudaMemcpyDeviceToDevice,
                         st));
      ++ps.res->res_n;
      shortconv_pass(spc.post_raw, H, spc.conv_w, nullptr, spc.win, spc.win_stride, spc.in_sel, spc.next_sel, spc.skip,
                     spc.replay, spc.replay_stride, spc.replay_sel, spc.ch_off, spc.ch_total, H, spc.conv, 0, s->P, R,
                     dst, H, st, false);
    } else {
      shortconv_pass(spc.post_raw, H, spc.conv_w, nullptr, spc.win, spc.win_stride, spc.in_sel, spc.next_sel, spc.skip,
                     spc.replay, spc.replay_stride, spc.replay_sel, spc.ch_off, spc.ch_total, H, spc.conv, 0, s->P, R,
                     dst, H, st, true, dst == x_cur ? nullptr : x_cur);
    }
    ps.res->pending = {};
    ps.res->x_cur = dst;
    return;
  }
  if (require_in_x && x_cur != rv.b.x) {
    CK(cudaMemcpyAsync(rv.b.x, x_cur, row_bytes, cudaMemcpyDeviceToDevice, st));
    ps.res->x_cur = rv.b.x;
  }
}

// The expert stream rejoins the main stream at the end of this pass.
static void warm_side(const Pass& ps, const L2Ranges& r) {
  auto* s = ps.s;
  const cudaStream_t st = ps.v.L.st;
  if (!r.count || !ps.res || !s->expert_side.st || st != s->st) return;  // a sequence's own stream (batched passes: not)
  const SideStream& es = s->expert_side;
  CK(cudaEventRecord(es.fork, st));
  CK(cudaStreamWaitEvent(es.st, es.fork, 0));
  l2_warm(r, es.st);
  CK(cudaEventRecord(es.join, es.st));
  ps.res->expert_work = true;
}

// A linear's round weights (Lin::round) of n values: binary16 or float32.
static void add_round(L2Ranges& r, const Lin& w, size_t n) {
  if (w.round.b16) r.add(w.round.b16, n * 2);
  else r.add(w.round.f32, n * 4);
}

void rel_attn_layer(const Pass& ps, const RelAttnBlock& a, bool fused) {
  const RowView& rv = ps.v;
  auto* s = ps.s;
  const int H = a.hidden, R = ps.rows, nh = a.nh, nkv = a.nkv, hd = a.hd;
  const int q_dim = nh * hd, kv_dim = nkv * hd, dr = a.d_rel;
  const bool is_swa = a.window > 0;
  cudaStream_t st = rv.L.st;
  const Launch& L = rv.L;
  const KernelOptions& ko = opts(L);
  ShortConvBuf& sc = s->sconv[size_t(a.sconv)];
  const bool direct = ps.round && ps.p->width == 0;
  const int next_sel = ps.round ? (direct ? kSlotNext : -1) : kSlotCur;
  const int skip = ps.round && direct ? kSkipIfStopped : kSkipNever;
  float* rep = ps.round ? sc.replay : nullptr;
  const size_t rep_stride = ps.round ? sc.r_stride : 0;

  // Each guess consumes the residual on the expert stream. Only that read must finish before an
  // in-place ShortConv or the next layer can reuse the row/bank. Prefetching weights can continue.
  bool expert_read_pending = false;
  auto guess_experts = [&](const float* row, bool mix, int planes) {
    if (!ps.round || !ps.res || !ps.next_moe || !rv.b.moe_guess || !s->expert_side.st || st != s->st) return;
    const MoeBlock& f = *ps.next_moe;
    const bf16* rb = f.router.round.b16 ? f.router.round.b16 : (f.router.round.f32 ? nullptr : f.router.w);
    if ((!rb && !f.router.round.f32) || f.n_experts < 1 || f.n_experts > 256 ||
        f.top_k < 1 || f.top_k > std::min(f.n_experts, kMoeRouteMaxTopK) ||
        !f.gate_up.q || !f.gate_up.s || !f.down.q || !f.down.s || f.gate_up.packed() || f.down.packed() ||
        (mix && !f.res_query)) return;
    const SideStream& es = s->expert_side;
    CK(cudaEventRecord(es.fork, st));
    CK(cudaStreamWaitEvent(es.st, es.fork, 0));
    if (mix) {
      float* mixed = rv.b.moe_guess + f.n_experts;
      attn_res_mix(rv.b.res_bank, ps.res->res_n, row, f.res_query, 1, 1, H, mixed, es.st);
      CK(cudaEventRecord(s->expert_read, es.st));
      row = mixed;
    }
    moe_guess_scores(row, f.norm.w, f.norm.offset, rb, rb ? nullptr : f.router.round.f32,
                     f.router_bias, f.gating_sigmoid, f.n_experts, H, rv.b.moe_guess, es.st);
    if (!mix) CK(cudaEventRecord(s->expert_read, es.st));
    expert_read_pending = true;
    moe_prefetch(rv.b.moe_guess, f.n_experts, f.top_k, H, f.inter, f.gate_up, f.down, planes, ko.guess_k, es.st);
    CK(cudaEventRecord(es.join, es.st));
    ps.res->expert_work = true;
  };
  auto finish_residual_read = [&] {
    if (expert_read_pending) CK(cudaStreamWaitEvent(st, s->expert_read, 0));
    expert_read_pending = false;
  };
  const int prefetch = ko.expert_prefetch;
  if (prefetch && (prefetch != 2 || R > 1) && R <= FLYX_MAX_ROWS && ps.res)
    guess_experts(ps.res->x_cur ? ps.res->x_cur : rv.b.x, false, -1);

  // The loop messages read only the loop bank, which no kernel of the pass writes: on the side stream beside the
  // sublayer's mix, joined before the controllers read them (option side_loop_msg=0: in turn).  Same bits.
  bool loop_msg_side = false;
  if (a.loop_msg_query) {
    if (ko.side_loop_msg && ps.round && L.side && L.side->st && ko.side_stream) {
      CK(cudaEventRecord(L.side->fork, st));
      CK(cudaStreamWaitEvent(L.side->st, L.side->fork, 0));
      attn_res_mix(rv.b.loop_bank, ps.res->loop_n, rv.b.loop_bank_out, a.loop_msg_query, 4, R, H, rv.b.loop_msg,
                   L.side->st);
      CK(cudaEventRecord(L.side->join, L.side->st));
      loop_msg_side = true;
    } else {
      attn_res_mix(rv.b.loop_bank, ps.res->loop_n, rv.b.loop_bank_out, a.loop_msg_query, 4, R, H, rv.b.loop_msg, st);
    }
  }

  // A fused round's producers write a projection's input as its GEMM reads it, in the launch that computes it; the
  // generic layer stages it from float32 rows in a launch of its own for each reader.
  const bool round_q8 = fused && ps.round && R <= FLYX_MAX_ROWS;
  bool q_staged = false;
  bool q_normed = false;
  const float* in_raw[3] = {rv.b.x, rv.b.x, rv.b.x};

  const bool has_pending = ps.res && ps.res->pending.post_raw != nullptr;
  const bool can_fuse_sub = has_pending && round_q8 && a.res_query && rv.b.post_out && H <= 5120 &&
                            (ps.res->res_n + int(ps.res->pending.res_push) + 1) <= 64;
  if (can_fuse_sub) {
    const PendingPostConv pend = ps.res->pending;
    const SublayerPostConv spc = make_sublayer_post_conv(ps, pend);
    float* x_src = ps.res->x_cur ? ps.res->x_cur : rv.b.x;
    float* x_dst = (x_src == rv.b.x) ? rv.b.post_out : rv.b.x;
    const int nq = a.res_queries();
    const bool q_dual = a.loop_rank == 0 && a.stacked_qgr && a.w_qgr.q8_inputs() && (H % 32 == 0) &&
                        gemm_reads_split(rv.b.h, H, a.w_sink, H, L);
    const bool swa_reuse = is_swa && same_norm(a.q_norm_in, a.k_norm_in) && same_norm(a.q_norm_in, a.v_norm_in) &&
                           a.w_k.q8_inputs() && a.w_v.q8_inputs();
    const bool fuse_q_norm = q_dual && (!is_swa || swa_reuse);
    shortconv_res_mix_norm(spc, x_src, x_dst, rv.b.res_bank, ps.res->res_n, a.res_query, nq,
                           (fuse_q_norm && is_swa) ? nullptr : rv.b.attn_in,
                           fuse_q_norm ? SublayerNorm::AttnQ8Split : SublayerNorm::None,
                           fuse_q_norm ? a.q_norm_in.w : nullptr, a.q_norm_in.offset, a.q_norm_in.eps, nullptr, H,
                           nullptr, false, s->P, R, H, L);
    if (pend.res_push) ++ps.res->res_n;
    ps.res->pending = {};
    ps.res->x_cur = x_dst;
    // The next MoE's router, loop controller, query and norm (option warm, bits 1, 2, 4), while the attention runs:
    // RTX 5090, Inkling-2-Mini UD-IQ4_XS plain decoding 204.7 -> 207.5 tok/s with all three (the bank too: no gain).
    const int warm = ko.warm;
    if (warm && s->expert_side.st && ps.next_moe && ps.round) {
      const MoeBlock& f = *ps.next_moe;
      L2Ranges r;
      const size_t HH = size_t(f.hidden);
      if (warm & 1) add_round(r, f.router, size_t(f.n_experts) * HH);
      if ((warm & 2) && f.loop_rank > 0) {
        const size_t rk = size_t(f.loop_rank);
        add_round(r, f.loop_ctrl.down, rk * 2 * HH);
        if (f.loop_ctrl.stacked) {
          add_round(r, f.loop_ctrl.gain_forget, rk * 2 * HH);
        } else {
          add_round(r, f.loop_ctrl.gain, rk * HH);
          add_round(r, f.loop_ctrl.forget, rk * HH);
        }
      }
      if (warm & 4) {
        r.add(f.res_query, HH * 4);
        r.add(f.norm.w, HH * 4);
      }
      warm_side(ps, r);
    }
    // The one-row guess uses the mixed residual before the attention adds its contribution.
    if (ko.guess > 0 && R == 1) guess_experts(x_dst, true, ko.guess & 15);
    in_raw[0] = rv.b.attn_in;
    in_raw[1] = is_swa ? rv.b.attn_in : rv.b.attn_in + size_t(R) * H;
    in_raw[2] = is_swa ? rv.b.attn_in : rv.b.attn_in + size_t(2) * R * H;
    q_staged = fuse_q_norm;
  } else {
    if (has_pending) {
      finish_residual_read();
      flush_pending_post_conv(ps, false);
    }
    const float* cur_x = (ps.res && ps.res->x_cur) ? ps.res->x_cur : rv.b.x;
    in_raw[0] = in_raw[1] = in_raw[2] = cur_x;
    if (a.res_query) {
      const int nq = a.res_queries();
      attn_res_mix(rv.b.res_bank, ps.res->res_n, cur_x, a.res_query, nq, R, H, rv.b.attn_in, st);
      in_raw[0] = rv.b.attn_in;
      in_raw[1] = is_swa ? rv.b.attn_in : rv.b.attn_in + size_t(R) * H;
      in_raw[2] = is_swa ? rv.b.attn_in : rv.b.attn_in + size_t(2) * R * H;
    }
  }
  if (loop_msg_side) CK(cudaStreamWaitEvent(st, L.side->join, 0));
  if (a.loop_rank > 0 && loop_as_round(ps)) {
    // The q, k and v controllers at once, each writing where its input was mixed (attn_in + j R H).  An SWA block
    // mixes one input for all three there, which the first would write over while the others read it: that one
    // writes to ctrl_cat.
    LoopGate gates[3];
    for (int j = 0; j < 3; ++j) {
      float* dst = j == 0 && in_raw[1] == rv.b.attn_in ? rv.b.ctrl_cat : rv.b.attn_in + size_t(j) * R * H;
      gates[j] = loop_gate(a.loop_ctrl[j], in_raw[j], rv.b.loop_msg + size_t(j) * R * H, dst);
    }
    loop_control(gates, 3, R, H, a.loop_rank, rv.b.ctrl_scratch, st);
    for (int j = 0; j < 3; ++j) in_raw[j] = gates[j].out;
  } else if (a.loop_rank > 0) {
    // Reverse order so when is_swa shares in_raw[0..2] == attn_in, j=2,1 read attn_in before j=0 overwrites it.
    for (int j = 2; j >= 0; --j) {
      const float* msg = rv.b.loop_msg + size_t(j) * R * H;
      if (j == 0 && fused) {
        apply_loop_ctrl_norm(in_raw[0], msg, R, H, a.loop_rank, a.loop_ctrl[0], a.q_norm_in, rv.b.ctrl_cat,
                             rv.b.ctrl_gate, rv.b.ctrl_gf, rv.b.h, H, L);
        q_normed = true;
      } else {
        float* dst = rv.b.attn_in + size_t(j) * R * H;
        apply_loop_ctrl(in_raw[j], msg, R, H, a.loop_rank, a.loop_ctrl[j], rv.b.ctrl_cat, rv.b.ctrl_gate, rv.b.ctrl_gf,
                        dst, L);
        in_raw[j] = dst;
      }
    }
  }

  // The norm of x for two projections that read it in different forms, w in 8 bits and its companion (a sink or
  // dynamic-tap projection) as three bf16 parts: one launch writes both, which lie apart in the scratch.
  auto project_pair = [&](const float* x, const Norm& n, const Lin& w, int cols, float* out, int ld_out,
                          const Lin& companion, int companion_cols, float* companion_out) {
    const bool dual = round_q8 && w.q8_inputs() && gemm_reads_split(rv.b.h, H, companion, H, L);
    if (dual) {
      rmsnorm_q8_split(x, H, n.w, n.offset, n.eps, R, H, L);
      // The companion (the sink's 24 columns: a 3-block launch, ~3.5 us) beside the wide GEMM on the side stream: it
      // reads only the split rows, which nothing writes until the join after the wide GEMM, and writes only its
      // own output.  Option side_companion=0: in turn.  Same launches, same bits.
      if (ko.side_companion && ps.round && L.side && L.side->st && ko.side_stream) {
        Launch on = L;
        on.st = L.side->st;
        on.side = nullptr;
        CK(cudaEventRecord(L.side->fork, L.st));
        CK(cudaStreamWaitEvent(on.st, L.side->fork, 0));
        gemm_presplit(companion, companion_cols, H, companion_out, companion_cols, R, false, on);
        CK(cudaEventRecord(L.side->join, on.st));
        gemm_presplit(w, cols, H, out, ld_out, R, false, L);
        CK(cudaStreamWaitEvent(L.st, L.side->join, 0));
      } else {
        gemm_presplit(w, cols, H, out, ld_out, R, false, L);
        gemm_presplit(companion, companion_cols, H, companion_out, companion_cols, R, false, L);
      }
    } else {
      rmsnorm(x, H, n.w, n.offset, n.eps, rv.b.h, H, R, H, st);
      gemm(rv.b.h, H, w, cols, H, out, ld_out, R, false, L);
      gemm(rv.b.h, H, companion, companion_cols, H, companion_out, companion_cols, R, false, L);
    }
    return dual;
  };

  // A sliding-window round's K and V projections (their own norms of their own inputs) beside the q path on the side
  // stream, with the side's own 8-bit rows (side_scratch): they read inputs ready before the fork and write only
  // the K/V ShortConv rows, which rel_attn_prep reads after the join.  Option side_kv=0: in turn.  Same bits.
  const bool kv_shares_q = in_raw[0] == in_raw[1] && in_raw[0] == in_raw[2] && same_norm(a.q_norm_in, a.k_norm_in) &&
                           same_norm(a.q_norm_in, a.v_norm_in);
  // K and V weights in 8 bits only: their norms then write the side scratch's q8 rows, never the bf16 split rows
  // the q path's GEMMs share (rmsnorm_split of other weights writes Scratch::split).
  const bool kv_beside = ko.side_kv && ps.round && is_swa && fused && H % 64 == 0 && round_q8 && a.conv <= 64 &&
                         a.w_k.q8_inputs() && a.w_v.q8_inputs() &&
                         !kv_shares_q && L.side && L.side->st && ko.side_stream && !q_normed;
  if (kv_beside) {
    // The side's 8-bit rows in the SwiGLU's buffers (Launch::glu), which no attention block uses: the q path's
    // rows stay in q8 while the K and V norms write theirs.
    Scratch side_scr = side_scratch(*L.scratch);
    side_scr.q8 = side_scr.q8_glu;
    side_scr.q8_scale = side_scr.q8_glu_scale;
    side_scr.q8_sums = side_scr.q8_glu_sums;
    Launch on = L;
    on.st = L.side->st;
    on.scratch = &side_scr;
    on.side = nullptr;
    CK(cudaEventRecord(L.side->fork, L.st));
    CK(cudaStreamWaitEvent(on.st, L.side->fork, 0));
    norm_project(in_raw[1], rv.b.h, rv.b.kv_conv, kv_dim, R, a.k_norm_in, a.w_k, kv_dim, H, on, true);
    norm_project(in_raw[2], rv.b.h, rv.b.kv_conv + size_t(R) * kv_dim, kv_dim, R, a.v_norm_in, a.w_v, kv_dim, H, on,
                 true);
    CK(cudaEventRecord(L.side->join, on.st));
  }

  if (q_staged) {
    gemm_presplit(a.w_qgr, 2 * q_dim + nh * dr, H, rv.b.proj, rv.ld_proj, R, false, L);
    gemm_presplit(a.w_sink, nh, H, rv.b.sink, nh, R, false, L);
  } else if (q_normed) {
    if (a.stacked_qgr) {
      gemm(rv.b.h, H, a.w_qgr, 2 * q_dim + nh * dr, H, rv.b.proj, rv.ld_proj, R, false, L);
      gemm(rv.b.h, H, a.w_sink, nh, H, rv.b.sink, nh, R, false, L);
    } else {
      gemm(rv.b.h, H, a.w_q, q_dim, H, rv.b.proj, rv.ld_proj, R, false, L);
      gemm(rv.b.h, H, a.w_gate, q_dim, H, rv.b.proj + q_dim, rv.ld_proj, R, false, L);
      gemm(rv.b.h, H, a.w_rel, nh * dr, H, rv.b.proj + 2 * q_dim, rv.ld_proj, R, false, L);
      gemm(rv.b.h, H, a.w_sink, nh, H, rv.b.sink, nh, R, false, L);
    }
  } else if (a.stacked_qgr) {
    q_staged = project_pair(in_raw[0], a.q_norm_in, a.w_qgr, 2 * q_dim + nh * dr, rv.b.proj, rv.ld_proj,
                            a.w_sink, nh, rv.b.sink);
  } else {
    rmsnorm(in_raw[0], H, a.q_norm_in.w, a.q_norm_in.offset, a.q_norm_in.eps, rv.b.h, H, R, H, st);
    gemm(rv.b.h, H, a.w_q, q_dim, H, rv.b.proj, rv.ld_proj, R, false, L);
    gemm(rv.b.h, H, a.w_gate, q_dim, H, rv.b.proj + q_dim, rv.ld_proj, R, false, L);
    gemm(rv.b.h, H, a.w_rel, nh * dr, H, rv.b.proj + 2 * q_dim, rv.ld_proj, R, false, L);
    gemm(rv.b.h, H, a.w_sink, nh, H, rv.b.sink, nh, R, false, L);
  }

  // A sliding-window block's q, k and v read one input through one norm: q's 8-bit rows serve k and v too.  The
  // sink's GEMM, the packed GEMMs and the ShortConv that run in between leave those rows as they are.
  const bool reuse_q = q_staged && is_swa && in_raw[0] == in_raw[1] && in_raw[0] == in_raw[2] &&
                       same_norm(a.q_norm_in, a.k_norm_in) && same_norm(a.q_norm_in, a.v_norm_in) &&
                       a.w_k.q8_inputs() && a.w_v.q8_inputs();
  // In fused rounds (next_sel != kSlotCur), write raw K/V directly into rv.b.kv_conv and run both K & V ShortConvs
  // inside rel_attn_prep (eliminating 2 shortconv_pass launches + DRAM round-trips per layer).
  const bool fuse_kv_sc = round_q8 && a.conv <= 64;
  auto project_kv = [&](const float* x, const Norm& n, const Lin& w, const Lin& taps, const float* conv, int offset,
                        float* dyn_dst) {
    float* raw_dst = fuse_kv_sc ? (rv.b.kv_conv + size_t(R) * offset) : rv.b.act;
    float* dyn = nullptr;
    if (reuse_q) {
      gemm_presplit(w, kv_dim, H, raw_dst, kv_dim, R, false, L);
    } else if (is_swa) {
      norm_project(x, rv.b.h, raw_dst, kv_dim, R, n, w, kv_dim, H, L, fused && H % 64 == 0);
    } else {
      dyn = dyn_dst;
      project_pair(x, n, w, kv_dim, raw_dst, kv_dim, taps, nkv * a.conv, dyn);
    }
    if (!fuse_kv_sc) {
      shortconv_pass(rv.b.act, kv_dim, conv, dyn, sc.win, sc.w_stride, kSlotCur, next_sel, skip, rep, rep_stride,
                     kSlotCur, offset, sc.channels, kv_dim, a.conv, nkv, s->P, R, rv.b.kv_conv + size_t(R) * offset,
                     kv_dim, st, false);
    }
  };
  float* k_dyn = is_swa ? nullptr : rv.b.kv_dyn;
  float* v_dyn = is_swa ? nullptr : (fuse_kv_sc ? rv.b.kv_dyn + size_t(R) * nkv * a.conv : rv.b.kv_dyn);
  if (kv_beside) {
    CK(cudaStreamWaitEvent(L.st, L.side->join, 0));
  } else {
    project_kv(in_raw[1], a.k_norm_in, a.w_k, a.w_k_conv_proj, a.k_conv, 0, k_dyn);
    project_kv(in_raw[2], a.v_norm_in, a.w_v, a.w_v_conv_proj, a.v_conv, kv_dim, v_dyn);
  }

  const KvBuf& kb = s->kv[size_t(a.kv)];
  KvRef kv = kb.ref();
  // Stage aligned 16-row prefill Strata writes in dead device memory (rv.b.act) and flush contiguously
  // via cudaMemcpyAsync instead of scattering 4-byte writes across PCIe into host-pinned Strata buffers.
  const bool stage_strata =
      !ps.round && rv.b.act && kb.kq && kb.kq_host && kb.kq == kb.kq_host && kb.ring == 0 && R >= 16 && (R & 15) == 0 &&
      (s->position & 15) == 0;
  const bool stage_grids = kb.copy_slot >= 0;
  const size_t q_words = size_t(R) * size_t(nkv) * size_t(kb.words);
  const size_t grid_cnt = size_t(R) * size_t(nkv);
  const size_t need_bytes =
      2 * q_words * sizeof(uint32_t) + (stage_grids ? 2 * grid_cnt * sizeof(__half2) : 0);
  const bool use_stage = stage_strata && need_bytes <= size_t(R) * size_t(rv.ld_act) * sizeof(float);
  uint32_t *stg_kq = nullptr, *stg_vq = nullptr;
  __half2 *stg_kqs = nullptr, *stg_vqs = nullptr;
  const size_t goff = size_t(s->position) * size_t(nkv);
  if (use_stage) {
    stg_kq = reinterpret_cast<uint32_t*>(rv.b.act);
    stg_vq = stg_kq + q_words;
    kv.kq = stg_kq;
    kv.vq = stg_vq;
    if (stage_grids) {
      stg_kqs = reinterpret_cast<__half2*>(stg_vq + q_words);
      stg_vqs = stg_kqs + grid_cnt;
      kv.kqs = stg_kqs;
      kv.vqs = stg_vqs;
    } else {
      kv.kqs = kb.kqs + goff;
      kv.vqs = kb.vqs + goff;
    }
  }
  const KvShortConv kv_sc{a.k_conv, a.v_conv, k_dyn, v_dyn, sc.win, sc.w_stride, kSlotCur, next_sel, skip, rep,
                          rep_stride, kSlotCur, sc.channels, a.conv};
  rel_attn_prep(rv.b.proj, rv.ld_proj, rv.b.kv_conv, kv_dim, rv.b.kv_conv + size_t(R) * kv_dim, kv_dim,
                rv.b.proj + 2 * q_dim, rv.ld_proj, a.q_clip_w, a.rel_clip_w, a.qk_clip, a.rel_clip, a.eps,
                a.log_scale_floor, a.log_scale_alpha, nh, nkv, hd, dr, R, s->P, kPosPosition, is_swa, rv.b.qbuf,
                rv.b.rel_q, kv, st, 0, use_stage ? s->position : 0, fuse_kv_sc ? &kv_sc : nullptr);
  if (use_stage) {
    const size_t woff = strata_words_at(s->position, 0, nkv, kb.words);
    CK(cudaMemcpyAsync(kb.kq + woff, stg_kq, q_words * sizeof(uint32_t), cudaMemcpyDefault, st));
    CK(cudaMemcpyAsync(kb.vq + woff, stg_vq, q_words * sizeof(uint32_t), cudaMemcpyDefault, st));
    if (stage_grids) {
      CK(cudaMemcpyAsync(kb.kqs + goff, stg_kqs, grid_cnt * sizeof(__half2), cudaMemcpyDefault, st));
      CK(cudaMemcpyAsync(kb.vqs + goff, stg_vqs, grid_cnt * sizeof(__half2), cudaMemcpyDefault, st));
    }
  }

  const bool split = fused && !a.w_o.q8_inputs() && q_dim % 64 == 0;
  // The output projection's 8-bit rows from the attention's last kernel (a warp a group of 32 within a head); a
  // round that reads a low-bit copy ends in that copy's attention, which writes float32 rows.
  const bool q8_out = round_q8 && a.w_o.q8_inputs() && hd % 32 == 0 && !ps.reads_copy(kb);
  if (ps.reads_cache(kb)) {
    if (const int windows = ps.round ? 0 : s->windows(kb, R)) {
      rel_attn_windows(ps, a, kb, windows, split ? nullptr : rv.b.act, q_dim);
    } else {
      const bool direct_kv = s->reads_direct(kb, R, ps.round);
      if (kb.vk && !direct_kv) s->stage_before(kb, 0, R, st);
      rel_attention(rv.b.qbuf, rv.b.rel_q, (kb.vk && !direct_kv) ? kb.staged() : kb.ref(), a.rel_bias, rv.b.sink, nh,
                    rv.b.proj + q_dim, rv.ld_proj, nh, nkv, hd, dr, a.rel_extent, a.window, R, s->P,
                    split || q8_out ? nullptr : rv.b.act, q_dim, rv.b.part, ps.round ? kRoundSplits : kPrefillSplits,
                    L, 0, {}, q8_out);
      if (kb.vk && !direct_kv) s->stage_after(kb, st);
    }
  }
  if (ps.reads_copy(kb)) {
    const bool only = ps.reads_copy_only(kb);
    const cudaEvent_t full = only ? s->copy_before(kb, R, st, true) : nullptr;
    const float scale = 1.f / std::sqrt(float(hd));
    const int from = ps.strata == kStrataBoth ? s->strata_from : 0;
    if (R <= FLYX_MAX_ROWS) {
      rel_bias_table(rv.b.rel_q, a.rel_bias, nh, dr, a.rel_extent, R, rv.b.rel_btab, st);
      const RelStrata rel{rv.b.rel_btab, a.rel_extent, rv.b.sink, nh};
      attention_strata(rv.b.qbuf, kb.copy_read(only), s->strata, nh, nkv, hd, R, scale, s->P, rv.b.proj + q_dim,
                       rv.ld_proj, split ? nullptr : rv.b.act, q_dim, rv.b.part, kRoundSplits, L, from, full, nullptr,
                       rel);
    } else {
      const size_t q_row = size_t(q_dim), rq_row = size_t(nh) * dr;
      for (int a0 = 0; a0 < R; a0 += FLYX_MAX_ROWS) {
        const int k = std::min(FLYX_MAX_ROWS, R - a0);
        shift_params(s->P, s->P_shift, a0, st);
        rel_bias_table(rv.b.rel_q + a0 * rq_row, a.rel_bias, nh, dr, a.rel_extent, k, rv.b.rel_btab, st);
        const RelStrata rel{rv.b.rel_btab, a.rel_extent, rv.b.sink + size_t(a0) * nh, nh};
        attention_strata(rv.b.qbuf + a0 * q_row, kb.copy_read(only), s->strata, nh, nkv, hd, k, scale, s->P_shift,
                         rv.b.proj + size_t(a0) * rv.ld_proj + q_dim, rv.ld_proj, rv.b.act + a0 * q_row, q_dim,
                         rv.b.part, kRoundSplits, L, from, a0 == 0 ? full : nullptr, s->P, rel);
      }
    }
    if (only) s->stage_after(kb, st);
  }

  if (split || q8_out) gemm_presplit(a.w_o, H, q_dim, rv.b.post_raw, H, R, false, L);
  else gemm(rv.b.act, q_dim, a.w_o, H, q_dim, rv.b.post_raw, H, R, false, L);

  finish_residual_read();
  if (round_q8 && ps.res && a.post_conv && rv.b.post_out && H <= 5120) {
    ps.res->pending = PendingPostConv{rv.b.post_raw, a.post_conv, a.sconv, 2 * kv_dim, a.conv, a.res_push != 0};
    return;
  }
  float* x_cur = (ps.res && ps.res->x_cur) ? ps.res->x_cur : rv.b.x;
  if (a.res_push) {
    const size_t row_bytes = size_t(R) * size_t(H) * sizeof(float);
    CK(cudaMemcpyAsync(rv.b.res_bank + size_t(ps.res->res_n) * R * H, x_cur, row_bytes, cudaMemcpyDeviceToDevice, st));
    ++ps.res->res_n;
    shortconv_pass(rv.b.post_raw, H, a.post_conv, nullptr, sc.win, sc.w_stride, kSlotCur, next_sel, skip, rep,
                   rep_stride, kSlotCur, 2 * kv_dim, sc.channels, H, a.conv, 0, s->P, R, x_cur, H, st, false);
  } else {
    shortconv_pass(rv.b.post_raw, H, a.post_conv, nullptr, sc.win, sc.w_stride, kSlotCur, next_sel, skip, rep,
                   rep_stride, kSlotCur, 2 * kv_dim, sc.channels, H, a.conv, 0, s->P, R, x_cur, H, st, true);
  }
}

void hc_layer(const Pass& ps, const HcBlock& hc) {
  const auto& v = ps.v;
  const auto& b = v.b;
  const int R = ps.rows, H = hc.hidden, C = hc.streams, W = C * H;
  if (hc.phase == FLYX_HC_EXPAND || hc.phase == FLYX_HC_SCATTER) {
    hc_rows(hc.phase, b.x, b.hc_wide, nullptr, b.hc_inject, nullptr, R, C, H, v.L.st);
    return;
  }
  for (int c = 0; c < C; ++c)
    rmsnorm(b.hc_wide + c*H, W, hc.norm.w+c*H, hc.norm.offset, hc.norm.eps, b.hc_norm+c*H, W, R, H, v.L.st);
  gemm(b.hc_norm,W,hc.down,hc.rank,W,b.act,hc.rank,R,false,v.L);
  hc_down_silu(b.act, R*hc.rank, C, v.L.st);
  gemm(b.act,hc.rank,hc.up,W,hc.rank,b.proj,W,R,false,v.L);
  if (hc.phase == FLYX_HC_MIX) gemm(b.hc_norm,W,hc.inject,C,W,b.hc_inject,C,R,false,v.L);
  hc_rows(hc.phase,b.x,b.hc_wide,b.hc_norm,b.hc_inject,b.proj,R,C,H,v.L.st);
}

void moe_layer(const Pass& ps, const MoeBlock& f, bool fused) {
  const RowView& rv = ps.v;
  auto* s = ps.s;
  const int H = f.hidden, I = f.inter, E = f.n_experts, K = f.top_k, R = ps.rows;
  cudaStream_t st = rv.L.st;
  const Launch& L = rv.L;

  // A fused round's kernels write what the next one reads, where the generic layer stages it from float32 rows in
  // a launch between them: the norm the gate|up GEMM's 8-bit rows beside the router's float32 ones, and that GEMM
  // the down GEMM's bf16 rows (in moe_act: the scratch's own bf16 rows lie over the 8-bit ones it is reading).
  const MoeScratch* msc = rv.b.moe_sc.pairs ? &rv.b.moe_sc : nullptr;
  const bool round_q8 = !f.body_only && !f.separate_gate_up && fused && ps.round && moe_round_q8(f.gate_up, R, K, H, msc);
  bool norm_done = false;
  const float* cur = rv.b.x;

  const bool has_pending = ps.res && ps.res->pending.post_raw != nullptr;
  const bool can_fuse_sub = has_pending && fused && ps.round && R <= FLYX_MAX_ROWS && f.res_query && rv.b.post_out &&
                            H <= 5120 && (ps.res->res_n + int(ps.res->pending.res_push) + 1) <= 64;
  if (can_fuse_sub) {
    const PendingPostConv pend = ps.res->pending;
    const SublayerPostConv spc = make_sublayer_post_conv(ps, pend);
    float* x_src = ps.res->x_cur ? ps.res->x_cur : rv.b.x;
    float* x_dst = (x_src == rv.b.x) ? rv.b.post_out : rv.b.x;
    SublayerNorm nmode = SublayerNorm::None;
    if (f.loop_rank == 0) nmode = (round_q8 && H % 32 == 0) ? SublayerNorm::MoeQ8 : SublayerNorm::F32;
    shortconv_res_mix_norm(spc, x_src, x_dst, rv.b.res_bank, ps.res->res_n, f.res_query, 1,
                           nmode == SublayerNorm::None ? rv.b.attn_in : nullptr, nmode, f.norm.w, f.norm.offset,
                           f.norm.eps, rv.b.h, H, msc, moe_q8_mins(f.gate_up), s->P, R, H, L);
    if (pend.res_push) ++ps.res->res_n;
    ps.res->pending = {};
    ps.res->x_cur = x_dst;
    norm_done = (nmode != SublayerNorm::None);
    cur = rv.b.attn_in;
  } else {
    if (has_pending) flush_pending_post_conv(ps, false);
    const float* cur_x = (ps.res && ps.res->x_cur) ? ps.res->x_cur : rv.b.x;
    cur = cur_x;
    if (f.res_query) {
      attn_res_mix(rv.b.res_bank, ps.res->res_n, cur_x, f.res_query, 1, R, H, rv.b.attn_in, st);
      cur = rv.b.attn_in;
    }
  }
  // The next attention's controllers, queries, ShortConv taps, sink, norm, relative bias and Q clip (option warm,
  // bit 16), while this MoE runs: 207.5 -> 211.5 tok/s, 35k-token prompt 159.5 -> 163.3 (its K and V blocks too: a
  // loss).
  if (s->expert_side.st && (opts(L).warm & 16) && ps.next_attn && ps.round) {
    const RelAttnBlock& a = *ps.next_attn;
    const size_t HH = size_t(a.hidden), kvd = size_t(a.nkv) * a.hd;
    L2Ranges ctrl, small;  // L2Ranges holds 8 ranges
    if (a.loop_rank > 0)
      for (int j = 0; j < 3; ++j) {
        const size_t rk = size_t(a.loop_rank);
        add_round(ctrl, a.loop_ctrl[j].down, rk * 2 * HH);
        if (a.loop_ctrl[j].stacked) add_round(ctrl, a.loop_ctrl[j].gain_forget, rk * 2 * HH);
      }
    small.add(a.res_query, a.res_query_bytes());
    small.add(a.k_conv, kvd * a.conv * 4);
    small.add(a.v_conv, kvd * a.conv * 4);
    small.add(a.post_conv, HH * a.conv * 4);
    small.add(a.w_sink.w, size_t(a.nh) * HH * 2);
    small.add(a.q_norm_in.w, HH * 4);
    small.add(a.rel_bias, size_t(a.rel_extent) * a.d_rel * 4);
    small.add(a.q_clip_w, size_t(a.nh) * a.hd * 4);
    warm_side(ps, ctrl);
    warm_side(ps, small);
  }
  if (f.loop_rank > 0) {
    const float* msg = rv.b.loop_msg + size_t(3) * R * H;
    if (loop_as_round(ps)) {
      const LoopGate gate = loop_gate(f.loop_ctrl, cur, msg, rv.b.attn_in);
      loop_control(&gate, 1, R, H, f.loop_rank, rv.b.ctrl_scratch, st);
      cur = rv.b.attn_in;
    } else if (fused) {
      apply_loop_ctrl_norm(cur, msg, R, H, f.loop_rank, f.loop_ctrl, f.norm, rv.b.ctrl_cat, rv.b.ctrl_gate,
                           rv.b.ctrl_gf, rv.b.h, H, L);
      norm_done = true;
    } else {
      apply_loop_ctrl(cur, msg, R, H, f.loop_rank, f.loop_ctrl, rv.b.ctrl_cat, rv.b.ctrl_gate, rv.b.ctrl_gf,
                      rv.b.attn_in, L);
      cur = rv.b.attn_in;
    }
  }

  if (f.body_only) {
    CK(cudaMemcpyAsync(rv.b.h, cur, size_t(R) * H * sizeof(float), cudaMemcpyDeviceToDevice, st));
  } else if (!norm_done) {
    if (round_q8)
      rmsnorm_moe_q8(cur, H, f.norm.w, f.norm.offset, f.norm.eps, rv.b.h, H, R, H, *msc, moe_q8_mins(f.gate_up), st);
    else
      rmsnorm(cur, H, f.norm.w, f.norm.offset, f.norm.eps, rv.b.h, H, R, H, st);
  }
  gemm(rv.b.h, H, f.router, E, H, rv.b.proj, E, R, false, L);
  moe_route(rv.b.proj, f.router_bias, R, E, K, f.gating_sigmoid, f.norm_weights, f.norm_eps, f.route_scale,
            rv.b.moe_idx, rv.b.moe_w, L);

  bf16* act_bf16 =
      round_q8 && moe_round_bf16_down(f.down, R, K, I, msc, L) ? reinterpret_cast<bf16*>(rv.b.moe_act) : nullptr;
  // The grouping of the rows by expert that the gate|up GEMM leaves in the scratch, which the down GEMM takes over.
  int grouped = 0;
  if (f.separate_gate_up) {
    // Each unfused projection writes I rows of its own dtype into one half of [R, K, 2I].
    moe_gate_up(rv.b.h, H, f.gate_up, rv.b.moe_idx, R, K, I / 2, H, 0, 0, false, rv.b.moe_gu, 2*I, msc, L);
    grouped = moe_gate_up(rv.b.h, H, f.up, rv.b.moe_idx, R, K, I / 2, H, 0, 0, false, rv.b.moe_gu+I, 2*I, msc, L);
    swiglu(rv.b.moe_gu, 2*I, I, rv.b.moe_act, I, R*K, st);
  } else if (fused) {
    grouped = moe_gate_up(rv.b.h, H, f.gate_up, rv.b.moe_idx, R, K, I, H, f.situ_beta, f.situ_linear_beta, true,
                          rv.b.moe_act, I, msc, L, round_q8, act_bf16);
  } else {
    grouped = moe_gate_up(rv.b.h, H, f.gate_up, rv.b.moe_idx, R, K, I, H, f.situ_beta, f.situ_linear_beta, false,
                          rv.b.moe_gu, 2 * I, msc, L);
    situ_rows(rv.b.moe_gu, 2 * I, I, f.situ_beta, f.situ_linear_beta, rv.b.moe_act, I, R * K, st);
  }
  if (f.post_conv) {
    moe_down(rv.b.moe_act, I, f.down, rv.b.moe_idx, rv.b.moe_w, R, K, H, I, rv.b.post_raw, H, false, msc, L, act_bf16,
             grouped);
    if (fused && ps.round && R <= FLYX_MAX_ROWS && ps.res && rv.b.post_out && H <= 5120) {
      ps.res->pending = PendingPostConv{rv.b.post_raw, f.post_conv, f.sconv, f.sconv_offset, f.conv, false};
      return;
    }
    ShortConvBuf& sc = s->sconv[size_t(f.sconv)];
    const bool direct = ps.round && ps.p->width == 0;
    const int next_sel = ps.round ? (direct ? kSlotNext : -1) : kSlotCur;
    const int skip = ps.round && direct ? kSkipIfStopped : kSkipNever;
    float* rep = ps.round ? sc.replay : nullptr;
    const size_t rep_stride = ps.round ? sc.r_stride : 0;
    float* x_cur = (ps.res && ps.res->x_cur) ? ps.res->x_cur : rv.b.x;
    shortconv_pass(rv.b.post_raw, H, f.post_conv, nullptr, sc.win, sc.w_stride, kSlotCur, next_sel, skip, rep,
                   rep_stride, kSlotCur, f.sconv_offset, sc.channels, H, f.conv, 0, s->P, R, x_cur, H, st, true);
  } else {
    float* x_cur = (ps.res && ps.res->x_cur) ? ps.res->x_cur : rv.b.x;
    moe_down(rv.b.moe_act, I, f.down, rv.b.moe_idx, rv.b.moe_w, R, K, H, I, x_cur, H, !f.body_only, msc, L, act_bf16,
             grouped);
  }
  if (f.shared_inter) {
    const int I_shared = f.shared_inter;
    gemm(rv.b.h,H,f.shared_gu,2*I_shared,H,rv.b.proj,2*I_shared,R,false,L);
    swiglu(rv.b.proj,2*I_shared,I_shared,rv.b.act,I_shared,R,st);
    gemm(rv.b.act,I_shared,f.shared_down,H,I_shared,rv.b.post_raw,H,R,false,L);
    shared_expert_add(rv.b.x,rv.b.h,f.shared_router,rv.b.post_raw,R,H,st);
  }
}

void dspark_context(const Pass& ps, const DSparkBlock& b, bool /*fused*/) {
  const RowView& rv = ps.v;
  auto* s = ps.s;
  const auto* m = s->m;
  const int H = b.hidden, R = ps.rows, nkv = b.nkv, hd = b.hd;
  const int sel = ps.round ? kPosStart : kPosPosition;
  cudaStream_t st = rv.L.st;
  const Launch& L = rv.L;

  const float* feats = rv.b.feats + b.feats_offset;
  dspark_ctx_mix_norm(feats, rv.ld_feats, b.markov_ctx_mix, b.markov_ctx_norm.w, b.markov_ctx_norm.eps, b.n_taps,
                      R, H, rv.b.cx, st);
  if (!ps.round && s->pass_snap > 0 && s->ring_snap) {
    const int snap_row = s->pass_snap - s->position;
    const float* src = snap_row == 0 ? s->ring + size_t((s->pass_snap - 1) % kHring) * size_t(H)
                                     : rv.b.cx + size_t(snap_row - 1) * size_t(H);
    CK(cudaMemcpyAsync(s->ring_snap, src, size_t(H) * sizeof(float), cudaMemcpyDeviceToDevice, st));
  }
  hring_put(s->P, sel, ps.round, rv.b.cx, H, R, s->ring, H, st);

  for (int j = 0; j < b.layers(); ++j) {
    const auto& l = block<DSparkLayer>(m, b.first_layer + j);
    const KvBuf& kv = s->kv[size_t(l.kv)];
    dspark_ctx_mix_norm(feats, rv.ld_feats, l.ctx_mix, l.ctx_norm.w, l.ctx_norm.eps, b.n_taps, R, H, rv.b.cx, st);
    if (l.stacked_ctx_kv) {
      gemm(rv.b.cx, H, l.w_ctx_kv, 2 * nkv * hd, H, rv.b.ckv, rv.ld_ckv, R, false, L);
    } else {
      gemm(rv.b.cx, H, l.w_k_ctx, nkv * hd, H, rv.b.ckv, rv.ld_ckv, R, false, L);
      gemm(rv.b.cx, H, l.w_v_ctx, nkv * hd, H, rv.b.ckv + nkv * hd, rv.ld_ckv, R, false, L);
    }
    draft_ctx_kv(rv.b.ckv, rv.ld_ckv, l.k_ctx_norm.w, l.k_ctx_norm.eps, b.theta, 1, nkv, hd, R, s->P, sel, ps.round,
                 kv.k16(0), kv.v16(0), kv.layer_stride, st, kv.ring);
  }
}

void dspark_draft(flyx_seq* s, const DSparkBlock& b, int width, const Launch& L, bool fused, int start,
                  int total_width) {
  const auto* m = s->m;
  const auto& E = block<EmbedBlock>(m, b.embed);
  const auto& head = block<HeadBlock>(m, b.head);
  const int total = total_width > 0 ? total_width : width;
  const int B = std::min(b.block, total + 1);
  const int H = b.hidden, nh = b.nh, nkv = b.nkv, hd = b.hd, I = b.inter;
  const int V = head.vocab, Rk = b.markov_rank, nqkv = (nh + 2 * nkv) * hd;
  const int nr_total = std::min(total, B - 1);
  const int nr_end = std::min(width, nr_total);
  const bool two_stage = draft_head_additive_ready(head, b.markov_w2, b.markov_coarse, Rk);
  const int ld = two_stage ? std::max(V, kDraftPickFloats) : V;
  cudaStream_t st = L.st;

  if (start == 0) {
    // Row 0: raw anchor embedding -> DSpark tok_norm; rows 1..B-1: mask_embd[0..B-2]; init prev & Markov state.
    build_rows(s->P, nullptr, 0, s->dtok, 1, st);
    embed_tokens(E, s->dtok, s->ds_h, 1, st);
    dspark_prep_block(s->P, s->ds_h, b.tok_norm.w, b.tok_norm.eps, b.mask_embd, B, H, s->ds_x, s->ds_prev, s->ds_state,
                      Rk, st);

    const float scale = 1.f / std::sqrt(float(hd));
    for (int j = 0; j < b.layers(); ++j) {
      const auto& l = block<DSparkLayer>(m, b.first_layer + j);
      const KvBuf& kv = s->kv[size_t(l.kv)];
      const int eff_win = l.window > 0 ? l.window : kv.window;
      if (l.stacked_qkv) {
        norm_project(s->ds_x, s->ds_h, s->ds_proj, s->ld_ds_proj, B, l.norm, l.w_qkv, nqkv, H, L, fused);
      } else {
        rmsnorm(s->ds_x, H, l.norm.w, l.norm.offset, l.norm.eps, s->ds_h, H, B, H, st);
        gemm(s->ds_h, H, l.w_q, nh * hd, H, s->ds_proj, s->ld_ds_proj, B, false, L);
        gemm(s->ds_h, H, l.w_k, nkv * hd, H, s->ds_proj + nh * hd, s->ld_ds_proj, B, false, L);
        gemm(s->ds_h, H, l.w_v, nkv * hd, H, s->ds_proj + (nh + nkv) * hd, s->ld_ds_proj, B, false, L);
      }
      attn_prep(s->ds_proj, s->ld_ds_proj, 0, nh * hd, (nh + nkv) * hd, l.q_norm.w, l.k_norm.w, l.q_norm.offset,
                l.q_norm.eps, b.theta, hd, nh, nkv, hd, B, s->P, kPosPosition, false, s->ds_q, kv.draft_ref(0), st);
      dspark_block_attention(s->ds_q, kv.k16(0), kv.v16(0), nh, nkv, hd, B, eff_win, scale, s->P, s->ds_act, nh * hd,
                             s->own.part, kRoundSplits, L, kv.ring);
      gemm(s->ds_act, nh * hd, l.w_o, H, nh * hd, s->ds_x, H, B, true, L);
      if (l.stacked_gu) {
        norm_project(s->ds_x, s->ds_h, s->ds_proj, s->ld_ds_proj, B, l.post_norm, l.w_gu, 2 * I, H, L, fused);
      } else {
        rmsnorm(s->ds_x, H, l.post_norm.w, l.post_norm.offset, l.post_norm.eps, s->ds_h, H, B, H, st);
        gemm(s->ds_h, H, l.w_gate, I, H, s->ds_proj, s->ld_ds_proj, B, false, L);
        gemm(s->ds_h, H, l.w_up, I, H, s->ds_proj + I, s->ld_ds_proj, B, false, L);
      }
      if (fused) {
        swiglu_split(s->ds_proj, s->ld_ds_proj, I, B, L, l.w_down.q8_inputs());
        gemm_presplit(l.w_down, H, I, s->ds_x, H, B, true, L);
      } else {
        swiglu(s->ds_proj, s->ld_ds_proj, I, s->ds_act, s->ld_ds_act, B, st);
        gemm(s->ds_act, s->ld_ds_act, l.w_down, H, I, s->ds_x, H, B, true, L);
      }
    }

    // The drafts' logits are softcap(head(hn) + markov_w2(markov) + bias), a step's Markov vector made from the draft
    // before it.  With 1-bit copies of both heads (loader/weights.cpp) the rows' rough head scores are made at once and
    // each step adds its rough Markov scores, picks its candidates from the sum and scores only those exactly on both
    // heads (draft_head_pick_additive_step); picked from the head's rough scores alone, the drafts the Markov head
    // moves up were lost.  Otherwise both heads are read whole: the head once for every row, the Markov head a step.
    rmsnorm(s->ds_x + H, H, b.norm.w, b.norm.offset, b.norm.eps, s->ds_hn, H, nr_total, H, st);
    if (two_stage) gemm_sign_rough(s->ds_hn, H, head.coarse, V, H, s->logits, ld, nr_total, L);
    else draft_logits(head, s->ds_hn, H, s->logits, nr_total, L);
  }
  for (int step = start; step < nr_end; ++step) {
    if (b.markov_w1_blk.data) embed_blocks(s->ds_prev, b.markov_w1_blk, Rk, s->ds_joint + Rk, 1, st);
    else embed(s->ds_prev, b.markov_w1, Rk, s->ds_joint + Rk, 1, st);
    dspark_markov_pack(s->P, s->ds_state, nullptr, s->ds_hn + size_t(step) * H, s->ring, Rk, H, s->ds_joint, st);
    gemm(s->ds_joint, 2 * Rk + 2 * H, b.joint_proj, 3 * Rk, 2 * Rk + 2 * H, s->ds_jout, 3 * Rk, 1, false, L);
    dspark_markov_gate(s->ds_jout, b.joint_proj_b, Rk, s->ds_state, s->ds_markov, st);
    float* row_logits = s->logits + size_t(step) * ld;
    if (two_stage) {
      draft_head_pick_additive_step(head, s->ds_hn + size_t(step) * H, b.markov_w2, b.markov_coarse, b.markov_w2_b,
                                    s->ds_markov, Rk, b.softcap, row_logits, ld, s->ds_bias, 1, s->ds_prev, nullptr,
                                    s->draft_prob + step, s->draft_margin + step, L, s->P, 1 + step, s->G);
    } else {
      gemm(s->ds_markov, Rk, b.markov_w2, V, Rk, s->ds_bias, V, 1, false, L);
      dspark_combine_logits(row_logits, s->ds_bias, b.markov_w2_b, V, b.softcap, row_logits, st);
      topk_rows(row_logits, V, V, 1, s->ds_prev, nullptr, 1, st, s->topv, s->topi, s->top_rows, s->draft_prob + step,
                s->draft_margin + step);
      sample_rows(row_logits, V, V, 1, s->P, 1 + step, s->G, nullptr, s->cz, s->ci, s->ds_prev, st);
    }
    dspark_pick_step(s->P, s->oracle, step, s->ds_prev, s->drafts, st);
  }
}

void head(const Pass& ps, const HeadBlock& h, bool last_only) {
  const RowView& rv = ps.v;
  auto* s = ps.s;
  const int V = h.vocab;
  const int r0 = last_only ? ps.rows - 1 : 0, n = last_only ? 1 : ps.rows;
  cudaStream_t st = rv.L.st;
  if (ps.round && n > FLYX_MAX_ROWS) {
    // A wide round: the head and the choice 16 rows at a time (the logits hold a round's 16), each row's choice what
    // a round of 16 makes.  A tree's row chooses at its depth's position and its penalties count its ancestors'
    // drafts; for the relaxed rules each row's statistics, and each draft's logit in its parent's row (child_logits:
    // what accept_tree reads of the logits), are kept as the rows go.
    const TreeRows* tree = tree_of(ps);
    for (int a = 0; a < n; a += FLYX_MAX_ROWS) {
      const int k = std::min(FLYX_MAX_ROWS, n - a);
      head_logits(h, rv, a, k, ps.cert, s->cert, s->m->opt.kernels);
      if (h.softcap > 0.f) softcap_rows(rv.logits, V, V, k, h.softcap, st);
      if (ps.pen) penalize_rows(rv.logits, V, V, k, s->G, s->reply, s->drafts, st, tree, a);
      choose_rows(rv.logits, V, V, k, s->P, a + 1, s->G, a == 0 ? s->mask : nullptr, s->cz, s->ci, s->argv, s->argi,
                  rv.b.choice + a, st, tree, a);  // the mask is the round's first row's
      if (ps.relaxed) {
        row_stats(rv.logits, V, V, k, s->G, s->accept_stats + 3 * a, st);
        child_logits(rv.logits, V, a, k, s->drafts, ps.p->width, tree, s->child_z, st);
      }
    }
    return;
  }
  head_logits(h, rv, r0, n, ps.cert, s->cert, s->m->opt.kernels, residual16(ps));
  if (h.softcap > 0.f) softcap_rows(rv.logits, V, V, n, h.softcap, st);
  // Penalties first (a request that has one): in a round, row r was given the
  // drafts before it, which count as the reply for it.
  // A tree's row: its ancestors' drafts, its depth's position.
  const TreeRows* tree = tree_of(ps);
  if (ps.pen) penalize_rows(rv.logits, V, V, n, s->G, s->reply, ps.round ? s->drafts : nullptr, st, tree);
  // Row r of the logits picks the token at position P->position + r0 + 1 + r.
  choose_rows(rv.logits, V, V, n, s->P, r0 + 1, s->G, s->mask, s->cz, s->ci, s->argv, s->argi, rv.b.choice, st, tree);
  if (ps.relaxed) row_stats(rv.logits, V, V, n, s->G, s->accept_stats, st);
}

void draft_context(const Pass& ps, const DFlashBlock& b, bool fused) {
  const RowView& rv = ps.v;
  auto* s = ps.s;
  const KvBuf& kv = s->kv[size_t(b.kv)];
  const int H = b.hidden, R = ps.rows, F = b.n_taps * b.target_hidden;
  cudaStream_t st = rv.L.st;
  const Launch& L = rv.L;
  gemm(rv.b.feats + b.feats_offset, rv.ld_feats, b.fc, H, F, rv.b.cx, H, R, false, L);
  norm_project(rv.b.cx, rv.b.h, rv.b.ckv, rv.ld_ckv, R, b.hidden_norm, b.ctx, b.ctx_width(), H, L, fused);
  draft_ctx_kv(rv.b.ckv, rv.ld_ckv, b.k_norms.w, b.k_norms.eps, b.theta, b.layers(), b.nkv, b.hd, R, s->P,
               ps.round ? kPosStart : kPosPosition, ps.round, kv.k16(), kv.v16(), kv.layer_stride, st, kv.ring);
}

// The selector's table over the slots the tree reaches (slots - 1 at most: the last slot never has candidates after
// a slide), then the tree.  A canvas that follows another round drafts DV's first tree; the test's oracle a chain.
void dv_draft(flyx_seq* s, const DvBlock& v, int width, const Launch& L) {
  flyx_seq::DvRows& d = s->dv;
  const int depths = std::min(v.slots - 1, width);
  cudaStream_t st = L.st;
  dv_selector_table(s->P, d.canvas, v.sel_ta, v.sel_tb, v.sel_rank, v.sel_alpha, v.sel_temp, depths, d.table, st);
  if (v.tree == FLYX_DV_TREE_BEAM)
    dv_tree_beam(s->P, d.canvas, d.table, width, v.beam, v.cov, depths, v.mask_token, s->oracle, s->drafts, nullptr,
                 s->tree, st);
  else
    dv_tree_pool(s->P, d.canvas, d.table, width, v.pool_ranks, depths, v.mask_token, s->oracle, s->drafts, nullptr,
                 s->tree, st);
}

// The canvas pass: slot w a row at P->start + 1 + w (P->start the round's anchor, its K/V written by the round's
// pass and never moved by the compaction), through the target's attention and MLP blocks with their base weights
// (never an adapter of the target's own: bifill-dv's canvas rows call the base linears) and the layer's canvas
// norms; each linear's adapter as two more GEMMs whose product adds to the base's (y = W x + scale b (a x)).  Its K/V
// go to the scratch slots, its attention reads the text up to the anchor and those slots (CanvasKeys), never the
// low-bit copy.  K/V in tiers: through the
// views, the text's host part staged once a layer (stage_text) and the slots written to the staging itself (only this
// pass reads them); read in place, every canvas row fetched each of the text's keys over the bus (MiniCPM5-1B, 4273
// tokens in host memory: a round of 221 ms, 35 staged).  Then fnormc, the head's weight over the slots in float32,
// what they leave, and the slide.
// The pass's GEMMs past 16 rows take bf16 inputs (Launch::draft): only drafting reads the canvas, which must be
// deterministic, not row-count invariant (numerics.md).  Past 16 slots and up to 64, on bf16 weights, each input is
// rounded to bf16 once (by its producer: rmsnorm_split, swiglu_split, the adapters' a; the attention's output by
// stage_bf16) and a layer's GEMMs on the same rows go in one launch each (gemm_draft): a base weight with its adapter's
// a, the adapters' b side by side.  A fused region as the others are: the arithmetic of each GEMM on its own (one part,
// the chunks K gives it, whatever the table), so option fusion=0 runs them one by one for the same canvas.
namespace {
MtOut draft_out(const Lin& W, int N, float* y, int ldy, bool accumulate, int x_col = 0) {
  return MtOut{W.w, y, nullptr, N, ldy, x_col, int(accumulate), 0};
}
MtOut draft_out16(const Lin& W, int N, bf16* y16, int ldy) { return MtOut{W.w, nullptr, y16, N, ldy, 0, 0, 0}; }
MtGroups draft_group(std::initializer_list<MtOut> outs) {
  MtGroups g;
  for (const MtOut& o : outs) g.g[g.n++] = o;
  return g;
}
bool bf16_lin(const Lin& W) { return W.w && !W.q && !W.packed(); }
bool drafts_grouped(const flyx_bmodel* m, const DvBlock& v) {
  if (!m->opt.fusion || v.slots <= 16 || v.slots > 64) return false;
  for (int j = 0; j < v.n_layers; ++j) {
    const DvLayer& l = block<DvLayer>(m, v.first_layer + j);
    const AttnBlock& a = block<AttnBlock>(m, l.attn);
    const MlpBlock& f = block<MlpBlock>(m, l.mlp);
    if (l.scale != 1.f) return false;  // the a's write bf16 products, no scale step (bifill-dv's adapters have none)
    for (const Lin* W : {&a.w_in, &a.w_out, &f.w_gu, &f.w_down, &l.a_qkv, &l.a_o, &l.a_gu, &l.a_down, &l.b_q, &l.b_k,
                         &l.b_v, &l.b_o, &l.b_gate, &l.b_up, &l.b_down})
      if (!bf16_lin(*W)) return false;
    if (a.hidden % 32 || a.nh * a.hd % 32 || f.inter % 32 || l.rank % 32) return false;
  }
  return true;
}
}  // namespace

void dv_context(const Pass& ps, const DvBlock& v) {
  auto* s = ps.s;
  const auto* m = s->m;
  flyx_seq::DvRows& d = s->dv;
  const auto& E = block<EmbedBlock>(m, v.embed);
  const auto& head = block<HeadBlock>(m, v.head);
  const int N = v.slots, H = v.hidden, ldp = m->need.proj, lda = m->need.act, ldl = d.ld_lora;
  Launch L{ps.v.L.st, &d.scratch, ps.v.L.opt, nullptr, ps.v.L.side};
  L.draft = true;
  cudaStream_t st = L.st;
  const CanvasKeys keys{d.s0};
  const bool grouped = drafts_grouped(m, v);
  const bf16* rows16 = d.scratch.abuf;  // where the producers leave a grouped GEMM's bf16 input
  canvas_input(s->P, d.canvas, N, v.mask_token, E.table, v.mask_emb, v.res_in.w, v.res_out.w, H, d.ug, d.x, L);
  for (int j = 0; j < v.n_layers; ++j) {
    const DvLayer& l = block<DvLayer>(m, v.first_layer + j);
    const AttnBlock& a = block<AttnBlock>(m, l.attn);
    const MlpBlock& f = block<MlpBlock>(m, l.mlp);
    const int r = l.rank, nh = a.nh, nkv = a.nkv, hd = a.hd, qw = nh * hd, kvw = nkv * hd, I = f.inter;
    const KvBuf& kb = s->kv[size_t(a.kv)];
    const KvRef kv = kb.staged();  // the views, FP8 scales past the device part in their staging
    if (grouped) {
      rmsnorm_split(d.x, H, l.norm.w, l.norm.offset, l.norm.eps, N, H, L, false);
      gemm_draft(rows16, H, H, draft_group({draft_out(a.w_in, a.n_in(), d.proj, ldp, false),
                                            draft_out16(l.a_qkv, 3 * r, d.lora16, ldl)}), N, L);
      gemm_draft(d.lora16, ldl, r, draft_group({draft_out(l.b_q, qw, d.proj, ldp, true),
                                                draft_out(l.b_k, kvw, d.proj + qw, ldp, true, r),
                                                draft_out(l.b_v, kvw, d.proj + qw + kvw, ldp, true, 2 * r)}), N, L);
    } else {
      rmsnorm(d.x, H, l.norm.w, l.norm.offset, l.norm.eps, d.h, H, N, H, st);
      gemm(d.h, H, a.w_in, a.n_in(), H, d.proj, ldp, N, false, L);
      lora_down(d.h, H, l.a_qkv.w, 3 * r, H, l.scale, d.lora, ldl, N, L);
      gemm(d.lora, ldl, l.b_q, qw, r, d.proj, ldp, N, true, L);
      gemm(d.lora + r, ldl, l.b_k, kvw, r, d.proj + qw, ldp, N, true, L);
      gemm(d.lora + 2 * r, ldl, l.b_v, kvw, r, d.proj + qw + kvw, ldp, N, true, L);
    }
    attn_prep(d.proj, ldp, 0, qw, qw + kvw, a.q_norm.w, a.k_norm.w, a.q_norm.offset, a.q_norm.eps, a.theta, a.rot, nh,
              nkv, hd, N, s->P, kPosStart, false, d.q, kv, st, 0, nullptr, 0, keys, a.mrope());
    s->stage_text(kb, st);
    attention(d.q, kv, nh, nkv, hd, N, v.causal, N, 1.f / std::sqrt(float(hd)), s->P, nullptr, 0, d.act, qw, d.part,
              d.splits, L, 0, 0, nullptr, keys);
    if (grouped) {
      gemm_draft(stage_bf16(d.act, qw, N, qw, L), qw, qw,
                 draft_group({draft_out(a.w_out, H, d.x, H, true), draft_out16(l.a_o, r, d.lora16, ldl)}), N, L);
      gemm_draft(d.lora16, ldl, r, draft_group({draft_out(l.b_o, H, d.x, H, true)}), N, L);
      rmsnorm_split(d.x, H, l.post_norm.w, l.post_norm.offset, l.post_norm.eps, N, H, L, false);
      gemm_draft(rows16, H, H, draft_group({draft_out(f.w_gu, 2 * I, d.proj, ldp, false),
                                            draft_out16(l.a_gu, 2 * r, d.lora16, ldl)}), N, L);
      gemm_draft(d.lora16, ldl, r, draft_group({draft_out(l.b_gate, I, d.proj, ldp, true),
                                                draft_out(l.b_up, I, d.proj + I, ldp, true, r)}), N, L);
      swiglu_split(d.proj, ldp, I, N, L, false);
      gemm_draft(rows16, I, I, draft_group({draft_out(f.w_down, H, d.x, H, true),
                                            draft_out16(l.a_down, r, d.lora16, ldl)}), N, L);
      gemm_draft(d.lora16, ldl, r, draft_group({draft_out(l.b_down, H, d.x, H, true)}), N, L);
      continue;
    }
    gemm(d.act, qw, a.w_out, H, qw, d.x, H, N, true, L);
    lora_down(d.act, qw, l.a_o.w, r, qw, l.scale, d.lora, ldl, N, L);
    gemm(d.lora, ldl, l.b_o, H, r, d.x, H, N, true, L);
    rmsnorm(d.x, H, l.post_norm.w, l.post_norm.offset, l.post_norm.eps, d.h, H, N, H, st);
    gemm(d.h, H, f.w_gu, 2 * I, H, d.proj, ldp, N, false, L);
    lora_down(d.h, H, l.a_gu.w, 2 * r, H, l.scale, d.lora, ldl, N, L);
    gemm(d.lora, ldl, l.b_gate, I, r, d.proj, ldp, N, true, L);
    gemm(d.lora + r, ldl, l.b_up, I, r, d.proj + I, ldp, N, true, L);
    swiglu(d.proj, ldp, I, d.act, lda, N, st);
    gemm(d.act, lda, f.w_down, H, I, d.x, H, N, true, L);
    lora_down(d.act, lda, l.a_down.w, r, I, l.scale, d.lora, ldl, N, L);
    gemm(d.lora, ldl, l.b_down, H, r, d.x, H, N, true, L);
  }
  rmsnorm(d.x, H, v.norm.w, v.norm.offset, v.norm.eps, d.h, H, N, H, st);
  gemm(d.h, H, head.w, v.vocab, H, d.logits, v.vocab, N, false, L);
  canvas_stats(d.logits, v.vocab, v.vocab, E.table, d.h, H, H, v.sel_h, v.sel_rank, N, d.stats, st);
  dv_slide(s->P, d.stats, N, H, v.sel_rank, v.fill, v.mask_token, d.canvas, st);
}

// The canvas pass of a round captured as a graph, as the body of a conditional node that dv_live sets from P->noop: a
// round launched behind the one that ended the reply (it commits nothing, and its pass would leave the canvas as it
// was) skips it (the reply's last round else costs a canvas pass more).  Launched eagerly, it runs (dv_slide moves
// nothing then).
void dv_context_unless_noop(const Pass& ps, const DvBlock& v) {
  auto* s = ps.s;
  cudaStream_t st = ps.v.L.st;
  cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
  cudaGraph_t graph = nullptr;
  const cudaGraphNode_t* deps = nullptr;
  size_t n_deps = 0;
  if (!s->dv.body || !s->dv.skip_noop || cudaStreamIsCapturing(st, &capturing) != cudaSuccess ||
      capturing != cudaStreamCaptureStatusActive)
    return dv_context(ps, v);
  cudaGraphConditionalHandle live{};
  CK(cudaStreamGetCaptureInfo(st, &capturing, nullptr, &graph));
  CK(cudaGraphConditionalHandleCreate(&live, graph, 0, 0));
  dv_live(live, s->P, st);
  CK(cudaStreamGetCaptureInfo(st, &capturing, nullptr, &graph, &deps, nullptr, &n_deps));
  cudaGraphNodeParams cond{};
  cond.type = cudaGraphNodeTypeConditional;
  cond.conditional.handle = live;
  cond.conditional.type = cudaGraphCondTypeIf;
  cond.conditional.size = 1;
  cudaGraphNode_t node = nullptr;
  CK(cudaGraphAddNode(&node, graph, deps, nullptr, n_deps, &cond));
  CK(cudaStreamUpdateCaptureDependencies(st, &node, nullptr, 1, cudaStreamSetCaptureDependencies));
  CK(cudaStreamBeginCaptureToGraph(s->dv.body, cond.conditional.phGraph_out[0], nullptr, nullptr, 0,
                                   cudaStreamCaptureModeThreadLocal));
  Pass body = ps;
  body.v.L.st = s->dv.body;
  cudaGraph_t captured = nullptr;
  try {
    dv_context(body, v);
  } catch (...) {
    cudaStreamEndCapture(s->dv.body, &captured);
    throw;
  }
  CK(cudaStreamEndCapture(s->dv.body, &captured));
}

void compact_path(const Pass& ps) {
  const RowView& rv = ps.v;
  auto* s = ps.s;
  const auto* m = s->m;
  cudaStream_t st = rv.L.st;
  compact_kv(s->tree_kv, s->n_tree_kv, s->tree_kv_bytes, s->P, s->tree, st);
  if (s->gdn_layers) {
    gdn_compact(s->gdn_layers, int(s->gdn.size()), s->gdn[0].replay, s->P, s->tree, st);
  } else {
    for (const auto& g : s->gdn) gdn_compact(nullptr, 1, g.replay, s->P, s->tree, st);
  }
  // What the drafters' context reads: the rows' tokens, final hidden states and taps.
  compact_rows(rv.b.tokens, 1, 1, s->P, s->tree, st);
  compact_rows(rv.b.x, size_t(m->hidden), m->hidden, s->P, s->tree, st);
  if (rv.b.feats) compact_rows(rv.b.feats, size_t(rv.ld_feats), rv.ld_feats, s->P, s->tree, st);
}

// Fold every recurrent state to the round's accepted rows.
void commit(const Pass& ps) {
  const RowView& rv = ps.v;
  auto* s = ps.s;
  if (ps.p->width == 0) return;  // plain rounds wrote the next window and state in the pass
  if (!s->sconv.empty()) {
    if (s->sconv_alike) {
      const ShortConvBuf& f = s->sconv[0];
      shortconv_fold(s->sconv_layers, int(s->sconv.size()), f.w_stride, f.r_stride, kSlotLast, kSlotAfterLast,
                     kSlotLast, f.channels, f.conv, s->P, 0, &s->P->last_n, true, rv.L.st);
    } else {
      for (size_t i = 0; i < s->sconv.size(); ++i) {
        const ShortConvBuf& sc = s->sconv[i];
        shortconv_fold(s->sconv_layers + i, 1, sc.w_stride, sc.r_stride, kSlotLast, kSlotAfterLast, kSlotLast,
                       sc.channels, sc.conv, s->P, 0, &s->P->last_n, true, rv.L.st);
      }
    }
  }
  if (s->gdn.empty()) return;
  // Every layer in two launches when they are alike (the same arithmetic as a layer at a time).
  if (s->gdn_layers) {
    const GdnBuf& g = s->gdn[0];
    if (gdn_fold_layers(s->gdn_layers, int(s->gdn.size()), g.s_stride, g.w_stride, g.replay, kSlotLast, kSlotAfterLast,
                        kSlotLast, s->P, &s->P->last_n, rv.L.st))
      return;
  }
  for (auto& g : s->gdn) {
    gdn_recur(g.S, g.s_stride, kSlotLast, kSlotAfterLast, g.replay, kSlotLast, s->P, nullptr, 0, &s->P->last_n,
              kSkipIfNoop, rv.L.st);
    gdn_conv_fold(g.win, g.w_stride, kSlotLast, kSlotAfterLast, g.replay, kSlotLast, s->P, 0, &s->P->last_n, true,
                  rv.L.st);
  }
}

}  // namespace flyx::cuda
