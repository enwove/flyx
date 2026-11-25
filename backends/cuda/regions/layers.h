// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// What each program node runs on its block (regions/layers.cpp).  fused = the
// override plan.cpp may pick: the producers write the next GEMM's input already
// rounded (split for 1..16 rows, bf16 for more), so the GEMM reads it as is; the
// result is the generic one bit for bit.
#pragma once

#include "backends/cuda/backend.h"

namespace flyx::cuda {

// Shared layer building blocks used by both single-sequence (layers.cpp) and multi-sequence batch (batch.cpp) paths.
inline void embed_tokens(const EmbedBlock& e, const int* tokens, float* x, int rows, cudaStream_t st) {
  if (e.blk.data) embed_blocks(tokens, e.blk, e.hidden, x, rows, st);
  else embed(tokens, e.table, e.hidden, x, rows, st);
}
// A prefill pass whose residual rows are bf16 (as vLLM, SGLang and basalt keep them): more than 16 rows, the fused
// producers, NVFP4 inputs in the scratch, and a model whose every residual add is a W4A4 GEMM (layers.cpp).  The
// rows sit in RowBufs::x's memory (x16_of).
bool residual16(const Pass& ps);
// The model's part of it, decided once (flyx_bmodel::res16): open_model reports it (-v).
bool residual16_model(flyx_bmodel* m);
inline bf16* x16_of(const RowView& v) { return reinterpret_cast<bf16*>(v.b.x); }
void norm_project(const float* x, float* h, float* out, int ld_out, int R, const Norm& n, const Lin& w, int N, int H,
                  const Launch& L, bool fused);
void norm_and_project(const RowView& v, int R, const Norm& n, const Lin& w, int N, int H, bool fused);
void mtp_draft_attention(flyx_seq* s, const AttnBlock& a, const KvBuf& kv, int pos_offset, const float* proj,
                         int ld_proj, float* q, float* out, float* part, const Launch& L);
void mlp_body(float* x, const MlpBlock& f, int R, float* h, float* proj, int ld_proj, float* act, int ld_act,
              const Launch& L, bool fused);
void gdn_out_project(const RowView& v, const GdnBlock& g, int R, bool fused);
void head_logits(const HeadBlock& h, const RowView& v, int r0, int n, bool cert, const CertScratch& cs,
                 const KernelOptions& ko, bool x16 = false);
// Where a draft head's pick keeps its scores and its choice's scratch: a sequence's own, or (logits given) a batch's
// scores with one of its sequences' choice scratch, the batch's rows sharing a launch.
struct DraftPickScratch {
  float* logits;         // rows x max(vocab, kDraftPickFloats)
  float* argv;           // argmax_rows'
  int* argi;
  float* topv;           // topk_rows'
  int* topi;
  int top_rows;
};
inline DraftPickScratch pick_scratch(const flyx_seq* s, float* logits = nullptr) {
  return {logits ? logits : s->logits, s->argv, s->argi, s->topv, s->topi, s->top_rows};
}
// The drafts' top k of nr rows from the head (k 1 without values: their argmax): its 1-bit copy's candidates scored
// exactly on a packed or a bf16 head, else its draft logits whole.  exact_1row: each of several rows gets the rough
// scores it has alone (a batch's rows, one a sequence: the drafts a sequence has by itself).
void draft_head_pick(const HeadBlock& head, const float* x, int nr, int k, int* idx, float* val,
                     const DraftPickScratch& sc, const Launch& L, bool exact_1row = false, float* prob = nullptr,
                     float* margin = nullptr);
inline void draft_head_pick(const HeadBlock& head, const float* x, int nr, int k, int* idx, float* val, flyx_seq* s,
                            const Launch& L, float* prob = nullptr, float* margin = nullptr) {
  draft_head_pick(head, x, nr, k, idx, val, pick_scratch(s), L, false, prob, margin);
}
// Shared two-stage rough+exact pick for additive draft heads: softcap(head(x_base) + extra(x_extra) + bias).
bool draft_head_additive_ready(const HeadBlock& head, const Lin& extra, const SignHead& extra_coarse, int K_extra);
void draft_head_pick_additive_step(const HeadBlock& head, const float* x_base, const Lin& extra,
                                   const SignHead& extra_coarse, const float* extra_bias, const float* x_extra,
                                   int K_extra, float softcap, float* cand_row, int ld, float* scratch_add, int k,
                                   int* idx_out, float* val_out, float* prob_out, float* margin_out, const Launch& L);

// Drafting works on the sequence's own buffers and scratch, on the stream of L (the sequence's, or a batch round's).
// DFlash2: one block of drafts; tree: `width` of its candidates as a tree (FLYX_DRAFT_TREE), the TreeRows with them
void draft_block(flyx_seq* s, const DFlashBlock& b, int width, const Launch& L, bool tree = false, bool fused = false);
void mtp_draft(flyx_seq* s, const MtpBlock& b, int width, const Launch& L, bool fused = false, int start = 0);  // MTP: draft steps [start, width)
void mtp_context(const Pass& ps, const MtpBlock& b, bool fused = false);  // MTP K/V for the rows of a pass
void gdn_layer(const Pass& ps, const GdnBlock& g, bool fused = false);
void attn_layer(const Pass& ps, const AttnBlock& a, bool fused = false);
void rel_attn_layer(const Pass& ps, const RelAttnBlock& a, bool fused = false);
void moe_layer(const Pass& ps, const MoeBlock& f, bool fused = false);
void dspark_draft(flyx_seq* s, const DSparkBlock& b, int width, const Launch& L, bool fused = false);
void dspark_context(const Pass& ps, const DSparkBlock& b, bool fused = false);
// A block's own adapters (ABI 43) on the linears that read one input (x null: a fused producer's staged rows; else
// ldx apart), added to rv.b.proj's columns in turn (out[i] each), and on its last linear, added to rv.b.x; ldu:
// RowNeeds::lora.
void lora_in(const RowView& rv, const Adapters& l, int linears, const int* out, const float* x, int ldx, int K, int R,
             int ldu, const Launch& L);
void lora_out(const RowView& rv, const Adapters& l, int linear, const float* x, int ldx, int K, int H, int R, int ldu,
              const Launch& L);
void mlp_layer(const Pass& ps, const MlpBlock& f, bool fused = false);
void head(const Pass& ps, const HeadBlock& h, bool last_only);  // logits and the token choice
void draft_context(const Pass& ps, const DFlashBlock& b, bool fused = false);  // DFlash2 context K/V from the target taps
// DV (docs/design/dv.md): the round's tree of `width` drafts from the canvas (DRAFT), the TreeRows with them; after
// the round's accept, the canvas pass and the slide by the tokens it committed (CONTEXT).
void dv_draft(flyx_seq* s, const DvBlock& v, int width, const Launch& L);
void dv_context(const Pass& ps, const DvBlock& v);
void dv_context_unless_noop(const Pass& ps, const DvBlock& v);
void commit(const Pass& ps);  // fold the accepted rows into the recurrent state
// A tree round's tree (null: a chain round's pass).
inline const TreeRows* tree_of(const Pass& ps) { return ps.round && ps.p->tree ? ps.s->tree : nullptr; }
// After a tree round's accept: its accepted path to rows 0 .. n in every cache and buffer read after it.
void compact_path(const Pass& ps);

}  // namespace flyx::cuda
