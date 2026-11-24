// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Prefill passes and rounds (graphs, the token ring), waiting for the device, rollback, checkpoints and their
// snapshots in host memory.
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>

#include "backends/common/snapshot_format.h"
#include "backends/cuda/backend.h"
#include "backends/cuda/runtime/strata_snapshot.h"

namespace flyx::cuda {

namespace {

void copy_ring(flyx_seq* s, cudaStream_t st) {
  CK(cudaMemcpyAsync(s->ring_host, s->ring_dev, sizeof(flyx_round_result) * kRing, cudaMemcpyDeviceToHost, st));
}

void copy_ring(flyx_seq* s) { copy_ring(s, s->st); }

template <class Fn>
int guarded(Fn&& fn) {
  try {
    return fn();
  } catch (const std::exception& e) {
    return fail(e.what());
  }
}

// What `record` queues on the stream, captured as a graph (instantiate() makes it one to launch).
template <class Fn>
bool capture_graph(cudaStream_t st, Fn&& record, cudaGraph_t* out) {
  cudaGraph_t graph = nullptr;
  bool ok = cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
  if (ok) {
    try {
      record();
    } catch (const std::exception&) {
      ok = false;
    }
    ok = (cudaStreamEndCapture(st, &graph) == cudaSuccess) && ok && graph;
  }
  if (!ok) {
    if (graph) cudaGraphDestroy(graph);
    cudaGetLastError();
    return false;
  }
  *out = graph;
  return true;
}

// A captured graph, freed with its holder (an exception between the capture and the instantiation leaks none).
struct Captured {
  cudaGraph_t graph = nullptr;
  Captured() = default;
  Captured(const Captured&) = delete;
  Captured& operator=(const Captured&) = delete;
  ~Captured() {
    if (graph) cudaGraphDestroy(graph);
  }
};

// The captured graph made one to launch (what that takes of the device's memory: graph_size); the capture freed.
bool instantiate(Captured& c, cudaGraphExec_t* out) {
  const bool ok = cudaGraphInstantiate(out, c.graph, 0) == cudaSuccess;
  cudaGraphDestroy(c.graph);
  c.graph = nullptr;
  if (!ok) cudaGetLastError();
  return ok;
}

// A graph made for s (its own, or a batch graph it is in): its stats.
void note_graph(flyx_seq* s, const GraphSize& size) {
  flyx_graph_stats& st = s->graph_stats;
  ++st.captured;
  st.params_most = std::max<uint64_t>(st.params_most, size.params_most);
  st.params_unknown += size.params_unknown;
}

// -v: what a graph holds, and what it is counted at.
void say_graph(const flyx_bmodel* m, const char* what, int rows, int key, const GraphSize& g) {
  if (!m->verbose) return;
  std::fprintf(stderr, "flyx-cuda: %s graph (%d rows, head %d): %zu kernels (parameters %zu bytes at most), %zu "
                       "copies, %zu memsets, %zu other nodes (%zu graphs in them, %zu not given): %.0f KiB at most\n",
               what, rows, key, g.kernels, g.params_most, g.copies, g.sets, g.others, g.bodies, g.bodies_unknown,
               g.bytes / 1024.0);
}

void ensure_checkpoint_slot(flyx_seq* s, int c) {
  if (s->ckp_arena[c] || s->ckp_slot_bytes == 0) return;
  void* p = nullptr;
  CK(cudaHostAlloc(&p, s->ckp_slot_bytes, cudaHostAllocDefault));
  s->host_allocs.push_back(p);
  auto* base = static_cast<unsigned char*>(p);
  s->ckp_arena[c] = base;
  size_t off = 0;
  auto take_raw = [&](size_t bytes) -> void* {
    const size_t aligned = (bytes + 255) / 256 * 256;
    void* ptr = base + off;
    off += aligned;
    return ptr;
  };
  auto take = [&](size_t bytes) -> float* { return static_cast<float*>(take_raw(bytes)); };
  for (auto& g : s->gdn) {
    g.ckp_S[c] = take(g.s_stride * sizeof(float));
    g.ckp_win[c] = take(g.w_stride * sizeof(float));
  }
  for (auto& sc : s->sconv) sc.ckp_win[c] = take(sc.w_stride * sizeof(float));
  for (size_t i = 0; i < s->kv.size(); ++i) {
    auto& kb = s->kv[i];
    if (!kb.swa_ring) continue;
    const KvSpec& ks = s->m->states.kv[i];
    const size_t elem = ks.fp8 ? 1 : sizeof(kv16);
    const size_t bytes = size_t(kb.ring) * size_t(ks.nkv) * size_t(ks.hd) * elem;
    kb.ckp_k[c] = take_raw(bytes);
    kb.ckp_v[c] = take_raw(bytes);
    if (ks.fp8) {
      const size_t sbytes = size_t(kb.ring) * size_t(ks.nkv) * sizeof(float);
      kb.ckp_ks[c] = take(sbytes);
      kb.ckp_vs[c] = take(sbytes);
    }
  }
  if (const int w = s->m->states.ring_width) {
    s->ring_ckp[c] = take(size_t(w) * sizeof(float));
    std::memset(s->ring_ckp[c], 0, size_t(w) * sizeof(float));
  }
}

void copy_checkpoint_state(flyx_seq* s, int c, int sl, int pos, bool save, bool snap = false) {
  auto move = [&](void* host, void* dev, size_t bytes) {
    if (bytes == 0) return;
    CK(cudaMemcpyAsync(save ? host : dev, save ? dev : host, bytes, cudaMemcpyDefault, s->st));
  };
  for (auto& g : s->gdn) {
    move(g.ckp_S[c], g.S + sl * g.s_stride, g.s_stride * sizeof(float));
    move(g.ckp_win[c], g.win + sl * g.w_stride, g.w_stride * sizeof(float));
  }
  for (auto& sc : s->sconv) move(sc.ckp_win[c], sc.win + sl * sc.w_stride, sc.w_stride * sizeof(float));
  for (size_t i = 0; i < s->kv.size(); ++i) {
    auto& kb = s->kv[i];
    if (!kb.swa_ring) continue;
    const int n_slots = std::min(pos, kb.ring);
    if (n_slots <= 0) continue;
    const KvSpec& ks = s->m->states.kv[i];
    const size_t elem = ks.fp8 ? 1 : sizeof(kv16);
    const size_t bytes = size_t(n_slots) * size_t(ks.nkv) * size_t(ks.hd) * elem;
    move(kb.ckp_k[c], kb.k, bytes);
    move(kb.ckp_v[c], kb.v, bytes);
    if (ks.fp8) {
      const size_t sbytes = size_t(n_slots) * size_t(ks.nkv) * sizeof(float);
      move(kb.ckp_ks[c], kb.ks, sbytes);
      move(kb.ckp_vs[c], kb.vs, sbytes);
    }
  }
  const size_t w = size_t(s->m->states.ring_width);
  if (s->ring && pos > 0) {  // snap: saved before hring_put overwrote kHring slots; else the ring holds pos - 1
    float* dev = (save && snap && s->ring_snap) ? s->ring_snap : s->ring + size_t((pos - 1) % kHring) * w;
    move(s->ring_ckp[c], dev, sizeof(float) * w);
  }
}

}  // namespace


// ---- waiting for the device ----------------------------------------------------------------
// Every CUDA wait spins a core on WSL2, whatever the scheduling flags (a
// blocking-sync event costs 212 ms of CPU for a 200 ms kernel), so waits poll:
// cudaEventQuery, then sleep (sleep_us: on Windows a high-resolution timer).  A
// round result: with two rounds in flight the device is busy with the next round
// meanwhile, so waking a little late costs no throughput; the first sleep covers
// most of the usual wait, short polls end it.  Option wait=spin:
// cudaEventSynchronize (lowest latency, one core).  wait=auto (Windows' default):
// a sleep there ends ~0.5 ms late at best, as long as a small model's whole
// round (the tiny test model's: 0.19 ms), so a round's result the last waits say
// comes within kSpinBelowUs is spun for and the rest polled: the speed of
// spinning, the CPU of polling.
namespace {
constexpr double kSpinBelowUs = 1500;
}  // namespace

void poll_event(cudaEvent_t e, double first_sleep_us) {
  cudaError_t r = cudaEventQuery(e);
  if (r == cudaErrorNotReady && first_sleep_us > 0) sleep_us(int64_t(first_sleep_us));
  int step_us = 20;
  while ((r = cudaEventQuery(e)) == cudaErrorNotReady) {
    sleep_us(step_us);
    step_us = std::min(step_us * 2, 200);  // long waits (prefill) settle at 200 us
  }
  CK(r);
}

void wait_round(flyx_seq* s, cudaEvent_t e) {
  const WaitMode w = s->m->opt.wait;
  const auto t0 = std::chrono::steady_clock::now();
  if (w == WaitMode::spin || (w == WaitMode::automatic && s->round_wait_us < kSpinBelowUs))
    CK(cudaEventSynchronize(e));
  else
    poll_event(e, std::min(0.75 * s->round_wait_us - 200.0, 4000.0));
  const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
  s->round_wait_us = 0.8 * s->round_wait_us + 0.2 * us;
}

// Everything queued on the session's stream has finished.
void sync_stream(flyx_seq* s) {
  if (s->m->opt.wait == WaitMode::spin) {
    CK(cudaStreamSynchronize(s->st));
    return;
  }
  CK(cudaEventRecord(s->idle_ev, s->st));
  poll_event(s->idle_ev, 0);
}

// The session's stream does not wait for the legacy default stream, and a
// cudaMemcpy from pageable memory may return before its data lands: anything
// the stream's kernels read is written through the stream itself.  (A reset
// session's prefill could read the last request's round from P, leave a
// layer's state in another slot, and every token after the first was wrong.)
// Everything before oracle_len is the round's position; oracle_len stays as
// set_oracle or set_drafts left it.
void set_params(flyx_seq* s, int position, int anchor, int round) {
  s->snap_pos = -1;
  sync_stream(s);  // nothing in flight reads P or the staging
  s->forget_hot();  // positions from here on may be written again
  DevParams& p = *s->params_host;
  p = DevParams{};
  p.position = position;
  p.anchor = anchor;
  p.round = round;
  p.seq_no = s->seq_no;
  CK(cudaMemcpyAsync(s->P, &p, offsetof(DevParams, oracle_len), cudaMemcpyHostToDevice, s->st));
  // A rewind, restore or snapshot load: the canvas no longer follows the text (a prompt edited back to the position
  // a slide left would draft from a canvas of text no longer there).
  forget_canvas(s, s->st);
  s->position = position;
  s->round = round;
  s->update_write_hi();
}

void forget_canvas(flyx_seq* s, cudaStream_t st) {
  if (s->dv.canvas.next_start) CK(cudaMemsetAsync(s->dv.canvas.next_start, 0xff, sizeof(int), st));  // -1
}

// Empty recurrent state; the positional state (K/V caches, the hidden ring) needs no reset.
void reset_state(flyx_seq* s, int anchor) {
  sync_stream(s);
  for (auto& g : s->gdn) {
    CK(cudaMemsetAsync(g.S, 0, FLYX_ROUND_SLOTS * g.s_stride * sizeof(float), s->st));
    CK(cudaMemsetAsync(g.win, 0, FLYX_ROUND_SLOTS * g.w_stride * sizeof(float), s->st));
  }
  for (auto& sc : s->sconv)
    CK(cudaMemsetAsync(sc.win, 0, FLYX_ROUND_SLOTS * sc.w_stride * sizeof(float), s->st));
  for (auto& mt : s->meta) mt = {};
  set_params(s, 0, anchor, 0);
}

// Upload generation parameters when they changed (rounds read them on the device).
void set_gen(flyx_seq* s, const flyx_gen_params* g) {
  DevGen dg{};
  dg.temperature = g->temperature > 0.f ? g->temperature : 0.f;
  dg.top_p = (g->top_p > 0.f && g->top_p < 1.f) ? g->top_p : 1.f;
  dg.top_k = g->top_k > 0 ? g->top_k : 0;
  dg.seed = g->seed;
  dg.n_stop = std::min(g->n_stop_ids, FLYX_MAX_STOP_IDS);
  for (int i = 0; i < dg.n_stop; ++i) dg.stop[i] = g->stop_ids[i];
  dg.limit = g->limit;
  dg.max_ctx = s->max_ctx;
  dg.presence = g->presence_penalty;
  dg.frequency = g->frequency_penalty;
  dg.accept_min_p = g->accept_min_p > 0.f ? std::min(g->accept_min_p, 1.f) : 0.f;
  dg.accept_typical = g->accept_typical > 0.f ? g->accept_typical : 0.f;
  dg.mask = g->mask != nullptr;
  if (g->mask) {  // rare (a token the grammar rejected is chosen again): staged on every such launch
    sync_stream(s);  // the pinned staging must not race a pending copy
    std::memcpy(s->mask_host, g->mask, sizeof(uint32_t) * size_t(s->mask_words));
    CK(cudaMemcpyAsync(s->mask, s->mask_host, sizeof(uint32_t) * size_t(s->mask_words), cudaMemcpyHostToDevice,
                       s->st));
  }
  if (s->gen_valid && std::memcmp(&dg, &s->gen_last, sizeof(dg)) == 0) return;
  sync_stream(s);  // the pinned copy below must not race a pending one
  *s->gen_host = dg;
  CK(cudaMemcpyAsync(s->G, s->gen_host, sizeof(DevGen), cudaMemcpyHostToDevice, s->st));
  s->gen_last = dg;
  s->gen_valid = true;
}

// A round's drafts in place of the drafter's (gen.drafts): written into the
// oracle at their positions, with the oracle length for this round; a round
// without them gets the test oracle's length back.
void set_drafts(flyx_seq* s, const flyx_gen_params* g) {
  const bool given = g->drafts && g->n_drafts > 0;
  if (!given && !s->drafts_given) return;
  int* slot = s->drafts_host + (s->drafts_slot++ % flyx_seq::kDraftSlots) * (FLYX_MAX_WIDE_ROWS + 1);
  int len = s->test_oracle_len;
  if (given) {
    const int n = std::min(g->n_drafts, s->m->wide_rows - 1);
    if (g->drafts_at < 0 || g->drafts_at + n > s->oracle_cap) throw std::runtime_error("drafts outside the context");
    std::memcpy(slot, g->drafts, sizeof(int) * size_t(n));
    CK(cudaMemcpyAsync(s->oracle + g->drafts_at, slot, sizeof(int) * size_t(n), cudaMemcpyHostToDevice, s->st));
    len = g->drafts_at + n;
  }
  slot[FLYX_MAX_WIDE_ROWS] = len;
  CK(cudaMemcpyAsync(&s->P->oracle_len, slot + FLYX_MAX_WIDE_ROWS, sizeof(int), cudaMemcpyHostToDevice, s->st));
  s->drafts_given = given;
}

bool penalized(const flyx_gen_params* g) { return g->presence_penalty != 0.f || g->frequency_penalty != 0.f; }
bool relaxed(const flyx_gen_params* g) { return g->accept_min_p > 0.f || g->accept_typical > 0.f; }

// At the first tree round, the table of the K/V caches the compaction moves.
void tree_kv_table(flyx_seq* s, const flyx_plan* p) {
  if (s->n_tree_kv >= 0) return;
  std::vector<KvRows> kv;
  int most = 0;
  for (int i : p->tree_kv) {
    const KvBuf& b = s->kv[size_t(i)];
    const KvSpec& k = s->m->states.kv[size_t(i)];
    const int bytes = k.nkv * k.hd * (k.fp8 ? 1 : 2);
    kv.push_back(KvRows{bytes, k.nkv, b.ref()});
    most = std::max(most, bytes);
  }
  if (!kv.empty()) {
    CK(cudaMemcpyAsync(s->tree_kv, kv.data(), kv.size() * sizeof(KvRows), cudaMemcpyHostToDevice, s->st));
    CK(cudaStreamSynchronize(s->st));  // kv is the host's
  }
  s->n_tree_kv = static_cast<int>(kv.size());
  s->tree_kv_bytes = most;
}

// A tree round's tree (gen.parents) as the kernels read it (TreeRows): every row's depth and the row of its ancestor
// at each depth, through a ring of pinned slots as set_drafts'.  A drafter's tree (FLYX_DRAFT_TREE) the round
// writes itself (selector_tree).
void set_tree(flyx_seq* s, const flyx_plan* p, const flyx_gen_params* g) {
  tree_kv_table(s, p);
  if (p->drafted_tree) return;
  TreeRows* t = s->tree_host + (s->tree_slot++ % flyx_seq::kDraftSlots);
  const int rows = p->width + 1;
  *t = TreeRows{};
  t->n = rows;
  for (int r = 1; r < rows; ++r) {
    const int i = r - 1;  // the row's draft
    const int parent = g->parents && g->drafts && i < g->n_drafts ? g->parents[i] + 1 : r - 1;
    if (parent < 0 || parent >= r) throw std::runtime_error("tree: a draft's parent must come before it");
    t->depth[r] = t->depth[parent] + 1;
    for (int d = 0; d < t->depth[parent] + 1; ++d) t->path[r][d] = t->path[parent][d];
    t->path[r][t->depth[r]] = static_cast<unsigned char>(r);
  }
  CK(cudaMemcpyAsync(s->tree, t, sizeof(TreeRows), cudaMemcpyHostToDevice, s->st));
}

// The certified head chooses: option head_cert, greedy, no mask (a masked
// choice reads the full head, whose scores of the survivors are the same
// bits) and no penalty (the screen bounds the head's scores, not the
// penalized ones).
bool certified(const flyx_seq* s, const flyx_gen_params* g) {
  return s->cert.cols && g->temperature <= 0.f && !g->mask && !penalized(g) && !relaxed(g);  // relaxed: every score
}

void check_idle(flyx_seq* s) {
  if (!s->pending.empty()) throw std::runtime_error("rounds still in flight");
  sync_stream(s);
}

int prefill(flyx_plan* p, flyx_seq* s, const int32_t* tokens, int32_t n, const flyx_gen_params* sample) {
  if (!s) return no_sequence("prefill");
  const int sample_last = sample != nullptr;
  if (p->phase != FLYX_PHASE_PREFILL) return fail("not a prefill plan");
  if (n <= 0 || n > p->max_rows || n > s->rows_cap) return fail("prefill chunk size out of range");
  return guarded([&] {
    check_idle(s);  // also guarantees the staging buffer is free
    if (!flyx_room(s->position + n, s->max_ctx)) return fail("context is full");
    if (sample) set_gen(s, sample);
    // Borrow the model's prefill buffers for this pass: after any previous prefill or strata session using the slab.
    auto* m = const_cast<flyx_bmodel*>(s->m);
    std::lock_guard<std::mutex> lock(m->prefill_mu);
    if (m->prefill_owner) {
      flyx_seq* prev = m->prefill_owner;
      if (prev->uses_shared_prefill()) {
        prev->copy_home(false);
        prev->prefill_clobbered = true;
        CK(cudaEventRecord(m->prefill_ev, prev->st));
      }
      m->prefill_owner = nullptr;
    }
    CK(cudaStreamWaitEvent(s->st, m->prefill_ev, 0));
    if (m->prefill_dirty && m->prefill_zero_ptr && m->prefill_zero_bytes > 0) {
      CK(cudaMemsetAsync(m->prefill_zero_ptr, 0, m->prefill_zero_bytes, s->st));
      m->prefill_dirty = false;
    }
    std::memcpy(s->tok_host, tokens, sizeof(int) * n);
    CK(cudaMemcpyAsync(m->prefill_bufs.tokens, s->tok_host, sizeof(int) * n, cudaMemcpyHostToDevice, s->st));
    s->meta[flyx_round_slot(s->round)].valid = false;
    // A multiple of prefill_align inside the pass (the pass then ends past it): the last one, or prefill_snapshot's;
    // its GDN state and window go to the round slot after the current one, for checkpoint() (the bits a pass ending
    // there gives: numerics.md, K).  That slot held the oldest round's start; it is no longer a rollback source.
    const int align = m->prefill_align;
    const bool wanted = s->want_snap >= s->position && s->want_snap < s->position + n && s->want_snap % align == 0;
    const int boundary = wanted ? s->want_snap : (s->position + n) / align * align;
    s->want_snap = -1;
    s->snap_pos = -1;
    s->pass_snap = align > 1 && boundary >= s->position && boundary < s->position + n ? boundary : -1;
    if (s->pass_snap >= 0) {
      s->snap_slot = flyx_round_slot(s->round + 1);
      s->meta[s->snap_slot].valid = false;
    }
    Pass ps{p, s, n, false, sample_last != 0, sample && certified(s, sample), sample && penalized(sample)};
    ps.v = rows_of(s, &m->prefill_bufs);  // the prefill buffers it borrows
    // A prefill reads the caches through the stagings: the copy at home in host memory.
    s->copy_home(false);
    s->copy_written(s->position);
    s->staged(s->position + n);
    forward(ps);
    s->snap_pos = s->pass_snap;
    s->pass_snap = -1;
    prefill_done(s->P, n, m->prefill_bufs.choice, sample_last, s->ring_dev, s->G, s->reply, s->st);
    CK(cudaEventRecord(m->prefill_ev, s->st));
    s->position += n;
    if (sample_last) {
      copy_ring(s);
      ++s->seq_no;
      CK(cudaEventRecord(s->ev[s->seq_no % kRing], s->st));
      s->pending.push_back({'P', 0});
    }
    s->update_write_hi();
    CK(cudaGetLastError());
    return 0;
  });
}

}  // namespace flyx::cuda

void flyx_seq::forget_hot() {
  if (dry) return;
  for (const flyx::cuda::KvBuf& b : kv)
    if (b.hot_tag) {
      CK(cudaMemsetAsync(b.hot_tag, 0xff, b.hot_tags * sizeof(int32_t), st));      // -1: no pair
      CK(cudaMemsetAsync(b.hot_claim, 0x7f, b.hot_tags * sizeof(int32_t), st));    // kHotFree
    }
}

// The rows of a round launched now: the first at the position the host last read or past it (the
// rounds in flight move it on by at most their rows), the last at most their rows plus this round's further.
int flyx_seq::strata_phase(int rows) const {
  using namespace flyx::cuda;
  if (kv_method != FLYX_KV_STRATA) return kStrataNone;
  if (position >= strata_from) return kStrataAll;
  const long long last = (long long)position + (long long)in_flight_rows() + rows - 1;
  return last < strata_from ? kStrataNone : kStrataBoth;
}

namespace flyx::cuda {

// Where a round's copy is: in the pool for a round that reads only the copy, else at home in host
// memory, the stagings free for the caches; and what the round writes, and stages.  When the session shares the
// model's VMM prefill slab, coordinate ownership under prefill_mu (at most one session holds it at a time).
// Callers holding prefill_mu across submission pass prefill_locked = true.
void copy_for(flyx_seq* s, int strata, int rows, bool allow_prefill_pool = true, bool prefill_locked = false) {
  const bool copy_only = strata == kStrataAll;
  const long long reach = (long long)s->position + s->in_flight_rows() + rows;
  // A round reading the caches stays inside their stagings (they reach auto's switch and the rows of the
  // rounds in flight past it, 4 x 16 rows: a wide round's up to 64 run alone, and a tree's only below the switch).
  if (!copy_only && s->stage_window > 0)
    for (const KvBuf& b : s->kv)
      if (b.vk && reach > b.staged_cap())
        throw std::logic_error("cuda: a round would read a cache past its staging");
  const bool shared = s->uses_shared_prefill();
  const bool want_dev = copy_only && (allow_prefill_pool || !shared);
  if (shared) {
    auto* m = const_cast<flyx_bmodel*>(s->m);
    std::unique_lock<std::mutex> lock(m->prefill_mu, std::defer_lock);
    if (!prefill_locked) lock.lock();
    if (want_dev) {
      if (m->prefill_owner != s) {
        if (m->prefill_owner) {
          flyx_seq* prev = m->prefill_owner;
          if (prev->uses_shared_prefill()) {
            prev->copy_home(false);
            prev->prefill_clobbered = true;
            CK(cudaEventRecord(m->prefill_ev, prev->st));
          }
        }
        CK(cudaStreamWaitEvent(s->st, m->prefill_ev, 0));
        m->prefill_owner = s;
      }
      m->prefill_dirty = true;
      s->copy_home(true);
    } else {
      s->copy_home(false);
      if (m->prefill_owner == s) {
        s->prefill_clobbered = true;
        CK(cudaEventRecord(m->prefill_ev, s->st));
        m->prefill_owner = nullptr;
      }
    }
  } else {
    s->copy_home(want_dev);
  }
  s->copy_written(s->position);
  if (!copy_only) s->staged(int(std::min<long long>(reach, s->cap)));
}

namespace {

// A round of plan p on s as launch() runs it: its pass (its parameters uploaded, the copies where it reads them), its
// graph's key (the plan and what its head does), and the positions its graph's fixed-size copies take.
struct RoundPass {
  Pass ps;
  std::pair<const flyx_plan*, int> key;
  int prefix;
};

void set_stop_policy(flyx_seq* s, const flyx_plan* p, const flyx_gen_params* g) {
  flyx_stop_policy* slot = s->stop_host + (s->stop_slot++ % flyx_seq::kDraftSlots);
  const int max_buckets = std::min(FLYX_MAX_STOP_BUCKETS, int(p->checkpoints.size()));
  if (g && g->stop_policy) {
    *slot = *g->stop_policy;
    slot->n_buckets = std::clamp(slot->n_buckets, 1, std::max(1, max_buckets));
    for (int i = 0; i < slot->n_buckets; ++i) {
      const int cap = i < int(p->checkpoints.size()) ? std::min(p->checkpoints[size_t(i)], FLYX_MAX_ROWS) : FLYX_MAX_ROWS;
      slot->widths[i] = std::clamp(slot->widths[i], 0, cap);
    }
  } else {
    *slot = flyx_stop_policy{};
    slot->n_buckets = max_buckets;
    for (int i = 0; i < slot->n_buckets; ++i) slot->widths[i] = std::min(p->checkpoints[size_t(i)], FLYX_MAX_ROWS);
  }
  CK(cudaMemcpyAsync(s->stop_dev, slot, sizeof(flyx_stop_policy), cudaMemcpyHostToDevice, s->st));
}

RoundPass round_pass(flyx_plan* p, flyx_seq* s, const flyx_gen_params* g, bool prefill_locked = false) {
  set_gen(s, g);  // the same for every width: switching widths uploads nothing
  set_drafts(s, g);
  if (p->tree) set_tree(s, p, g);
  if (p->adaptive()) set_stop_policy(s, p, g);
  const int rows = p->width + 1;
  RoundPass r{Pass{p, s, rows, true, false, certified(s, g), penalized(g)}, {}, 0};
  Pass& ps = r.ps;
  ps.relaxed = p->width > 0 && relaxed(g);
  ps.strata = s->strata_phase(rows);
  ps.v = rows_of(s);
  copy_for(s, ps.strata, rows, true, prefill_locked);
  r.key = std::make_pair(static_cast<const flyx_plan*>(p),
                         (ps.cert ? 1 : 0) | (ps.pen ? 2 : 0) | (ps.relaxed ? 4 : 0) | (ps.strata << 3));
  // The host knows this prefix is committed even with rounds in flight. A code tile can be copied ahead
  // only when all its 16 positions precede the pass. The staging kernel reads the actual end from P.
  const bool dma = s->m->opt.stage_dma && s->m->opt.kernels.stage_ahead && s->copy_slots && ps.strata == kStrataAll;
  r.prefix = dma ? s->position / 16 * 16 : 0;
  return r;
}

#if CUDART_VERSION >= 12040
// Flat GPU-resident conditional graph for an adaptive round plan:
// - Sequential drafters (MTP, EAGLE-4): chunk 0 -> decide -> IF(draft chunk 1 -> decide) -> ... -> IF(verify w_j).
// - Parallel block drafters (DFlash2, DSpark): full block draft -> decide -> IF(verify w_j).
// Zero CPU sync; full FLYX_MAX_IN_FLIGHT=2 pipelining; handles reset to 0 on every launch.
bool capture_adaptive_graph(flyx_plan* p, flyx_seq* s, const RoundPass& r, cudaGraph_t* out, GraphSize* size_out) {
  cudaGraph_t root = nullptr;
  if (cudaGraphCreate(&root, 0) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  Captured guard;
  guard.graph = root;
  const int M = int(p->checkpoints.size());
  const bool seq_draft = sequential_draft(p);

  std::vector<cudaGraphConditionalHandle> h_ver(size_t(M), 0);
  std::vector<cudaGraphConditionalHandle> h_draft(size_t(M), 0);
  for (int j = 0; j < M; ++j) {
    if (cudaGraphConditionalHandleCreate(&h_ver[size_t(j)], root, 0, cudaGraphCondAssignDefault) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    if (seq_draft && j > 0 &&
        cudaGraphConditionalHandleCreate(&h_draft[size_t(j)], root, 0, cudaGraphCondAssignDefault) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
  }

  auto add_size = [](GraphSize& dst, const GraphSize& sub) {
    dst.kernels += sub.kernels;
    dst.copies += sub.copies;
    dst.sets += sub.sets;
    dst.others += sub.others;
    dst.params_unknown += sub.params_unknown;
    dst.params_most = std::max(dst.params_most, sub.params_most);
    dst.bytes += sub.bytes;
  };

  auto capture_into = [&](cudaGraph_t target, const std::vector<cudaGraphNode_t>& deps, auto&& fn,
                          std::vector<cudaGraphNode_t>* leaves_out) -> bool {
    if (cudaStreamBeginCaptureToGraph(s->st, target, deps.empty() ? nullptr : deps.data(), nullptr, deps.size(),
                                      cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    bool ok = true;
    try {
      fn();
    } catch (const std::exception&) {
      ok = false;
    }
    if (ok && leaves_out) {
      cudaStreamCaptureStatus cs{};
      const cudaGraphNode_t* leaves = nullptr;
      size_t n_leaves = 0;
      if (cudaStreamGetCaptureInfo(s->st, &cs, nullptr, nullptr, &leaves, nullptr, &n_leaves) == cudaSuccess) {
        leaves_out->assign(leaves, leaves + n_leaves);
      } else {
        cudaGetLastError();
        ok = false;
      }
    }
    cudaGraph_t ended = nullptr;
    if (cudaStreamEndCapture(s->st, &ended) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    return ok;
  };

  std::vector<cudaGraphNode_t> prev;
  if (seq_draft) {
    if (!capture_into(root, {}, [&] {
          draft_step_range(r.ps, 0, p->checkpoints[0]);
          stop_decide_step(s->draft_prob, s->draft_margin, s->stop_dev, s->stop_state, 0, 0, p->checkpoints[0],
                           static_cast<unsigned long long>(h_draft[1]), static_cast<unsigned long long>(h_ver[0]),
                           s->st);
        }, &prev))
      return false;
  } else {
    StopCondHandles handles{};
    for (int j = 0; j < M; ++j) handles.h[j] = static_cast<unsigned long long>(h_ver[size_t(j)]);
    if (!capture_into(root, {}, [&] {
          draft_step_range(r.ps, 0, p->width);
          stop_decide_block(s->draft_prob, s->draft_margin, s->stop_dev, s->stop_state, handles, s->st);
        }, &prev))
      return false;
  }

  std::vector<GraphSize> bodies;  // as captured: priced here where the driver gives no body
  if (seq_draft) {
    for (int j = 1; j < M; ++j) {
      cudaGraphNodeParams np{};
      np.type = cudaGraphNodeTypeConditional;
      np.conditional.handle = h_draft[size_t(j)];
      np.conditional.type = cudaGraphCondTypeIf;
      np.conditional.size = 1;
      cudaGraphNode_t cnode = nullptr;
      if (cudaGraphAddNode(&cnode, root, prev.empty() ? nullptr : prev.data(), nullptr, prev.size(), &np) !=
          cudaSuccess) {
        cudaGetLastError();
        return false;
      }
      cudaGraph_t body = np.conditional.phGraph_out[0];
      const int w0 = p->checkpoints[size_t(j - 1)];
      const int w1 = p->checkpoints[size_t(j)];
      const unsigned long long h_nxt = (j + 1 < M) ? static_cast<unsigned long long>(h_draft[size_t(j + 1)]) : 0ull;
      const unsigned long long h_v = static_cast<unsigned long long>(h_ver[size_t(j)]);
      if (!capture_into(body, {}, [&] {
            draft_step_range(r.ps, w0, w1);
            stop_decide_step(s->draft_prob, s->draft_margin, s->stop_dev, s->stop_state, j, w0, w1, h_nxt, h_v, s->st);
          }, nullptr))
        return false;
      bodies.push_back(graph_size(body));
      prev = {cnode};
    }
  }

  for (int j = 0; j < M; ++j) {
    cudaGraphNodeParams np{};
    np.type = cudaGraphNodeTypeConditional;
    np.conditional.handle = h_ver[size_t(j)];
    np.conditional.type = cudaGraphCondTypeIf;
    np.conditional.size = 1;
    cudaGraphNode_t vnode = nullptr;
    if (cudaGraphAddNode(&vnode, root, prev.empty() ? nullptr : prev.data(), nullptr, prev.size(), &np) !=
        cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    cudaGraph_t body = np.conditional.phGraph_out[0];
    Pass ps_j = r.ps;
    ps_j.rows = p->checkpoints[size_t(j)] + 1;
    ps_j.accept_width = p->checkpoints[size_t(j)];
    if (!capture_into(body, {}, [&] {
          forward_target(ps_j, r.prefix);
          copy_ring(s);
        }, nullptr))
      return false;
    bodies.push_back(graph_size(body));
  }
  // graph_size reads each conditional node's body, a graph of its own; a driver that gives none of them
  // (bodies_unknown) gets them priced as captured, so each body is counted once.
  GraphSize total = graph_size(root);
  if (total.bodies_unknown == bodies.size()) {
    for (const GraphSize& b : bodies) {
      add_size(total, b);
      ++total.bodies;
    }
    total.bodies_unknown = 0;
  }
  *out = guard.graph;
  guard.graph = nullptr;
  if (size_out) *size_out = total;
  return true;
}
#endif

}  // namespace

GraphSize round_graph_size(flyx_plan* p, flyx_seq* s) {
  const flyx_gen_params greedy{};
  const RoundPass r = round_pass(p, s, &greedy);
#if CUDART_VERSION >= 12040
  if (p->adaptive()) {
    Captured c;
    GraphSize size{};
    if (!capture_adaptive_graph(p, s, r, &c.graph, &size)) return {};
    say_graph(s->m, "counted adaptive round", p->width + 1, r.key.second, size);
    return size;
  }
#endif
  // As launch: a conditional canvas pass the driver refuses, or whose body it does not give, counted inline.
  for (;;) {
    Captured c;
    const bool ok = capture_graph(s->st, [&] { forward(r.ps, r.prefix); copy_ring(s); }, &c.graph);
    const GraphSize size = ok ? graph_size(c.graph) : GraphSize{};
    if ((!ok || size.bodies_unknown) && s->dv.body && s->dv.skip_noop) {
      s->dv.skip_noop = false;
      continue;
    }
    if (!ok) return {};
    say_graph(s->m, "counted round", p->width + 1, r.key.second, size);
    return size;
  }
}

int launch(flyx_plan* p, flyx_seq* s, const flyx_gen_params* g) {
  if (!s) return no_sequence("launch");
  if (p->phase != FLYX_PHASE_ROUND) return fail("not a round plan");
  const int rows = p->width + 1;
  // A wide round's strata attention goes 16 rows at a time as a chain's rows (shift_params): a tree's rows past 16
  // read every position exactly.
  if (rows > FLYX_MAX_ROWS && p->tree && s->strata_phase(rows) != kStrataNone)
    return fail("a tree round of more than " + std::to_string(FLYX_MAX_ROWS) + " rows reads no strata copy");
  s->snap_pos = -1;  // rounds write the round slots
  return guarded([&] {
    // Hold prefill_mu across submission when sharing the VMM prefill slab so another session cannot evict
    // prefill_owner before this round's work is queued on s->st (lock order: prefill_mu before batch->mu).
    const bool shared = s->uses_shared_prefill();
    std::unique_lock<std::mutex> prefill_lock(s->m->prefill_mu, std::defer_lock);
    if (shared) prefill_lock.lock();
    const RoundPass r = round_pass(p, s, g, shared);
    const Pass& ps = r.ps;
    const auto& key = r.key;
    const int prefix = r.prefix;
    auto it = s->graph.find(key);
    // A rewind may shorten the safe prefix. As it grows, recapture at most once per 4096 positions per
    // graph so a long generation also benefits.  The old graph goes once its launches are done:
    // destroyed in flight, what it holds would stay the driver's beside the new one's until they end.
    if (it != s->graph.end() && (prefix < it->second.copy_prefix || prefix - it->second.copy_prefix >= 4096)) {
      if (it->second.used > s->read_no) sync_stream(s);
      CK(cudaGraphExecDestroy(it->second.exec));
      s->graph_held -= it->second.bytes;
      s->graph.erase(it);
      it = s->graph.end();
    }
    auto record = [&] {
      forward(ps, prefix);
      copy_ring(s);
    };
    // A capture that found no room waits kGraphRecent rounds before the next: the graphs it would take
    // the place of are by then ones the session ran lately, or not.
    if (it == s->graph.end() && s->graphs && s->seq_no >= s->graph_retry) {
      cudaGraphExec_t exec = nullptr;
      // A driver that refuses DV's conditional canvas pass (dv_context_unless_noop), in the capture or the
      // instantiation, or does not give its body to count (graph_size), gets the round with the pass inline.
      for (bool again = true; again;) {
        again = false;
        bool failed = false;
        Captured c;
        GraphSize size{};
        bool captured = false;
#if CUDART_VERSION >= 12040
        if (p->adaptive()) {
          captured = capture_adaptive_graph(p, s, r, &c.graph, &size);
        } else
#endif
        {
          captured = capture_graph(s->st, record, &c.graph);
          if (captured) size = graph_size(c.graph);
        }
        if (!captured || (size.bodies_unknown && s->dv.body && s->dv.skip_noop)) {
          failed = true;
        } else {
          plan_counted(const_cast<flyx_bmodel*>(s->m), p, size.bytes);  // the plan's first graph: its count
          if (!make_graph_room(s, size.bytes)) {
            s->graph_retry = s->seq_no + kGraphRecent;
            if (s->m->verbose && !s->graph_short)
              std::fprintf(stderr, "flyx-cuda: a round without its graph: %.2f MiB needed, %.2f of the session's "
                                   "%.2f held by graphs it ran lately\n", size.bytes / 1048576.0,
                           double(s->graph_held) / 1048576.0, graph_keep(*s) / 1048576.0);
            s->graph_short = true;
          } else if (instantiate(c, &exec)) {
            it = s->graph.emplace(key, flyx_seq::RoundGraph{exec, prefix, size.bytes}).first;
            s->graph_held += size.bytes;
            note_graph(s, size);
            say_graph(s->m, p->adaptive() ? "adaptive round" : "round", ps.rows, key.second, size);
          } else {
            failed = true;
          }
        }
        if (failed && s->dv.body && s->dv.skip_noop) {
          again = true;
          s->dv.skip_noop = false;
          if (s->m->verbose)
            std::fprintf(stderr, "flyx-cuda: a round graph with a conditional canvas pass failed: the pass inline\n");
        } else if (failed) {
          if (!p->adaptive()) s->graphs = false;
          if (s->m->verbose) std::fprintf(stderr, "flyx-cuda: graph capture failed, running rounds eagerly\n");
        }
      }
    }
    if (it != s->graph.end()) {
      CK(cudaGraphLaunch(it->second.exec, s->st));
      it->second.used = s->seq_no + 1;
      ++s->graph_stats.from_graph;
    } else {
      record();
      ++s->graph_stats.without_graph;
    }
    ++s->seq_no;
    CK(cudaEventRecord(s->ev[s->seq_no % kRing], s->st));
    s->pending.push_back({'R', ps.rows});
    s->update_write_hi();
    CK(cudaGetLastError());
    return 0;
  });
}

// ---- rounds of several sequences ---------------------------------------------------------------------
namespace {

// Two round plans that run the same nodes after their drafting, by the same implementations (accept takes each
// plan's own width).
bool same_target(const flyx_plan* a, const flyx_plan* b) {
  size_t i = target_start(a), j = target_start(b);
  if (a->nodes.size() - i != b->nodes.size() - j) return false;
  for (; i < a->nodes.size(); ++i, ++j) {
    const flyx_node &x = a->nodes[i], &y = b->nodes[j];
    if (x.kind != y.kind || x.block != y.block || (x.kind != FLYX_NODE_ACCEPT && x.arg0 != y.arg0) ||
        a->impl[i] != b->impl[j])
      return false;
  }
  return true;
}

// A sequence's slice of the batch's rows, from row `first`; the scratch buffers (attention partials, the GEMM
// staging) are shared, as one sequence's kernels at a time use them.
RowView slice(const RowView& v, const flyx_bmodel* m, int first) {
  const RowNeeds& n = m->need;
  const size_t r = size_t(first), H = size_t(m->hidden);
  RowView s = v;
  s.b.tokens += r;
  s.b.choice += r;
  s.b.x += r * H;
  s.b.h += r * H;
  s.b.proj += r * size_t(v.ld_proj);
  s.b.act += r * size_t(v.ld_act);
  s.b.gdn_o += r * size_t(n.gdn_o);
  s.b.qbuf += r * size_t(n.q);
  if (s.b.feats) s.b.feats += r * size_t(v.ld_feats);
  if (s.b.cx) s.b.cx += r * size_t(n.cx);
  if (s.b.ckv) s.b.ckv += r * size_t(v.ld_ckv);
  if (s.b.mcat) s.b.mcat += r * 2 * H;
  if (s.b.mx) s.b.mx += r * H;
  if (s.b.mhn) s.b.mhn += r * H;
  if (s.b.lora) s.b.lora += r * size_t(n.lora);
  s.logits += r * size_t(m->vocab);
  return s;
}

}  // namespace

int launch_batch(flyx_plan* const* plans, flyx_seq* const* seqs, const flyx_gen_params* const* gens,
                 int32_t n) {
  if (n < 1) return fail("launch_batch: no sequences");
  for (int i = 0; i < n; ++i)
    if (!seqs[i]) return no_sequence("launch_batch");
  if (n == 1) return launch(plans[0], seqs[0], gens[0]);
  flyx_bmodel* m = seqs[0]->m;
  if (m->need.max_bank > 0 || m->dspark_block >= 0) {
    for (int i = 0; i < n; ++i)
      if (const int r = launch(plans[i], seqs[i], gens[i]); r != 0) return r;
    return 0;
  }
  int rows = 0;
  for (int i = 0; i < n; ++i) {
    if (plans[i]->phase != FLYX_PHASE_ROUND || plans[i]->m != m || seqs[i]->m != m)
      return fail("launch_batch: round plans of the sequences' model");
    if (!same_target(plans[0], plans[i])) return fail("launch_batch: plans that run different target nodes");
    if (plans[i]->tree || plans[i]->adaptive() || relaxed(gens[i]))
      return fail("launch_batch: tree, adaptive, and relaxed rounds run alone");
    for (int j = 0; j < i; ++j)
      if (seqs[j] == seqs[i]) return fail("launch_batch: a sequence twice");
    rows += plans[i]->width + 1;
  }
  if (rows > FLYX_MAX_ROWS) return fail("launch_batch: more than FLYX_MAX_ROWS rows");
  // Canonicalize sequence order by pointer so equivalent batches share one cached graph regardless of caller order.
  int ord[FLYX_MAX_ROWS];
  for (int i = 0; i < n; ++i) ord[i] = i;
  std::sort(ord, ord + n, [&](int a, int b) { return uintptr_t(seqs[a]) < uintptr_t(seqs[b]); });

  BatchRounds& B = *m->batch;
  bool any_shared = false;
  for (int i = 0; i < n; ++i) any_shared = any_shared || seqs[ord[i]]->uses_shared_prefill();
  // Lock order: prefill_mu before B.mu (make_graph_room acquires B.mu while launch() may hold prefill_mu).
  std::unique_lock<std::mutex> prefill_lock(m->prefill_mu, std::defer_lock);
  if (any_shared) prefill_lock.lock();
  std::lock_guard<std::mutex> lock(B.mu);
  return guarded([&] {
    flyx_seq* lead = seqs[ord[0]];
    BatchPass bp;
    bp.rows = rows;
    bp.cert = true;
    bp.v.b = lead->own;
    bp.v.logits = lead->logits;
    bp.v.ld_proj = m->need.proj;
    bp.v.ld_act = m->need.act;
    bp.v.ld_feats = m->need.feats;
    bp.v.ld_ckv = m->need.ckv;
    bp.v.L = Launch{B.st, &lead->own.scratch, &m->opt.kernels, nullptr, &B.side};
    BatchKey key;
    key.n = n;
    bool prefill_claimed = false;
    for (int i = 0, at = 0; i < n; ++i) {
      const int idx = ord[i];
      flyx_seq* s = seqs[idx];
      const int seq_rows = plans[idx]->width + 1;
      const int strata = s->strata_phase(seq_rows);
      s->snap_pos = -1;  // rounds write the round slots
      set_gen(s, gens[idx]);
      set_drafts(s, gens[idx]);
      const bool allow_shared = (strata == kStrataAll) && s->uses_shared_prefill() && !prefill_claimed;
      if (allow_shared) prefill_claimed = true;
      copy_for(s, strata, seq_rows, allow_shared, any_shared);  // on the sequence's stream, before the batch's waits
      // The batch's stream after everything on the sequence's (its rounds in flight, the uploads above).
      CK(cudaEventRecord(s->ready_ev, s->st));
      CK(cudaStreamWaitEvent(B.st, s->ready_ev, 0));
      Pass ps{plans[idx], s, seq_rows, true, false, certified(s, gens[idx]), penalized(gens[idx])};
      ps.strata = strata;
      ps.v = slice(bp.v, m, at);
      bp.cert = bp.cert && ps.cert;
      bp.seq.push_back(ps);
      bp.first.push_back(at);
      at += ps.rows;
      key.seqs[i] = {uintptr_t(s), uintptr_t(plans[idx]),
                     (ps.cert ? 1 : 0) | (ps.pen ? 2 : 0) | (ps.strata << 2) | (s->copy_dev ? 16 : 0)};
    }
    key.cert = bp.cert;
    ++B.launched;
    B.rows += rows;
    B.sequences += n;
    auto record = [&] {
      run_batch(bp);
      for (int i = 0; i < n; ++i) copy_ring(seqs[ord[i]], B.st);
    };
    auto it = B.graphs.find(key);
    if (it == B.graphs.end() && B.graphs_ok && m->opt.graphs) {
      // Only capture once a batch key repeats enough to amortize graph instantiation cost.
      if (B.hits.size() >= BatchRounds::kMaxHitEntries && B.hits.find(key) == B.hits.end()) B.hits.clear();
      const uint32_t seen = ++B.hits[key];
      if (seen >= BatchRounds::kGraphWarmHits) {
        B.hits.erase(key);
        ++B.captured;
        Captured c;
        cudaGraphExec_t exec = nullptr;
        std::vector<flyx_seq*> members;
        for (int i = 0; i < n; ++i) members.push_back(seqs[ord[i]]);
        // DV's canvas passes inline in every member's rounds from now on (as launch does for one sequence): where a
        // conditional one was refused or its body not given, before the batches give up their graphs.
        auto inline_canvas = [&] {
          bool any = false;
          for (flyx_seq* q : members)
            if (q->dv.body && q->dv.skip_noop) q->dv.skip_noop = false, any = true;
          return any;
        };
        bool ok = capture_graph(B.st, record, &c.graph);
        GraphSize size = ok ? graph_size(c.graph) : GraphSize{};
        if ((!ok || size.bodies_unknown) && inline_canvas()) {
          if (c.graph) cudaGraphDestroy(c.graph);
          c.graph = nullptr;
          ok = capture_graph(B.st, record, &c.graph);
          size = ok ? graph_size(c.graph) : GraphSize{};
        }
        // A share of what it takes in each sequence's room for graphs; no room: this batch without
        // (it comes kGraphWarmHits times more before the next capture).
        const size_t share = (size.bytes + size_t(n) - 1) / size_t(n);
        const bool room = ok && make_batch_room(B, members.data(), n, share);
        if (room && B.graphs.size() >= BatchRounds::kMaxGraphs) {  // before the new one is made: nothing to leak
          auto victim = B.graphs.begin();
          for (auto g = B.graphs.begin(); g != B.graphs.end(); ++g)
            if (g->second.used < victim->second.used) victim = g;
          bool synced = false;
          drop_batch_graph(B, victim, &synced);
        }
        if (room && instantiate(c, &exec)) {
          it = B.graphs.emplace(key, BatchRounds::CachedGraph{exec, uint64_t(B.launched), share}).first;
          for (flyx_seq* s : members) {
            s->graph_held += share;
            note_graph(s, size);
          }
          say_graph(m, "batch round", rows, int(key.cert), size);
        } else if ((!ok || room) && !inline_canvas()) {  // refused (no room: this batch without); a conditional
          B.graphs_ok = false;                              // canvas pass: the next capture with it inline
          if (m->verbose) std::fprintf(stderr, "flyx-cuda: batch graph capture failed, running batches eagerly\n");
        }
      }
    }
    if (it != B.graphs.end()) {
      it->second.used = uint64_t(B.launched);
      for (int i = 0; i < n; ++i) it->second.ran[i] = seqs[ord[i]]->seq_no + 1;  // the seq_no this launch takes
      CK(cudaGraphLaunch(it->second.exec, B.st));
    } else {
      record();
    }
    for (int i = 0; i < n; ++i) {
      auto& stats = seqs[i]->graph_stats;
      ++(it != B.graphs.end() ? stats.from_graph : stats.without_graph);
    }
    CK(cudaEventRecord(B.done, B.st));
    for (int i = 0; i < n; ++i) {
      flyx_seq* s = seqs[ord[i]];
      ++s->seq_no;
      CK(cudaEventRecord(s->ev[s->seq_no % kRing], B.st));
      s->pending.push_back({'R', bp.seq[size_t(i)].rows});
      s->update_write_hi();
      CK(cudaStreamWaitEvent(s->st, B.done, 0));  // what the sequence does next, after this round
    }
    CK(cudaGetLastError());
    return 0;
  });
}

int read_result(flyx_seq* s, flyx_round_result* out) {
  if (!s) return no_sequence("read");
  if (s->pending.empty()) return fail("no round in flight");
  return guarded([&] {
    const unsigned long long seq = s->read_no + 1;
    if (s->pending.front().kind == 'R')
      wait_round(s, s->ev[seq % kRing]);
    else if (s->m->opt.wait == WaitMode::spin)
      CK(cudaEventSynchronize(s->ev[seq % kRing]));
    else
      poll_event(s->ev[seq % kRing], 0);
    *out = s->ring_host[seq % kRing];
    if (out->seq_no != seq) return fail("ring entry out of order (" + std::to_string(out->seq_no) + ")");
    s->read_no = seq;
    char kind = s->pending.front().kind;
    s->pending.pop_front();
    if (kind == 'R' && out->n_tokens > 0) {
      s->meta[flyx_round_slot(s->round)] = {true, out->position - out->n_tokens, out->n_tokens};
      s->round += 1;
      s->meta[flyx_round_slot(s->round)].valid = false;
    }
    s->position = out->position;
    s->update_write_hi();
    return 0;
  });
}

// Row `row` of the last round's head (bench/quality/kld.cpp): the certified head scores only the rows it must.
int read_logits(flyx_seq* s, int32_t row, float* out) {
  if (!s) return no_sequence("logits");
  if (row < 0 || row >= FLYX_MAX_ROWS) return fail("logits: row out of range");
  if (s->m->opt.kernels.head_cert && !s->m->opt.kernels.head_fp8)
    return fail("logits: the certified head scores only the rows it must (backend option head_cert=0)");
  return guarded([&] {
    sync_stream(s);
    CK(cudaMemcpy(out, s->logits + size_t(row) * s->m->vocab, sizeof(float) * size_t(s->m->vocab),
                  cudaMemcpyDeviceToHost));
    return 0;
  });
}

// The oldest ring entry not yet read is there; with none pending, the stream is idle.
int ready(flyx_seq* s) {
  if (!s) return no_sequence("ready");
  return guarded([&] {
    cudaError_t r;
    if (!s->pending.empty()) {
      r = cudaEventQuery(s->ev[(s->read_no + 1) % kRing]);
    } else {
      CK(cudaEventRecord(s->idle_ev, s->st));
      r = cudaEventQuery(s->idle_ev);
    }
    if (r == cudaErrorNotReady) return 0;
    CK(r);
    return 1;
  });
}

// The reply so far as the penalties count it: the table and the list of distinct tokens.
int set_reply(flyx_seq* s, const int32_t* tokens, int32_t n) {
  if (!s) return no_sequence("set_reply");
  return guarded([&] {
    check_idle(s);
    const int V = s->m->vocab;
    std::vector<int> count(n == 0 ? 0 : size_t(V));
    std::vector<int> seen;
    for (int32_t i = 0; i < n; ++i) {
      const int32_t t = tokens[i];
      if (t < 0 || t >= V) return fail("set_reply: a token outside the vocabulary");
      if (count[size_t(t)]++ == 0) seen.push_back(t);
    }
    const int n_seen = static_cast<int>(seen.size());
    if (n_seen > s->cap) return fail("set_reply: more tokens than the context holds");
    // Through the session's stream (set_params), and waited for: the sources are this call's.
    if (n == 0) CK(cudaMemsetAsync(s->reply.count, 0, sizeof(int) * size_t(V), s->st));
    else
      CK(cudaMemcpyAsync(s->reply.count, count.data(), sizeof(int) * size_t(V), cudaMemcpyHostToDevice, s->st));
    if (n_seen > 0)
      CK(cudaMemcpyAsync(s->reply.seen, seen.data(), sizeof(int) * size_t(n_seen), cudaMemcpyHostToDevice, s->st));
    CK(cudaMemcpyAsync(s->reply.n_seen, &n_seen, sizeof(int), cudaMemcpyHostToDevice, s->st));
    sync_stream(s);
    return 0;
  });
}

int rewind(flyx_seq* s, int32_t pos, int32_t anchor) {
  if (!s) return no_sequence("rewind");
  return guarded([&] {
    check_idle(s);
    if (pos > s->position) return fail("cannot rewind forward");
    if (pos == 0) {
      reset_state(s, anchor);  // a failed snapshot load may have written state before advancing the position
      return 1;
    }
    if (pos == s->position) {
      set_params(s, pos, anchor, s->round);
      return 1;
    }
    for (int back = 1; back < FLYX_ROUND_SLOTS && s->round - back >= 0; ++back) {
      const int r = s->round - back, sl = flyx_round_slot(r), to = flyx_round_slot(r + 1);
      const int rsl = r % kGdnReplaySlots;
      const auto& mt = s->meta[sl];
      if (!mt.valid || pos < mt.start || pos > mt.start + mt.rows) continue;
      const int rows = pos - mt.start;
      for (auto& g : s->gdn) {
        gdn_recur(g.S, g.s_stride, kSlotExplicit + sl, kSlotExplicit + to, g.replay, kSlotExplicit + rsl, s->P, nullptr,
                  rows, nullptr, false, s->st);
        gdn_conv_fold(g.win, g.w_stride, kSlotExplicit + sl, kSlotExplicit + to, g.replay, kSlotExplicit + rsl, s->P,
                      rows, nullptr, false, s->st);
      }
      if (!s->sconv.empty()) {
        if (s->sconv_alike) {
          const ShortConvBuf& f = s->sconv[0];
          shortconv_fold(s->sconv_layers, int(s->sconv.size()), f.w_stride, f.r_stride, kSlotExplicit + sl,
                         kSlotExplicit + to, kSlotExplicit + rsl, f.channels, f.conv, s->P, rows, nullptr, false,
                         s->st);
        } else {
          for (size_t i = 0; i < s->sconv.size(); ++i) {
            const ShortConvBuf& sc = s->sconv[i];
            shortconv_fold(s->sconv_layers + i, 1, sc.w_stride, sc.r_stride, kSlotExplicit + sl, kSlotExplicit + to,
                           kSlotExplicit + rsl, sc.channels, sc.conv, s->P, rows, nullptr, false, s->st);
          }
        }
      }
      s->meta[sl].rows = rows;  // tokens past pos are discarded; later rewinds must not replay stale rows
      for (int k = 1; k <= back; ++k) s->meta[flyx_round_slot(r + k)].valid = false;
      set_params(s, pos, anchor, r + 1);
      CK(cudaGetLastError());
      return 1;
    }
    reset_state(s, anchor);
    return 0;
  });
}

// The state at the current position, or at the chunk boundary the last
// prefill pass crossed (what a fresh prefill reproduces bit for bit).
int checkpoint(flyx_seq* s, int32_t c) {
  if (!s) return no_sequence("checkpoint");
  if (c < 0 || c >= FLYX_CHECKPOINTS) return fail("bad checkpoint slot");
  return guarded([&] {
    check_idle(s);
    ensure_checkpoint_slot(s, c);
    const bool snap = s->snap_pos >= 0;
    const int pos = snap ? s->snap_pos : s->position;
    const int sl = snap ? s->snap_slot : flyx_round_slot(s->round);
    copy_checkpoint_state(s, c, sl, pos, true, snap);
    s->ckp[c].valid = true;
    s->ckp[c].pos = pos;
    return pos;
  });
}

int prefill_snapshot(flyx_seq* s, int32_t pos) {
  if (!s) return no_sequence("prefill_snapshot");
  s->want_snap = pos;
  return 0;
}

int restore(flyx_seq* s, int32_t c, int32_t anchor) {
  if (!s) return no_sequence("restore");
  if (c < 0 || c >= FLYX_CHECKPOINTS || !s->ckp[c].valid) return -1;
  return guarded([&] {
    check_idle(s);
    copy_checkpoint_state(s, c, flyx_round_slot(s->round), s->ckp[c].pos, false);
    for (auto& mt : s->meta) mt = {};
    set_params(s, s->ckp[c].pos, anchor, s->round);
    return s->position;
  });
}

// ---- snapshots: a checkpoint in host memory ---------------------------------------------------------
// A header, then per K/V cache of the state table and per stacked layer the positions [begin, pos) of k and v (and of
// their FP8 scales), then per gated delta net the checkpoint's state and convolution window, then the hidden ring's
// row before pos.  The header names the layout (the state table's dimensions and formats); loading checks it.
namespace {

using flyx::backends::HostCursor;
using flyx::backends::kDeltaVersion;
using flyx::backends::kSnapshotMagic;
using flyx::backends::kSnapshotVersion;
using flyx::backends::pieces_hold;
using flyx::backends::SnapshotHeader;

uint64_t layout_hash(const flyx_seq* s) {
  const flyx_bmodel* m = s->m;
  uint64_t h = flyx::backends::kFnv1aOffset;  // FNV-1a over the dimensions, eight bytes each
  auto mix = [&](uint64_t v) { flyx::backends::fnv1a64_mix(&h, v); };
  for (size_t i = 0; i < m->states.kv.size(); ++i) {
    const KvSpec& k = m->states.kv[i];
    mix(uint64_t(k.layers));
    mix(uint64_t(k.nkv));
    mix(uint64_t(k.hd));
    mix(k.fp8);
    const KvBuf& b = s->kv[i];
    if (k.window) mix(uint64_t(k.window) << 32 | 79);
    if (b.ring) mix(uint64_t(b.ring) << 32 | 75);  // a ring of slots: saved as its slots are
    if (b.kq) {  // the strata copy, saved with the cache: its format too
      mix(uint64_t(b.bits));
      mix(uint64_t(strata_words(k.hd, b.bits)));
      mix(uint64_t(b.grid));
      mix(71);  // strata copy layout version
    }
  }
  mix(~0ull);
  for (const GdnDims& g : m->states.gdn) {
    mix(uint64_t(g.nv));
    mix(uint64_t(g.dv));
    mix(uint64_t(g.dk));
    mix(uint64_t(g.ck));
    mix(uint64_t(g.C()));
  }
  if (!m->states.sconv.empty()) {
    mix(0x53434f4e56ull);
    for (const ShortConvSpec& sc : m->states.sconv) {
      mix(uint64_t(sc.conv));
      mix(uint64_t(sc.channels));
    }
  }
  mix(uint64_t(m->states.ring_width));
  return h;
}

uint64_t snapshot_size(const flyx_seq* s, int begin, int pos) {
  uint64_t n = sizeof(SnapshotHeader);
  for (size_t i = 0; i < s->m->states.kv.size(); ++i) {
    const KvSpec& k = s->m->states.kv[i];
    const KvBuf& b = s->kv[i];
    const uint64_t per_pos = uint64_t(k.nkv) * k.hd * (k.fp8 ? 1 : sizeof(kv16)) + (k.fp8 ? k.nkv * sizeof(float) : 0);
    n += 2 * uint64_t(k.layers) * uint64_t(b.ring ? std::min(pos, b.ring) : pos - begin) * per_pos;
    if (b.kq) {  // the strata copy's codes 16 positions at a time (StrataLayout), its grids position by position
      n += 2 * uint64_t(strata_span(pos) - begin / 16 * 16) * k.nkv * strata_words(k.hd, b.bits) * sizeof(uint32_t);
      n += 2 * uint64_t(pos - begin) * k.nkv * sizeof(__half2);
    }
  }
  for (const GdnBuf& g : s->gdn) n += (g.s_stride + g.w_stride) * sizeof(float);
  for (const ShortConvBuf& sc : s->sconv) n += sc.w_stride * sizeof(float);
  return n + uint64_t(s->m->states.ring_width) * sizeof(float);
}

// Every part of a snapshot of the state at `pos`, between the host pieces and the device in the layout above.
// Saving reads checkpoint c's recurrent part; loading writes the current round slot's, as restore() does.
void snapshot_parts(flyx_seq* s, int c, int begin, int pos, HostCursor& host, bool save) {
  auto copy = [&](void* dev, size_t bytes) {
    host.take(bytes, [&](uint8_t* h, uint64_t o, uint64_t len) {
      uint8_t* d = static_cast<uint8_t*>(dev) + o;
      // cudaMemcpyDefault: the checkpoints' recurrent part is in host memory, the K/V on the device
      CK(cudaMemcpyAsync(save ? static_cast<void*>(h) : static_cast<void*>(d), save ? static_cast<const void*>(d) : h,
                         len, cudaMemcpyDefault, s->st));
    });
  };
  bool host_synced = false;
  for (size_t i = 0; i < s->kv.size(); ++i) {
    const KvSpec& k = s->m->states.kv[i];
    const KvBuf& b = s->kv[i];
    // A ring as its slots are: position p in slot p % ring, the last ring positions there.
    const int first = b.ring ? 0 : begin, end = b.ring ? std::min(pos, b.ring) : pos;
    const int slots = b.ring ? b.ring : s->cap;
    const size_t elem = k.fp8 ? 1 : sizeof(kv16), width = size_t(k.nkv) * k.hd;
    if (b.swa_ring && save) {
      const size_t bytes = size_t(end - first) * width * elem;
      copy(b.ckp_k[c], bytes);
      copy(b.ckp_v[c], bytes);
      if (k.fp8) {
        const size_t sbytes = size_t(end - first) * k.nkv * sizeof(float);
        copy(b.ckp_ks[c], sbytes);
        copy(b.ckp_vs[c], sbytes);
      }
    } else {
      for (int l = 0; l < k.layers; ++l) {
        const size_t offset = (size_t(l) * b.layer_stride + size_t(first) * width) * elem;
        const size_t bytes = size_t(end - first) * width * elem;
        copy(static_cast<uint8_t*>(b.k) + offset, bytes);
        copy(static_cast<uint8_t*>(b.v) + offset, bytes);
        if (k.fp8 && b.ks_host) {  // the device part's scales, then the others' from host memory
          const int split = std::clamp(b.device_positions, first, end);
          auto scales = [&](float* device, float* far) {
            if (first < split) copy(device + size_t(first) * k.nkv, size_t(split - first) * k.nkv * sizeof(float));
            if (split < end) copy(far + size_t(split) * k.nkv, size_t(end - split) * k.nkv * sizeof(float));
          };
          scales(b.ks, b.ks_host);
          scales(b.vs, b.vs_host);
        } else if (k.fp8) {
          const size_t offset_scales = (size_t(l) * slots + first) * k.nkv;
          copy(b.ks + offset_scales, size_t(end - first) * k.nkv * sizeof(float));
          copy(b.vs + offset_scales, size_t(end - first) * k.nkv * sizeof(float));
        }
      }
    }
    if (b.kq) {  // one layer's (strata attention): its codes 16 positions at a time, its scales
      // A partially filled tile in the parent is overwritten with the completed tile from the child.
      const int tile_begin = begin / 16 * 16;
      const size_t offset = size_t(tile_begin) * k.nkv * b.words;
      const int file_words = strata_words(k.hd, b.bits);
      const size_t codes = size_t(strata_span(pos) - tile_begin) * k.nkv * file_words * sizeof(uint32_t);
      const size_t scales = size_t(pos - begin) * k.nkv * sizeof(__half2);
      if (b.words != file_words) {  // a permanently hosted 256d three-bit copy, saved and loaded in canonical form
        if (!host_synced) {
          sync_stream(s);  // finish any queued copy_home / stream writes before CPU accesses b.kq / b.vq
          host_synced = true;
        }
        host.take(codes, [&](uint8_t* p, uint64_t off, uint64_t n) {
          strata_snapshot_span<true>(b.kq + offset, p, off, n, save);
        });
        host.take(codes, [&](uint8_t* p, uint64_t off, uint64_t n) {
          strata_snapshot_span<false>(b.vq + offset, p, off, n, save);
        });
      } else {
        copy(b.kq + offset, codes);
        copy(b.vq + offset, codes);
      }
      copy(b.kqs + size_t(begin) * k.nkv, scales);
      copy(b.vqs + size_t(begin) * k.nkv, scales);
    }
  }
  const int sl = flyx_round_slot(s->round);
  for (GdnBuf& g : s->gdn) {
    copy(save ? g.ckp_S[c] : g.S + sl * g.s_stride, g.s_stride * sizeof(float));
    copy(save ? g.ckp_win[c] : g.win + sl * g.w_stride, g.w_stride * sizeof(float));
  }
  for (ShortConvBuf& sc : s->sconv)
    copy(save ? sc.ckp_win[c] : sc.win + sl * sc.w_stride, sc.w_stride * sizeof(float));
  const size_t w = size_t(s->m->states.ring_width) * sizeof(float);
  if (w && pos > 0) copy(save ? s->ring_ckp[c] : s->ring + size_t((pos - 1) % kHring) * (w / sizeof(float)), w);
  else if (w && save)  // no row before position 0
    host.take(w, [](uint8_t* h, uint64_t, uint64_t len) { std::memset(h, 0, len); });
}

}  // namespace

int snapshot_bytes(flyx_seq* s, int32_t c, int32_t begin, uint64_t* out) {
  if (!s) return no_sequence("snapshot_bytes");
  if (c < 0 || c >= FLYX_CHECKPOINTS || !s->ckp[c].valid) return fail("no checkpoint in that slot");
  if (begin < 0 || begin > s->ckp[c].pos) return fail("snapshot starts outside the checkpoint");
  *out = snapshot_size(s, begin, s->ckp[c].pos);
  return 0;
}

int snapshot_save(flyx_seq* s, int32_t c, int32_t begin, const flyx_host_pieces* dst, uint64_t bytes) {
  uint64_t need = 0;
  if (snapshot_bytes(s, c, begin, &need) != 0) return -1;
  const int pos = s->ckp[c].pos;
  if (bytes < need || !pieces_hold(dst, need))
    return fail("a snapshot of this checkpoint takes " + std::to_string(need) + " bytes");
  return guarded([&] {
    check_idle(s);
    const SnapshotHeader h{kSnapshotMagic, begin ? kDeltaVersion : kSnapshotVersion, layout_hash(s), pos, begin, need};
    HostCursor host{dst};
    host.put(&h, sizeof h);
    s->copy_home(false);  // the copy saved from its host home
    snapshot_parts(s, c, begin, pos, host, true);
    sync_stream(s);
    return pos;
  });
}

int snapshot_load(flyx_seq* s, const flyx_host_pieces* src, uint64_t bytes, int32_t anchor) {
  if (!s) return no_sequence("snapshot_load");
  SnapshotHeader h{};
  std::string err;
  if (!flyx::backends::validate_snapshot_header(
          src, bytes, layout_hash(s), s->max_ctx, s->position,
          [&](int begin, int pos) { return snapshot_size(s, begin, pos); }, &h, &err))
    return fail(err);
  return guarded([&] {
    check_idle(s);
    HostCursor host{src, sizeof h};
    s->copy_home(false);  // the copy loaded to its host home, every position changed
    snapshot_parts(s, -1, h.begin, h.pos, host, false);
    s->copy_written(h.begin);
    for (auto& mt : s->meta) mt = {};
    set_params(s, h.pos, anchor, s->round);
    sync_stream(s);  // the bytes are the caller's
    return s->position;
  });
}

int set_oracle(flyx_seq* s, const int32_t* tokens, int32_t n) {
  if (!s) return no_sequence("set_oracle");
  return guarded([&] {
    check_idle(s);
    // The buffer is allocated once in seq_new: captured graphs hold its address.
    n = std::min(n, s->oracle_cap);
    s->test_oracle_len = std::max(n, 0);
    // Through the session's stream (set_params), and waited for: the sources are the caller's.
    if (n > 0) CK(cudaMemcpyAsync(s->oracle, tokens, sizeof(int) * n, cudaMemcpyHostToDevice, s->st));
    CK(cudaMemcpyAsync(&s->P->oracle_len, &n, sizeof(int), cudaMemcpyHostToDevice, s->st));
    sync_stream(s);
    return 0;
  });
}

}  // namespace flyx::cuda
