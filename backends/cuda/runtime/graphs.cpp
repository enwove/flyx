// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The device memory of the round graphs: what an instantiated graph takes at most, by its nodes; the
// room seq_bytes counts for a session's graphs; the session's own graphs and its shares of the batch graphs it is in
// kept inside that room.
#include <cuda.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "backends/cuda/backend.h"

namespace flyx::cuda {

namespace {

// The bytes a kernel's parameters take (the driver's cuFuncGetParamInfo, from CUDA 12.4; through the runtime, as
// runtime/tiers.cpp takes the driver's functions); -1 where the driver has no answer.  Cached per function pointer:
// a kernel's parameter layout is fixed at compile time, avoiding thousands of driver queries per graph capture.
long long kernel_param_bytes(void* func) {
  static std::mutex mu;
  static std::unordered_map<void*, long long> cache;
  {
    std::lock_guard<std::mutex> lock(mu);
    if (const auto it = cache.find(func); it != cache.end()) return it->second;
  }
  using ParamInfo = CUresult (*)(CUfunction, size_t, size_t*, size_t*);
  static const ParamInfo info = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q{};
    if (cudaGetDriverEntryPointByVersion("cuFuncGetParamInfo", &p, 12040, cudaEnableDefault, &q) != cudaSuccess ||
        q != cudaDriverEntryPointSuccess) {
      cudaGetLastError();
      return ParamInfo(nullptr);
    }
    return reinterpret_cast<ParamInfo>(p);
  }();
  long long bytes = -1;
  cudaFunction_t f = nullptr;
  if (info && cudaGetFuncBySymbol(&f, func) == cudaSuccess) {
    size_t end = 0;
    for (size_t i = 0;; ++i) {
      size_t offset = 0, size = 0;
      if (info(reinterpret_cast<CUfunction>(f), i, &offset, &size) != CUDA_SUCCESS) break;
      end = std::max(end, offset + size);
    }
    bytes = static_cast<long long>(end);
  } else {
    cudaGetLastError();
  }
  std::lock_guard<std::mutex> lock(mu);
  cache.emplace(func, bytes);
  return bytes;
}

// A conditional node's body graphs (the driver's cuGraphNodeGetParams, from CUDA 13.2, through the runtime so
// builds against older CUDA 12/13.0/13.1 headers compile and still query conditional bodies on 13.2+ drivers);
// false where the driver has no answer.
bool conditional_bodies(cudaGraphNode_t node, unsigned* count, cudaGraph_t** graphs) {
  static_assert(std::is_same_v<CUgraph, cudaGraph_t> && std::is_same_v<CUgraphNode, cudaGraphNode_t>);
  using GetParams = CUresult(CUDAAPI*)(CUgraphNode, CUgraphNodeParams*);
  static const GetParams get_params = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q{};
    if (cudaGetDriverEntryPointByVersion("cuGraphNodeGetParams", &p, 13020, cudaEnableDefault, &q) != cudaSuccess ||
        q != cudaDriverEntryPointSuccess) {
      cudaGetLastError();
      return GetParams(nullptr);
    }
    return reinterpret_cast<GetParams>(p);
  }();
  if (!get_params) return false;
  CUgraphNodeParams c{};
  if (get_params(node, &c) != CUDA_SUCCESS || c.type != CU_GRAPH_NODE_TYPE_CONDITIONAL || !c.conditional.phGraph_out)
    return false;
  *count = c.conditional.size;
  *graphs = c.conditional.phGraph_out;
  return true;
}

}  // namespace

namespace {

// g's nodes into out, each priced by its kind; the graphs a node holds (a conditional node's bodies, as DV's canvas
// pass is one: regions/layers.cpp dv_context_unless_noop; a child graph) each a graph of its own, kGraphBytes and its
// nodes the same way (cuda_bench graphs: a body of 400 kernels takes 128 KiB more than a chain of 400).
void add_nodes(cudaGraph_t g, const GraphPrices& price, GraphSize& out) {
  size_t n = 0;
  CK(cudaGraphGetNodes(g, nullptr, &n));
  std::vector<cudaGraphNode_t> nodes(n);
  CK(cudaGraphGetNodes(g, nodes.data(), &n));
  out.bytes += kGraphBytes;
  // forks: each node's edges out past its first (programmatic ones among them: their data asked for too, which the
  // driver otherwise refuses to leave out)
  size_t n_edges = 0;
#if CUDART_VERSION >= 13000
  CK(cudaGraphGetEdges(g, nullptr, nullptr, nullptr, &n_edges));
#else
  CK(cudaGraphGetEdges_v2(g, nullptr, nullptr, nullptr, &n_edges));
#endif
  if (n_edges > 0) {
    std::vector<cudaGraphNode_t> from(n_edges), to(n_edges);
    std::vector<cudaGraphEdgeData> data(n_edges);
#if CUDART_VERSION >= 13000
    CK(cudaGraphGetEdges(g, from.data(), to.data(), data.data(), &n_edges));
#else
    CK(cudaGraphGetEdges_v2(g, from.data(), to.data(), data.data(), &n_edges));
#endif
    from.resize(n_edges);
    std::sort(from.begin(), from.end());
    size_t forks = 0;
    for (size_t i = 1; i < from.size(); ++i) forks += from[i] == from[i - 1];
    if (forks) out.bytes += kGraphForkBytes + (forks - 1) * kGraphForkMoreBytes;
    out.forks += forks;
  }
  for (cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type{};
    CK(cudaGraphNodeGetType(node, &type));
    if (type == cudaGraphNodeTypeKernel) {
      cudaKernelNodeParams p{};
      CK(cudaGraphKernelNodeGetParams(node, &p));
      const long long known = kernel_param_bytes(p.func);
      const size_t params = known < 0 ? kGraphParamBytes : size_t(known);  // unknown: priced as at the bound
      ++out.kernels;
      out.params_unknown += known < 0 ? 1 : 0;
      out.params_most = std::max(out.params_most, known < 0 ? 0 : params);
      out.bytes += price.kernel + kGraphParamCost * (params > kGraphParamBytes ? params - kGraphParamBytes : 0);
    } else if (type == cudaGraphNodeTypeMemcpy) {
      ++out.copies;
      out.bytes += price.copy;
    } else if (type == cudaGraphNodeTypeMemset) {
      ++out.sets;
      out.bytes += kGraphSetBytes;
    } else {
      ++out.others;
      out.bytes += kGraphOtherBytes;
      if (type == cudaGraphNodeTypeConditional) {
        unsigned body_count = 0;
        cudaGraph_t* bodies = nullptr;
        if (!conditional_bodies(node, &body_count, &bodies)) {
          ++out.bodies_unknown;  // priced as its node alone: launch and the count capture DV's round again flat
          continue;
        }
        for (unsigned b = 0; b < body_count; ++b) {
          ++out.bodies;
          add_nodes(bodies[b], price, out);
        }
      } else if (type == cudaGraphNodeTypeGraph) {
        cudaGraph_t child = nullptr;
        CK(cudaGraphChildGraphNodeGetGraph(node, &child));
        ++out.bodies;
        add_nodes(child, price, out);
      }
    }
  }
}

}  // namespace

GraphSize graph_size(cudaGraph_t g, const GraphPrices& price) {
  GraphSize out;
  add_nodes(g, price, out);
  return out;
}

void plan_lowered(flyx_bmodel* m, const flyx_plan* p) {
  PlanGraph g;
  for (size_t i = 0; i < p->nodes.size(); ++i)
    if (p->nodes[i].kind == FLYX_NODE_DRAFT && p->block[i] && std::holds_alternative<MtpBlock>(*p->block[i]))
      g.mtp_steps += std::max(0, p->nodes[i].arg0);
  std::lock_guard<std::mutex> lock(m->graph_mu);
  m->round_graphs[p] = g;
}

void plan_counted(flyx_bmodel* m, const flyx_plan* p, size_t bytes) {
  std::lock_guard<std::mutex> lock(m->graph_mu);
  const auto it = m->round_graphs.find(p);
  if (it == m->round_graphs.end() || it->second.counted) return;
  it->second.bytes = bytes;
  it->second.counted = true;
}

void forget_graphs(flyx_bmodel* m, const flyx_plan* p) {
  std::lock_guard<std::mutex> lock(m->graph_mu);
  m->round_graphs.erase(p);
}

namespace {

// The round plans no session captured yet, sized: a greedy round of each captured on a session of its own (every
// position on the GPU and read exactly), not instantiated, nothing run.  graph_mu held.  Only seq_size asks for it
// (the load's rounds count every plan the engine runs; a plan lowered past them, or a load that ran none, comes
// here); the session runs no batch, so it goes without seq_free's look at the batch graphs.
void count_round_graphs(flyx_bmodel* m) {
  bool all = true;
  for (const auto& [plan, g] : m->round_graphs) all = all && g.counted;
  if (all) return;
  flyx_seq* s = nullptr;
  if (const int rc = seq_new(m, kGraphCountCtx, &s); rc != 0) {
    const auto why = "sizing the round graphs: " + g_error;
    if (rc == FLYX_ERR_OOM) throw OutOfDeviceMemory(why);
    throw std::runtime_error(why);
  }
  const std::unique_ptr<flyx_seq> scratch(s);
  for (auto& [plan, g] : m->round_graphs)
    if (!g.counted) {
      g.bytes = round_graph_size(const_cast<flyx_plan*>(plan), s).bytes;
      g.counted = true;
    }
}

}  // namespace

size_t graph_keep(const flyx_seq& s, bool count) {
  auto* m = const_cast<flyx_bmodel*>(s.m);
  if (!m->opt.graphs) return 0;
  // What a round takes past its plan's count: other heads, the K/V in tiers, rounds that read low-bit copies.
  size_t kernels = kGraphHeadNodes, copies = 0, others = 0;
  bool window = false;
  for (const KvBuf& b : s.kv) {
    kernels += b.vk ? kTierNodes : 0;
    window = window || b.window > 0;
  }
  if (s.kv_method == FLYX_KV_STRATA) {
    kernels += kStrataNodes * size_t(std::max(0, s.copies));
    copies += kStrataCopyNodes * size_t(std::clamp(s.copy_host, 0, std::max(0, s.copies)));
  }
  const size_t page_n = s.page_spill_auto ? 0 : s.page_layers.size();
  if (page_n > 0) {
    kernels += 1 + page_n;  // kv_drain_pageable + extra claim/list/install kernel per pageable layer
    others += page_n;       // cudaLaunchHostFunc node per pageable layer
  }
  std::lock_guard<std::mutex> lock(m->graph_mu);
  if (count) count_round_graphs(m);
  size_t room = 0, most = 0;
  for (const auto& [plan, g] : m->round_graphs) {
    if (!g.bytes) continue;  // not counted yet, or no graph of it can be made
    const size_t steps = window ? size_t(g.mtp_steps) : 0;
    size_t round = 0;
    if (s.page_spill_auto) {
      // Auto mode may instantiate either all-pinned adaptive graphs or flat pageable graphs (up to s.copies layers).
      const size_t ad_branches = plan->adaptive() ? std::max<size_t>(1, plan->checkpoints.size()) : 1;
      const size_t round_pinned = g.bytes + (ad_branches * kernels + kMtpWindowNodes * steps) * m->graph_prices.kernel +
                                  ad_branches * copies * m->graph_prices.copy + ad_branches * others * kGraphOtherBytes;
      const size_t page_k = s.copies > 0 ? 1 + size_t(s.copies) : 0;
      const size_t page_o = s.copies > 0 ? size_t(s.copies) : 0;
      const size_t round_paged = g.bytes + (kernels + page_k + kMtpWindowNodes * steps) * m->graph_prices.kernel +
                                 copies * m->graph_prices.copy + (others + page_o) * kGraphOtherBytes;
      round = std::max(round_pinned, round_paged);
    } else {
      const size_t branches = (plan->adaptive() && s.page_layers.empty())
                                  ? std::max<size_t>(1, plan->checkpoints.size())
                                  : 1;
      round = g.bytes + (branches * kernels + kMtpWindowNodes * steps) * m->graph_prices.kernel +
              branches * copies * m->graph_prices.copy + branches * others * kGraphOtherBytes;
    }
    room += round;
    most = std::max(most, round);
  }
  // And a share of a batch graph beside them, so that a session going from rounds alone to rounds with another and
  // back keeps both graphs: a batch graph of n sessions' rounds holds fewer nodes than their n rounds alone (each
  // weight's GEMM is one launch for all their rows), so its share is at most the largest round's.
  return room + (m->batch ? most : 0);
}

size_t graph_room(const flyx_seq& s, bool count) {
  // What the session's graphs may hold, and a kGraphPoolPart-th more that they may not: the driver's pool keeps what
  // the graphs a session gave up held and gives it to later graphs only where they fit.
  const size_t keep = graph_keep(s, count);
  return keep + keep / kGraphPoolPart;
}

namespace {

// Room for `bytes` more in the session's graphs: the graph it ran least lately goes first, its own or a batch graph it
// is in alike (both by its own rounds' seq_no: a batch graph a session left behind for rounds alone gives way to them
// however few batches run since), once that launch was read (finished: what the graph holds is the driver's pool's
// again at once, for the next graph) and it ran the graph in none of its last kGraphRecent rounds.  B: the model's,
// its lock held.  False: no room.
bool make_room(flyx_seq* s, size_t bytes, BatchRounds* B) {
  const size_t room = graph_keep(*s);
  auto idle = [&](unsigned long long ran) { return ran <= s->read_no && ran + kGraphRecent <= s->seq_no; };
  bool synced = false;
  while (s->graph_held + bytes > room) {
    unsigned long long oldest = ~0ull;
    auto own = s->graph.end();
    for (auto g = s->graph.begin(); g != s->graph.end(); ++g)
      if (idle(g->second.used) && g->second.used < oldest) own = g, oldest = g->second.used;
    bool shared = false;
    BatchGraphs::iterator batch;
    if (B)
      for (auto g = B->graphs.begin(); g != B->graphs.end(); ++g)
        for (int j = 0; j < g->first.n; ++j)
          if (g->first.seqs[j].seq == uintptr_t(s) && idle(g->second.ran[j]) && g->second.ran[j] < oldest)
            shared = true, batch = g, oldest = g->second.ran[j];
    if (shared) {
      drop_batch_graph(*B, batch, &synced);
    } else if (own != s->graph.end()) {
      CK(cudaGraphExecDestroy(own->second.exec));
      s->graph_held -= own->second.bytes;
      s->graph.erase(own);
    } else {
      return false;
    }
  }
  return true;
}

}  // namespace

bool make_graph_room(flyx_seq* s, size_t bytes) {
  if (s->graph_held + bytes <= graph_keep(*s)) return true;
  BatchRounds* B = s->m->batch.get();
  if (!B) return make_room(s, bytes, nullptr);
  std::lock_guard<std::mutex> lock(B->mu);
  return make_room(s, bytes, B);
}

BatchGraphs::iterator drop_batch_graph(BatchRounds& B, BatchGraphs::iterator it, bool* synced, bool quiet) {
  // An error throws; where the caller cannot throw (seq_free, quiet) it is let go, as the driver leaves it.
  auto check = [&](cudaError_t e, const char* what) {
    if (e == cudaSuccess) return;
    cudaGetLastError();
    if (!quiet) ck(e, what);
  };
  if (!*synced) check(cudaStreamSynchronize(B.st), "cudaStreamSynchronize(B.st)");  // no launch of it in flight
  *synced = true;
  const cudaGraphExec_t exec = it->second.exec;
  for (int i = 0; i < it->first.n; ++i)
    reinterpret_cast<flyx_seq*>(it->first.seqs[i].seq)->graph_held -= it->second.share;
  it = B.graphs.erase(it);
  check(cudaGraphExecDestroy(exec), "cudaGraphExecDestroy(exec)");
  return it;
}

bool make_batch_room(BatchRounds& B, flyx_seq* const* seqs, int n, size_t share) {
  for (int i = 0; i < n; ++i)
    if (!make_room(seqs[i], share, &B)) return false;
  return true;
}

int graph_stats(flyx_seq* s, flyx_graph_stats* out) {
  if (!s) return no_sequence("graph_stats");
  try {
    *out = s->graph_stats;
    out->held = s->graph_held;
    out->room = graph_room(*s);
    return 0;
  } catch (const std::exception& e) {
    return fail(e.what());
  }
}

}  // namespace flyx::cuda
