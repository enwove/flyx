// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "backends/cuda/runtime/offload/offload_plan.h"

#include <map>
#include <set>

namespace flyx::cuda {

void OffloadPlan::fit_pinned(uint64_t budget) {
  if (pinned_bytes <= budget) return;
  if (stream.host_bytes && !bounce_layers) {
    pinned_bytes = size_add(pinned_bytes - stream.host_bytes, stream.ring_bytes);
    pageable_bytes = size_add(pageable_bytes, stream.host_bytes);
    bounce_layers = true;
  }
  if (pinned_bytes > budget) throw std::runtime_error("offload staging exceeds pinned host budget");
}

OffloadPlan plan_offload(const flyx_model_desc& d, const BackendOptions& o) {
  if (d.n_blocks < 0 || (d.n_blocks && !d.blocks) || d.n_weights < 0 || (d.n_weights && !d.weights))
    throw std::runtime_error("invalid offload model tables");
  OffloadPlan plan;
  plan.experts.resize(size_t(d.n_weights));
  std::vector<char> draft(size_t(d.n_blocks), 0);
  std::map<int, int> partner;
  auto link = [&](int a, int b) {
    auto [it, inserted] = partner.emplace(a, b);
    if (!inserted && it->second != b) it->second = -1;
  };
  auto weight = [&](int i) -> const flyx_weight& {
    if (i < 0 || i >= d.n_weights || !d.weights[i].t) throw std::runtime_error("missing MoE weight");
    return d.weights[i];
  };
  auto dimensions = [&](int w, std::initializer_list<int> shape) {
    const auto& t = *weight(w).t;
    if (!t.data || t.ndim != int(shape.size()) || !std::equal(shape.begin(), shape.end(), t.shape))
      throw std::runtime_error("MoE projection dimensions do not match block");
  };
  for (int i = 0; i < d.n_blocks; ++i) {
    const auto& b = d.blocks[i];
    if (o.gpu_layers != INT_MAX && b.kind == FLYX_BLOCK_MTP) {
      draft.at(size_t(b.mtp.attn)) = draft.at(size_t(b.mtp.mlp)) = 1;
    }
    if (b.kind != FLYX_BLOCK_MOE) continue;
    const auto& f = b.moe;
    if (f.hidden < 1 || f.inter < 1 || f.inter > INT_MAX / 2 || f.n_experts < 1 || f.n_experts > kMoeRouteMaxExperts ||
        f.top_k < 1 || f.top_k > std::min(f.n_experts, kMoeRouteMaxTopK))
      throw std::runtime_error("invalid MoE dimensions");
    dimensions(f.router, {f.n_experts, f.hidden});
    check_linear_payload(weight(f.router));
    dimensions(f.gate_up, {f.n_experts, 2 * f.inter, f.hidden});
    dimensions(f.down, {f.n_experts, f.hidden, f.inter});
    if (f.router_bias >= 0) {
      dimensions(f.router_bias, {f.n_experts});
      check_linear_payload(weight(f.router_bias));
    }
    if ((d.hidden && f.hidden != d.hidden) || int64_t(2) * f.inter * f.top_k > INT_MAX ||
        int64_t(f.hidden) * f.top_k > INT_MAX)
      throw std::runtime_error("MoE routed row exceeds CUDA row dimensions");
    if (o.gpu_experts < f.n_experts) plan.cold.include(f.hidden, f.inter, f.top_k);
    link(f.gate_up, f.down);
    link(f.down, f.gate_up);
    for (int w : {f.gate_up, f.down}) {
      if (plan.experts[w]) continue;
      plan.experts[w] = 1;
      const auto& src = weight(w);
      const auto layout = expert_layout(src, o.packed_blocks);
      const size_t e = size_t(f.n_experts), hot = std::min(e, size_t(o.gpu_experts));
      plan.resident_expert_bytes = size_add(plan.resident_expert_bytes, size_mul(hot, layout.bytes() / e));
      if (hot == e) continue;
      if (o.keep_expert_archive) {
        const auto bank = ExpertBankLayout::from(layout, e, e - hot);
        plan.cold.bank_bytes = std::max(plan.cold.bank_bytes, bank.bytes(bank.count));
        plan.pageable_bytes = size_add(plan.pageable_bytes, bank.archive_bytes(e - hot));
      }
      const size_t raw = layout.format == ExpertLayout::bf16_rows ? layout.payload_bytes() : src.t->nbytes;
      plan.pageable_bytes =
          size_add(plan.pageable_bytes, size_mul(e - hot, raw / e + (src.scale2 ? sizeof(float) : 0)));
    }
  }
  std::set<int> placed, global_sizes;
  for (int i = 0; i < d.n_blocks; ++i) {
    if (d.blocks[i].kind != FLYX_BLOCK_MOE) continue;
    const auto& f = d.blocks[i].moe;
    if (!o.keep_expert_archive || !o.expert_adapt || !o.gpu_experts || o.gpu_experts >= f.n_experts ||
        f.gate_up == f.down || partner[f.gate_up] != f.down || partner[f.down] != f.gate_up ||
        !placed.insert(f.gate_up).second)
      continue;
    plan.placement_blocks.push_back(i);
    const size_t n = f.n_experts;
    plan.workspace_bytes = size_add(plan.workspace_bytes, n * (sizeof(int) + sizeof(uint32_t)));
    plan.pageable_bytes = size_add(plan.pageable_bytes, n * (2 * sizeof(int) + sizeof(uint64_t) + sizeof(uint32_t)));
    if (o.expert_global && global_sizes.insert(f.n_experts).second)
      plan.pageable_bytes = size_add(plan.pageable_bytes, n * sizeof(uint64_t));
  }
  plan.stream = estimate_stream(d, draft, o);
  plan.workspace_bytes = size_add(plan.workspace_bytes, size_add(plan.stream.ring_bytes, plan.cold.device_bytes()));
  plan.pageable_bytes = size_add(plan.pageable_bytes, plan.cold.pageable_bytes());
  plan.pinned_bytes = size_add(plan.cold.pinned_bytes(), plan.stream.host_bytes);
  return plan;
}
}  // namespace flyx::cuda
