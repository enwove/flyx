// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "backends/cuda/runtime/offload/expert_store.h"
#include "backends/cuda/backend.h"

namespace flyx::cuda {
void pack_expert_nvfp4(const uint8_t* in, size_t blocks, uint8_t* qo, uint8_t* so) {
  for (size_t k = 0; k < blocks; ++k, in += 36) {
    const uint8_t* d = in;
    const uint8_t* qs = in + 4;
    for (int g = 0; g < 4; ++g) {
      if (d[g] == 0xFF) throw std::runtime_error("an NVFP4 scale of 0xFF (no e4m3 holds it)");
      so[4 * k + size_t(g)] = d[g] == 0x7F ? 0 : uint8_t(d[g] & 0x7F);
      auto code = [&](int l) { return l < 8 ? qs[8 * g + l] & 15 : qs[8 * g + l - 8] >> 4; };
      for (int i = 0; i < 8; ++i) qo[32 * k + 8 * size_t(g) + size_t(i)] = uint8_t(code(2 * i) | code(2 * i + 1) << 4);
    }
  }
}

void ColdMatrix::pack_banks(const ExpertLayout& layout, size_t all) {
  const size_t cold = all - first;
  bank = ExpertBankLayout::from(layout, all, cold);
  archive.resize(bank.archive_bytes(cold));
  size_t offset = 0;
  for (size_t e = 0; e < cold; e += bank.count) {
    const size_t n = std::min(bank.count, cold - e);
    uint8_t* dst = archive.data() + offset;
    const uint8_t* src = data.data() + e * size_t(rows) * stride;
    auto* scale = dst + bank.scale_offset(n);
    if (nvfp4)
      pack_expert_nvfp4(src, n * size_t(rows) * (cols / 64), dst, scale);
    else if (layout.format == ExpertLayout::ggml_blocks)
      blocks_pack(dtype, src, n * size_t(rows) * (cols / blocks_layout(dtype).values), dst,
                  reinterpret_cast<uint16_t*>(scale));
    else
      std::memcpy(dst, src, n * bank.payload);
    if (bank.gains) {
      auto* g = reinterpret_cast<float*>(dst + bank.gain_offset(n));
      if (scales.empty())
        std::fill_n(g, n, 1.f);
      else
        std::copy_n(scales.data() + e, n, g);
    }
    offset += bank.bytes(n);
  }
}

void ColdMatrix::exchange(int hot_slot, int cold_slot, int total, uint8_t* bounce) {
  if (!bank.count || hot_slot < 0 || hot_slot >= first || cold_slot < first || cold_slot >= total || !bounce)
    throw std::logic_error("invalid expert exchange");
  const int cold = cold_slot - first;
  const size_t group = size_t(cold) / bank.count;
  const size_t lane = size_t(cold) % bank.count;
  const size_t n = std::min(bank.count, size_t(total - first) - group * bank.count);
  uint8_t* image = archive.data() + group * bank.bytes(bank.count);
  auto plane = [&](void* gpu, size_t bytes, size_t image_offset, size_t bounce_offset) {
    if (!bytes) return;
    auto* device = static_cast<uint8_t*>(gpu) + size_t(hot_slot) * bytes;
    auto* host = image + image_offset + lane * bytes;
    CK(cudaMemcpy(bounce + bounce_offset, device, bytes, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice));
    std::memcpy(host, bounce + bounce_offset, bytes);
  };
  plane(resident.payload, bank.payload, 0, 0);
  plane(resident.scale, bank.scale, bank.scale_offset(n), bank.scale_offset(1));
  plane(resident.gains, bank.gains, bank.gain_offset(n), bank.gain_offset(1));
  auto* raw = data.data() + size_t(cold) * rows * stride;
  auto* scale = bounce + bank.scale_offset(1);
  if (q8 || a16h) {
    blocks_unpack(dtype, bounce, reinterpret_cast<uint16_t*>(scale),
                  size_t(rows) * (cols / blocks_layout(dtype).values), raw);
  } else if (nvfp4) {
    for (size_t i = 0; i < size_t(rows) * (cols / 64); ++i) {
      std::memcpy(raw, scale + i * 4, 4);
      for (int g = 0; g < 4; ++g) {
        const auto* q = bounce + 32 * i + 8 * g;
        for (int j = 0; j < 8; ++j) {
          const int shift = (j & 1) * 4;
          raw[4 + 8 * g + j] = uint8_t(((q[j / 2] >> shift) & 15) | ((q[4 + j / 2] >> shift) << 4));
        }
      }
      raw += 36;
    }
  } else
    std::memcpy(raw, bounce, bank.payload);
  if (!scales.empty()) std::memcpy(&scales[cold], bounce + bank.gain_offset(1), sizeof(float));
}

}  // namespace flyx::cuda
