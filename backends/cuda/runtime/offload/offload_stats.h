// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <cstdio>
#include <map>
#include <utility>

namespace flyx::cuda {
struct MoeBlock;
struct ExpertTimes {
  uint64_t rows = 0, visits = 0, cold = 0, pairs = 0, bytes = 0;
  double handoff = 0, cpu = 0, hot_gpu = 0, upload = 0, reduce = 0;
};

struct OffloadStats {
  struct CopyStats {
    uint64_t bytes = 0, samples = 0;
    double milliseconds = 0;
  } copies[2];  // prefill and round
  bool verbose = false, profile = false;
  std::map<std::pair<const MoeBlock*, bool>, ExpertTimes> times;
  uint64_t decode_slots = 0, decode_cold = 0, last_decode_slots = 0, last_decode_cold = 0;
  uint64_t visits = 0, chunks = 0, slots = 0, cpu_pairs = 0;
  double route_seconds = 0, cpu_seconds = 0, decode_handoff = 0, decode_cpu = 0;
  uint64_t decode_visits = 0, decode_bytes = 0;
  void report(FILE* file, size_t threads) const {
    for (int round = 0; round < 2; ++round) {
      const CopyStats& phase = copies[round];
      if (phase.milliseconds > 0)
        std::fprintf(file, "flyx-cuda: %s pinned layer H2D %.2f GB/s (%llu sampled transfers, %llu bytes)\n",
                     round ? "round" : "prefill", phase.bytes / (phase.milliseconds * 1e6),
                     static_cast<unsigned long long>(phase.samples), static_cast<unsigned long long>(phase.bytes));
    }
    for (const auto& [key, t] : times) {
      if (!t.visits) continue;
      std::fprintf(file,
                   "flyx-cuda: expert profile block=%p phase=%s visits=%llu rows=%llu "
                   "handoff=%.2f cpu=%.2f hot_gpu=%.2f h2d=%.2f reduce=%.2f us/visit "
                   "cold_reads=%.2f GB/s resident=%.2f%%\n",
                   static_cast<const void*>(key.first), key.second ? "round" : "prefill",
                   static_cast<unsigned long long>(t.visits), static_cast<unsigned long long>(t.rows),
                   t.handoff / t.visits, t.cpu / t.visits, t.hot_gpu / t.visits, t.upload / t.visits,
                   t.reduce / t.visits, t.cpu ? t.bytes / (t.cpu * 1e3) : 0., 100. * (t.pairs - t.cold) / t.pairs);
    }
    if (verbose && decode_visits)
      std::fprintf(file, "flyx-cuda: decode %.2f us/visit route handoff, %.2f us/visit CPU, %.2f GB/s cold reads\n",
                   decode_handoff * 1e6 / decode_visits, decode_cpu * 1e6 / decode_visits,
                   decode_cpu > 0 ? decode_bytes / decode_cpu / 1e9 : 0.);
    if (verbose && decode_slots)
      std::fprintf(file, "flyx-cuda: decode expert cache hit rate %.2f%% (%llu/%llu routed slots)\n",
                   100. * (decode_slots - decode_cold) / decode_slots,
                   static_cast<unsigned long long>(decode_slots - decode_cold),
                   static_cast<unsigned long long>(decode_slots));
    if (verbose && visits)
      std::fprintf(file,
                   "flyx-cuda: hybrid expert route/activation wait %.1f us per visit, CPU work %.1f us per visit "
                   "(%llu visits, %llu chunks, %llu of %llu routed pairs on CPU, %zu pool threads)\n",
                   route_seconds * 1e6 / visits, cpu_seconds * 1e6 / visits, static_cast<unsigned long long>(visits),
                   static_cast<unsigned long long>(chunks), static_cast<unsigned long long>(cpu_pairs),
                   static_cast<unsigned long long>(slots), threads);
  }
};
}  // namespace flyx::cuda
