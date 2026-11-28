// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "core/session/session.h"
#include "nlohmann/json.hpp"
#include "tests/common/check.h"

using namespace flyx;
using flyx::test::check;
namespace {
void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}
void check_rounds(Engine& engine, const std::vector<int32_t>& prompt, bool sampled,
                  std::vector<float>* decode_trace) {
  const auto* api = engine.api();
  const int at = int(prompt.size());
  std::string error;
  Session plain(&engine), spec(&engine), loaded(&engine);
  for (auto* s : {&plain, &spec, &loaded}) require(s->init(&error), error);
  flyx_gen_params g{};
  g.temperature = sampled ? .8f : 0.f;
  g.top_p = 1;
  g.seed = 1123;
  auto read = [&](Session& s) {
    flyx_round_result r{};
    require(api->read(s.seq(), &r) == 0, api->last_error());
    return r;
  };
  auto round = [&](Session& s, flyx_plan* plan, const flyx_gen_params& gen) {
    require(api->launch(plan, s.seq(), &gen) == 0, api->last_error());
    return read(s);
  };
  auto plain_plan = engine.round_plan(0);
  std::vector<int> want;
  int anchor = -1;
  for (auto* s : {&plain, &spec}) {
    require(api->prefill(engine.prefill_plan(), s->seq(), prompt.data(), at, &g) == 0, api->last_error());
    auto r = read(*s);
    require(r.n_tokens == 1, "prefill returns an anchor");
    if (anchor >= 0) require(r.tokens[0] == anchor, "same prefix anchor");
    anchor = r.tokens[0];
  }
  require(api->checkpoint(spec.seq(), 0) == at, "checkpoint position");
  uint64_t bytes = 0;
  require(api->snapshot_bytes(spec.seq(), 0, 0, &bytes) == 0, api->last_error());
  std::vector<uint8_t> storage(static_cast<size_t>(bytes));
  void* data = storage.data();
  flyx_host_pieces pieces{&data, 1, 0, std::max<uint64_t>(4096, bytes)};
  require(api->snapshot_save(spec.seq(), 0, 0, &pieces, bytes) == at, api->last_error());
  require(api->snapshot_load(loaded.seq(), &pieces, bytes, anchor) == at, api->last_error());
  for (int j = 0; j < 14; ++j) {
    want.push_back(round(plain, plain_plan, g).tokens[0]);
    if (decode_trace) {
      const auto start = decode_trace->size();
      decode_trace->resize(start + engine.model().desc().vocab);
      require(api->logits(plain.seq(), 0, decode_trace->data() + start) == 0, api->last_error());
    }
  }
  if (decode_trace) {
    flyx_graph_stats stats{};
    require(api->graph_stats && api->graph_stats(plain.seq(), &stats) == 0, "CUDA graph statistics");
    require(stats.captured > 0 && stats.from_graph == 14 && stats.captured < stats.from_graph,
            "plain decode captures and replays its CUDA graph");
    std::printf("decode graph launches %llu, captures %llu\n", (unsigned long long)stats.from_graph,
                (unsigned long long)stats.captured);
  }
  for (int j = 0; j < 3; ++j)
    require(round(loaded, plain_plan, g).tokens[0] == want[j], "snapshot preserves sampled anchor");
  auto plan_for = [&](bool tree) {
    const auto& widths = tree ? engine.tree_widths() : engine.lookup_widths();
    auto it = std::find_if(widths.begin(), widths.end(), [](int n) { return n >= 3 && n <= 15; });
    require(it != widths.end(), "missing given-draft verification plan");
    const size_t i = size_t(it - widths.begin());
    return std::make_pair(*it, tree ? engine.tree_plan(i) : engine.lookup_plan(i));
  };
  for (bool tree : {false, true}) {
    const auto [width, plan] = plan_for(tree);
    for (bool reject : {false, true}) {
      require(api->restore(spec.seq(), 0, anchor) == at, "restore prefix");
      std::vector<int32_t> drafts(width, (want[0] + 1) % engine.model().desc().vocab), parents(width, -1);
      if (!reject) {
        const int first = tree ? 1 : 0;
        drafts[first] = want[0];
        drafts[first + 1] = want[1];
        parents[first + 1] = first;
        if (!tree && width >= 3) drafts[2] = (want[2] + 1) % engine.model().desc().vocab;
      }
      auto oracle = g;
      oracle.drafts = drafts.data();
      oracle.n_drafts = width;
      oracle.drafts_at = at + 1;
      oracle.parents = tree ? parents.data() : nullptr;
      auto r = round(spec, plan, oracle);
      require(r.n_accepted == (reject ? 0 : 2), "accepted path length");
      require(r.n_tokens == r.n_accepted + 1 && std::equal(r.tokens, r.tokens + r.n_tokens, want.begin()),
              "verified path agrees with plain");
      const int n = r.n_tokens;
      require(round(spec, plain_plan, g).tokens[0] == want[n], "post-commit continuation");
      require(api->rewind(spec.seq(), at + 1, want[0]) == 1, "rewind inside verified round");
      require(round(spec, plain_plan, g).tokens[0] == want[1], "rewound continuation");
    }
  }
}

std::vector<float> reference_trace(const std::string& stem, const std::vector<int32_t>& prompt) {
  EngineOptions opt;
  opt.backend = "cpu_ref";
  opt.max_ctx = 128;
  opt.ple_placement = FLYX_PLE_HOST;
  Engine engine;
  std::string error;
  require(engine.load(stem + ".gguf", opt, &error), error);
  Session session(&engine);
  require(session.init(&error), error);
  const auto* api = engine.api();
  const int vocab = engine.model().desc().vocab;
  std::vector<float> trace(prompt.size() * vocab);
  flyx_gen_params gen{};
  for (size_t i = 0; i < prompt.size(); ++i) {
    flyx_round_result result{};
    require(api->prefill(engine.prefill_plan(), session.seq(), &prompt[i], 1, &gen) == 0, api->last_error());
    require(api->read(session.seq(), &result) == 0, api->last_error());
    require(api->logits(session.seq(), 0, trace.data() + i * vocab) == 0, api->last_error());
  }
  return trace;
}

void fixture(const std::string& stem, bool parent, const std::string& backend, bool fused) {
  const auto golden = nlohmann::json::parse(std::ifstream(stem + ".json"));
  const auto prompt = golden.at("prompt").get<std::vector<int32_t>>();
  const auto wanted = golden.at("logits").get<std::vector<std::vector<float>>>();
  const auto precise =
      golden.at("reference64").get<std::vector<std::vector<double>>>();
  const auto cpu_logits = backend == "cuda" ? reference_trace(stem, prompt) : std::vector<float>();
  std::vector<float> baseline, decode_baseline;
  for (int tier : {FLYX_PLE_GPU, FLYX_PLE_HOST, FLYX_PLE_DISK, FLYX_PLE_HYBRID}) {
    if (!parent && tier != FLYX_PLE_GPU) continue;
    EngineOptions opt;
    opt.backend = backend;
    if (backend == "cuda")
      opt.backend_options = fused ? "head_cert=0" : "head_cert=0,fusion=0";
    opt.prefill_chunk = 32;
    opt.max_ctx = 128;
    opt.ple_placement = tier;
    opt.ple_hot_bytes = 1024;
    Engine engine;
    std::string err;
    require(engine.load(stem + ".gguf", opt, &err), err);
    const auto& desc = engine.model().desc();
    const auto states_of = [&](int kind) {
      return std::count_if(desc.states, desc.states + desc.n_states, [kind](const flyx_state& s) { return s.kind == kind; });
    };
    check(states_of(FLYX_STATE_PLE) == int(parent), "PLE state exists exactly when a table is present");
    check(states_of(FLYX_STATE_INDEXER) == int(parent), "dense attention reserves no indexer state");
    check(engine.tokenizer().prefix().empty(), "a GGUF BOS ID alone adds no prefix");
    const int text_end = desc.vocab - 1;
    check(engine.model_file().eos_ids() == std::vector<int32_t>{text_end - 1, text_end}, "both named stop tokens");
    std::vector<float> decoded;
    check_rounds(engine, prompt, false, backend == "cuda" ? &decoded : nullptr);
    check_rounds(engine, prompt, true, backend == "cuda" ? &decoded : nullptr);
    if (decode_baseline.empty()) decode_baseline = decoded;
    else check(same_bits(decoded, decode_baseline), "graph-replayed logits match across every placement");
    const auto* api = engine.api();
    std::vector<float> trace;
    double maximum = 0, max_double_error = 0, max_cpu_error = 0;
    for (int chunk : {1, 5, 32}) {
      Session s(&engine);
      require(s.init(&err), err);
      flyx_gen_params gen{};
      for (int i = 0; i < int(prompt.size()); i += chunk) {
        int n = std::min(chunk, int(prompt.size()) - i);
        require(api->prefill(engine.prefill_plan(), s.seq(), prompt.data() + i, n, &gen) == 0, api->last_error());
        flyx_round_result result{};
        require(api->read(s.seq(), &result) == 0, api->last_error());
        std::vector<float> logits(wanted.back().size());
        // CUDA projects only the final prefill row and places its logits in
        // slot zero.
        require(api->logits(s.seq(), backend == "cuda" ? 0 : n - 1,
                            logits.data()) == 0,
                api->last_error());
        const auto& ref = wanted[i + n - 1];
        for (size_t j = 0; j < logits.size(); ++j) {
          require(std::isfinite(logits[j]), "finite output logits");
          if (!cpu_logits.empty())
            max_cpu_error = std::max(max_cpu_error, std::abs(double(logits[j]) - cpu_logits[(i + n - 1) * logits.size() + j]));
          maximum = std::max(maximum, std::abs(double(logits[j]) - ref[j]));
          max_double_error =
              std::max(max_double_error,
                       std::abs(double(logits[j]) - precise[i + n - 1][j]));
        }
        if (chunk == 1)
          trace.insert(trace.end(), logits.begin(), logits.end());
        else if (backend == "cpu_ref")
          check(std::equal(logits.begin(), logits.end(), trace.begin() + size_t(i + n - 1) * logits.size()),
                "chunk and step logits match exactly");
      }
    }
    std::printf("%s tier %d max absolute logits error %.8g\n", stem.c_str(), tier, maximum);
    std::printf("  float64 reference logits max absolute error %.8g\n", max_double_error);
    if (!cpu_logits.empty()) {
      std::printf("  cpu_ref logits max absolute error %.8g\n", max_cpu_error);
      check(max_cpu_error < 0.04, "CUDA logits versus cpu_ref");
    }
    if (backend == "cpu_ref") {
      check(max_double_error < 5e-6, "float64 reference logits");
    }
    // CUDA stores attention K/V in binary16 and wide prefill GEMMs round their inputs.
    check(maximum < (backend == "cpu_ref" ? 0.0002 : 0.04), "eager Transformers logits");
    if (baseline.empty())
      baseline = trace;
    else
      check(same_bits(trace, baseline), "placement preserves every logit at every position");
  }
}
}  // namespace
int main(int argc, char** argv) {
  if (argc < 2 || argc > 4) return 2;
  const std::string backend = argc >= 3 ? argv[2] : "cpu_ref";
  if (argc == 4 && std::string(argv[3]) != "unfused") return 2;
  const bool fused = argc < 4;
  try {
    fixture(std::string(argv[1]) + "/qwen4exp_coder", false, backend, fused);
    fixture(std::string(argv[1]) + "/qwen4exp_parent", true, backend, fused);
    fixture(std::string(argv[1]) + "/qwen4exp_mixed", true, backend, fused);
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "%s\n", ex.what());
    return 1;
  }
  return flyx::test::finish("qwen4exp");
}
