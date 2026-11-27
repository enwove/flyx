// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// How far one setting's next-token distributions are from another's, position by position, on texts fed to the model
// token by token (teacher forcing): KL divergence, top-1 agreement, perplexity.
//
//   kld MODEL --text FILE [--text FILE ...] [--tokens N] [--start S] [--backend-options O] [--activations exact]
//       [--kv fp8|16]           the K/V cache's format (the model's default otherwise)
//       [--kv-method M]         how rounds read it (exact, strata[:BITS])
//       [--kv-ram GB]           host memory for K/V the GPU does not hold (the engine's default otherwise)
//       [--ctx N]               context length (default: auto)
//       [--prefill-chunk N]     prefill chunk size (default: auto)
//       [--mmproj FILE]         multimodal vision projector GGUF
//       [--lora DIR]            a LoRA adapter as Hugging Face PEFT saves it (EngineOptions::lora_file)
//       [--image FILE ...]      pair each --text (assistant reply) 1:1 with an image prompt
//       [--prompt TEXT]         user prompt sent with each --image (default: "Describe this image in detail.")
//       [--no-think]            disable thinking in multimodal chat template
//       [--vision-device D]     disk | host | gpu | auto (default: disk)
//       [--vision-quant Q]      q8 | exact | nvfp4 (default: q8)
//       --save REF.bin          the distributions, for comparisons against them
//       --ref REF.bin           compare with a file --save wrote (the same model's vocabulary, texts and tokens)
//
// Each text: a fresh session prefills its first S tokens, then every position up to S + N is one plain round (one row:
// the kernels plain decoding runs, and by rule 4 every round's rows, bit for bit) whose input is the text's token,
// whatever the model chose (rewind() puts the text's token in).  The head's logits come back through the backend's
// logits() (backend option head_cert=0 is added).  --save writes, per position, the log-probabilities over the
// vocabulary (float32) after a header with the texts' tokens; --ref prints, per text and in all: the mean KL
// divergence D(ref || this) and its 90th / 99th percentiles, how often both put the same token first, and each one's
// perplexity on the text.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "core/exec/engine.h"
#include "core/exec/generation.h"
#include "core/exec/request.h"
#include "core/session/session.h"
#include "core/vision/image.h"

using namespace flyx;

namespace {

constexpr char kMagic[8] = {'F', 'L', 'Y', 'X', 'K', 'L', 'D', '1'};

// log-softmax in double, written as float
void log_softmax(const std::vector<float>& z, std::vector<float>* out) {
  double m = -INFINITY;
  for (float v : z) m = std::max(m, double(v));
  double sum = 0;
  for (float v : z) sum += std::exp(double(v) - m);
  const double lse = m + std::log(sum);
  out->resize(z.size());
  for (size_t i = 0; i < z.size(); ++i) (*out)[i] = float(double(z[i]) - lse);
}

double percentile(std::vector<double> v, double q) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, size_t(q * double(v.size())))];
}

struct Totals {
  std::vector<double> kl;
  long same_top = 0, n = 0;
  double nll = 0, nll_ref = 0;
  void print(const char* name) const {
    double mean = 0;
    for (double k : kl) mean += k;
    mean /= std::max<size_t>(1, kl.size());
    std::printf("%-28s %6ld positions  KL mean %.5f  p90 %.5f  p99 %.5f  same top-1 %6.2f%%  ppl %.4f (ref %.4f)\n",
                name, n, mean, percentile(kl, 0.9), percentile(kl, 0.99), 100.0 * double(same_top) / double(n),
                std::exp(nll / double(n)), std::exp(nll_ref / double(n)));
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: kld MODEL --text FILE [--text FILE ...] [--tokens N] [--start S] [--backend-options O] "
                         "[--activations exact] (--save REF.bin | --ref REF.bin)\n");
    return 2;
  }
  const std::string model = argv[1];
  std::vector<std::string> texts, images;
  std::string save, ref, activations, backend_options, kv, kv_method;
  std::string mmproj, lora, vision_device, vision_quant, prompt_text = "Describe this image in detail.";
  double kv_ram = -1;
  int tokens = 1024, start = 32, ctx = 0, prefill_chunk = 0;
  bool enable_thinking = true;
  for (int i = 2; i < argc; ++i) {
    const std::string k = argv[i];
    if (k == "--no-think") {
      enable_thinking = false;
      continue;
    }
    if (i + 1 >= argc) {
      std::fprintf(stderr, "kld: missing value for %s\n", k.c_str());
      return 2;
    }
    const std::string v = argv[++i];
    if (k == "--text") texts.push_back(v);
    else if (k == "--image") images.push_back(v);
    else if (k == "--prompt") prompt_text = v;
    else if (k == "--mmproj") mmproj = v;
    else if (k == "--lora") lora = v;
    else if (k == "--vision-device") vision_device = v;
    else if (k == "--vision-quant") vision_quant = v;
    else if (k == "--tokens") tokens = std::atoi(v.c_str());
    else if (k == "--start") start = std::atoi(v.c_str());
    else if (k == "--ctx") ctx = std::atoi(v.c_str());
    else if (k == "--prefill-chunk") prefill_chunk = std::atoi(v.c_str());
    else if (k == "--save") save = v;
    else if (k == "--ref") ref = v;
    else if (k == "--activations") activations = v;
    else if (k == "--backend-options") backend_options = v;
    else if (k == "--kv") kv = v;
    else if (k == "--kv-method") kv_method = v;
    else if (k == "--kv-ram") kv_ram = std::atof(v.c_str());
    else {
      std::fprintf(stderr, "kld: unknown flag %s\n", k.c_str());
      return 2;
    }
  }
  if (texts.empty() || save.empty() == ref.empty()) {
    std::fprintf(stderr, "kld: one --text at least, and --save or --ref\n");
    return 2;
  }
  if (!images.empty() && images.size() != texts.size()) {
    std::fprintf(stderr, "kld: --image count (%zu) must match --text count (%zu)\n", images.size(), texts.size());
    return 2;
  }
  EngineOptions o;
  o.speculative = false;
  o.lookup = false;
  o.max_ctx = ctx > 0 ? ctx : (!images.empty() ? std::max(2048, tokens + 1024) : start + tokens + 64);
  o.ctx_given = true;
  if (prefill_chunk > 0) o.prefill_chunk = prefill_chunk;
  o.activations = activations;
  if (!kv.empty()) o.kv = kv == "fp8" ? FLYX_KV_FP8 : FLYX_KV_16;
  if (!kv_method.empty()) o.kv_method = kv_method;
  if (kv_ram >= 0) o.kv_ram = int64_t(kv_ram * 1e9);
  if (!mmproj.empty()) o.mmproj_file = mmproj;
  o.lora_file = lora;
  if (!vision_device.empty()) o.vision_device = vision_device;
  if (!vision_quant.empty()) o.vision_quant = vision_quant;
  if (images.empty() && mmproj.empty()) o.vision = false;
  o.backend_options = backend_options.empty() ? "head_cert=0" : backend_options + ",head_cert=0";
  Engine e;
  std::string err;
  if (!e.load(model, o, &err)) {
    std::fprintf(stderr, "load: %s\n", err.c_str());
    return 1;
  }
  if (!images.empty() && (!e.has_vision() || !e.chat().ok())) {
    std::fprintf(stderr, "kld: --image requires a model with a chat template and a loaded vision encoder\n");
    return 1;
  }
  const auto* api = e.api();
  if (!api->logits) {
    std::fprintf(stderr, "kld: the backend has no logits()\n");
    return 1;
  }
  const int V = e.model().desc().vocab;
  flyx_plan* plain = nullptr;
  for (size_t i = 0; i < e.widths().size(); ++i)
    if (e.widths()[i] == 0) plain = e.round_plan(i);
  if (!plain) {
    std::fprintf(stderr, "kld: no plain round plan\n");
    return 1;
  }

  // Tokenize texts: text-only mode needs start + tokens + 1; multimodal mode teacher-forces tokens + 1 reply tokens.
  const bool multimodal = !images.empty();
  const int need = multimodal ? (tokens + 1) : (start + tokens + 1);
  std::vector<std::vector<int32_t>> toks;
  for (const auto& path : texts) {
    std::ifstream f(path, std::ios::binary);
    const std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto t = multimodal ? e.tokenizer().encode(s) : e.tokenizer().encode_prompt(s);
    if (t.size() < size_t(need)) {
      std::fprintf(stderr, "kld: %s has %zu tokens, fewer than %d\n", path.c_str(), t.size(), need);
      return 1;
    }
    t.resize(size_t(need));
    toks.push_back(std::move(t));
  }

  std::FILE* out = nullptr;
  std::FILE* in = nullptr;
  if (!save.empty()) {
    out = std::fopen(save.c_str(), "wb");
    const int32_t head[3] = {V, int32_t(toks.size()), int32_t(toks[0].size())};
    std::fwrite(kMagic, 1, 8, out);
    std::fwrite(head, sizeof(int32_t), 3, out);
    for (const auto& t : toks) std::fwrite(t.data(), sizeof(int32_t), t.size(), out);
  } else {
    in = std::fopen(ref.c_str(), "rb");
    char magic[8] = {};
    int32_t head[3] = {};
    if (!in || std::fread(magic, 1, 8, in) != 8 || std::memcmp(magic, kMagic, 8) != 0 ||
        std::fread(head, sizeof(int32_t), 3, in) != 3 || head[0] != V || head[1] != int32_t(toks.size()) ||
        head[2] != int32_t(toks[0].size())) {
      std::fprintf(stderr, "kld: %s is not a file of these texts and this vocabulary\n", ref.c_str());
      return 1;
    }
    for (const auto& t : toks) {
      std::vector<int32_t> r(t.size());
      if (std::fread(r.data(), sizeof(int32_t), r.size(), in) != r.size() || r != t) {
        std::fprintf(stderr, "kld: %s holds other tokens\n", ref.c_str());
        return 1;
      }
    }
  }

  Totals all;
  std::vector<float> z(size_t(V), 0.f), lp, lr(size_t(V), 0.f);
  for (size_t ti = 0; ti < toks.size(); ++ti) {
    const auto& t = toks[ti];
    Session s(&e);
    if (!s.init(&err)) {
      std::fprintf(stderr, "session: %s\n", err.c_str());
      return 1;
    }
    int base_pos = start;
    int t_start = start;
    if (multimodal) {
      auto img = std::make_shared<interfaces::ImageInput>();
      if (!vision::load_image_file(images[ti], img.get(), &err)) {
        std::fprintf(stderr, "image %s: %s\n", images[ti].c_str(), err.c_str());
        return 1;
      }
      Request req;
      req.chat = true;
      req.chat_options.enable_thinking = enable_thinking;
      ChatMessage msg;
      msg.role = "user";
      msg.parts.push_back(interfaces::ContentPart::make_image(img));
      msg.parts.push_back(interfaces::ContentPart::make_text(prompt_text));
      req.messages.push_back(std::move(msg));
      RenderedChat rc;
      if (!render_request(e.chat(), e.tokenizer(), req, &rc, &err, false)) {
        std::fprintf(stderr, "render: %s\n", err.c_str());
        return 1;
      }
      PreparedRequest pr;
      if (!prepare_request(e, s, req, rc, &pr, &err)) {
        std::fprintf(stderr, "prepare: %s\n", err.c_str());
        return 1;
      }
      base_pos = int(pr.gen.prompt.size());
      t_start = 0;
      pr.gen.max_new_tokens = 1;
      pr.gen.ignore_eos = true;
      GenerateResult res;
      if (generate(s, pr.gen, TextSink{}, &res, &err) != 0) {
        std::fprintf(stderr, "prefill: %s\n", err.c_str());
        return 1;
      }
    } else {
      GenerateRequest rq;
      rq.prompt.assign(t.begin(), t.begin() + start);
      rq.max_new_tokens = 1;
      rq.ignore_eos = true;
      GenerateResult res;
      if (generate(s, rq, TextSink{}, &res, &err) != 0) {
        std::fprintf(stderr, "prefill: %s\n", err.c_str());
        return 1;
      }
    }
    Totals one;
    flyx_gen_params g{};
    g.top_p = 1.f;
    flyx_round_result r{};
    for (int step = 0; step < tokens; ++step) {
      const int seq_pos = base_pos + step;
      const size_t ti_pos = size_t(t_start + step);
      // the text's token is the input at position seq_pos, whatever the model chose there
      if (api->rewind(s.seq(), seq_pos, t[ti_pos]) != 1 || api->launch(plain, s.seq(), &g) != 0 ||
          api->read(s.seq(), &r) != 0 || api->logits(s.seq(), 0, z.data()) != 0) {
        std::fprintf(stderr, "kld: position %d: %s\n", seq_pos, api->last_error());
        return 1;
      }
      log_softmax(z, &lp);
      const int32_t next = t[ti_pos + 1];
      if (out) {
        std::fwrite(lp.data(), sizeof(float), lp.size(), out);
        one.nll -= lp[size_t(next)];
        ++one.n;
        continue;
      }
      if (std::fread(lr.data(), sizeof(float), lr.size(), in) != lr.size()) {
        std::fprintf(stderr, "kld: %s ends early\n", ref.c_str());
        return 1;
      }
      double kl = 0;
      for (int i = 0; i < V; ++i) kl += std::exp(double(lr[size_t(i)])) * (double(lr[size_t(i)]) - double(lp[size_t(i)]));
      one.kl.push_back(std::max(0.0, kl));
      const auto top = std::max_element(lp.begin(), lp.end()) - lp.begin();
      const auto top_ref = std::max_element(lr.begin(), lr.end()) - lr.begin();
      one.same_top += top == top_ref;
      one.nll -= lp[size_t(next)];
      one.nll_ref -= lr[size_t(next)];
      ++one.n;
    }
    if (out) {
      std::printf("%-28s %6ld positions  ppl %.4f\n", texts[ti].c_str(), one.n, std::exp(one.nll / double(one.n)));
    } else {
      one.print(texts[ti].c_str());
      all.kl.insert(all.kl.end(), one.kl.begin(), one.kl.end());
      all.same_top += one.same_top;
      all.nll += one.nll;
      all.nll_ref += one.nll_ref;
      all.n += one.n;
    }
    std::fflush(stdout);
  }
  if (in) all.print("all");
  if (out) std::fclose(out);
  if (in) std::fclose(in);
  return 0;
}
