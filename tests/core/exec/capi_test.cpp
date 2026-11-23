// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The C API (include/flyx/flyx.h) on the tiny model: a chat request is
// rendered with the model file's template, or with the one the load options
// give (chat_template), and the two give different prompts.
//
//   capi_test TINY.flyx BACKEND
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "flyx/flyx.h"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("  %-4s %s\n", ok ? "ok" : "FAIL", what.c_str());
  if (!ok) ++failures;
}

// A template of our own: ChatML after a line of plain text, so its prompt differs from the file's template.
const char* kTemplate =
    "Rendered by the override.\n"
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
const char* kRendered = "Rendered by the override.\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n";

struct Prompt {
  int tokens = -1;       // the request's prompt tokens (flyx_usage)
  int override_ids = -1;  // the tokens of kRendered, by this engine's tokenizer
};

// One chat message on an engine loaded with `tmpl` (null: the file's template).
Prompt prompt_of(const char* model, const char* backend, const char* tmpl, std::string* err) {
  Prompt p;
  flyx_load_options o{};
  o.max_ctx = 256;
  o.backend = backend;
  o.speculative = 0;
  o.chat_template = tmpl;
  flyx_error e{};
  flyx_engine* eng = flyx_load(model, &o, &e);
  if (!eng) {
    *err = e.message;
    return p;
  }
  p.override_ids = flyx_tokenize(eng, kRendered, nullptr, 0);
  flyx_session* s = flyx_session_new(eng);
  flyx_message m{"user", "hi", nullptr};
  flyx_request rq{};
  rq.messages = &m;
  rq.n_messages = 1;
  rq.max_new_tokens = tmpl ? 1 : 0;
  rq.ignore_eos = 1;
  flyx_usage u{};
  if (!s) *err = "no session";
  else if (flyx_generate(s, &rq, nullptr, nullptr, &u, &e) != 0) *err = e.message;
  else {
    p.tokens = u.prompt_tokens;
    if (!tmpl) check(u.new_tokens > 1 && u.finish_reason == FLYX_FINISH_LENGTH,
                     "a zero C API output limit generates until the context is full");
  }
  if (s) flyx_session_free(s);
  flyx_unload(eng);
  return p;
}

void check_handle_lifetime(const char* model, const char* backend) {
  flyx_load_options opts{};
  opts.max_ctx = 64;
  opts.backend = backend;
  flyx_error err{};
  flyx_engine* engine = flyx_load(model, &opts, &err);
  if (!engine) {
    check(false, err.message);
    return;
  }
  flyx_session* first = flyx_session_new(engine);
  flyx_session* second = flyx_session_new(engine);
  flyx_unload(engine);
  check(first && second, "two sessions share the loaded model");
  if (first && second) {
    const int32_t tokens[80]{};
    auto reject = [&](flyx_session* s, int n, flyx_error* error) {
      flyx_request rq{};
      rq.tokens = tokens;
      rq.n_tokens = n;
      return flyx_generate(s, &rq, nullptr, nullptr, nullptr, error);
    };
    int result = 0;
    flyx_error worker_err{};
    std::thread worker([&] { result = reject(first, 64, &worker_err); });
    const int other = reject(second, 80, &err);
    worker.join();
    check(result != 0 && other != 0 && std::string(worker_err.message).find("64 tokens") != std::string::npos &&
              std::string(err.message).find("80 tokens") != std::string::npos,
          "concurrent callers keep the error for their own request after the engine handle is released");
  }
  flyx_session_free(first);
  if (second) {
    flyx_request rq{};
    rq.text = "hello";
    rq.max_new_tokens = 4;
    rq.ignore_eos = 1;
    flyx_usage usage{};
    check(flyx_generate(second, &rq, nullptr, nullptr, &usage, &err) == 0 && usage.new_tokens == 4,
          "the last session can generate after its engine handle and sibling are released");
    flyx_session_free(second);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: capi_test TINY.flyx BACKEND\n");
    return 2;
  }
  check(flyx_api_version() == FLYX_API_VERSION, "the library answers the header's API version");
  {
    flyx_error e{};
    check(flyx_load(nullptr, nullptr, &e) == nullptr, "flyx_load rejects null path");
    check(flyx_session_new(nullptr) == nullptr, "flyx_session_new rejects null engine");
    check(std::string(flyx_backend_name(nullptr)).empty(), "flyx_backend_name handles null engine");
    check(flyx_tokenize(nullptr, "hi", nullptr, 0) < 0, "flyx_tokenize rejects null engine");
    check(flyx_generate(nullptr, nullptr, nullptr, nullptr, nullptr, &e) != 0, "flyx_generate rejects null args");
    flyx_cancel(nullptr);
    flyx_session_free(nullptr);
    flyx_unload(nullptr);
  }
  check_handle_lifetime(argv[1], argv[2]);
  std::string err;
  const Prompt own = prompt_of(argv[1], argv[2], nullptr, &err);
  check(own.tokens > 0, "a chat request with the file's template: " + std::to_string(own.tokens) + " prompt tokens " +
                            err);
  err.clear();
  const Prompt over = prompt_of(argv[1], argv[2], kTemplate, &err);
  check(over.tokens > 0 && over.tokens == over.override_ids && over.tokens != own.tokens,
        "chat_template in the load options renders the request (" + std::to_string(over.tokens) +
            " prompt tokens, the override's text " + std::to_string(over.override_ids) + ") " + err);
  std::printf("capi: %s\n", failures ? "FAILED" : "ok");
  return failures ? 1 : 0;
}
