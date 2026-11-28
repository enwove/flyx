// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// flyx command line: the one executable, the HTTP server one of its commands.
//
//   flyx run      MODEL -p "text" [--chat] [--no-think] [-n N]   generate once
//   flyx chat     MODEL                                          interactive, multi-turn
//   flyx bench    MODEL [-n N] [--json [FILE]]                   speed and draft acceptance
//   flyx bench    MODEL --prompts FILE.jsonl [--suffix S] [--limit N] [--json [FILE]]   ... on a prompt set
//   flyx tokenize MODEL "text"                                   print token ids
//   flyx template MODEL MESSAGES.json [--no-think]               render the chat template
//   flyx serve    MODEL [--port 8080] [options]                  OpenAI / Anthropic HTTP API (apps/flyx-server/)
//   flyx monitor  [URL]                                          live panel of a running server
//
// Called by the name flyx-server (the release package's link to it), it is `flyx serve`.
//
// The engine's options (--ctx, --backend, --no-spec, ...) are apps/common/engine_flags.h's, shared with
// flyx serve; sampling: --temperature T, --top-p P, --top-k K, --seed S, --presence-penalty P,
// --frequency-penalty F; relaxed acceptance of drafts (lossy): --accept-min-p P, --accept-typical E
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "apps/common/engine_flags.h"
#include "apps/common/model_source.h"
#include "apps/common/numbers.h"
#include "apps/common/version.h"
#include "apps/flyx-cli/monitor.h"
#include "core/exec/engine.h"
#include "core/exec/request.h"
#include "core/exec/telemetry.h"
#include "core/os/os.h"
#include "core/os/path.h"
#include "core/session/session.h"
#include "core/render/chat.h"
#include "core/render/chat_json.h"
#include "nlohmann/json.hpp"

using namespace flyx;

namespace flyx::server {
int run(int argc, char** argv);  // apps/flyx-server/main.cpp: argv[1] the model
}
#ifdef FLYX_STANDALONE_EXE
namespace flyx::apps {
void register_builtin_backends();  // apps/flyx-cli/builtin_backends.cpp
}
#endif

namespace {

struct Args {
  std::string cmd, model, prompt, extra, rounds;
  bool chat = false, ignore_eos = false, json = false;
  ChatOptions chat_options;
  int n = 256, verbose = 0;
  bool n_given = false;     // -n was on the command line (a prompt set's default is its own)
  std::string json_file;    // --json FILE: the record goes there, not to stdout
  std::string prompts, suffix;  // bench --prompts FILE.jsonl, --suffix S
  int limit = 0;                // bench --limit N: the first N prompts of the set (0: all)
  std::vector<std::string> images;   // run --image FILE (repeatable)
  std::vector<std::string> command;  // the command line, for a prompt set's record
  EngineOptions engine;     // the engine's options (apps/common/engine_flags.h)
  SamplingParams sampling;  // greedy unless --temperature
};

// A prompt set's answers get as many tokens as bifill-dv's GSM8K runs gave theirs (the references DV is held to).
constexpr int kPromptSetMaxNew = 512;

// The usage: asked for (--help) it goes to stdout and the exit status is 0; printed for a command line that names
// no command or model, to stderr with status 2.
[[noreturn]] void usage(bool asked = false) {
  std::fprintf(asked ? stdout : stderr,
               "usage: flyx {run|chat|bench|tokenize|template} MODEL [options]\n"
               "       flyx serve [MODEL] [options]   (the HTTP API; flyx serve --help lists its options)\n"
               "       flyx --version\n"
               "  MODEL                a .gguf or .flyx file, or a Hugging Face name (org/name:QUANT), fetched into\n"
               "                       the Hugging Face cache\n"
               "       flyx monitor [URL] [--key KEY] [--interval S] [--once]   (a running server)\n"
               "  -p TEXT              prompt (run)\n"
               "  -f FILE              the prompt from a file (run; a long one: a command line holds 128 KB at most)\n"
               "  --image FILE         attach an image file to the prompt (run, repeatable; implies --chat)\n"
               "  --chat               wrap the prompt as a user message (run)\n"
               "  --no-think           disable the thinking block in chat mode\n"
               "  --chat-template-kwargs JSON  template variables for run --chat, chat and template (JSON object)\n"
               "  -n N                 max new tokens (default 256; 0: the model's own limit if its file has one,\n"
               "                       else up to the end of the context)\n"
               "  --ignore-eos         do not stop at end-of-sequence tokens\n"
               "  --temperature T, --top-p P, --top-k K, --seed S   sampling (default: greedy)\n"
               "  --presence-penalty P, --frequency-penalty F       what a token already in the reply loses\n"
               "                       from its logit, once and per time it was chosen (default 0)\n"
               "  --accept-min-p P, --accept-typical E   relaxed acceptance of drafts (lossy: the output may differ\n"
               "                       from plain decoding): a draft within P of the top token's probability, or\n"
               "                       above min(E, sqrt(E) exp(-entropy)) (default 0: off)\n"
               "  --json [FILE]        bench: one JSON record (every prompt, the totals, the device's clocks) on\n"
               "                       stdout, or in FILE (opened before the run) when the next argument is not an\n"
               "                       option\n"
               "  --prompts FILE       bench: a prompt set, a JSON object a line, each prompt in a fresh session\n"
               "                       up to the model's end token (default -n 512): {\"ids\": [...]} the prompt\n"
               "                       as tokens, {\"messages\": [...]} a chat (OpenAI's format), or\n"
               "                       {\"question\": \"...\"} one user message, the question then --suffix; chats\n"
               "                       without thinking. Every prompt is checked before the first runs. The record\n"
               "                       has each prompt's tokens, its answer's (the end token last) and every\n"
               "                       round's [tokens, accepted drafts, 1 when its last token was a draft]\n"
               "  --suffix S           bench --prompts: after each question (\\n in S a newline, \\\\ a backslash)\n"
               "  --limit N            bench --prompts: the set's first N prompts\n"
               "  --rounds FILE        run: every round the device gave, a JSON line each (position, tokens, drafts,\n"
               "                       accepted)\n"
               "  -v                   verbose\n"
               "the engine:\n%s",
               apps::engine_flags_help(EngineOptions{}.max_ctx).c_str());
  std::exit(asked ? 0 : 2);
}

// A command line that cannot be followed: what is wrong with it, not the sixty lines of usage it would be lost in.
[[noreturn]] void bad_usage(const std::string& why) {
  std::fprintf(stderr, "error: %s (flyx --help lists the commands and options)\n", why.c_str());
  std::exit(2);
}

bool asks_help(const std::string& s) { return s == "-h" || s == "--help"; }

Args parse(int argc, char** argv) {
  Args a;
  if (argc >= 2 && (asks_help(argv[1]) || std::strcmp(argv[1], "help") == 0)) usage(true);
  if (argc < 2) usage();
  a.cmd = argv[1];
  static const char* const kCommands[] = {"run", "chat", "bench", "tokenize", "template"};
  if (std::none_of(std::begin(kCommands), std::end(kCommands), [&](const char* c) { return a.cmd == c; }))
    bad_usage("unknown command '" + a.cmd + "'");
  if (argc < 3) bad_usage(a.cmd + " needs a model: a .gguf or .flyx file, or a Hugging Face name");
  a.model = argv[2];
  if (asks_help(a.model)) usage(true);
  a.command.assign(argv, argv + argc);
  apps::SamplingOverrides sampling_overrides;
  bool no_think = false;
  ojson chat_defaults = ojson::object();
  for (int i = 3; i < argc; ++i) {
    std::string s = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) bad_usage(s + " needs a value");
      return argv[++i];
    };
    if (asks_help(s)) usage(true);
    if (s == "-p") a.prompt = next();
    else if (s == "-f") {
      const std::string path = next();
      if (!apps::read_text_file(path, &a.prompt))
        std::fprintf(stderr, "error: cannot read %s\n", path.c_str()), std::exit(2);
    } else if (s == "--chat") a.chat = true;
    else if (s == "--image") a.images.push_back(next());
    else if (s == "--no-think") no_think = true;
    else if (s == "--chat-template-kwargs") {
      if (a.cmd != "run" && a.cmd != "chat" && a.cmd != "template")
        bad_usage("--chat-template-kwargs applies to run, chat and template");
      chat_defaults = apps::chat_template_kwargs_flag(next());
    }
    else if (s == "-n") a.n = apps::whole_flag("-n", next(), 0, INT_MAX), a.n_given = true;
    else if (s == "--ignore-eos") a.ignore_eos = true;
    else if (apps::sampling_flag(s, next, &sampling_overrides, true)) continue;
    else if (s == "-v") a.verbose = 1;
    else if (s == "--json") {
      a.json = true;
      if (i + 1 < argc && argv[i + 1][0] != '-') a.json_file = next();
    } else if (s == "--prompts") a.prompts = next();
    else if (s == "--suffix") a.suffix = next();
    else if (s == "--limit") a.limit = apps::whole_flag("--limit", next(), 0, INT_MAX);
    else if (s == "--rounds") a.rounds = next();
    else if (apps::engine_flag(s, next, &a.engine)) continue;
    else if (a.extra.empty() && !s.empty() && s[0] != '-') a.extra = s;
    else bad_usage(!s.empty() && s[0] == '-' ? "unknown option " + s : "unexpected argument '" + s + "'");
  }
  sampling_overrides.apply(&a.sampling);
  if (no_think) chat_defaults["enable_thinking"] = false;
  a.chat_options = apps::resolve_chat_options(chat_defaults, ojson::object());
  if (!a.images.empty() && a.cmd != "run") bad_usage("--image is only supported with 'flyx run'");
  // What the command needs from the command line, said before the model is looked for (a Hub name is fetched).
  if (a.cmd == "template" && a.extra.empty()) bad_usage("template needs a MESSAGES.json path after the model");
  if (a.cmd == "run" && a.prompt.empty() && !a.chat && a.images.empty())
    bad_usage("run needs a prompt: -p TEXT or -f FILE (with --chat, an empty one is a user message)");
  return a;
}

void print(std::string_view text) {
  std::fwrite(text.data(), 1, text.size(), stdout);
  std::fflush(stdout);
}

void report(const Engine& e, const GenerateResult& res) {
  const flyx_usage& u = res.usage;
  // The tokens the model chose, its stop token among them (GenerateResult::output_tokens): the first with the prefill,
  // the others from the rounds, which the rate and the tokens a round count, as the server's request log does.
  const int out = res.output_tokens();
  const double tps = apps::decode_rate(out, u.decode_ms);
  double acc = u.rounds && out > 1 ? double(out - 1) / u.rounds : 0;
  // Which drafter drafted how many rounds, when there are several, and the lookup rounds.
  std::string by;
  if (e.model().drafters() > 1)
    for (size_t d = 0; d < res.drafter_rounds.size(); ++d)
      by += std::string(by.empty() ? " | " : ", ") + e.model().drafter_name(int(d)) + " " +
            std::to_string(res.drafter_rounds[d]);
  if (res.lookup_rounds > 0) by += std::string(by.empty() ? " | " : ", ") + "lookup " + std::to_string(res.lookup_rounds);
  std::fprintf(stderr,
               "\n[prompt %d (reused %d) | prefill %.1f ms | %d tokens in %.1f ms = %.1f tok/s | %d rounds, %.2f "
               "tokens/round, %.1f drafts/round%s | finish %d]\n",
               u.prompt_tokens, u.reused_tokens, u.prefill_ms, out, u.decode_ms, tps, u.rounds, acc,
               u.rounds ? double(u.drafted) / u.rounds : 0.0, by.c_str(), u.finish_reason);
}

// A prompt's prefill on a terminal: once it has taken a second, how far it is, on one line of standard error redrawn
// at most twice a second and wiped when the last pass goes in (a long document's prefill takes minutes, and said
// nothing).  Not on a terminal: nothing.
//
// The time left: a token at position p costs a + b p (its attention reads the p before it), so the time a pass ends
// is c + a (p - p0) + b (p^2 - p0^2) / 2 from position p0 on, c the time the first passes were launched ahead of the
// device; c, a and b fitted (least squares) to the passes past the first second.  A rate alone says a long document's
// prefill ends sooner than it does.
bool solve3(double m[3][3], double r[3], double out[3]) {  // Gaussian elimination, partial pivoting
  int row[3] = {0, 1, 2};
  for (int c = 0; c < 3; ++c) {
    int best = c;
    for (int i = c + 1; i < 3; ++i)
      if (std::abs(m[row[i]][c]) > std::abs(m[row[best]][c])) best = i;
    std::swap(row[c], row[best]);
    const double piv = m[row[c]][c];
    if (std::abs(piv) < 1e-12) return false;
    for (int i = c + 1; i < 3; ++i) {
      const double f = m[row[i]][c] / piv;
      for (int j = c; j < 3; ++j) m[row[i]][j] -= f * m[row[c]][j];
      r[row[i]] -= f * r[row[c]];
    }
  }
  for (int c = 2; c >= 0; --c) {
    double v = r[row[c]];
    for (int j = c + 1; j < 3; ++j) v -= m[row[c]][j] * out[j];
    out[c] = v / m[row[c]][c];
  }
  return true;
}

std::function<void(int, int)> prefill_progress() {
  if (!os::stderr_is_terminal()) return {};
  struct Shown {
    double start = 0, last = 0;
    double p0 = -1;                     // the tokens reused (the first call's)
    double mm[3][3] = {}, mr[3] = {};   // the fit's normal equations, over (1, u, v): u, v the terms scaled to 0..1
    int fitted = 0;                     // passes in them
    size_t width = 0;                   // of the line on the terminal, 0: none
  };
  auto st = std::make_shared<Shown>();
  return [st](int done, int total) {
    const double now = os::now_ms();
    if (st->p0 < 0) st->p0 = done, st->start = now;
    const double p0 = st->p0, span = std::max(1.0, total - p0);
    const double sq = std::max(1.0, (double(total) * total - p0 * p0) / 2);
    const double f[3] = {1, (done - p0) / span, (double(done) * done - p0 * p0) / 2 / sq};
    const double t = (now - st->start) / 1000;
    if (t >= 1) {
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) st->mm[i][j] += f[i] * f[j];
        st->mr[i] += f[i] * t;
      }
      ++st->fitted;
    }
    std::string line;
    if (done < total) {
      if (t < 1 || now - st->last < 500) return;
      double m[3][3], r[3], k[3] = {0, 0, 0};
      std::copy(&st->mm[0][0], &st->mm[0][0] + 9, &m[0][0]);
      std::copy(st->mr, st->mr + 3, r);
      double end = 0;  // when the last pass ends: the fit at u = v = 1, or, too few passes to tell, the rate alone
      if (st->fitted >= 4 && solve3(m, r, k) && k[1] >= 0 && k[2] >= 0) end = k[0] + k[1] + k[2];
      else if (done > p0) end = t * span / (done - p0);
      const int left = int(std::max(0.0, end - t));
      char buf[160];
      std::snprintf(buf, sizeof(buf), "prefill %d / %d tokens (%d%%), %.0f tokens/s, about %d:%02d left", done, total,
                    int(100.0 * done / total), (done - p0) / t, left / 60, left % 60);
      line = buf;
    } else if (st->width == 0) {
      return;
    }
    st->last = now;
    // Over the last line: the new one, blanks where the last was longer; the cursor back at the start once wiped.
    const std::string blanks(st->width > line.size() ? st->width - line.size() : 0, ' ');
    std::fprintf(stderr, "\r%s%s%s", line.c_str(), blanks.c_str(), line.empty() ? "\r" : "");
    st->width = line.size();
  };
}

std::string widths_str(const Engine& e) {
  std::string s;
  for (size_t i = 0; i < e.widths().size(); ++i) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%s%s (%.1f ms)", i ? ", " : "", e.round_name(i).c_str(), e.round_ms()[i]);
    s += buf;
  }
  return s;
}

std::unique_ptr<Engine> load(const Args& a, bool elastic = true) {
  EngineOptions o = a.engine;
  o.verbose = a.verbose;
  o.elastic_load = elastic;
  auto e = std::make_unique<Engine>();
  std::string err;
  if (!e->load(a.model, o, &err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    std::exit(1);
  }
  if (a.verbose)
    std::fprintf(stderr, "backend %s, widths %s\n", e->backend_name().c_str(), widths_str(*e).c_str());
  return e;
}

ChatMessage msg(const std::string& role, const std::string& content) {
  ChatMessage m;
  m.role = role;
  m.content = content;
  return m;
}

// A request made ready for a session by the steps every entry point shares
// (core/exec/request.h).  branch_points: a later request may continue this
// conversation (checkpoints are kept where it could branch).
PreparedRequest prepare(Engine& e, const Session& s, const Request& rq, bool branch_points) {
  std::string err;
  RenderedChat rc;
  PreparedRequest p;
  if (!render_request(e.chat(), e.tokenizer(), rq, &rc, &err, branch_points) ||
      !prepare_request(e, s, rq, rc, &p, &err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    std::exit(1);
  }
  return p;
}

int cmd_run(const Args& a) {
  std::string err;
  std::vector<std::shared_ptr<const interfaces::ImageInput>> loaded_images;
  loaded_images.reserve(a.images.size());
  for (const std::string& img_path : a.images) {
    interfaces::ImageInput img;
    if (!vision::load_image_file(img_path, &img, &err))
      return std::fprintf(stderr, "error: %s\n", err.c_str()), 1;
    loaded_images.push_back(std::make_shared<interfaces::ImageInput>(std::move(img)));
  }
  auto e = load(a);
  Session s(e.get());
  if (!s.init(&err)) return std::fprintf(stderr, "error: %s\n", err.c_str()), 1;
  Request rq;
  rq.chat = a.chat || !loaded_images.empty();
  rq.split_output = false;  // the reply as the model wrote it, thinking included
  if (!loaded_images.empty()) {
    ChatMessage um = msg("user", a.prompt);
    for (const auto& img : loaded_images) um.parts.push_back(interfaces::ContentPart::make_image(img));
    if (!a.prompt.empty()) um.parts.push_back(interfaces::ContentPart::make_text(a.prompt));
    rq.messages = {std::move(um)};
  } else if (a.chat) {
    rq.messages = {msg("user", a.prompt)};
  } else {
    rq.text = a.prompt;
  }
  rq.chat_options = a.chat_options;
  rq.gen.max_new_tokens = a.n;
  rq.gen.ignore_eos = a.ignore_eos;
  rq.gen.sampling = a.sampling;
  PreparedRequest p = prepare(*e, s, rq, false);
  p.gen.on_prefill = prefill_progress();
  GenerateResult res;
  TextSink sink = [](std::string_view text, const int32_t*, int) {
    print(text);
    return true;
  };
  std::vector<flyx_round_result> trace;
  std::vector<int> which;  // the round each entry was (Engine::round_name's index; -1 lookup, -2 the prefill's)
  if (!a.rounds.empty()) s.trace = &trace, s.trace_rounds = &which;
  if (generate(s, p.gen, sink, &res, &err) != 0) return std::fprintf(stderr, "error: %s\n", err.c_str()), 1;
  s.trace = nullptr;
  s.trace_rounds = nullptr;
  report(*e, res);
  if (!a.rounds.empty()) {  // the prefill's result first, then a round's each
    std::ofstream f(os::to_path(a.rounds));
    for (size_t k = 0; k < trace.size(); ++k) {
      const flyx_round_result& r = trace[k];
      const int w = k < which.size() ? which[k] : -3;
      const std::string round = w >= 0 ? e->round_name(size_t(w)) : w == -1 ? "lookup" : w == -2 ? "prefill" : "?";
      nlohmann::json row{{"position", r.position}, {"tokens", r.n_tokens}, {"drafts", r.n_drafts},
                         {"accepted", r.n_accepted}, {"round", round}};
      const int nc = std::min(r.n_drafts, FLYX_MAX_ROWS);
      if (nc > 0 && r.draft_prob[0] > 0.f) {
        row["prob"] = std::vector<float>(r.draft_prob, r.draft_prob + nc);
        row["margin"] = std::vector<float>(r.draft_margin, r.draft_margin + nc);
      }
      f << row.dump() << "\n";
    }
    if (!f) return std::fprintf(stderr, "error: cannot write %s\n", a.rounds.c_str()), 1;
  }
  return 0;
}

int cmd_chat(const Args& a) {
  auto e = load(a);
  Session s(e.get());
  std::string err;
  if (!s.init(&err)) return std::fprintf(stderr, "error: %s\n", err.c_str()), 1;
  std::vector<ChatMessage> msgs;
  std::string line;
  std::fprintf(stderr, "(empty line to quit)\n");
  while (std::fprintf(stdout, "\n> "), std::fflush(stdout), std::getline(std::cin, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) break;
    msgs.push_back(msg("user", line));
    Request rq;
    rq.messages = msgs;
    rq.chat_options = a.chat_options;
    rq.gen.max_new_tokens = a.n;
    rq.gen.sampling = a.sampling;
    PreparedRequest p = prepare(*e, s, rq, true);
    p.gen.on_prefill = prefill_progress();
    // The reply as the next turn's history needs it: thinking apart from the answer.
    ChatMessage reply = msg("assistant", "");
    bool thinking = false;
    auto show = [&](const std::vector<OutputEvent>& events) {
      for (const auto& ev : events) {
        if (ev.kind != OutputEvent::kReasoning && ev.kind != OutputEvent::kContent) continue;  // no tools are offered here
        const bool reasoning = ev.kind == OutputEvent::kReasoning;
        if (thinking && !reasoning) print("\n</think>\n\n");
        thinking = reasoning;
        if (reasoning) reply.reasoning = reply.reasoning.value_or("") + ev.text;
        else reply.content += ev.text;
        print(ev.text);
      }
    };
    TextSink sink = [&](std::string_view text, const int32_t*, int) {
      if (p.parser) show(p.parser->feed(text));
      else show({{OutputEvent::kContent, std::string(text), {}}});
      return true;
    };
    GenerateResult res;
    if (generate(s, p.gen, sink, &res, &err) != 0) {
      std::fprintf(stderr, "error: %s\n", err.c_str());
      return 1;
    }
    if (p.parser) show(p.parser->finish());
    report(*e, res);
    msgs.push_back(reply);
  }
  return 0;
}

// The device's clocks, power and temperature, and what limits its clocks (for benchmark records); null when the
// backend has none.
nlohmann::json print_device(const Engine& e, const char* when) {
  flyx_device_stats d{};
  if (!e.device_stats(&d)) return nullptr;
  const std::string limits = clock_limit_names(d.clock_limits);
  std::fprintf(stderr, "device %s: sm %d MHz, mem %d MHz, %.0f of %.0f W, %d C%s%s\n", when, d.sm_clock_mhz,
               d.mem_clock_mhz, d.power_w, d.power_limit_w, d.temp_c, limits.empty() ? "" : ", clocks limited by: ",
               limits.c_str());
  return {{"sm_mhz", d.sm_clock_mhz}, {"mem_mhz", d.mem_clock_mhz}, {"power_w", d.power_w},
          {"power_limit_w", d.power_limit_w}, {"temp_c", d.temp_c}, {"clocks_limited_by", limits}};
}

// --json FILE, opened (emptied) before anything runs: a path that cannot be written fails at once, not after the
// whole run.
bool open_record(const Args& a, std::ofstream* f) {
  if (!a.json || a.json_file.empty()) return true;
  f->open(os::to_path(a.json_file), std::ios::binary);
  if (!*f) std::fprintf(stderr, "error: cannot write %s\n", a.json_file.c_str());
  return bool(*f);
}

// A bench's JSON record, on stdout or in the file open_record opened.
bool write_record(const std::string& text, const Args& a, std::ofstream& f) {
  if (a.json_file.empty()) {
    print(text);
    return true;
  }
  f << text;
  f.flush();
  if (!f) std::fprintf(stderr, "error: cannot write %s\n", a.json_file.c_str());
  return bool(f);
}

// The text --suffix S stands for: \n a newline, \\ a backslash, any other backslash itself (S is often a LaTeX
// instruction such as GSM8K's "\boxed{}").
std::string unescape_suffix(const std::string& s) {
  std::string out;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '\\' && i + 1 < s.size() && (s[i + 1] == 'n' || s[i + 1] == '\\')) {
      out += s[i + 1] == 'n' ? '\n' : '\\';
      ++i;
    } else {
      out += s[i];
    }
  }
  return out;
}

const char* kv_name(int kv) {
  return kv == FLYX_KV_FP8 ? "fp8" : kv == FLYX_KV_16 ? "16" : "default";
}

// One line of a prompt set as the request it asks for (see usage: --prompts).  Chats without thinking, as the
// built-in prompts; the rest (limits, sampling) is the command line's.
bool prompt_set_request(const ojson& j, const std::string& suffix, Request* rq, std::string* err) {
  rq->chat_options.enable_thinking = false;
  rq->split_output = false;
  if (!j.is_object()) return *err = "not a JSON object", false;
  if (j.contains("ids")) {
    const ojson& ids = j["ids"];
    if (!ids.is_array() || ids.empty()) return *err = "\"ids\" is not a list of token ids", false;
    for (const ojson& t : ids) {
      if (!t.is_number_integer() || t.get<int64_t>() < 0 || t.get<int64_t>() > INT32_MAX)
        return *err = "\"ids\" holds " + t.dump() + ", not a token id", false;
      rq->tokens.push_back(t.get<int32_t>());
    }
    rq->chat = false;
    return true;
  }
  if (j.contains("messages")) return messages_from_openai(j["messages"], &rq->messages, err);
  if (j.contains("question") && j["question"].is_string()) {
    rq->messages = {msg("user", j["question"].get<std::string>() + suffix)};
    return true;
  }
  return *err = "a line needs \"ids\", \"messages\" or \"question\" (a string)", false;
}

// Whether a round's last token was one of its drafts.  A draft that is the target's choice and a stop token ends the
// round with itself, and the ring counts it among the round's tokens but not its accepted drafts
// (ops::accepted_drafts, ops::accept_tree), as it counts the target's own token after the drafts it accepted; other
// counts (bifill-dv's: check_dv.py) take it for an accepted draft.  The last token was a draft when a child of the
// last accepted draft (of the anchor when none was) carries it: siblings are distinct tokens, and a child that
// carries the target's choice is taken unless it stops the reply.
bool last_token_drafted(const flyx_round_result& r) {
  if (r.n_tokens != r.n_accepted + 1) return false;
  const int drafts = std::min<int>(r.n_drafts, FLYX_MAX_WIDE_ROWS);
  int at = -1;  // the row last taken: -1 the anchor, else a draft
  for (int k = 0; k <= r.n_accepted; ++k) {
    int child = -2;
    for (int c = at + 1; c < drafts && child == -2; ++c)
      if (r.parents[c] == at && r.drafts[c] == r.tokens[k]) child = c;
    if (child == -2) return false;  // token k is the target's own (only the last can be)
    at = child;
  }
  return true;
}

// bench --prompts FILE.jsonl: every prompt of a set in a fresh session, one after another, until the model's end
// token (or -n).  A line a prompt on stderr, then the totals; the JSON record (--json) has every prompt's tokens and
// the generated ones, and every round's [tokens, accepted drafts, its last token a draft, the round it was] (the
// ring's, as `run --rounds` writes them; the round an index into the record's "rounds", -1 a lookup round), for
// scoring the answers and counting rounds the way other implementations count theirs (tests/e2e/check_dv.py).  When
// a prompt fails to run, the record still has the prompts before it, and the error.
int cmd_bench_prompts(const Args& a) {
  std::ifstream in(os::to_path(a.prompts), std::ios::binary);
  if (!in) return std::fprintf(stderr, "error: cannot read %s\n", a.prompts.c_str()), 1;
  const std::string suffix = unescape_suffix(a.suffix);
  // Every line read before the model loads, and every prompt checked before the first runs: a mistake in line 900
  // is found before the first 899 have run.  Errors name the file's line; the record numbers prompts from 0.
  std::vector<Request> set;
  std::vector<int> line_of;
  std::string line;
  for (int line_no = 1; std::getline(in, line) && (a.limit == 0 || int(set.size()) < a.limit); ++line_no) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.find_first_not_of(" \t") == std::string::npos) continue;
    Request rq;
    std::string err;
    bool ok = false;
    try {
      ok = prompt_set_request(ojson::parse(line), suffix, &rq, &err);
    } catch (const std::exception& ex) {
      err = ex.what();
    }
    if (!ok) return std::fprintf(stderr, "error: %s line %d: %s\n", a.prompts.c_str(), line_no, err.c_str()), 1;
    set.push_back(std::move(rq));
    line_of.push_back(line_no);
  }
  if (set.empty()) return std::fprintf(stderr, "error: %s has no prompts\n", a.prompts.c_str()), 1;
  std::ofstream out;
  if (!open_record(a, &out)) return 1;

  auto e = load(a, /*elastic=*/false);
  const int max_new = a.n_given ? a.n : kPromptSetMaxNew;
  // The prompts as tokens (a fresh session's: no history to continue), each one the engine can run.  Each stays with
  // its PreparedRequest, which owns the reply grammar its GenerateRequest points to: a chat without thinking has one
  // where only a grammar keeps the thinking out (Inkling-2's format).
  std::vector<PreparedRequest> prepared(set.size());
  {
    Session fresh(e.get());
    for (size_t i = 0; i < set.size(); ++i) {
      Request& rq = set[i];
      rq.gen.max_new_tokens = max_new;
      rq.gen.ignore_eos = a.ignore_eos;
      rq.gen.sampling = a.sampling;
      RenderedChat rc;
      std::string err;
      if (!render_request(e->chat(), e->tokenizer(), rq, &rc, &err, false) ||
          !prepare_request(*e, fresh, rq, rc, &prepared[i], &err) || !prompt_runs(*e, prepared[i].gen.prompt, &err))
        return std::fprintf(stderr, "error: %s line %d: %s\n", a.prompts.c_str(), line_of[i], err.c_str()), 1;
    }
  }
  std::vector<std::string> round_names;
  for (size_t i = 0; i < e->widths().size(); ++i) round_names.push_back(e->round_name(i));
  nlohmann::json record = {{"model", a.model},
                           {"backend", e->backend_name()},
                           {"device", std::string(e->device().name)},
                           {"command", a.command},
                           {"prompt_set", a.prompts},
                           {"suffix", suffix},
                           {"max_new_tokens", max_new},
                           {"ignore_eos", a.ignore_eos},
                           {"eos_ids", e->eos()},
                           {"sampling", {{"temperature", a.sampling.temperature},
                                         {"top_p", a.sampling.top_p},
                                         {"top_k", a.sampling.top_k},
                                         {"seed", a.sampling.seed},
                                         {"presence_penalty", a.sampling.presence_penalty},
                                         {"frequency_penalty", a.sampling.frequency_penalty},
                                         {"accept_min_p", a.sampling.accept_min_p},
                                         {"accept_typical", a.sampling.accept_typical}}},
                           {"ctx", e->options().max_ctx},
                           {"backend_options", e->options().backend_options},
                           {"kv", kv_name(e->options().kv)},
                           {"lookup", e->options().lookup},
                           {"drafter", e->drafter()},
                           {"drafter_refused", e->refused()},
                           {"widths", e->widths()},
                           {"rounds", round_names},
                           {"round_ms", e->round_ms()},
                           {"device_before", print_device(*e, "before")}};
  std::vector<std::string> lines;  // a prompt's record each
  long tok = 0, rounds = 0, ended = 0;
  double ms = 0;
  const double cpu0 = os::process_usage().cpu_seconds, wall0 = os::now_ms();
  // The totals, then the record (with `error` when a prompt failed: the prompts before it).
  auto finish = [&](const std::string& error) {
    record["device_after"] = print_device(*e, "after ");
    const double cpu_ms = (os::process_usage().cpu_seconds - cpu0) * 1000.0, wall_ms = os::now_ms() - wall0;
    const double tok_s = ms > 0 ? tok * 1000.0 / ms : 0, per_round = rounds ? double(tok) / rounds : 0;
    record["decode_tok_s"] = tok_s;
    record["tokens_per_round"] = per_round;
    record["cpu_us_per_round"] = rounds ? cpu_ms * 1000.0 / rounds : 0;
    record["cpu_core_percent"] = wall_ms > 0 ? 100.0 * cpu_ms / wall_ms : 0;
    record["wall_s"] = wall_ms / 1000.0;
    if (!error.empty()) record["error"] = error;
    // On stdout unless the record is all of it.
    std::fprintf(a.json && a.json_file.empty() ? stderr : stdout,
                 "%zu prompts, %ld ended: %ld tokens after the first in %ld rounds = %.3f tokens/round, %.1f tok/s "
                 "(decode), %.0f s in all (%s, widths %s)\n",
                 lines.size(), ended, tok, rounds, per_round, tok_s, wall_ms / 1000.0, e->backend_name().c_str(),
                 widths_str(*e).c_str());
    std::fflush(stdout);
    if (!a.json) return error.empty() ? 0 : 1;
    // The record with a prompt a line, so that it reads (and diffs) by prompt.
    std::string text = record.dump(1, ' ', false, nlohmann::json::error_handler_t::replace);
    text.resize(text.size() - 2);  // the closing "\n}"
    text += ",\n \"prompts\": [\n";
    for (size_t i = 0; i < lines.size(); ++i) text += "  " + lines[i] + (i + 1 < lines.size() ? ",\n" : "\n");
    text += " ]\n}\n";
    return write_record(text, a, out) && error.empty() ? 0 : 1;
  };
  for (size_t i = 0; i < prepared.size(); ++i) {
    const PreparedRequest ran = std::move(prepared[i]);  // its grammar's state goes with it once the prompt has run
    const GenerateRequest& gen = ran.gen;
    std::string err;
    Session s(e.get());
    // The reply's tokens as they are shown (no stop strings here: all of them), then the end token that ended it.
    std::vector<int32_t> ids;
    TextSink sink = [&ids](std::string_view, const int32_t* t, int n) {
      ids.insert(ids.end(), t, t + n);
      return true;
    };
    std::vector<flyx_round_result> trace;
    std::vector<int> which;
    s.trace = &trace;
    s.trace_rounds = &which;
    GenerateResult res;
    if (!s.init(&err) || generate(s, gen, sink, &res, &err) != 0) {
      const std::string error = a.prompts + " line " + std::to_string(line_of[i]) + ": " + err;
      std::fprintf(stderr, "error: %s\n", error.c_str());
      return finish(error);
    }
    s.trace = nullptr;
    s.trace_rounds = nullptr;
    if (res.stop_token >= 0) ids.push_back(res.stop_token), ++ended;
    // The ring's entries after the prefill's (which gave the first token): one a round, in order, the rounds
    // still in flight when the reply ended included, as usage.rounds counts them; each its tokens, its drafts
    // accepted, whether its last token was a draft, and the round it was (an index into "rounds"; -1 a lookup round).
    nlohmann::json steps = nlohmann::json::array();
    for (size_t k = 1; k < trace.size(); ++k)
      steps.push_back({trace[k].n_tokens, trace[k].n_accepted, last_token_drafted(trace[k]) ? 1 : 0,
                       k < which.size() ? which[k] : -3});
    const flyx_usage& u = res.usage;
    const char* finish_reason = u.finish_reason == FLYX_FINISH_STOP     ? "stop"
                                : u.finish_reason == FLYX_FINISH_LENGTH ? "length"
                                                                        : "other";
    std::fprintf(stderr,
                 "%4zu/%zu | prompt %3d tok | prefill %6.1f ms | %3d tok %7.1f ms = %6.1f tok/s | %.2f tok/round "
                 "| %s\n",
                 i + 1, prepared.size(), u.prompt_tokens, u.prefill_ms, u.new_tokens, u.decode_ms,
                 u.decode_ms > 0 ? (u.new_tokens - 1) * 1000.0 / u.decode_ms : 0,
                 u.rounds ? double(u.new_tokens - 1) / u.rounds : 0, finish_reason);
    tok += u.new_tokens - 1;
    ms += u.decode_ms;
    rounds += u.rounds;
    // What the rounds cost by now at the context this reply ended at, learned from the rounds read (null: the load's
    // costs throughout, FLYX_LEARN_COSTS=0, or nothing to choose among: one plan, no lookup).
    nlohmann::json learned = nullptr;
    if (const RoundCosts* costs = e->round_costs()) {
      learned = nlohmann::json::array();
      const int at = u.prompt_tokens + u.new_tokens;
      for (size_t k = 0; k < e->widths().size(); ++k) learned.push_back(costs->cost(k, at));
    }
    lines.push_back(nlohmann::json{{"i", i},
                                   {"prompt_tokens", u.prompt_tokens},
                                   {"prefill_ms", u.prefill_ms},
                                   {"new_tokens", u.new_tokens},
                                   {"decode_ms", u.decode_ms},
                                   {"rounds", u.rounds},
                                   {"drafted", u.drafted},
                                   {"accepted_drafts", u.accepted_drafts},
                                   {"drafter_rounds", res.drafter_rounds},
                                   {"round_choices", res.round_choices},
                                   {"round_accepted", res.round_accepted},
                                   {"lookup_rounds", res.lookup_rounds},
                                   {"round_ms_learned", learned},
                                   {"finish", finish_reason},
                                   {"prompt_ids", gen.prompt},
                                   {"ids", ids},
                                   {"trace", steps}}
                        .dump());
  }
  return finish("");
}

int cmd_bench(const Args& a) {
  if (!a.prompts.empty()) return cmd_bench_prompts(a);
  std::ofstream out;
  if (!open_record(a, &out)) return 1;
  auto e = load(a, /*elastic=*/false);
  std::vector<std::string> round_names;
  for (size_t i = 0; i < e->widths().size(); ++i) round_names.push_back(e->round_name(i));
  nlohmann::json record = {{"model", a.model},
                           {"backend", e->backend_name()},
                           {"device", std::string(e->device().name)},
                           {"max_new_tokens", a.n},
                           {"drafter", e->drafter()},
                           {"drafter_refused", e->refused()},
                           {"widths", e->widths()},
                           {"rounds", round_names},
                           {"round_ms", e->round_ms()},
                           {"device_before", print_device(*e, "before")},
                           {"prompts", nlohmann::json::array()}};
  const char* prompts[] = {
      "Write a Python function that parses an ISO 8601 date string without using datetime, with tests.",
      "Explain how a CPU cache works to a new engineer, with a concrete example of a cache miss.",
      "Copy this list exactly, one item per line: alpha, bravo, charlie, delta, echo, foxtrot, golf, hotel, india, "
      "juliet, kilo, lima, mike, november, oscar, papa, quebec, romeo, sierra, tango, uniform, victor, whiskey.",
  };
  double tok = 0, ms = 0, rounds = 0;
  // The process's CPU time (all threads) against wall time, over the requests together: os::process_usage, since
  // std::clock is wall time with MSVC (every wait 100% a core), and on Windows it counts whole clock ticks of 15.6 ms,
  // so one reading for the run rather than one a request.
  const double cpu0 = os::process_usage().cpu_seconds, wall0 = os::now_ms();
  for (const char* p : prompts) {
    Session s(e.get());
    std::string err;
    if (!s.init(&err)) return std::fprintf(stderr, "error: %s\n", err.c_str()), 1;
    Request rq;
    rq.messages = {msg("user", p)};
    rq.chat_options.enable_thinking = false;
    rq.split_output = false;
    rq.gen.max_new_tokens = a.n;
    rq.gen.ignore_eos = true;
    rq.gen.sampling = a.sampling;
    PreparedRequest pr = prepare(*e, s, rq, false);
    GenerateResult res;
    if (generate(s, pr.gen, TextSink(), &res, &err) != 0)
      return std::fprintf(stderr, "error: %s\n", err.c_str()), 1;
    const flyx_usage& u = res.usage;
    std::fprintf(stderr,
                 "prompt %3d tok | prefill %7.1f ms | decode %4d tok %8.1f ms = %6.1f tok/s | %.2f tok/round, "
                 "%.1f drafts/round\n",
                 u.prompt_tokens, u.prefill_ms, u.new_tokens, u.decode_ms, apps::decode_rate(u.new_tokens, u.decode_ms),
                 u.rounds && u.new_tokens > 1 ? double(u.new_tokens - 1) / u.rounds : 0,
                 u.rounds ? double(u.drafted) / u.rounds : 0.0);
    tok += std::max(0, u.new_tokens - 1);
    ms += u.decode_ms;
    rounds += u.rounds;
    record["prompts"].push_back({{"prompt_tokens", u.prompt_tokens},
                                 {"prefill_ms", u.prefill_ms},
                                 {"new_tokens", u.new_tokens},
                                 {"decode_ms", u.decode_ms},
                                 {"rounds", u.rounds},
                                 {"drafted", u.drafted},
                                 {"accepted_drafts", u.accepted_drafts},
                                 {"drafter_rounds", res.drafter_rounds},
                                 {"round_choices", res.round_choices},
                                 {"round_accepted", res.round_accepted},
                                 {"lookup_rounds", res.lookup_rounds}});
    record["device_after"] = print_device(*e, "after ");
  }
  // CPU time of the whole run (prefill and the requests' setup included), charged to the rounds.
  const double cpu_ms = (os::process_usage().cpu_seconds - cpu0) * 1000.0, wall_ms = os::now_ms() - wall0;
  const double tok_s = ms > 0 ? tok * 1000.0 / ms : 0, per_round = rounds ? tok / rounds : 0;
  const double cpu_us = rounds ? cpu_ms * 1000.0 / rounds : 0, core = wall_ms > 0 ? 100.0 * cpu_ms / wall_ms : 0;
  if (a.json) {
    record["decode_tok_s"] = tok_s;
    record["tokens_per_round"] = per_round;
    record["cpu_us_per_round"] = cpu_us;
    record["cpu_core_percent"] = core;
    if (!write_record(record.dump(1) + "\n", a, out)) return 1;
    if (a.json_file.empty()) return 0;  // the record is all of stdout
  }
  std::printf("decode: %.1f tok/s, %.2f tokens/round (%s, widths %s)\n", tok_s, per_round, e->backend_name().c_str(),
              widths_str(*e).c_str());
  std::printf("cpu: %.0f us per round, %.1f%% of one core\n", cpu_us, core);
  return 0;
}

int cmd_tokenize(const Args& a) {
  ModelFile file;
  Tokenizer tok;
  std::string err;
  if (!file.open(a.model, &err) || !tok.load(file, &err)) return std::fprintf(stderr, "error: %s\n", err.c_str()), 1;
  auto ids = tok.encode_prompt(a.extra.empty() ? a.prompt : a.extra);  // what `flyx run` feeds for this text
  for (size_t i = 0; i < ids.size(); ++i) std::printf(i ? " %d" : "%d", ids[i]);
  std::printf("\n");
  return 0;
}

int cmd_template(const Args& a) {
  if (a.extra.empty()) {
    std::fprintf(stderr, "error: template requires a MESSAGES.json path\n");
    return 2;
  }
  ModelFile file;
  Tokenizer tok;
  ChatTemplate chat;
  std::string err;
  if (!file.open(a.model, &err) || !tok.load(file, &err) || !chat.load(file, a.engine.chat_template, &err))
    return std::fprintf(stderr, "error: %s\n", err.c_str()), 1;
  std::ifstream f(os::to_path(a.extra));
  if (!f) return std::fprintf(stderr, "error: cannot read %s\n", a.extra.c_str()), 1;
  // Either a list of OpenAI-format messages or {"messages": [...], "tools": [...]}.
  auto j = ojson::parse(f, nullptr, false);
  if (j.is_discarded() || (!j.is_array() && (!j.is_object() || !j.contains("messages"))))
    return std::fprintf(stderr, "error: %s is not valid messages JSON\n", a.extra.c_str()), 1;
  const ojson& jm = j.is_object() ? j["messages"] : j;
  std::vector<ChatMessage> msgs;
  std::vector<ojson> tools;
  if (!messages_from_openai(jm, &msgs, &err)) return std::fprintf(stderr, "error: %s\n", err.c_str()), 1;
  if (j.is_object() && j.contains("tools") && !tools_from_openai(j["tools"], &tools, &err))
    return std::fprintf(stderr, "error: %s\n", err.c_str()), 1;
  RenderedChat rc;
  if (!chat.render(msgs, tools, a.chat_options, &rc, &err)) return std::fprintf(stderr, "error: %s\n", err.c_str()), 1;
  auto ids = tok.encode(rc.text, &rc.literal);
  nlohmann::json out = {{"text", rc.text},
                        {"ids", ids},
                        {"boundaries", token_positions(tok, ids, rc.boundaries)},
                        {"boundary_bytes", rc.boundaries},
                        {"format", chat.format()},
                        {"native", chat.native()},
                        {"in_reasoning", rc.in_reasoning}};
  std::printf("%s\n", out.dump().c_str());
  return 0;
}

// Whether the executable was called by this name (the last part of argv[0], without a ".exe").
bool called_as(const char* argv0, std::string_view name) {
  std::string_view s = argv0 ? argv0 : "";
  if (const size_t slash = s.find_last_of("/\\"); slash != std::string_view::npos) s.remove_prefix(slash + 1);
  if (s.ends_with(".exe")) s.remove_suffix(4);
  return s == name;
}

}  // namespace

int run_command(int argc, char** argv);

// Nothing a command throws ends the process without a word: a model file or a request no check foresaw is an error
// with its reason and status 1, not an abort.
int main(int argc, char** argv) {
  try {
    return run_command(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
  } catch (...) {
    std::fprintf(stderr, "error: an unexpected failure\n");
  }
  return 1;
}

int run_command(int argc, char** argv) {
#ifdef FLYX_STANDALONE_EXE
  flyx::apps::register_builtin_backends();
#endif
  flyx::os::raw_utf8_stdio();
  std::vector<std::string> args = flyx::os::utf8_arguments(argc, argv);  // Windows: from UTF-16, not the code page
  std::vector<char*> utf8_argv;
  for (std::string& a : args) utf8_argv.push_back(a.data());
  utf8_argv.push_back(nullptr);
  argc = int(args.size());
  argv = utf8_argv.data();
  if (argc == 2 && (std::strcmp(argv[1], "--version") == 0 || std::strcmp(argv[1], "-V") == 0 ||
                    std::strcmp(argv[1], "version") == 0)) {
    std::printf("flyx %s\n", apps::kVersion);
    return 0;
  }
  if (called_as(argv[0], "flyx-server")) return flyx::server::run(argc, argv);
  // Double-clicking flyx.exe in Windows Explorer starts with no arguments in a console of its own:
  // launch the server (which opens the web dashboard) instead of flashing usage text and exiting.
  if (argc == 1 && flyx::os::owns_console()) return flyx::server::run(argc, argv);
  if (argc >= 2 && std::strcmp(argv[1], "serve") == 0) return flyx::server::run(argc - 1, argv + 1);
  if (argc >= 2 && std::strcmp(argv[1], "monitor") == 0) return flyx::monitor::run(argc, argv);
  Args a = parse(argc, argv);
  {  // a Hugging Face name: its files fetched; the drafters only for the commands that draft
    std::string err;
    const bool engine = a.cmd == "run" || a.cmd == "chat" || a.cmd == "bench";
    EngineOptions files_only;
    files_only.speculative = false;
    files_only.vision = false;
    if (!apps::fetch_named(&a.model, engine ? &a.engine : &files_only, &err)) {
      std::fprintf(stderr, "error: %s\n", err.c_str());
      return 1;
    }
  }
  if (a.cmd == "run") return cmd_run(a);
  if (a.cmd == "chat") return cmd_chat(a);
  if (a.cmd == "bench") return cmd_bench(a);
  if (a.cmd == "tokenize") return cmd_tokenize(a);
  if (a.cmd == "template") return cmd_template(a);
  usage();
}
