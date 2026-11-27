// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The Scheduler (core/exec/scheduler.h) on the tiny model: requests that run at
// once take turns on the GPU thread and each gives what it gives alone; a
// request whose session is busy waits without holding up the ones behind it;
// the same conversation again cancels the running one, while raw prompts of
// different clients that are the same or extend one another all run (raw-clients);
// a queued job can be cancelled; a prompt longer than the context is refused
// when it starts.  With `dv`, a DV file's rounds in requests at once (dv_main).
//
//   scheduler_test MODEL.flyx BACKEND [dv | queue | history | raw-clients | ...]
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "core/exec/scheduler.h"
#include "core/os/os.h"
#include "tests/common/check.h"

using namespace flyx;

extern "C" void flyx_test_delegate(const flyx_backend_api*);

namespace {

// A job's events, read on a thread of their own as an HTTP handler would.
struct Reply {
  std::shared_ptr<Job> job;
  std::string text, error;
  std::string reuse;  // how its prompt's prefix was reused
  int reused = 0;
  bool done = false;
  bool superseded = false;
  bool unavailable = false;
  int finish = 0;
  int new_tokens = 0;
  double first_ms = 0, last_ms = 0;  // the first piece of text, the last event
  std::atomic<bool> started{false};  // text has arrived: the request is running
  std::thread reader;

  void read() {
    JobEvent ev;
    while (job->next(&ev, 120000)) {
      last_ms = os::now_ms();
      if (ev.kind == JobEvent::kContent) {
        if (first_ms == 0) first_ms = last_ms;
        text += ev.text;
        started.store(true);
      } else if (ev.kind == JobEvent::kDone) {
        done = true;
        superseded = ev.superseded;
        finish = ev.result.usage.finish_reason;
        new_tokens = ev.result.usage.new_tokens;
        reuse = ev.result.reuse;
        reused = ev.result.usage.reused_tokens;
        return;
      } else if (ev.kind == JobEvent::kError) {
        error = ev.text;
        superseded = ev.superseded;
        unavailable = ev.unavailable;
        return;
      }
    }
    error = "no event for two minutes";
  }
  void wait() {
    if (reader.joinable()) reader.join();
  }
  // Until its first text is here (or it ended without any).
  void wait_started() {
    while (!started.load() && !job->finished()) std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  ~Reply() { wait(); }
};

// Greedy, a fixed number of tokens (as many as the context leaves): the same text every time.
Request raw(const Scheduler& sched, const std::string& label, const std::string& text, int n) {
  Request rq;
  rq.chat = false;
  rq.text = text;
  rq.label = label;
  rq.gen.max_new_tokens = std::max(1, std::min(n, sched.max_ctx() - sched.count_text_tokens(text) - 24));
  rq.gen.ignore_eos = true;
  return rq;
}

std::unique_ptr<Reply> send(Scheduler& sched, const Request& rq, double arrived_at = -1) {
  auto r = std::make_unique<Reply>();
  std::string err;
  int status = 0;
  r->job = sched.submit(rq, &err, &status, arrived_at);
  if (!r->job) {
    r->error = "submit: " + err;
    return r;
  }
  r->reader = std::thread([p = r.get()] { p->read(); });
  return r;
}

std::string words(const char* word, int n) {
  std::string s;
  for (int i = 0; i < n; ++i) s += std::string(word) + " ";
  return s;
}

// The labels of the last n finished requests, oldest first.
std::vector<std::string> finished(const Scheduler& sched, size_t n) {
  TelemetrySnapshot t;
  flyx_device_stats d{};
  bool device_ok = false;
  sched.telemetry(&t, &d, &device_ok);
  std::vector<std::string> out;
  for (size_t i = std::min(n, t.recent.size()); i-- > 0;) out.push_back(t.recent[i].label);
  return out;
}

// What the scheduler recorded of the newest finished request with this label, and how many requests finished after
// it (-1: none with the label).
struct Finished {
  RequestRecord record;
  int after = -1;
};
Finished finished_as(const Scheduler& sched, const std::string& label) {
  TelemetrySnapshot t;
  flyx_device_stats d{};
  bool device_ok = false;
  sched.telemetry(&t, &d, &device_ok);
  for (size_t i = 0; i < t.recent.size(); ++i)
    if (t.recent[i].label == label) return {t.recent[i], int(i)};
  return {};
}

void check_queue(Scheduler& sched) {
  auto busy = send(sched, raw(sched, "busy", "Keep generating.", 1500));
  busy->wait_started();
  Request again = raw(sched, "replaced", "Wait here.", 1);
  again.conversation_id = "queued client";  // a raw prompt is replaced only by its own conversation
  auto replaced = send(sched, again);
  again.label = "front";
  auto head = send(sched, again);
  auto cancelled = send(sched, raw(sched, "cancelled", "Cancel this.", 1));
  cancelled->job->cancel();
  cancelled->wait();
  replaced->wait();
  check(replaced->superseded && replaced->error == "cancelled", "a newer request replaced the queued original");
  check(!cancelled->superseded && cancelled->error == "cancelled" && !busy->job->finished() && !head->job->finished(),
        "a cancellation behind the queue head returns while the only slot remains busy");
  head->wait();
  busy->wait();
  check(head->error == "timed out waiting in the queue" && busy->done,
        "the queued request reaches its timeout while the running one completes");
  TelemetrySnapshot t;
  flyx_device_stats device{};
  bool device_ok = false;
  sched.telemetry(&t, &device, &device_ok);
  bool saw_cancel = false, saw_timeout = false, saw_replaced = false;
  for (const auto& r : t.recent) {
    if (r.session || r.output_tokens) continue;
    if (r.label == "replaced") saw_replaced = r.status == "cancelled" && r.superseded;
    if (r.label == "cancelled") saw_cancel = r.status == "cancelled" && !r.superseded;
    if (r.label == "front") saw_timeout = r.status == "error" && r.error == head->error && r.queue_ms >= 100;
  }
  check(saw_replaced && saw_cancel && saw_timeout && t.totals.requests == 4 && t.totals.cancelled == 2 && t.totals.errors == 1,
        "recent requests and totals include a queue cancellation and timeout without a session");
}

// Raw prompts of different clients: the same text, or a text that extends another client's, is another request, and
// each reaches its own token limit; only a shared conversation id makes the later one replace the earlier.  One
// session (main sets max_sessions 1), so a later request waits in the queue while the earlier one runs.
void check_raw_clients(Scheduler& sched, int long_reply) {
  auto own_end = [](const Reply& r, const Request& rq) {
    return r.done && !r.superseded && r.finish == FLYX_FINISH_LENGTH && r.new_tokens == rq.gen.max_new_tokens;
  };
  auto superseded = [](const Reply& r) { return r.done && r.superseded && r.finish == FLYX_FINISH_CANCEL; };
  // Both queued behind a third client's request.
  {
    const Request busy_rq = raw(sched, "busy", "Keep generating.", long_reply);
    const Request cat = raw(sched, "cat", "cat", 6), catalog = raw(sched, "catalog", "catalog", 6);
    auto busy = send(sched, busy_rq);
    busy->wait_started();
    auto first = send(sched, cat);
    auto second = send(sched, catalog);
    const bool queued = !busy->job->finished();
    busy->wait();
    first->wait();
    second->wait();
    check(queued && own_end(*busy, busy_rq) && own_end(*first, cat) && own_end(*second, catalog),
          "two queued raw prompts, one extending the other's: both reach their own limits (" + first->error +
              second->error + ")");
  }
  // The first running when the second arrives.
  {
    const Request cat = raw(sched, "cat", "cat", long_reply), catalog = raw(sched, "catalog", "catalog", 6);
    auto first = send(sched, cat);
    first->wait_started();
    auto second = send(sched, catalog);
    const bool running = !first->job->finished();
    first->wait();
    second->wait();
    check(running && own_end(*first, cat) && own_end(*second, catalog),
          "a raw prompt extending another client's running one leaves it running: both reach their own limits (" +
              first->error + second->error + ")");
  }
  // The same anonymous prompt, sampled with another seed: queued, and while the first runs.
  {
    Request one = raw(sched, "seed 1", "Pick a number:", long_reply);
    one.gen.sampling = SamplingParams{0.9f, 0.9f, 40, 1};
    Request two = one;
    two.label = "seed 2";
    two.gen.sampling.seed = 2;
    two.gen.max_new_tokens = 6;
    Request three = two;
    three.label = "seed 3";
    three.gen.sampling.seed = 3;
    auto first = send(sched, one);
    first->wait_started();
    auto second = send(sched, two);
    auto third = send(sched, three);
    const bool running = !first->job->finished();
    first->wait();
    second->wait();
    third->wait();
    check(running && own_end(*first, one) && own_end(*second, two) && own_end(*third, three),
          "the same anonymous raw prompt with other seeds, running and queued: each reaches its own limit (" +
              first->error + second->error + third->error + ")");
  }
  // Another client's id: the same prompt is still another request.
  {
    Request mine = raw(sched, "mine", "Tell the story.", long_reply);
    mine.conversation_id = "client 1";
    Request theirs = raw(sched, "theirs", "Tell the story.", 6);
    theirs.conversation_id = "client 2";
    auto first = send(sched, mine);
    first->wait_started();
    auto second = send(sched, theirs);
    first->wait();
    second->wait();
    check(own_end(*first, mine) && own_end(*second, theirs),
          "the same raw prompt under two conversation ids: both reach their own limits");
  }
  // The documented replacement: the same conversation id, the same prompt or one that extends it.
  for (const char* next : {"Tell the story.", "Tell the story. Then another."}) {
    Request told = raw(sched, "told", "Tell the story.", long_reply);
    told.conversation_id = "client 1";
    Request retold = raw(sched, "retold", next, 6);
    retold.conversation_id = told.conversation_id;
    auto first = send(sched, told);
    first->wait_started();
    auto second = send(sched, retold);
    first->wait();
    second->wait();
    check(superseded(*first) && own_end(*second, retold),
          std::string("a raw prompt with the running one's conversation id replaces it (") + next + ")");
  }
}

void check_prewarmed_history(Scheduler& sched) {
  const std::string prefix = words("Shared input", 31);
  auto branch = send(sched, raw(sched, "long branch", prefix + "Alpha " + words("Detailed old branch", 55), 4));
  branch->wait();
  Request warm = raw(sched, "warm prefix", prefix, 1);
  warm.gen.prefill_only = true;
  auto ready = send(sched, warm);
  ready->wait();
  const double arrived_at = os::now_ms() - 1000;  // time spent in a caller before submit
  auto follow = send(sched, raw(sched, "follow", prefix + "Beta and a new detail.", 4), arrived_at);
  follow->wait();
  check(branch->done && ready->done && follow->done && follow->reused == sched.count_text_tokens(prefix),
        "an extension of an entire prewarmed history uses its live state even when an older branch matches as many "
        "bytes (" + std::to_string(follow->reused) + " reused, " + follow->reuse + ")");
  TelemetrySnapshot t;
  flyx_device_stats device{};
  bool device_ok = false;
  sched.telemetry(&t, &device, &device_ok);
  check(t.recent.size() == 3 && t.recent[0].ttft_ms >= 1000 && t.recent[0].ttft_ms <= follow->first_ms - arrived_at &&
            t.recent[1].ttft_ms < 0 && t.recent[2].ttft_ms >= t.recent[2].queue_ms,
        "TTFT includes time before submit, is absent on prewarm, and defaults to submit for a direct caller");
}

// Pause the fourth pass while a retry replaces the prewarm. Its next request must find all four
// completed passes in the original slot even when a second slot is available.
const flyx_backend_api* cpu = nullptr;
std::promise<void> fourth_pass, resume_pass;
int passes_until_pause = 0;
int gate_prefill(flyx_plan* plan, flyx_seq* seq, const int32_t* tokens, int32_t n, const flyx_gen_params* params) {
  if (passes_until_pause > 0 && --passes_until_pause == 0) {
    fourth_pass.set_value();
    resume_pass.get_future().wait();
  }
  return cpu->prefill(plan, seq, tokens, n, params);
}

void check_interrupted_prewarm(const char* model, EngineOptions eo, const SchedulerOptions& so, const std::string& mode) {
  const bool reassign = mode == "reassigned-prewarm";
  std::unique_ptr<os::DynLib> lib;
  flyx_device_info device{};
  std::string err;
  if (!open_backend(eo, &lib, &cpu, &device, &err)) return check(false, err);
  auto gated = *cpu;
  gated.prefill = gate_prefill;
  flyx_test_delegate(&gated);
  eo.backend = "test-host-kv";
  eo.speculative = false;
  Scheduler sched(model, eo, so);
  if (!sched.ok(&err)) return check(false, err);
  passes_until_pause = 4;  // exclude the model loader's probe
  Request prompt = raw(sched, "prewarm", words("long input", 24), 4);
  prompt.gen.prefill_only = true;
  auto warm = send(sched, prompt);
  const bool started = fourth_pass.get_future().wait_for(std::chrono::seconds(10)) == std::future_status::ready;
  std::unique_ptr<Reply> title, other;
  if (reassign) {
    title = send(sched, raw(sched, "title", "Name this document.", 1));
    other = send(sched, raw(sched, "other", "Write something else.", 240));
  }
  prompt.gen.prefill_only = mode == "repeated-prewarm";
  prompt.label = "replaced";
  // The anonymous prewarm hands its work to the prompt that extends it; the retry below replaces a raw
  // generation, which takes the conversation id they share.
  prompt.conversation_id = "prewarmed client";
  auto replaced = send(sched, prompt);
  prompt.label = "retry";
  auto reply = send(sched, prompt);  // a retry before the interrupted pass completes
  resume_pass.set_value();
  warm->wait();
  replaced->wait();
  reply->wait();
  check(started && warm->done && warm->superseded && replaced->superseded && reply->done,
        "the latest request replaces the prewarm and an earlier retry");
  if (!reassign) {
    if (prompt.gen.prefill_only) check(reply->text.empty(), "the repeated prewarm returns no text");
    check(reply->reused == 4 * eo.prefill_chunk,
          "the retry continues all four completed passes of a replaced prewarm with two slots (" +
              std::to_string(reply->reused) + " reused)");
    return;
  }
  title->wait();
  other->wait();
  TelemetrySnapshot t;
  flyx_device_stats d{};
  bool device_ok = false;
  sched.telemetry(&t, &d, &device_ok);
  auto session = [&](const char* label) {
    for (const auto& r : t.recent)
      if (r.label == label) return r.session;
    return 0;
  };
  check(other->done && title->done && session("prewarm") && session("prewarm") == session("other") &&
            session("title") && session("title") == session("retry") && session("retry") != session("other") &&
            finished(sched, 2) == std::vector<std::string>{"retry", "other"},
        "when another job takes the prewarm slot, the retry uses the freed title slot while the other job runs");
}

// A DV file's rounds in requests that run at once: every round DV's warm-up (EngineOptions::cycle),
// so that the sessions' rounds of several sessions are all warm-ups (launch_batch: plain decoding's rows together,
// the canvas pass a sequence at a time); a cycle of its warm-up, its trees and plain decoding (the trees alone, the
// others together where they meet); and the rounds chosen (DV does not draft while the rounds are batched).  Greedy,
// sampled for one seed and with penalties, each request gives plain decoding's text, alone and beside the others.
int dv_main(const char* model, const char* backend) {
  EngineOptions eo;
  eo.backend = backend;
  eo.max_ctx = 1024;
  eo.prefill_chunk = 16;
  eo.kv = FLYX_KV_16;
  eo.drafter = "dv";
  size_t plain = 0, warm = 0, narrow = 0, wide = 0;
  {
    Engine e;
    std::string err;
    if (!e.load(model, eo, &err)) {
      std::printf("load failed: %s\n", err.c_str());
      return 1;
    }
    int lo = 1 << 30, hi = 0;
    for (size_t i = 0; i < e.widths().size(); ++i) {
      const int w = e.widths()[i];
      if (e.round_drafters()[i] < 0) plain = i;
      else if (w == 0) warm = i;
      else if (w < lo) lo = w, narrow = i;
      if (e.round_drafters()[i] >= 0 && w > hi) hi = w, wide = i;
    }
    check(e.widths().size() == 4 && e.round_drafters()[warm] == 0 && lo < hi,
          "DV's rounds: plain decoding, its warm-up, trees of " + std::to_string(lo) + " and " + std::to_string(hi));
  }
  SchedulerOptions so;
  so.max_sessions = 3;
  so.verbose = std::getenv("FLYX_TEST_VERBOSE") ? 1 : 0;
  EngineOptions warmed = eo, cycled = eo, plain_eo = eo;
  warmed.cycle = {warm};
  cycled.cycle = {warm, plain, warm, narrow, warm, wide, plain, plain, warm, narrow, narrow, wide};
  plain_eo.speculative = false;
  plain_eo.drafter = "none";
  for (EngineOptions* o : {&eo, &warmed, &cycled, &plain_eo}) o->verbose = so.verbose;  // the backend's batch count
  Scheduler sched_plain(model, plain_eo, so), sched_warmed(model, warmed, so), sched_cycled(model, cycled, so),
      sched_chosen(model, eo, so);
  std::string err;
  for (Scheduler* sched : {&sched_plain, &sched_warmed, &sched_cycled, &sched_chosen})
    if (!sched->ok(&err)) {
      std::printf("load failed: %s\n", err.c_str());
      return 1;
    }
  const int N = 40;
  const std::vector<std::string> prompts = {"alpha beta gamma delta " + words("one two three", 4),
                                            "The quick brown fox " + words("jumps over", 6),
                                            "An unrelated request " + words("of its own", 6)};
  const std::vector<SamplingParams> samplings = {SamplingParams{}, SamplingParams{0.9f, 0.9f, 40, 4242},
                                                 SamplingParams{0.8f, 1.f, 0, 7, 1.5f, 0.25f}};
  const char* kinds[] = {"greedy", "sampled for one seed", "sampled, with penalties"};
  for (size_t s = 0; s < samplings.size(); ++s) {
    auto request = [&](Scheduler& sched, size_t k) {
      Request rq = raw(sched, "p" + std::to_string(k), prompts[k], N);
      rq.gen.sampling = samplings[s];
      return rq;
    };
    std::vector<std::string> want;
    for (size_t k = 0; k < prompts.size(); ++k) {
      auto r = send(sched_plain, request(sched_plain, k));
      r->wait();
      want.push_back(r->text);
    }
    for (Scheduler* sched : {&sched_warmed, &sched_cycled, &sched_chosen}) {
      const std::string which = sched == &sched_warmed   ? "every round DV's warm-up"
                                : sched == &sched_cycled ? "rounds cycled (warm-ups, trees, plain)"
                                                   : "rounds chosen";
      bool alone = true, together = true, ended = true;
      for (size_t k = 0; k < prompts.size(); ++k) {
        auto r = send(*sched, request(*sched, k));
        r->wait();
        alone = alone && r->text == want[k];
      }
      std::vector<std::unique_ptr<Reply>> at_once;
      for (size_t k = 0; k < prompts.size(); ++k) at_once.push_back(send(*sched, request(*sched, k)));
      for (size_t k = 0; k < prompts.size(); ++k) {
        at_once[k]->wait();
        ended = ended && at_once[k]->done;
        together = together && at_once[k]->text == want[k];
      }
      check(!want[0].empty() && alone && ended && together,
            std::string("DV, ") + kinds[s] + ", " + which + ": plain decoding's text alone and three at once");
    }
  }
  return flyx::test::finish("scheduler_test (DV)");
}

}  // namespace

int main(int argc, char** argv) {
  flyx::test::print_passes = true;
  if (argc < 3) {
    std::fprintf(stderr, "usage: scheduler_test MODEL.flyx BACKEND [dv]\n");
    return 2;
  }
  if (argc > 3 && std::string(argv[3]) == "dv") return dv_main(argv[1], argv[2]);
  EngineOptions eo;
  eo.backend = argv[2];
  eo.max_ctx = 2048;
  eo.prefill_chunk = 16;  // several passes per prompt
  // The generation marker varies with the last message, so the one-message probe cannot locate its start.
  eo.chat_template = "{% for m in messages %}{{ m.role }}:{{ m.content }}|{% endfor %}"
                     "{% if add_generation_prompt %}{{ messages[-1].content }}?{% endif %}";
  // A request that must still be running when the next one arrives: the tiny
  // model writes hundreds of tokens in the time a thread may be held up.  The
  // float64 backend is slower a round, but its lookup rounds take 8 to 11 of
  // these repeating prompts' tokens each (some 4000 tokens a second): 300 of
  // them were over in 44 to 74 ms, which a loaded machine outlasted (2 failed
  // runs in at most 21 beside other tests at -j 6): 1200, some 300 ms.
  const int kLong = eo.backend == "cpu_ref" ? 1200 : 1500;
  SchedulerOptions so;
  so.max_sessions = 2;
  so.host_cache_bytes = SnapshotStore::kPieceBytes;  // enough for one tiny checkpoint
  const bool queue_test = argc > 3 && std::string(argv[3]) == "queue";
  if (queue_test) {
    so.max_sessions = 1;
    so.queue_timeout_ms = 100;
  }
  so.verbose = std::getenv("FLYX_TEST_VERBOSE") ? 1 : 0;  // the scheduler's request log
  const std::string mode = argc > 3 ? argv[3] : "";
  if (mode == "raw-clients") so.max_sessions = 1;
  if (mode == "interrupted-prewarm" || mode == "reassigned-prewarm" || mode == "repeated-prewarm") {
    check_interrupted_prewarm(argv[1], eo, so, mode);
    return flyx::test::finish("scheduler_test");
  }
  Scheduler sched(argv[1], eo, so);
  std::string err;
  if (!sched.ok(&err)) {
    std::printf("load failed: %s\n", err.c_str());
    return 1;
  }
  if (queue_test) {
    check_queue(sched);
    return flyx::test::finish("scheduler_test");
  }
  if (argc > 3 && std::string(argv[3]) == "history") {
    check_prewarmed_history(sched);
    return flyx::test::finish("scheduler_test");
  }
  if (mode == "raw-clients") {
    check_raw_clients(sched, kLong);
    return flyx::test::finish("scheduler_test");
  }
  if (argc > 3 && std::string(argv[3]) == "marked-system") {
    Request main;
    main.gen.max_new_tokens = 1000;
    main.gen.ignore_eos = true;
    main.messages = {{"system", words("shared instructions", 28), {}, {}, ""}, {"user", "Explain the story.", {}, {}, ""}};
    main.cache_points = {{1, main.messages[0].content.size()}};
    auto foreground = send(sched, main);
    foreground->wait_started();
    Request title = main;
    title.messages.back().content = "Name the story.";
    title.gen.max_new_tokens = 4;
    auto side = send(sched, title);
    flyx_usage usage{};
    const bool concurrent = side->job && side->job->wait_prompt_usage(&usage, 60000) && !foreground->job->finished();
    side->wait();
    foreground->wait();
    check(concurrent && foreground->done && side->done && usage.reused_tokens >= 256,
          "the title reuses marked instructions while the first conversation is still generating (" +
          std::to_string(usage.reused_tokens) + " tokens)");
    return flyx::test::finish("scheduler_test");
  }
  {  // A shared system prompt can be longer than both conversations' actual turns.
    Request main;
    main.gen.max_new_tokens = 5;
    main.gen.ignore_eos = true;
    main.messages = {{"system", words("fixed instructions", 25), std::nullopt, {}, ""},
                     {"user", "Alpha analyzes a delivery.", std::nullopt, {}, ""}};
    auto first = send(sched, main);
    first->wait();
    Request aux = main;
    aux.messages.back().content = "Beta writes a title.";
    auto title = send(sched, aux);
    title->wait();
    main.messages.push_back({"assistant", first->text, std::nullopt, {}, ""});
    main.messages.push_back({"user", "Continue.", std::nullopt, {}, ""});
    auto follow = send(sched, main);
    follow->wait();
    TelemetrySnapshot state;
    flyx_device_stats device{};
    bool device_ok = false;
    sched.telemetry(&state, &device, &device_ok);
    check(first->done && title->done && follow->done && state.recent.size() >= 3 &&
              state.recent[0].session == state.recent[2].session && state.recent[1].session != state.recent[2].session,
          "a title with the same system instructions uses the free slot and leaves the main conversation resident");
    check(title->reused >= 256 && follow->reused >= title->reused,
          "the title reuses the system checkpoint and the follow-up keeps the main slot's history");
    aux.label = "idle title";
    aux.messages.back().content = "Draft a brief title.";
    auto idle_title = send(sched, aux);
    idle_title->wait();
    sched.telemetry(&state, &device, &device_ok);
    check(idle_title->done && state.recent.size() >= 3 && state.recent[0].session == state.recent[2].session &&
              state.recent[0].session != state.recent[1].session,
          "with both equally close siblings idle, a new title reuses the older title slot");
    main.messages.push_back({"assistant", follow->text, std::nullopt, {}, ""});
    main.messages.push_back({"user", "Explain in detail.", std::nullopt, {}, ""});
    main.label = "working";
    main.gen.max_new_tokens = kLong;
    auto working = send(sched, main);
    flyx_usage usage{};
    const bool main_started = working->job && working->job->wait_prompt_usage(&usage, 60000);
    aux.label = "new title";
    aux.messages.back().content = "Choose a different title.";
    auto another = send(sched, aux);
    // Admission is the property under test. The title's prefill can take longer than the remaining decode.
    const bool concurrent = main_started && another->job && another->job->wait_prompt_usage(&usage, 60000) &&
                            !working->job->finished();
    another->wait();
    working->wait();
    check(working->done && working->reused > 0 && working->reuse.find("snapshot from") == std::string::npos,
          "the next main turn resumes its resident state (" + working->reuse + ")");
    check(concurrent && another->done && working->done,
          "a title matching both resident conversations equally uses the idle slot while the main one runs");
  }
  // Long enough that two requests sent one after the other run at once (they overlap in time, below): 40 tokens were
  // 10 ms on cpu_ref, less than the test's thread may wait between the two sends on a loaded machine.  200 were 16 to
  // 19 ms on CUDA (RTX 5070 Ti, the two batched), which a loaded machine outlasted too (scheduler_cuda in the gpu
  // preset at -j 8; with every core busy a reader thread read one request's whole text after the other had ended):
  // 1500 there, 62 to 65 ms alone.
  const int N = eo.backend == "cpu_ref" ? 200 : 1500;
  // A reply a conversation's history is made of: short enough that kLong more tokens still fit the context.
  const int kReply = 200;
  const std::string a = "alpha beta gamma delta " + words("one two three", 4);
  const std::string b = "The quick brown fox " + words("jumps over", 6);
  std::printf("prompts of %d and %d tokens, context %d\n", sched.count_text_tokens(a), sched.count_text_tokens(b),
              sched.max_ctx());

  // Each alone: what it must give beside another, too.
  std::string alone_a, alone_b;
  {
    auto ra = send(sched, raw(sched, "a", a, N));
    ra->wait();
    auto rb = send(sched, raw(sched, "b", b, N));
    rb->wait();
    alone_a = ra->text;
    alone_b = rb->text;
    check(ra->done && rb->done, "two requests, one after the other " + ra->error + rb->error);
  }
  // At once: both run, on a session each, and neither changes the other's reply.
  {
    auto ra = send(sched, raw(sched, "a", a, N)), rb = send(sched, raw(sched, "b", b, N));
    ra->wait();
    rb->wait();
    check(ra->done && rb->done, "two requests at once: both end " + ra->error + rb->error);
    check(ra->text == alone_a && rb->text == alone_b, "two requests at once: each gives what it gives alone");
    check(ra->first_ms < rb->last_ms && rb->first_ms < ra->last_ms, "two requests at once: they overlap in time");
  }
  // Sampled, with a seed: the same again beside another request.
  {
    Request sampled = raw(sched, "sampled", b, N);
    sampled.gen.sampling = SamplingParams{0.9f, 0.9f, 40, 4242};
    auto first = send(sched, sampled);
    first->wait();
    auto ra = send(sched, raw(sched, "a", a, N)), again = send(sched, sampled);
    ra->wait();
    again->wait();
    check(first->done && again->done && !first->text.empty() && again->text == first->text && ra->text == alone_a,
          "a sampled request beside another: the same reply for its seed");
  }
  // A request whose session is running another request waits for it, and does
  // not hold up the one behind it.  `more` and `other` continue the same
  // history (a and its reply) in different ways; `third` is new and takes the
  // free session.  They end in the order third, more, other.
  {
    auto warm = send(sched, raw(sched, "a", a, kReply));  // a session whose history is a and its reply
    warm->wait();
    const std::string history = a + warm->text;
    auto more = send(sched, raw(sched, "more", history + words(" and then some more of the same", 4), kLong));
    auto other = send(sched, raw(sched, "other", history + words(" but something else entirely", 4), 8));
    auto third = send(sched, raw(sched, "third", "An unrelated request " + words("of its own", 6), 8));
    more->wait();
    other->wait();
    third->wait();
    check(more->done && other->done && third->done,
          "three requests, two of them for one session: all end " + more->error + other->error + third->error);
    const auto order = finished(sched, 3);
    check(order == std::vector<std::string>{"third", "more", "other"},
          "the one whose session was busy ran after it; the one behind it waited for neither (" + order[0] + ", " +
              order[1] + ", " + order[2] + ")");
  }
  // The same conversation again (a raw prompt says so with its conversation id): the request that was running is
  // cancelled.
  {
    const std::string story = "Once upon a time " + words("there was", 8);
    Request told = raw(sched, "story", story, kLong);
    told.conversation_id = "storyteller";
    auto first = send(sched, told);
    first->wait_started();
    Request retold = raw(sched, "story again", story, 8);
    retold.conversation_id = told.conversation_id;
    auto again = send(sched, retold);
    first->wait();
    again->wait();
    check(first->done && first->finish == FLYX_FINISH_CANCEL && first->superseded && !again->superseded,
          "the same conversation again cancels the running request " + first->error);
    check(again->done && again->finish == FLYX_FINISH_LENGTH && !again->text.empty(),
          "and runs to its own end " + again->error);
  }
  {
    Request rq = raw(sched, "foreground", "The story of a fox " + words("on the hill", 8), kLong);
    auto foreground = send(sched, rq);
    foreground->wait_started();
    rq.gen.prefill_only = true;
    auto prewarm = send(sched, rq);
    foreground->wait();
    prewarm->wait();
    check(foreground->done && foreground->finish == FLYX_FINISH_LENGTH && !foreground->text.empty() &&
              prewarm->done && prewarm->finish == FLYX_FINISH_LENGTH && prewarm->text.empty(),
          "prewarming an active prompt lets the foreground reply finish");
  }
  // A dynamic template: a different short question leaves the first running. A retry and a later turn
  // with the same history still supersede it, even when its generation marker changes in the follow-up.
  {
    Request old;
    old.messages = {{"user", "Alpha", std::nullopt, {}, ""}};
    old.gen.max_new_tokens = kLong;
    old.gen.ignore_eos = true;
    auto check_next = [&](std::vector<ChatMessage> messages, int expected, const char* label) {
      Request next = old;
      next.messages = std::move(messages);
      next.gen.max_new_tokens = 8;
      auto first = send(sched, old);
      first->wait_started();
      auto second = send(sched, next);
      first->wait();
      second->wait();
      check(first->done && first->finish == expected && second->done && second->finish == FLYX_FINISH_LENGTH, label);
    };
    check_next({{"user", "Beta", std::nullopt, {}, ""}}, FLYX_FINISH_LENGTH, "a different short chat leaves the first running");
    check_next(old.messages, FLYX_FINISH_CANCEL, "an identical short chat cancels the first");
    check_next({old.messages.front(), {"assistant", "A", std::nullopt, {}, ""}, {"user", "Go on", std::nullopt, {}, ""}},
               FLYX_FINISH_CANCEL, "a short chat follow-up cancels the first across a change of generation marker");
  }
  // A queued job that is cancelled says so and never runs.
  {
    auto warm = send(sched, raw(sched, "b", b, kReply));
    warm->wait();
    const std::string history = b + warm->text;
    auto busy = send(sched, raw(sched, "busy", history + words(" on and on it goes", 5), kLong));
    busy->wait_started();
    auto queued = send(sched, raw(sched, "queued", history + words(" or it stops right here", 5), 8));
    queued->job->cancel();
    busy->wait();
    queued->wait();
    check(busy->done && queued->error == "cancelled" && queued->text.empty(),
          "a queued job cancelled: an error, no text (" + queued->error + ")");
  }
  // Both sessions hold other conversations now.  Two new ones that share only
  // a few opening words (as every chat prompt shares its template's first
  // bytes) are tied to no session: the second takes the idle one and runs
  // beside the first.
  {
    auto cats = send(sched, raw(sched, "cats", "Tell me a story about " + words("a cat", 8), kLong));
    auto dogs = send(sched, raw(sched, "dogs", "Tell me a story about " + words("the dog", 8), 8));
    cats->wait();
    dogs->wait();
    // Where each ran is the scheduler's doing and what it recorded; which ended first is the machine's as well (the
    // long one outlasts the short one by tens of milliseconds on cpu_ref, less than a busy machine's hiccup).
    const Finished c = finished_as(sched, "cats"), g = finished_as(sched, "dogs");
    check(cats->done && dogs->done && c.record.session > 0 && g.record.session > 0 &&
              c.record.session != g.record.session,
          "two unrelated requests with the same opening words run at once, a session each (" +
              std::to_string(c.record.session) + ", " + std::to_string(g.record.session) + "; " +
              (g.after > c.after ? "dogs" : "cats") + " ended first)");
  }
  // A sibling: the same long preamble, another question.  With every slot
  // taken it takes its sibling's place, and waits while the sibling runs
  // rather than push out the unrelated conversation in the other session,
  // which a follow-up then still finds there.
  {
    const std::string preamble = words("You are a careful assistant that answers briefly.", 8);  // > 256 bytes
    const std::string story = "Tell me a story about " + words("the dog", 8);
    auto dogs = send(sched, raw(sched, "dogs", story, 8));  // the other session's conversation, as it is now
    dogs->wait();
    // Longer than twice the preamble, so the sibling shares less than half of it.
    const std::string asked = preamble + "First question: " + words("why", 110);
    auto first = send(sched, raw(sched, "first", asked, 8));
    first->wait();  // takes the place of the least recently used: the cats
    auto long_one = send(sched, raw(sched, "long", asked + first->text + words(" go on", 6), kLong));
    long_one->wait_started();
    auto sibling = send(sched, raw(sched, "sibling", preamble + "Second question: " + words("how", 6), 8));
    auto follow = send(sched, raw(sched, "follow", story + dogs->text + words(" and then", 4), 8));
    long_one->wait();
    sibling->wait();
    follow->wait();
    // The sibling ran where the long one ran and ended after it; the follow-up ran in the other session and found
    // its conversation still there (it reuses the story and the reply to it).  Whether the follow-up or the long one
    // ended first is not the scheduler's doing, and under load it was the long one.
    const auto order = finished(sched, 3);
    const Finished l = finished_as(sched, "long"), sib = finished_as(sched, "sibling");
    const Finished f = finished_as(sched, "follow");
    const int kept = sched.count_text_tokens(story + dogs->text);
    check(long_one->done && sibling->done && follow->done && l.record.session > 0 &&
              sib.record.session == l.record.session && sib.after >= 0 && sib.after < l.after &&
              f.record.session > 0 && f.record.session != l.record.session && f.record.reused_tokens >= kept,
          "a sibling waits for its busy sibling's session, the other session's conversation stays (" + order[0] + ", " +
              order[1] + ", " + order[2] + "; the follow-up reused " + std::to_string(f.record.reused_tokens) + " of " +
              std::to_string(kept) + ")");
    TelemetrySnapshot now;
    flyx_device_stats dd{};
    bool ok = false;
    sched.telemetry(&now, &dd, &ok);
    int reused = -1;
    for (const auto& r : now.recent)
      if (r.label == "follow") reused = r.reused_tokens;
    check(reused > 0, "and the unrelated conversation was left where it was (" + std::to_string(reused) +
                          " prompt tokens of its follow-up reused)");
  }
  // More conversations than sessions: three take turns on two. The first comes back with its next
  // turn after the other two took both sessions. Saving the displaced conversation must preserve the one
  // checkpoint the returning request needs. Its prompt up to that checkpoint comes from host memory. (That
  // the state loaded gives what a fresh prefill gives: golden_test.)
  {
    const std::string one = "First conversation: " + words("apples and pears", 100);
    const std::string two = "Second conversation: " + words("rivers and lakes", 100);
    const std::string three = "Third conversation: " + words("stars and planets", 100);
    auto r1 = send(sched, raw(sched, "one", one, 8));
    r1->wait();
    auto r2 = send(sched, raw(sched, "two", two, 8));
    r2->wait();
    auto r3 = send(sched, raw(sched, "three", three, 8));  // takes the session of `one`
    r3->wait();
    const std::string next_turn = one + r1->text + words(" and then", 4);
    auto back = send(sched, raw(sched, "one again", next_turn, 8));
    back->wait();
    check(r1->done && r2->done && r3->done && back->done && !back->text.empty(),
          "three conversations on two sessions: the first comes back " + back->error);
    check(back->reuse.find("snapshot from host memory") != std::string::npos && back->reused >= 256,
          "  with its checkpoint from host memory (" + back->reuse + ", " + std::to_string(back->reused) + " of " +
              std::to_string(sched.count_text_tokens(next_turn)) + " tokens)");
    flyx_usage prompt{};
    check(back->job->wait_prompt_usage(&prompt, 0) && prompt.reused_tokens == back->reused &&
              prompt.prompt_tokens > prompt.reused_tokens && prompt.new_tokens == 0,
          "the prompt usage before decoding includes the restored prefix");
  }
  // A prompt longer than the context: refused when it starts, before anything runs, with its tokens and the most the
  // context takes (a server answers it with an error status); the next request starts and runs as ever.
  {
    const int limit = prompt_limit(sched.max_ctx());
    check(flyx_room(limit, sched.max_ctx()) && !flyx_room(limit + 1, sched.max_ctx()),
          "the longest prompt the context takes is the one flyx_room leaves a round's room after (" +
              std::to_string(limit) + " in " + std::to_string(sched.max_ctx()) + ")");
    Request big = raw(sched, "too long", words("long", sched.max_ctx() + 64), 8);
    std::string e;
    int status = 0;
    auto job = sched.submit(big, &e, &status);
    JobEvent ev;
    const bool started = job && job->wait_start(&ev, 60000);
    check(job && !started && ev.kind == JobEvent::kError && ev.prompt_tokens > limit && ev.prompt_limit == limit,
          "a prompt longer than the context: refused at the start (" + ev.text + ")");
    flyx_usage prompt{};
    check(job && job->wait_prompt_usage(&prompt, 0) && prompt.prompt_tokens == 0,
          "a rejected request also wakes a reader waiting for prompt usage");
    auto after = send(sched, raw(sched, "after", a, 8));
    JobEvent none;
    check(after->job && after->job->wait_start(&none, 60000), "a prompt that fits starts");
    after->wait();
    check(after->done && !after->text.empty(), "and runs " + after->error);
  }
  TelemetrySnapshot t;
  flyx_device_stats d{};
  bool device_ok = false;
  sched.telemetry(&t, &d, &device_ok);
  check(t.running == 0 && t.queued == 0 && t.totals.requests == 43 && t.totals.errors == 1 && t.totals.cancelled == 4,
        "afterwards: nothing running or queued; " + std::to_string(t.totals.requests) + " requests, " +
            std::to_string(t.totals.errors) + " errors, " + std::to_string(t.totals.cancelled) + " cancelled");

  // Four concurrent sessions sharing rounds (greedy + seeded sampled + repeated text + relaxed acceptance):
  // every exact request produces the exact same output as alone, and a relaxed-acceptance request runs alongside
  // without breaking batching for the others.
  {
    SchedulerOptions so4 = so;
    so4.max_sessions = 4;
    Scheduler sched4(argv[1], eo, so4);
    std::string err4;
    if (!sched4.ok(&err4)) {
      check(false, "4-session scheduler load failed: " + err4);
    } else {
      const int M = 32;
      const std::string p[4] = {
          "alpha beta gamma " + words("one two three", 4),
          "The quick brown fox " + words("jumps over", 5),
          "repeat block: " + words("echo delta", 6) + "repeat block: " + words("echo delta", 3),
          "final sequence " + words("north south east west", 3),
      };
      Request rq[4] = {
          raw(sched4, "s0", p[0], M),
          raw(sched4, "s1", p[1], M),
          raw(sched4, "s2", p[2], M),
          raw(sched4, "s3", p[3], M),
      };
      rq[1].gen.sampling = SamplingParams{0.8f, 0.9f, 40, 1337};
      std::string solo[4];
      bool solo_ok = true;
      for (int i = 0; i < 4; ++i) {
        auto r = send(sched4, rq[i]);
        r->wait();
        solo_ok = solo_ok && r->done && !r->text.empty();
        solo[i] = r->text;
      }
      std::unique_ptr<Reply> conc[4];
      std::atomic<bool> polling{true};
      std::atomic<int> polls{0};
      std::thread poller([&] {
        while (polling.load()) {
          TelemetrySnapshot ts;
          flyx_device_stats ds{};
          bool ok = false;
          sched4.telemetry(&ts, &ds, &ok);
          (void)sched4.is_loaded();
          (void)sched4.max_sessions();
          polls.fetch_add(1);
          std::this_thread::yield();
        }
      });
      for (int i = 0; i < 4; ++i) conc[i] = send(sched4, rq[i]);
      bool conc_ok = true, same = true;
      for (int i = 0; i < 4; ++i) {
        conc[i]->wait();
        conc_ok = conc_ok && conc[i]->done;
        same = same && conc[i]->text == solo[i];
      }
      polling.store(false);
      poller.join();
      check(solo_ok && conc_ok && same && polls.load() > 0,
            "four requests at once (greedy, seeded sampled, repeated text) with concurrent telemetry polling: bit-identical");

      // One relaxed-acceptance request alongside two exact requests: exact replies stay identical.
      // (cpu_ref does not implement relaxed acceptance for width > 0; verifying its rejection still leaves
      // the two batched exact peers unaffected.)
      Request relaxed_rq = raw(sched4, "relaxed", p[3], M);
      relaxed_rq.gen.sampling.accept_min_p = 0.2f;
      auto r0 = send(sched4, rq[0]), r1 = send(sched4, rq[1]), rr = send(sched4, relaxed_rq);
      r0->wait();
      r1->wait();
      rr->wait();
      const bool relaxed_ok =
          eo.backend == "cpu_ref" ? (rr->done || rr->error.find("relaxed acceptance") != std::string::npos)
                                  : (rr->done && !rr->text.empty());
      check(r0->done && r1->done && relaxed_ok && r0->text == solo[0] && r1->text == solo[1],
            "a relaxed-acceptance request alongside exact requests isolates cleanly without altering exact replies");
    }
  }

  // Default max_sessions (0) resolves to kDefaultSessions (8), capped by sessions_that_fit().
  {
    SchedulerOptions so_def;
    Scheduler sched_def(argv[1], eo, so_def);
    std::string err_def;
    if (!sched_def.ok(&err_def)) {
      check(false, "default-sessions scheduler load failed: " + err_def);
    } else {
      const int expected =
          std::max(1, std::min(SchedulerOptions::kDefaultSessions, sched_def.engine()->sessions_that_fit()));
      check(sched_def.max_sessions() == expected &&
                (eo.backend != "cpu_ref" || sched_def.max_sessions() == SchedulerOptions::kDefaultSessions),
            "default max_sessions resolves to kDefaultSessions (8) capped by available memory");
      TelemetrySnapshot ts;
      flyx_device_stats ds{};
      bool dev_ok = false;
      sched_def.telemetry(&ts, &ds, &dev_ok);
      check(ts.loaded && !ts.model.device.empty() && !ts.model.backend.empty() && ts.model.chat_ok &&
                ts.model.max_sessions == expected,
            "ModelInfo in TelemetrySnapshot carries device, backend, chat_ok, and max_sessions");
    }
  }

  // Queued jobs cancelled by Scheduler::close() report unavailable=true (HTTP 503), while client cancels do not.
  {
    SchedulerOptions so_close;
    so_close.max_sessions = 1;
    so_close.queue_timeout_ms = 60000;
    Scheduler sched_close(argv[1], eo, so_close);
    std::string err_close;
    if (!sched_close.ok(&err_close)) {
      check(false, "close scheduler load failed: " + err_close);
    } else {
      auto busy = send(sched_close, raw(sched_close, "busy", "Keep generating.", 1500));
      busy->wait_started();
      auto client_cancel = send(sched_close, raw(sched_close, "client_cancel", "Cancel this.", 1));
      auto queued_close = send(sched_close, raw(sched_close, "queued_close", "Queued for close.", 1));
      client_cancel->job->cancel();
      client_cancel->wait();
      sched_close.close();
      queued_close->wait();
      busy->wait();
      std::string sub_err;
      int sub_status = 0;
      auto rejected = sched_close.submit(raw(sched_close, "after_close", "Too late.", 1), &sub_err, &sub_status);
      check(!client_cancel->unavailable && client_cancel->error == "cancelled",
            "client cancel of a queued job is not marked unavailable");
      check(queued_close->unavailable && !queued_close->done && !queued_close->error.empty(),
            "Scheduler::close() marks queued jobs unavailable (503)");
      check(!rejected && sub_status == 503, "submit after Scheduler::close() returns 503");
    }
  }

  // An idle reload followed by a request rejected in start() refreshes last_active_ and stays loaded.
  if (eo.backend == "cpu_ref") {
    EngineOptions eo_idle = eo;
    eo_idle.max_ctx = 128;
    SchedulerOptions so_idle;
    so_idle.max_sessions = 1;
    so_idle.keep_alive_ms = 2200;
    Scheduler sched_idle(argv[1], eo_idle, so_idle);
    std::string err_idle;
    if (!sched_idle.ok(&err_idle)) {
      check(false, "idle-reload scheduler load failed: " + err_idle);
    } else {
      const double deadline = os::now_ms() + 6000;
      while (sched_idle.is_loaded() && os::now_ms() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      const bool unloaded = !sched_idle.is_loaded();
      Request too_long;
      too_long.chat = false;
      too_long.label = "too_long";
      too_long.text = words("overflow", 200);
      too_long.gen.max_new_tokens = 1;
      auto r = send(sched_idle, too_long);
      r->wait();
      const bool reloaded = sched_idle.is_loaded();
      // Wait past one 1s scheduler idle wakeup tick (still well below keep_alive_ms = 2200 ms).
      std::this_thread::sleep_for(std::chrono::milliseconds(1300));
      const bool stayed_loaded = sched_idle.is_loaded();
      check(unloaded && !r->done && !r->error.empty() && reloaded && stayed_loaded,
            "idle reload refreshes last_active_ even when the triggering request fails in start()");
    }
  }

  return flyx::test::finish("scheduler_test");
}
