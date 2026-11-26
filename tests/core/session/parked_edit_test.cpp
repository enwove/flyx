// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Editing a long branch when a finite cache already holds the checkpoint it should resume.
#include <cstdio>
#include <stdexcept>
#include <string>

#include "core/exec/scheduler.h"

using namespace flyx;

namespace {
struct Reply {
  std::string text;
  GenerateResult result;
};

Reply ask(Scheduler& scheduler, const std::string& prompt) {
  Request rq;
  rq.chat = false;
  rq.text = prompt;
  rq.gen.max_new_tokens = 4;
  rq.gen.ignore_eos = true;
  std::string err;
  int status = 0;
  auto job = scheduler.submit(std::move(rq), &err, &status);
  if (!job) throw std::runtime_error(err);
  Reply reply;
  JobEvent ev;
  while (job->next(&ev, 30000)) {
    if (ev.kind == JobEvent::kError) throw std::runtime_error(ev.text);
    if (ev.kind == JobEvent::kContent) reply.text += ev.text;
    if (ev.kind == JobEvent::kDone) {
      reply.result = std::move(ev.result);
      return reply;
    }
  }
  throw std::runtime_error("timed out");
}

std::string repeat(const char* text, int n) {
  std::string out;
  while (n-- > 0) out += text;
  return out;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  EngineOptions engine;
  engine.max_ctx = 2048;
  engine.backend = "cpu_ref";
  engine.speculative = false;
  SchedulerOptions options;
  options.max_sessions = 1;
  options.host_cache_bytes = SnapshotStore::kPieceBytes;
  const std::string base = repeat(" Common instruction.", 28);
  const std::string common = base + " Additional shared detail.";
  const std::string edited = common + " Corrected ending.";
  try {
    Scheduler scheduler(argv[1], engine, options);
    std::string err;
    if (!scheduler.ok(&err)) throw std::runtime_error(err);
    const Reply known = ask(scheduler, base);
    ask(scheduler, repeat(" Independent title.", 30));
    const Reply branch = ask(scheduler, common + repeat(" Much longer original history.", 35));
    const Reply back = ask(scheduler, edited);
    const int checkpoint = known.result.usage.prompt_tokens;
    std::printf("checkpoint %d, original branch reused %d, deep edit reused %d (%s); parking: %s\n",
                checkpoint, branch.result.usage.reused_tokens, back.result.usage.reused_tokens,
                back.result.reuse.c_str(), back.result.parked.c_str());
    if (checkpoint < 256 || branch.result.usage.reused_tokens != checkpoint ||
        back.result.parked.empty() || back.result.usage.reused_tokens != checkpoint) {
      std::fprintf(stderr, "FAIL the branch and the deep edit both reuse the checkpoint of at least 256 tokens, the "
                           "edit with the branch parked\n");
      return 1;
    }
    Scheduler cold(argv[1], engine, options);
    if (!cold.ok(&err)) throw std::runtime_error(err);
    const Reply fresh = ask(cold, edited);
    if (back.result.usage.new_tokens != 4 || fresh.result.usage.new_tokens != 4 || back.text.empty() ||
        back.text != fresh.text) {
      std::fprintf(stderr, "FAIL the edited prompt's reply (%d tokens) is a cold start's (%d tokens, 4 each): \"%s\" "
                           "vs \"%s\"\n", back.result.usage.new_tokens, fresh.result.usage.new_tokens,
                   back.text.c_str(), fresh.text.c_str());
      return 1;
    }
    std::printf("parked_edit: ok\n");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
