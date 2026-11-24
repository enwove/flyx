// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Generation: one request running on one session, advanced a step at a time.
//
// A step is one unit of work on the device: a prefill pass launched, or
// rounds launched and one result handled.  Whoever owns the GPU thread calls
// advance() until it says done, and may advance other generations in between
// (core/exec/scheduler): what a generation has done and what it does next is
// in this object, not on a call stack.  Asked not to wait, advance() does
// only what the device is ready for, so the thread can turn to another
// generation while the device works for this one.
//
//   start        checks the request, brings the session to the longest usable
//                prefix of the prompt
//   prefill      a pass per step, cut at the checkpoint positions; the last
//                pass also chooses the first token
//   first token  read, checked against the grammar, shown; the prompt's
//                checkpoint taken
//   rounds       launched as the session's RoundChooser names them, two in
//                flight; each result recorded, checked, shown.  A token the
//                grammar rejects: the rounds behind it are read, the state is
//                rolled back and the token is chosen again under the mask
//   done         held-back text shown, the session's history set
//
// The tokens a generation produces depend on its request and its session
// only, never on what else runs between its steps.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "core/exec/engine.h"
#include "core/exec/reply_text.h"
#include "core/session/session.h"
#include "core/speculation/chooser.h"
#include "flyx/backend_abi.h"
#include "flyx/flyx.h"

namespace flyx {

class GrammarMatcher;

struct SamplingParams {
  float temperature = 0;     // 0 = greedy
  float top_p = 1;
  int top_k = 0;
  uint64_t seed = 0;
  // What a token already chosen in this reply loses from its logit: once
  // (presence), and per time it was chosen (frequency).  0 = off.  The choice
  // stays exact under speculation (ops/sampling/sampling.h).
  float presence_penalty = 0;
  float frequency_penalty = 0;
  // Relaxed acceptance of drafts (flyx_gen_params; lossy: then speculation changes the output).  0 = off.
  float accept_min_p = 0;    // a draft within this fraction of the top token's probability is accepted
  float accept_typical = 0;  // Medusa's typical acceptance: probability above min(eps, sqrt(eps) exp(-entropy))
};

struct GenerateRequest {
  std::vector<int32_t> prompt;            // the whole conversation, as tokens
  std::vector<int32_t> checkpoints;       // prompt positions worth a checkpoint (message boundaries)
  std::vector<int32_t> cache_prefixes;    // explicit client cache breakpoints in ascending order
  int32_t system_end = 0;                 // system instructions and tools precede the first user
  int max_new_tokens = 0;                 // 0 = until a stop or the context is full
  bool prefill_only = false;              // prepare the prompt's reusable state without choosing an output token
  bool park_previous = false;             // another conversation is giving up this slot
  int share_system_prefix = -1;           // last client-marked system boundary another resident slot can use
  bool ignore_eos = false;
  SamplingParams sampling;
  std::vector<std::string> stop;          // stop strings: generation ends before the first one
  std::vector<int32_t> stop_tokens;       // extra stop tokens (besides the model's end-of-sequence)
  const std::atomic<bool>* cancel = nullptr;  // set from another thread to stop (also between prefill passes)
  // The grammar the reply must match (core/grammar), or null.  Checked token
  // by token; a token it rejects is chosen again with its mask.
  GrammarMatcher* grammar = nullptr;
  // Tests: mask every round's first token, one round in flight.  On an engine
  // without a drafter this is constrained decoding by its definition, which
  // the checked path must reproduce exactly.
  bool grammar_reference = false;
  // A prompt position whose checkpoint is kept through later requests (-1:
  // none): the end of the latest user message.  A new question makes the
  // template drop the thinking of the turns since that message, so the next
  // question's prompt differs from the session right after it.
  int32_t keep_checkpoint = -1;
  // Monitoring: called as the prompt goes to the device with the tokens done
  // (the first call: those reused) and the prompt's length.
  std::function<void(int done, int total)> on_prefill;
  // Monitoring: called as each round is read, with the drafts it verified (its width) and the ones accepted.
  std::function<void(int drafted, int accepted)> on_round;
  // Optional multimodal prefill payload (empty for text-only requests).
  std::vector<interfaces::MediaSpan> media_spans;
  std::vector<float> prompt_embeds;          // [prompt_embed_rows.size() * hidden], row-major
  std::vector<int32_t> prompt_embed_rows;    // sorted ascending absolute token indices in `prompt`
  std::vector<int32_t> pos_3d;               // optional [prompt.size() * 3] = (t, h, w) per token
  std::vector<int32_t> pos_rope_deltas;      // optional [prompt.size() + 1] decode rope_delta per boundary
  int32_t rope_delta = 0;                    // scalar decode offset after prompt when has_mrope
  bool has_mrope = false;
  std::string rendered_text;                 // unexpanded template text (for multimodal history_text_)
  std::vector<interfaces::RenderedImage> rendered_images;  // for multimodal PrefixCache routing
};

struct GenerateResult {
  flyx_usage usage{};
  std::string stop_string;                // the stop string that ended the generation, if any
  int32_t stop_token = -1;                // the stop token that ended it, if any
  std::string reuse;                      // how the prompt prefix was reused, or why it was not
  std::string parked;                     // checkpoints kept when this request displaced a conversation
  int grammar_fixes = 0;                  // tokens the grammar rejected and that were chosen again
  int lookup_rounds = 0;                  // rounds whose drafts were looked up in the sequence
  int lookup_accepted = 0;                // their drafts accepted
  std::vector<int> drafter_rounds;        // rounds each drafter drafted (Model::drafter_name), by number
  // Rounds of each of the engine's rounds to choose from (Engine::widths(), round_name()), and the drafts they
  // accepted: what the speculation controller chose (lookup rounds apart).
  std::vector<int> round_choices, round_accepted;
};

// The longest prompt a context of max_ctx positions takes (flyx_room keeps a round's room after it), and what a
// request whose prompt is longer is told.
inline int prompt_limit(int max_ctx) { return max_ctx - FLYX_MAX_ROWS; }
std::string prompt_too_long(int tokens, int max_ctx);

class Generation {
 public:
  // `req` (and the grammar and cancel flag it points to) must outlive the
  // generation.  sink may be empty.  The session is in use until the
  // generation is destroyed.
  Generation(Session* s, const GenerateRequest& req, TextSink sink);
  ~Generation();

  enum class Progress { kWorked, kBusy, kDone };
  // The next unit of work.  wait: wait for the device when the unit needs it.
  // Without it, only what does not have to wait is done: kBusy when the
  // device is still working for this generation and there is nothing else to do.
  Progress advance(bool wait);
  bool done() const { return phase_ == Phase::kDone; }
  // Once done: it ended as a reply (how: result().usage.finish_reason), or
  // failed with error().
  bool failed() const { return failed_; }
  const std::string& error() const { return error_; }
  const GenerateResult& result() const { return result_; }
  Session* session() const { return s_; }

  // ---- rounds of several sessions -----------------------------------------
  // Batched, a generation does not launch its rounds: the scheduler asks each for the rounds it
  // could launch now (round_candidates: the chooser's, its state unchanged), picks one each and
  // which go out together (core/speculation/batch_schedule.h), has each take its pick
  // (commit_round: its plan and parameters), launches them together (launch_batch) and tells
  // each (launched, or launch_failed).  Reading the results stays the generation's.  A tree
  // runs alone; DV's warm-up (no tree) may go out beside other sessions' warm-ups, its canvas
  // pass a sequence at a time (the backend's).  DV does not draft while batched, so only a
  // --cycle that names its warm-up batches it: plain rounds of several sessions read the
  // weights once for all of them.
  struct NextRound {
    flyx_plan* plan = nullptr;
    flyx_gen_params g{};  // its mask and given drafts point into this generation, valid until it advances
    RoundChoice round{};
    int rows = 0;
  };
  void set_batched(bool on) {
    batched_ = on;
    s_->rounds().batched(on);
  }
  // In the round phase with its reply going on: it will want rounds (round_ready: one now).
  bool in_rounds() const { return phase_ == Phase::kRounds && launching_; }
  bool round_ready() const;
  // False: no round to launch now (not in the round phase, its reply over, or its rounds in flight at the limit).
  bool next_round(NextRound* out);
  // Candidate rounds for joint batch scheduling (core/speculation/batch_schedule.h); does not advance chooser state.
  // Follow with commit_round() for the selected candidate before launching.
  bool round_candidates(std::vector<RoundCandidate>* out);
  void commit_round(const RoundChoice& choice, NextRound* out);
  void launched(const NextRound& r) { launched_[in_flight_++] = r.round; }
  void launch_failed(const std::string& why) { fail("launch: " + why, true); }

 private:
  enum class Phase { kStart, kPrefill, kFinishPrefill, kRounds, kDone };

  bool start();
  bool prefill();
  bool finish_prefill();
  bool checkpoint_prompt_end();
  bool round(bool wait, bool* busy);
  bool finish_up();
  // The next call that waits for the device would not wait.
  bool device_ready() const;
  // The request is over with an error; reset: the device state is unknown.
  bool fail(std::string msg, bool reset);
  bool cancelled() const;
  bool penalized() const { return g_.presence_penalty != 0.f || g_.frequency_penalty != 0.f; }
  void prepare_round(const RoundChoice& round, bool masked, flyx_gen_params* params, flyx_plan** plan);
  bool launch(const RoundChoice& round, bool masked);
  // The index of the first token of `r` the grammar rejects, or -1; the tokens before it advance the grammar.
  int rejected(const flyx_round_result& r);
  // Counts and shows a result's tokens up to a stop; false when the reply has ended.
  bool deliver(const flyx_round_result& r);
  // Token `bad` of `r` does not fit the grammar: show what came before it,
  // undo the rest and choose it again under the mask.
  bool choose_again(const flyx_round_result& r, int bad);

  Session* s_;
  Engine* e_;
  const flyx_backend_api* api_;
  const GenerateRequest& req_;
  ReplyText reply_;
  GenerateResult result_;
  std::string error_;
  bool failed_ = false;
  Phase phase_ = Phase::kStart;

  flyx_gen_params g_{};
  std::vector<uint32_t> mask_;     // the grammar's allowed tokens, for a launch that needs them
  int n_ = 0;                      // prompt tokens
  int max_new_ = 0;
  double started_ms_ = 0, decoding_ms_ = 0;
  // prefill
  std::vector<int> cuts_;          // prompt positions where a pass ends and a checkpoint is taken
  size_t cut_ = 0;
  int pos_ = 0;                    // prompt tokens given to the device
  int keep_ = -1;                  // the cut whose checkpoint is kept
  bool checkpoint_due_ = false;    // the pass in flight ends at a cut, or crosses one: its checkpoint, once it has run
  int crossed_cut_ = -1;           // when >= 0, the intra-pass cut that checkpoint_due_ will capture
  bool due_keep_ = false;          // that checkpoint is the kept one (keep_)
  bool end_keep_ = false;          // the last pass crossed keep_: the checkpoint after the first token is the kept one
  // reply
  int emitted_ = 0;                // tokens counted against max_new
  int finish_ = 0;                 // flyx_finish_reason once the reply has ended
  // rounds
  bool every_ = false;             // GenerateRequest::grammar_reference
  bool launching_ = false;
  int64_t next_disk_stage_ = 0;    // next accepted position at which to stage generated KV (0: no disk tier)
  bool stage_due_ = false;         // drain outstanding rounds before reading their state
  bool batched_ = false;           // the scheduler launches its rounds (round_candidates, commit_round)
  int in_flight_ = 0;
  RoundChoice launched_[FLYX_MAX_IN_FLIGHT];  // the rounds in flight, oldest first
};

// Runs a request to its end on the calling thread (the C API, the command
// line, tests).  0, or -1 with *err.
int generate(Session& s, const GenerateRequest& req, const TextSink& sink, GenerateResult* result, std::string* err);

// Whether a prompt can run on the engine at all (not empty, fits the context, every token in the vocabulary), else
// why not: what a generation checks as it starts, for a caller that checks many prompts before running any.
bool prompt_runs(const Engine& e, const std::vector<int32_t>& prompt, std::string* why);

}  // namespace flyx
