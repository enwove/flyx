// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/exec/generation.h"

#include <algorithm>
#include <cmath>
#include <random>

#include "core/grammar/grammar.h"
#include "core/os/os.h"
#include "core/vision/media_span.h"

namespace flyx {

namespace {
constexpr int kAnchorTokens = 10 * 1024;
}

Generation::Generation(Session* s, const GenerateRequest& req, TextSink sink)
    : s_(s), e_(s->engine()), api_(s->engine()->api()), req_(req),
      reply_(s->engine()->tokenizer(), req.stop, std::move(sink)) {
  s_->set_in_use(true);
}

Generation::~Generation() { s_->set_in_use(false); }

bool Generation::device_ready() const {
  return api_->ready(s_->seq()) != 0;  // an error counts: the call that follows reports it
}

Generation::Progress Generation::advance(bool wait) {
  bool busy = false;
  switch (phase_) {
    case Phase::kStart:
      start();
      break;
    case Phase::kPrefill:  // the pass before must have run: prefill() and checkpoint() wait for it
    case Phase::kFinishPrefill:
      if (!wait && !device_ready()) return Progress::kBusy;
      if (phase_ == Phase::kPrefill) prefill();
      else finish_prefill();
      break;
    case Phase::kRounds:
      round(wait, &busy);
      break;
    case Phase::kDone:
      break;
  }
  return phase_ == Phase::kDone ? Progress::kDone : busy ? Progress::kBusy : Progress::kWorked;
}

std::string prompt_too_long(int tokens, int max_ctx) {
  // A context under FLYX_MAX_ROWS holds no prompt at all: 0, not the negative figure the subtraction gives.
  return "prompt does not fit the context (" + std::to_string(tokens) + " tokens, at most " +
         std::to_string(std::max(0, prompt_limit(max_ctx))) + " in a context of " + std::to_string(max_ctx) + ")";
}

bool Generation::fail(std::string msg, bool reset) {
  error_ = std::move(msg);
  failed_ = true;
  if (reset) s_->reset_after_error();
  phase_ = Phase::kDone;
  return false;
}

bool Generation::cancelled() const {
  return s_->cancel_requested() || (req_.cancel && req_.cancel->load());
}

bool prompt_runs(const Engine& e, const std::vector<int32_t>& prompt, std::string* why) {
  const int n = static_cast<int>(prompt.size()), max_ctx = e.options().max_ctx;
  if (n == 0) return *why = "empty prompt", false;
  if (!flyx_room(n, max_ctx)) return *why = prompt_too_long(n, max_ctx), false;
  for (int32_t t : prompt) {
    if (t < 0 || t >= e.model().desc().vocab) {
      *why = "token id " + std::to_string(t) + " is outside the vocabulary";
      return false;
    }
  }
  return true;
}

bool Generation::start() {
  s_->clear_cancel();
  // A session the engine refused has no sequence to run on: nothing goes to the backend.
  if (!s_->seq())
    return fail("the session has no sequence (" +
                    (s_->init_error().empty() ? std::string("Session::init was not called") : s_->init_error()) + ")",
                false);
  const auto& prompt = req_.prompt;
  n_ = static_cast<int>(prompt.size());
  const int max_ctx = e_->options().max_ctx;
  std::string why;
  if (!prompt_runs(*e_, prompt, &why)) return fail(why, false);

  auto finite = [](float v, float fallback = 0.f) { return std::isfinite(v) ? v : fallback; };
  g_ = {};
  const float temp = finite(req_.sampling.temperature);
  const float top_p = finite(req_.sampling.top_p, 1.f);
  g_.temperature = temp > 0 ? temp : 0.f;
  g_.top_p = (top_p > 0 && top_p < 1) ? top_p : 1.f;
  g_.top_k = std::max(0, req_.sampling.top_k);
  g_.seed = req_.sampling.seed;
  g_.presence_penalty = finite(req_.sampling.presence_penalty);
  g_.frequency_penalty = finite(req_.sampling.frequency_penalty);
  g_.accept_min_p = std::max(0.f, finite(req_.sampling.accept_min_p));
  g_.accept_typical = std::max(0.f, finite(req_.sampling.accept_typical));
  if (!req_.prefill_only && g_.temperature > 0 && g_.seed == 0)
    g_.seed = (uint64_t(std::random_device{}()) << 32) ^ std::random_device{}();
  auto add_stop = [&](int32_t id) {
    for (int i = 0; i < g_.n_stop_ids; ++i)
      if (g_.stop_ids[i] == id) return;
    if (g_.n_stop_ids < FLYX_MAX_STOP_IDS) g_.stop_ids[g_.n_stop_ids++] = id;
  };
  if (!req_.ignore_eos)
    for (int32_t id : e_->eos()) add_stop(id);
  for (int32_t id : req_.stop_tokens) add_stop(id);
  max_new_ = req_.max_new_tokens > 0 ? req_.max_new_tokens : max_ctx;
  g_.limit = static_cast<int32_t>(std::min<long long>(static_cast<long long>(n_) + max_new_ - 1, max_ctx));
  // A grammar: the first token is chosen under its mask (nothing to undo
  // before it); later tokens are checked after each round.
  if (!req_.prefill_only && req_.grammar) {
    mask_.resize(size_t((e_->model().desc().vocab + 31) / 32));
    req_.grammar->allowed(mask_.data());
    g_.mask = mask_.data();
  }

  // Park and restore in one scheduler step. Time the incoming prefill from its own preparation.
  const auto* media = req_.media_spans.empty() ? nullptr : &req_.media_spans;
  if (req_.park_previous) s_->park_for(prompt, &result_.parked, media);
  started_ms_ = os::now_ms();
  std::string err;
  const int reused = s_->prepare(prompt, &result_.reuse, &err, media);
  if (reused < 0) return fail(err, true);
  // Penalties count the reply's tokens, and a reply starts here with none.
  if (!req_.prefill_only && penalized() && api_->set_reply(s_->seq(), nullptr, 0) != 0)
    return fail(std::string("set_reply: ") + api_->last_error(), true);
  result_.usage.prompt_tokens = n_;
  result_.usage.reused_tokens = reused;
  if (req_.on_prefill) req_.on_prefill(reused, n_);

  // The rest of the prompt goes in passes cut at the checkpoint positions
  // (the last two message boundaries past the reused part); the last pass
  // chooses the first token when a reply is requested. Cuts and pass ends sit on multiples of the
  // backend's prefill alignment, so every pass after the first starts where a
  // fresh prefill would reproduce it bit for bit (numerics.md: K); a
  // checkpoint at a message boundary keeps the boundary before it.
  const int align = e_->prefill_align();
  for (int32_t at : req_.checkpoints) {
    // A boundary at the prompt's very end: a template that opens the reply with nothing of its own (Inkling-2's), so
    // that the conversation ends where the prompt does.  No pass is cut there; prefill() keeps the last pass's start.
    if (at == n_) ends_at_boundary_ = true;
    at = at / align * align;
    if (at > reused && at < n_ && !vision::inside_media_span(req_.media_spans, at)) cuts_.push_back(at);
  }
  std::sort(cuts_.begin(), cuts_.end());
  cuts_.erase(std::unique(cuts_.begin(), cuts_.end()), cuts_.end());
  if (cuts_.size() > 2) cuts_.erase(cuts_.begin(), cuts_.end() - 2);
  keep_ = req_.keep_checkpoint >= 0 ? req_.keep_checkpoint / align * align : -1;
  if (keep_ >= 0 && vision::inside_media_span(req_.media_spans, keep_)) keep_ = -1;
  pos_ = reused;
  phase_ = Phase::kPrefill;
  return true;
}

bool Generation::prefill() {
  const auto& prompt = req_.prompt;
  const auto* media = req_.media_spans.empty() ? nullptr : &req_.media_spans;
  if (cancelled()) {  // between passes: only expose live state if it does not split a MediaSpan
    if (vision::inside_media_span(req_.media_spans, pos_)) {
      s_->reset_after_error();
    } else {
      s_->advanced(prompt, pos_, media);
      if (checkpoint_due_)
        s_->take_checkpoint(prompt, pos_, due_keep_, Session::Anchor::None, false, media);
    }
    result_.usage.prefill_ms = os::now_ms() - started_ms_;
    result_.usage.finish_reason = FLYX_FINISH_CANCEL;
    phase_ = Phase::kDone;
    return false;
  }
  const int align = e_->prefill_align();
  while (cut_ < cuts_.size() && cuts_[cut_] <= pos_) ++cut_;
  int end = std::min(pos_ / align * align + e_->prefill_chunk(), n_);
  // A cut the pass crosses needs no pass of its own: the backend keeps the state there (checkpoint() gives it, the
  // bits a pass ending there gives), and the pass goes on to its end; the last multiple of the alignment it crosses,
  // or any one where the backend takes prefill_snapshot.  A short prompt cut at its last message boundary took two
  // passes, each reading every weight (the 27B: 52 ms for 44 tokens against 28 in one); a 3.6K-token chat whose
  // boundary fell 37 tokens before its end, 261 ms against 217.  A second cut in the same pass still ends it at the
  // first (one kept state a pass).
  int crossed = -1;
  if (cut_ < cuts_.size() && cuts_[cut_] < end) {
    const int c = cuts_[cut_];
    const bool alone = cut_ + 1 >= cuts_.size() || cuts_[cut_ + 1] >= end;
    if (align > 1 && (c == end / align * align || (api_->prefill_snapshot && alone))) crossed = c;
    else end = c;
  }
  // Inspect, but never split, the passes chosen above. A checkpoint just before the client's boundary is
  // usable even when the following message changes; inserting a short pass here could change the math.
  Session::Anchor anchor = Session::Anchor::None;
  if (pos_ > result_.usage.reused_tokens) {
    const auto mark = std::lower_bound(req_.cache_prefixes.begin(), req_.cache_prefixes.end(), pos_);
    if (pos_ <= req_.system_end && req_.system_end < end) anchor = Session::Anchor::System;
    else if ((mark != req_.cache_prefixes.end() && *mark < end) ||
             ((s_->disk_cache() || !req_.cache_prefixes.empty()) && (pos_ - 1) / kAnchorTokens != (end - 1) / kAnchorTokens))
      anchor = Session::Anchor::Prefix;
  }
  // An identical follow-up must run its last token to choose its own first output. If the end checkpoint
  // would include that token, keep the preceding prefill pass boundary: for a prompt only prefilled, and for one whose
  // conversation ends where it does (ends_at_boundary_), which has no boundary before its end for the same prompt
  // asked again (a retry, another sample) to start from; that request computed its whole prompt once the reply had
  // left the rollback window (6,303 tokens on Inkling-2-Mini: 2.9 s). It now runs this last pass again, the rows that
  // ran here, so its first token is computed as this one's was. Before the first pass, the reused
  // state can come directly from decode and must not become a restorable prefill checkpoint.
  const bool last = end == n_;
  const bool before_last = (req_.prefill_only || ends_at_boundary_) && last && n_ % align == 0 &&
                           pos_ > result_.usage.reused_tokens;
  if (checkpoint_due_ || anchor != Session::Anchor::None || before_last) {
    if (!vision::inside_media_span(req_.media_spans, pos_)) {
      s_->advanced(prompt, pos_, media);
      const bool share = pos_ <= req_.share_system_prefix && req_.share_system_prefix < end;
      // The question's checkpoint, kept through the tool calls that follow it: the cut at its end, or this pass's
      // start for a question that ends where the prompt does.
      const bool keep = (checkpoint_due_ && due_keep_) || (before_last && keep_ == n_);
      s_->take_checkpoint(prompt, pos_, keep, anchor, share, media);
    } else if (checkpoint_due_ && crossed_cut_ >= 0) {
      // Pass ended inside a MediaSpan, but its intra-pass snapshot sits at crossed_cut_ (outside any span).
      s_->take_checkpoint(prompt, pos_, due_keep_, anchor, false, media);
    }
    checkpoint_due_ = false;
    crossed_cut_ = -1;
  }
  const bool sample = last && !req_.prefill_only;
  // Never let the backend's implicit intra-pass snapshot land inside a MediaSpan: walk backward to an aligned
  // position outside all spans, or suppress intra-pass snapshotting so checkpoint_prompt_end() captures `end`.
  int snap_pos = crossed;
  if (align > 1 && api_->prefill_snapshot && snap_pos < 0 && !req_.media_spans.empty()) {
    const int nat = end / align * align;
    if (nat >= pos_ && nat < end && vision::inside_media_span(req_.media_spans, nat)) {
      int b = nat - align;
      while (b >= pos_ && vision::inside_media_span(req_.media_spans, b)) b -= align;
      snap_pos = (b > 0 && b >= pos_) ? b : FLYX_PREFILL_SNAP_NONE;
    }
  }
  if (align > 1 && api_->prefill_snapshot && api_->prefill_snapshot(s_->seq(), snap_pos) != 0)
    return fail(std::string("prefill_snapshot: ") + api_->last_error(), true);
  const bool multimodal = !req_.media_spans.empty() || !req_.prompt_embed_rows.empty() || req_.has_mrope;
  if (!multimodal) {
    if (api_->prefill(e_->prefill_plan(), s_->seq(), prompt.data() + pos_, end - pos_, sample ? &g_ : nullptr) != 0)
      return fail(std::string("prefill: ") + api_->last_error(), true);
  } else {
    if (!api_->prefill_ex) return fail("backend does not support multimodal prefill", true);
    flyx_prefill_input in{};
    in.tokens = prompt.data() + pos_;
    in.n = end - pos_;
    std::vector<int32_t> local_rows;
    if (!req_.prompt_embed_rows.empty()) {
      const auto lo = std::lower_bound(req_.prompt_embed_rows.begin(), req_.prompt_embed_rows.end(), pos_);
      const auto hi = std::lower_bound(lo, req_.prompt_embed_rows.end(), end);
      if (lo != hi) {
        const size_t first_idx = size_t(lo - req_.prompt_embed_rows.begin());
        local_rows.reserve(size_t(hi - lo));
        for (auto it = lo; it != hi; ++it) local_rows.push_back(*it - pos_);
        in.embeds = req_.prompt_embeds.data() + first_idx * size_t(e_->model().desc().hidden);
        in.embed_rows = local_rows.data();
        in.n_embeds = static_cast<int32_t>(local_rows.size());
      }
    }
    if (!req_.pos_3d.empty()) in.pos_3d = req_.pos_3d.data() + size_t(pos_) * 3;
    if (req_.has_mrope) {
      auto delta_at = [&](int at, int32_t* out_delta) -> bool {
        if (at >= 0 && size_t(at) < req_.pos_rope_deltas.size()) {
          *out_delta = req_.pos_rope_deltas[size_t(at)];
          return true;
        }
        if (at == n_) {
          *out_delta = req_.rope_delta;
          return true;
        }
        return false;
      };
      int32_t end_delta = 0;
      if (!vision::inside_media_span(req_.media_spans, end) && delta_at(end, &end_delta)) {
        in.set_rope_delta = 1;
        in.rope_delta = end_delta;
      }
      const int eff_snap =
          (align <= 1 || snap_pos == FLYX_PREFILL_SNAP_NONE)
              ? -1
              : (snap_pos >= pos_ && snap_pos < end && snap_pos % align == 0)
                    ? snap_pos
                    : ((end / align * align >= pos_ && end / align * align < end) ? end / align * align : -1);
      int32_t snap_delta = 0;
      if (eff_snap >= 0 && !vision::inside_media_span(req_.media_spans, eff_snap) &&
          delta_at(eff_snap, &snap_delta)) {
        in.has_snap_rope_delta = 1;
        in.snap_rope_delta = snap_delta;
      }
    }
    if (api_->prefill_ex(e_->prefill_plan(), s_->seq(), &in, sample ? &g_ : nullptr) != 0)
      return fail(std::string("prefill_ex: ") + api_->last_error(), true);
  }
  pos_ = end;
  if (req_.on_prefill) req_.on_prefill(pos_, n_);
  if (!last) {
    checkpoint_due_ = crossed >= 0 || (cut_ < cuts_.size() && cuts_[cut_] == pos_);
    crossed_cut_ = crossed;
    due_keep_ = (crossed >= 0 ? crossed : pos_) == keep_;
    return true;
  }
  end_keep_ = crossed >= 0 && crossed == keep_;
  s_->advanced(prompt, n_, media);
  g_.mask = nullptr;  // rounds run unmasked unless a token has to be chosen again
  phase_ = Phase::kFinishPrefill;
  return true;
}

int Generation::rejected(const flyx_round_result& r) {
  if (!req_.grammar || finish_ || grammar_over_) return -1;
  for (int i = 0; i < r.n_tokens; ++i) {
    if (!req_.grammar->accept(r.tokens[i])) return i;
    // A stop token the request goes on past (ignore_eos: a measurement's fixed number of tokens) is where the reply
    // the grammar holds ends.  What follows is no reply's, so nothing holds it; the matcher, which has ended, is not
    // asked again (it takes no token after a stop token, and has no mask to give).
    const std::vector<int32_t>& eos = e_->eos();
    if (req_.ignore_eos && std::find(eos.begin(), eos.end(), r.tokens[i]) != eos.end()) {
      grammar_over_ = true;
      return -1;
    }
  }
  return -1;
}

bool Generation::deliver(const flyx_round_result& r) {
  auto is_stop = [&](int32_t t) {
    for (int i = 0; i < g_.n_stop_ids; ++i)
      if (g_.stop_ids[i] == t) return true;
    return false;
  };
  int32_t shown[FLYX_MAX_WIDE_ROWS];
  int n_shown = 0;
  for (int i = 0; i < r.n_tokens && !finish_; ++i) {
    const int32_t t = r.tokens[i];
    if (is_stop(t)) {
      finish_ = FLYX_FINISH_STOP;
      result_.stop_token = t;
      break;
    }
    shown[n_shown++] = t;
    if (++emitted_ >= max_new_) finish_ = FLYX_FINISH_LENGTH;
  }
  if (n_shown > 0) {
    reply_.push(shown, n_shown);
    // A stop string inside these tokens ends the reply, whatever else did.
    if (!reply_.stop_string().empty()) finish_ = FLYX_FINISH_STOP;
    if (reply_.refused()) s_->cancel();
  }
  if (!finish_ && cancelled()) finish_ = FLYX_FINISH_CANCEL;
  if (!finish_ && r.stopped) finish_ = FLYX_FINISH_LENGTH;  // limit or context end
  return !finish_;
}

bool Generation::checkpoint_prompt_end() {
  // A checkpoint at the end of the prompt (nothing in flight now): a retry
  // with an edited reply, or a client that rewrites the reply, starts here. The backend keeps the last multiple of
  // its alignment the prompt reaches; when the last pass crossed the kept cut, that is it.
  // A follow-up can reuse a long decoded tail. Its end-of-prefill checkpoint can now own those staged
  // KV pages and a restorable recurrent state. Short continuations keep their usual cheap checkpoint.
  const auto* media = req_.media_spans.empty() ? nullptr : &req_.media_spans;
  const bool crossed_interval = n_ / kAnchorTokens > result_.usage.reused_tokens / kAnchorTokens;
  return s_->take_checkpoint(req_.prompt, n_, end_keep_,
                             s_->disk_cache() && crossed_interval ? Session::Anchor::Prefix : Session::Anchor::None,
                             false, media);
}

bool Generation::finish_prefill() {
  if (req_.prefill_only) {
    // No ring entry was requested. Capturing the end checkpoint waits for the last prefill pass.
    if (!checkpoint_prompt_end()) return fail("could not checkpoint the prefilled prompt", true);
    result_.usage.prefill_ms = os::now_ms() - started_ms_;
    result_.usage.finish_reason = cancelled() ? FLYX_FINISH_CANCEL : FLYX_FINISH_LENGTH;
    s_->publish(req_.prompt, {}, 0, -1, req_.media_spans.empty() ? nullptr : &req_.media_spans,
                req_.rendered_text.empty() ? nullptr : &req_.rendered_text,
                req_.rendered_images.empty() ? nullptr : &req_.rendered_images);
    phase_ = Phase::kDone;
    return false;
  }
  flyx_round_result r{};
  if (api_->read(s_->seq(), &r) != 0) return fail(std::string("read: ") + api_->last_error(), true);
  s_->record(r, -2);
  if (rejected(r) >= 0) return fail("the first token does not fit the reply's grammar", true);
  result_.usage.prefill_ms = os::now_ms() - started_ms_;
  // The first token goes out before the checkpoint: nothing is in flight until the rounds below, so the checkpoint
  // takes the same state after it, and its copies (and a slot's first use: pinning its host memory, ~30 ms on the
  // 27B) stay off the time to the first token.
  const bool more = deliver(r);
  checkpoint_prompt_end();
  decoding_ms_ = os::now_ms();

  // Rounds.  Which one to launch, and how many may be in flight, is the
  // session's RoundChooser's to say (core/speculation/chooser.h).
  RoundChooser& rounds = s_->rounds();
  if (!rounds.configured())
    rounds.configure(&e_->widths(), &e_->round_drafters(), &e_->round_ms(), &e_->lookup_widths(), &e_->round_trunks(),
                     &e_->lookup_ms(), &e_->adaptive_checkpoints(), &e_->round_ms_per_token(), e_->round_ms_at(),
                     e_->round_tokens());
  // The reference for grammar tests masks every round's first token, one round at a time.
  every_ = req_.grammar && req_.grammar_reference;
  const std::vector<size_t>& cycle = e_->options().cycle;
  if (!cycle.empty()) rounds.cycle(cycle);  // a measurement's rounds, from the reply's first on
  // Wide lookup rounds only without penalties, relaxed acceptance or a grammar: their head chooses plain rows.
  const bool wide = !req_.grammar && g_.presence_penalty == 0.f && g_.frequency_penalty == 0.f &&
                    g_.accept_min_p <= 0.f && g_.accept_typical <= 0.f;
  const bool adaptive = e_->options().adaptive && !every_ && g_.accept_min_p <= 0.f && g_.accept_typical <= 0.f;
  rounds.begin(e_->options().lookup && !every_ && cycle.empty(), wide, adaptive);
  in_flight_ = 0;
  launching_ = more;
  if (s_->disk_cache()) next_disk_stage_ = (int64_t(n_) / kAnchorTokens + 1) * kAnchorTokens;
  phase_ = Phase::kRounds;
  return true;
}

void Generation::prepare_round(const RoundChoice& round, bool masked, flyx_gen_params* params, flyx_plan** plan) {
  *params = g_;
  // No mask once the reply the grammar holds has ended (rejected()): its matcher has none to give.
  masked = masked && !grammar_over_;
  if (masked) {
    req_.grammar->allowed(mask_.data());
    params->mask = mask_.data();
  }
  if (round.lookup) {
    params->drafts = s_->rounds().drafts().data();
    params->n_drafts = round.width;
    params->drafts_at = static_cast<int32_t>(s_->tokens().size());
    *plan = e_->lookup_plan(round.index);
    return;
  }
  if (round.adaptive && !masked) {
    if (flyx_plan* ap = e_->adaptive_plan(round.drafter)) {
      if (const flyx_stop_policy* sp = s_->rounds().stop_policy(round.drafter)) {
        params->stop_policy = sp;
        *plan = ap;
        return;
      }
    }
  }
  *plan = e_->round_plan(round.index);
}

bool Generation::launch(const RoundChoice& round, bool masked) {
  flyx_gen_params params{};
  flyx_plan* plan = nullptr;
  prepare_round(round, masked, &params, &plan);
  if (api_->launch(plan, s_->seq(), &params) != 0) {
    fail(std::string("launch: ") + api_->last_error(), true);
    return false;
  }
  launched_[in_flight_++] = round;
  return true;
}

bool Generation::round_ready() const {
  return in_rounds() && !stage_due_ && in_flight_ < (every_ ? 1 : s_->rounds().in_flight_limit());
}

bool Generation::next_round(NextRound* out) {
  if (!round_ready()) return false;
  const RoundChoice round = s_->rounds().next(s_->tokens(), in_flight_);
  prepare_round(round, every_, &out->g, &out->plan);
  out->round = round;
  out->rows = round.width + 1;
  return true;
}

bool Generation::round_candidates(std::vector<RoundCandidate>* out) {
  if (!round_ready()) return false;
  s_->rounds().candidates(s_->tokens(), in_flight_, out);
  if (g_.accept_min_p > 0.f || g_.accept_typical > 0.f)
    for (auto& c : *out) c.batchable = false;  // relaxed acceptance runs alone (launch_batch)
  return !out->empty();
}

void Generation::commit_round(const RoundChoice& choice, NextRound* out) {
  s_->rounds().commit(choice, in_flight_);
  prepare_round(choice, every_, &out->g, &out->plan);
  out->round = choice;
  out->rows = choice.width + 1;
}

bool Generation::round(bool wait, bool* busy) {
  RoundChooser& rounds = s_->rounds();
  if (stage_due_ && !in_flight_) {
    // All the device's rows have now been read and checked against the grammar. Only published tokens
    // enter the staged prefix, excluding output held for a stop string and rounds after the reply ended.
    const int pos = s_->state_pos();
    if (launching_ && !cancelled() && pos >= next_disk_stage_ && size_t(pos) <= size_t(n_) + reply_.published())
      s_->take_checkpoint(s_->tokens(), pos, false, Session::Anchor::Staging, false,
                          req_.media_spans.empty() ? nullptr : &req_.media_spans);
    next_disk_stage_ = s_->disk_cache() ? (int64_t(pos) / kAnchorTokens + 1) * kAnchorTokens : 0;
    stage_due_ = false;
  }
  // Rounds in flight (two, FLYX_MAX_IN_FLIGHT): the device runs round k + 1 while the host handles round k.
  bool launched = false;
  while (!batched_ && round_ready()) {
    if (!launch(rounds.next(s_->tokens(), in_flight_), every_)) return false;
    launched = true;
  }
  if (in_flight_ == 0 && batched_ && launching_) {
    *busy = true;  // waiting for the scheduler to launch its next batched round
    return true;
  }
  if (in_flight_ == 0) return finish_up();
  if (!wait && !device_ready()) {  // the oldest round is still running
    *busy = !launched;
    return true;
  }

  flyx_round_result r{};
  if (api_->read(s_->seq(), &r) != 0) return fail(std::string("read: ") + api_->last_error(), true);
  const RoundChoice was = launched_[0];
  std::copy(launched_ + 1, launched_ + in_flight_, launched_);
  --in_flight_;
  if (r.n_tokens > 0) {
    s_->record(r, was.lookup ? -1 : static_cast<int>(was.index));
    flyx_usage& u = result_.usage;
    ++u.rounds;
    u.accepted_drafts += r.n_accepted;
    const int drafted = was.adaptive && r.n_drafts > 0 ? r.n_drafts : was.width;
    u.drafted += drafted;
    if (req_.on_round) req_.on_round(drafted, r.n_accepted);
    if (was.lookup) {
      ++result_.lookup_rounds;
      result_.lookup_accepted += r.n_accepted;
    } else {
      if (was.drafter >= 0) {
        if (result_.drafter_rounds.size() <= size_t(was.drafter)) result_.drafter_rounds.resize(size_t(was.drafter) + 1);
        ++result_.drafter_rounds[size_t(was.drafter)];
      }
      size_t choice_idx = was.index;
      if (was.adaptive && r.n_drafts > 0) {
        const auto& ws = e_->widths();
        const auto& ds = e_->round_drafters();
        const auto& tr = e_->round_trees();
        for (size_t i = 0; i < ws.size(); ++i)
          if (ds[i] == was.drafter && ws[i] == r.n_drafts && !tr[i]) {
            choice_idx = i;
            break;
          }
      }
      if (result_.round_choices.size() <= choice_idx) {
        result_.round_choices.resize(choice_idx + 1);
        result_.round_accepted.resize(choice_idx + 1);
      }
      ++result_.round_choices[choice_idx];
      result_.round_accepted[choice_idx] += r.n_accepted;
    }
  }
  // A tree's trunk (its first drafts, the drafter's chain) accepted the accepted tokens it starts with.
  int trunk_accepted = 0;
  if (!was.lookup && e_->round_trunks()[was.index] > 0)
    while (trunk_accepted < r.n_accepted && trunk_accepted < e_->round_trunks()[was.index] &&
           r.drafts[trunk_accepted] == r.tokens[trunk_accepted])
      ++trunk_accepted;
  rounds.result(was, r.n_tokens > 0, r.n_accepted, s_->tokens(), in_flight_, !finish_, trunk_accepted,
                was.adaptive ? r.n_drafts : -1, r.draft_prob, r.draft_margin);
  if (finish_) return true;  // the reply has ended: the rounds still in flight are read, nothing more is shown
  const int bad = rejected(r);
  if (bad >= 0) return choose_again(r, bad);
  if (!deliver(r) || r.stopped) launching_ = false;
  if (launching_ && next_disk_stage_ && s_->state_pos() >= next_disk_stage_) stage_due_ = true;
  return true;
}

bool Generation::choose_again(const flyx_round_result& r, int bad) {
  // Token `bad` was chosen by the row that read the token before it, at position `at`.
  const int at = r.position - r.n_tokens + bad;
  flyx_round_result head = r;
  head.n_tokens = bad;
  head.stopped = 0;
  const bool more = deliver(head);
  for (int k = 0; k < in_flight_; ++k) {  // the rounds behind it ran on what is now undone
    flyx_round_result behind{};
    if (api_->read(s_->seq(), &behind) != 0) return fail(std::string("read: ") + api_->last_error(), true);
    s_->record(behind, launched_[k].lookup ? -1 : static_cast<int>(launched_[k].index));
  }
  in_flight_ = 0;
  if (api_->rewind(s_->seq(), at, s_->tokens()[size_t(at)]) != 1)
    return fail(std::string("grammar rollback: ") + (api_->last_error() ? api_->last_error() : "outside the window"),
                true);
  s_->rolled_back(at);
  // The reply is shorter now: what the penalties count is its tokens up to the one at `at`.
  if (penalized() && api_->set_reply(s_->seq(), s_->tokens().data() + n_, at + 1 - n_) != 0)
    return fail(std::string("set_reply: ") + api_->last_error(), true);
  ++result_.grammar_fixes;
  if (!more || !flyx_room(at, e_->options().max_ctx)) return finish_up();
  if (!launch(s_->rounds().next_drafted(static_cast<int>(s_->tokens().size())), true)) return false;
  launching_ = true;
  return true;
}

bool Generation::finish_up() {
  if (!finish_) finish_ = FLYX_FINISH_LENGTH;
  if (finish_ != FLYX_FINISH_CANCEL) {
    reply_.end();  // the text still held back; it can complete a stop string
    if (!reply_.stop_string().empty()) finish_ = FLYX_FINISH_STOP;
  }
  result_.stop_string = reply_.stop_string();
  flyx_usage& u = result_.usage;
  u.decode_ms = os::now_ms() - decoding_ms_;
  u.new_tokens = emitted_;
  u.finish_reason = finish_;
  s_->publish(req_.prompt, reply_.tokens(), reply_.published(), result_.stop_token,
              req_.media_spans.empty() ? nullptr : &req_.media_spans,
              req_.rendered_text.empty() ? nullptr : &req_.rendered_text,
              req_.rendered_images.empty() ? nullptr : &req_.rendered_images);
  phase_ = Phase::kDone;
  return false;
}

int generate(Session& s, const GenerateRequest& req, const TextSink& sink, GenerateResult* result, std::string* err) {
  Generation g(&s, req, sink);
  while (g.advance(true) != Generation::Progress::kDone) {
  }
  if (result) *result = g.result();
  if (g.failed()) {
    *err = g.error();
    return -1;
  }
  return 0;
}

}  // namespace flyx
