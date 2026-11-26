// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Session: one conversation's state.  The device state and the tokens behind
// it, how a new prompt reuses it (prepare: continue, roll back, restore a
// checkpoint), its checkpoints, what the client saw of the last reply, and
// what the session has learned about its drafts.
//
// A session holds state and the rules for reusing it; it does not run
// anything.  A Generation (core/exec/generation.h) moves the state forward,
// through the calls under "the state, as a generation moves it".
#pragma once

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/exec/engine.h"
#include "core/resources/ledger.h"
#include "core/session/store.h"
#include "core/speculation/chooser.h"
#include "flyx/backend_abi.h"

namespace flyx {

class Session {
 public:
  explicit Session(Engine* e) : e_(e) {}
  ~Session();
  bool init(std::string* err);
  Engine* engine() const { return e_; }
  // Null where init failed (the engine had no room for the session, or seq_new failed) or was not called; a
  // generation on such a session fails with init_error() (empty: init was not called).
  flyx_seq* seq() { return seq_; }
  const std::string& init_error() const { return init_error_; }

  // ---- what a new prompt can keep of this session ----------------------------------------
  // Tokens for `text` (a whole rendered conversation).  Where the text
  // continues this session's published history (else that of a conversation
  // whose checkpoints are in the store), the history keeps its tokens and only
  // the rest is encoded: re-encoding a reply can split it into other tokens
  // than the ones generated, which would end the reusable prefix early.  Only
  // where `literal` reads the history's prompt as it was read (PromptEncoding);
  // else the whole text is encoded, as for a session without a history.
  // literal: where special tokens are read (RenderedChat::literal); null: everywhere.
  std::vector<int32_t> tokenize(const std::string& text, const std::vector<TextSpan>* literal = nullptr,
                                std::string* how = nullptr) const;
  // Common prefix of `prompt` (with optional multimodal spans) and the tokens this session has seen.
  size_t common_prefix(const std::vector<int32_t>& prompt,
                       const std::vector<interfaces::MediaSpan>* media = nullptr) const;
  // The last request's prompt and what the client saw of its reply.
  const std::vector<int32_t>& history() const { return history_; }
  const std::string& history_text() const { return history_text_; }
  const std::vector<interfaces::MediaSpan>& history_media() const { return history_media_; }
  const std::vector<interfaces::RenderedImage>& history_images() const { return history_images_; }
  const std::vector<interfaces::MediaSpan>& media() const { return media_; }
  size_t resident_tokens() const { return tokens_.size(); }

  // ---- the state, as a generation moves it ------------------------------------------------
  // Brings the device state to the longest usable prefix of `prompt`: the
  // live state, a rollback inside the last rounds, the newest checkpoint at
  // or before the common prefix, a snapshot in the store (set_store) that
  // holds more of it than those, else nothing.  Returns the number of prompt
  // tokens in the state, or -1 with *err; *how says which it was.
  int prepare(const std::vector<int32_t>& prompt, std::string* how, std::string* err,
              const std::vector<interfaces::MediaSpan>* media = nullptr);
  // Save a displaced conversation while preserving the incoming prompt's best stored checkpoint.
  void park_for(const std::vector<int32_t>& prompt, std::string* why,
                const std::vector<interfaces::MediaSpan>* media = nullptr);
  // Snapshots in host memory this session loads from (null: none).
  void set_store(SnapshotStore* store) {
    if (store_ != store) disk_tip_ = 0;
    store_ = store;
  }
  bool disk_cache() const { return store_ && store_->disk_enabled(); }
  // Before the session goes to another conversation: its checkpoints worth
  // keeping, into `store` (the one kept through tool turns, and the newest),
  // those of its current tokens only.  Returns how many tokens the longest
  // holds (0: none); *why says what was kept or why not.
  int save_to(SnapshotStore& store, std::string* why, const SnapshotStore::Entry* restore = nullptr);
  // A new conversation shares these instructions. Offer its canonical system checkpoint
  // to the other slot through the configured cache, leaving this conversation resident.
  void share_system_checkpoint();
  // Prefill brought the state to prompt[0, pos).
  void advanced(const std::vector<int32_t>& prompt, int pos,
                const std::vector<interfaces::MediaSpan>* media = nullptr);
  // A ring entry: its tokens follow the sequence, the state holds r.position tokens.  round: the round it was
  // (Engine::round_name's index; -1 a lookup round, -2 a prefill's first token, -3 not known), for trace_rounds.
  void record(const flyx_round_result& r, int round = -3);
  // The device state was rewound to `pos` tokens; token `pos` is its next input.
  void rolled_back(int pos);
  // A checkpoint of the state at prompt[0, pos) (or at the last position before
  // it that a fresh prefill reproduces exactly).  keep: it stays until another is kept.
  // Staging captures only the committed KV growth from generated tokens. A later canonical prefill checkpoint
  // may reference those pages on this execution lineage; its decode-time recurrent state is never a restore point.
  enum class Anchor { None, Prefix, System, Staging };
  // A marked system prefix can also be shared through spare RAM while this request is active.
  // Returns true when the backend captured the checkpoint.
  bool take_checkpoint(const std::vector<int32_t>& prompt, int pos, bool keep, Anchor anchor = Anchor::None,
                       bool share = false, const std::vector<interfaces::MediaSpan>* media = nullptr);
  // The state is unknown after a device error: start over next time.
  void reset_after_error();
  // The reply ended.  What the client saw of it (the first `published` of
  // `reply`, and the stop token when all of it was shown) becomes, after the
  // prompt, the history the next request is compared with.  encoding: how the
  // prompt's text was read (null: not known, as for a prompt given as tokens).
  void publish(const std::vector<int32_t>& prompt, const std::vector<int32_t>& reply, size_t published,
               int32_t stop_token, const std::vector<interfaces::MediaSpan>* media = nullptr,
               const std::string* rendered_text = nullptr,
               const std::vector<interfaces::RenderedImage>* rendered_images = nullptr,
               const PromptEncoding* encoding = nullptr);
  // Every token the state has seen or emitted; the last one is its next input.
  const std::vector<int32_t>& tokens() const { return tokens_; }
  int state_pos() const { return state_pos_; }
  RoundChooser& rounds() { return rounds_; }

  // A generation is running on this session: it is not handed to another
  // request (PrefixCache) until that one ends.
  bool in_use() const { return in_use_; }
  void set_in_use(bool on) { in_use_ = on; }

  // A request to stop the generation running on this session.  Any thread.
  void cancel() { cancel_.store(true); }
  bool cancel_requested() const { return cancel_.load(); }
  void clear_cancel() { cancel_.store(false); }

  // Testing hook: every ring entry, when set; and with trace_rounds, the round each was (record()'s `round`).
  std::vector<flyx_round_result>* trace = nullptr;
  std::vector<int>* trace_rounds = nullptr;

 private:
  void share_checkpoint(int slot, bool spare_only);
  // Slot and position of the longest local checkpoint the prompt can resume; {-1, 0} when none matches.
  std::pair<int, int> checkpoint_for(const std::vector<int32_t>& prompt, int limit,
                                     const std::vector<interfaces::MediaSpan>& media) const;
  std::optional<int> load_snapshot(const std::vector<int32_t>& prompt, int limit, int floor,
                                   const std::vector<interfaces::MediaSpan>& media, std::string* how,
                                   std::string* err);
  void cut_checkpoints(int pos);

  Engine* e_;
  flyx_seq* seq_ = nullptr;
  std::string init_error_;        // why init failed
  std::vector<int32_t> tokens_;   // every token this session's state has seen or emitted
  std::vector<interfaces::MediaSpan> media_;  // media spans within tokens_
  int state_pos_ = 0;             // tokens processed into the device state
  std::vector<int32_t> history_;  // prompt + published output of the last generation
  std::string history_text_;      // its text (unexpanded template text + reply for multimodal)
  PromptEncoding history_encoding_;  // how its prompt was read
  std::vector<interfaces::MediaSpan> history_media_;
  std::vector<interfaces::RenderedImage> history_images_;
  std::atomic<bool> cancel_{false};
  bool in_use_ = false;
  bool needs_reset_ = false;  // device rewind(0) failed after an error; retry before next prepare
  RoundChooser rounds_;  // which round next; what the session has learned about its drafts
  Lease lease_;  // device memory of this session's state
  SnapshotStore* store_ = nullptr;
  uint64_t disk_tip_ = 0;       // latest capture on this sequence's still-unchanged execution path
  struct Ckpt {
    bool valid = false;
    std::vector<int32_t> prefix;  // tokens [0, pos)
    std::vector<interfaces::MediaSpan> media;  // media spans within [0, pos)
    uint64_t stamp = 0;
    bool keep = false;            // not replaced until another is kept (GenerateRequest::keep_checkpoint)
    bool system = false;          // restore point shared by different branches near the start
  } ck_[FLYX_CHECKPOINTS];
  uint64_t ck_clock_ = 0;
};

}  // namespace flyx
