// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/session/session.h"

#include <algorithm>

namespace flyx {

Session::~Session() {
  if (seq_) {
    e_->api()->seq_free(seq_);
    e_->release_host_kv();
  }
}

bool Session::init(std::string* err) {
  static std::atomic<int> next_id{1};
  auto refused = [&](std::string why) {
    init_error_ = why;
    *err = std::move(why);
    return false;
  };
  std::string why;
  lease_ = e_->ledger().reserve("session", std::to_string(next_id++), e_->session_bytes(), &why);
  if (!lease_) return refused("not enough GPU memory for another session: " + why);
  if (!e_->reserve_host_kv()) {
    lease_.release();
    return refused("not enough host memory for another session's K/V (--kv-ram)");
  }
  if (e_->api()->seq_new_in(e_->bmodel(), &e_->kv_layout(), &seq_) != 0) {  // where its K/V go
    e_->release_host_kv();
    lease_.release();
    return refused(std::string("seq_new: ") + e_->api()->last_error());
  }
  init_error_.clear();
  return true;
}

size_t Session::common_prefix(const std::vector<int32_t>& prompt) const {
  size_t L = 0;
  while (L < prompt.size() && L < tokens_.size() && tokens_[L] == prompt[L]) ++L;
  return L;
}

std::vector<int32_t> Session::tokenize(const std::string& text, const std::vector<TextSpan>* literal,
                                       std::string* how) const {
  const auto& tok = e_->tokenizer();
  // A history whose text this one continues keeps its tokens: this session's, else that of a conversation whose
  // checkpoints are in the store (so they still match).
  const std::vector<int32_t>* kept = nullptr;
  size_t h = 0;
  if (!history_text_.empty() && text.size() >= history_text_.size() &&
      text.compare(0, history_text_.size(), history_text_) == 0) {
    kept = &history_;
    h = history_text_.size();
    if (how) *how = "history kept its tokens";
  } else if (const SnapshotStore::History* st = store_ ? store_->continued_by(text) : nullptr) {
    kept = &st->tokens;
    h = st->text.size();
    if (how) *how = "a stored conversation's history kept its tokens";
  }
  if (kept) {
    std::vector<int32_t> ids = *kept;
    std::vector<TextSpan> tail_literal;  // the spans past the history, shifted to the tail
    if (literal)
      for (const auto& s : *literal)
        if (s.end > h) tail_literal.push_back({s.begin > h ? s.begin - h : 0, s.end - h});
    auto tail = tok.encode(std::string_view(text).substr(h), literal ? &tail_literal : nullptr);
    ids.insert(ids.end(), tail.begin(), tail.end());
    return ids;
  }
  if (how) *how = history_text_.empty() ? "" : "history text changed: re-encoded";
  return tok.encode(text, literal);
}

void Session::reset_after_error() {
  // The state is unknown after a failure: reset to 0 now, or retry on the next prepare() if rewind fails.
  needs_reset_ = e_->api()->rewind(seq_, 0, 0) != 1;
  state_pos_ = 0;
  tokens_.clear();
  rounds_.forget();
  history_.clear();
  history_text_.clear();
  cut_checkpoints(0);
}

void Session::cut_checkpoints(int pos) {
  for (auto& ck : ck_)
    if (ck.valid && ck.prefix.size() > static_cast<size_t>(pos)) ck = {};
  if (disk_tip_) disk_tip_ = store_->ancestor(disk_tip_, static_cast<size_t>(pos));
}

void Session::advanced(const std::vector<int32_t>& prompt, int pos) {
  tokens_.assign(prompt.begin(), prompt.begin() + pos);
  state_pos_ = pos;
}

void Session::record(const flyx_round_result& r, int round) {
  if (trace) trace->push_back(r);
  if (trace && trace_rounds) trace_rounds->push_back(round);
  tokens_.insert(tokens_.end(), r.tokens, r.tokens + r.n_tokens);
  state_pos_ = r.position;
}

void Session::rolled_back(int pos) {
  tokens_.resize(size_t(pos) + 1);
  rounds_.cut(tokens_.size());
  state_pos_ = pos;
  cut_checkpoints(pos);
}

void Session::publish(const std::vector<int32_t>& prompt, const std::vector<int32_t>& reply, size_t published,
                      int32_t stop_token) {
  history_ = prompt;
  history_.insert(history_.end(), reply.begin(), reply.begin() + static_cast<std::ptrdiff_t>(published));
  if (stop_token >= 0 && published == reply.size()) history_.push_back(stop_token);
  history_text_ = e_->tokenizer().decode(history_);
}

bool Session::take_checkpoint(const std::vector<int32_t>& prompt, int pos, bool keep, Anchor anchor, bool share) {
  // An empty slot, else the least recently used one; a kept checkpoint stays.
  int slot = -1;
  for (int c = 0; c < FLYX_CHECKPOINTS; ++c) {
    if (ck_[c].keep || ck_[c].system) continue;
    if (!ck_[c].valid) {
      slot = c;
      break;
    }
    if (slot < 0 || ck_[c].stamp < ck_[slot].stamp) slot = c;
  }
  if (slot < 0) return false;
  // The backend may keep an earlier position (a chunk boundary a fresh prefill reproduces exactly).
  const int at = e_->api()->checkpoint(seq_, slot);
  if (at < 0 || at > pos) {
    ck_[slot] = {};
    return false;
  }
  if (keep)
    for (auto& c : ck_) c.keep = false;
  if (anchor == Anchor::System)
    for (auto& c : ck_) c.system = false;
  ck_[slot] = {true, {prompt.begin(), prompt.begin() + at}, ++ck_clock_, keep, anchor == Anchor::System};
  if (anchor != Anchor::None && disk_cache() && at > 0) {
    std::string why;
    store_->put(e_->api(), seq_, slot, ck_[slot].prefix, &why,
                {.disk_tip = &disk_tip_, .staging_only = anchor == Anchor::Staging});
  } else if (share) {
    share_checkpoint(slot, true);
  }
  if (anchor == Anchor::Staging) ck_[slot] = {};
  return true;
}

namespace {
// A checkpoint of fewer tokens is not kept in host memory: about what the 27B prefills in 90 ms on an RTX 5090,
// while its snapshot holds the gated delta nets' whole state (0.15 GB) whatever the tokens.
constexpr size_t kSnapshotWorth = 256;
}  // namespace

void Session::share_system_checkpoint() {
  if (in_use_) return;
  for (int slot = 0; slot < FLYX_CHECKPOINTS; ++slot)
    if (ck_[slot].valid && ck_[slot].system) {
      share_checkpoint(slot, false);
      return;
    }
}

void Session::share_checkpoint(int slot, bool spare_only) {
  if (!store_ || !store_->budget() || disk_cache()) return;  // disk anchors were saved during prefill
  const auto& prefix = ck_[slot].prefix;
  const size_t n = prefix.size();
  if (n < kSnapshotWorth || store_->find(prefix, n, n - 1)) return;
  uint64_t bytes = 0;
  if (spare_only && (e_->api()->snapshot_bytes(seq_, slot, 0, &bytes) != 0 ||
                     !store_->fits_without_eviction(bytes))) return;
  std::string why;
  store_->put(e_->api(), seq_, slot, prefix, &why);
}

int Session::save_to(SnapshotStore& store, std::string* why, const SnapshotStore::Entry* restore) {
  // The checkpoints of the tokens this session holds now: the one kept through tool turns, and the newest.
  auto current = [&](int c) {
    const Ckpt& k = ck_[c];
    return k.valid && k.prefix.size() >= kSnapshotWorth && k.prefix.size() <= tokens_.size() &&
           std::equal(k.prefix.begin(), k.prefix.end(), tokens_.begin());
  };
  int kept = -1, newest = -1;
  for (int c = 0; c < FLYX_CHECKPOINTS; ++c) {
    if (!current(c)) continue;
    if (ck_[c].keep) kept = c;
    if (newest < 0 || ck_[c].prefix.size() > ck_[newest].prefix.size()) newest = c;
  }
  int longest = 0;
  uint64_t current_tip = store_ == &store ? disk_tip_ : 0;
  why->clear();
  if (newest < 0) return 0;
  // Both checkpoints describe the same conversation when this session gives up its slot.
  auto history = history_text_.empty() ? nullptr : std::make_shared<const SnapshotStore::History>(history_, history_text_);
  // Disk ranges build from the earlier checkpoint. In RAM the last insertion survives eviction,
  // so give the protected question priority when the budget can hold only one of the two.
  const bool disk = store.disk_enabled();
  for (int c : {disk ? kept : newest, disk ? newest : kept}) {
    if (c < 0 || (c == kept && kept == newest && longest > 0)) continue;
    std::string w;
    bool on_disk = false;
    uint64_t tip = current_tip ? store.ancestor(current_tip, ck_[c].prefix.size()) : 0;
    const bool ok = store.put(e_->api(), seq_, c, ck_[c].prefix, &w,
                              {.history = history, .on_disk = &on_disk, .disk_tip = &tip, .restore = restore});
    if (ok && (c == newest || !current_tip)) current_tip = tip;
    if (ok) longest = std::max(longest, static_cast<int>(ck_[c].prefix.size()));
    *why += (why->empty() ? "" : "; ") + (ok ? "kept " + std::to_string(ck_[c].prefix.size()) + " tokens " +
                                                   (on_disk ? "on disk" : "in host memory")
                                             : "not kept: " + w);
  }
  if (store_ == &store) disk_tip_ = current_tip;
  return longest;
}

// A matching snapshot's restored position, or no value if none was tried. An incomplete read can already have
// changed the sequence, so rewind it to zero before trying a shorter snapshot or starting a fresh prefill.
std::optional<int> Session::load_snapshot(const std::vector<int32_t>& prompt, int limit, int floor, std::string* how,
                                         std::string* err) {
  if (!store_ || limit <= floor) return std::nullopt;
  const SnapshotStore::Entry* e = store_->find(prompt, size_t(limit), size_t(floor));
  if (!e) return std::nullopt;
  std::string first_failure;
  int pos = 0;
  bool from_disk = false;
  while (e) {
    size_t failed_end = e->prefix.size();
    pos = static_cast<int>(e->prefix.size());
    from_disk = e->disk_id || !e->file.empty();
    std::string why;
    if (store_->load(e_->api(), seq_, *e, prompt[size_t(pos)], &why, &failed_end) == pos) break;
    if (why.empty()) why = "the restored position did not match its checkpoint";
    if (first_failure.empty()) first_failure = why;
    state_pos_ = 0;
    tokens_.clear();
    rounds_.forget();
    cut_checkpoints(0);
    if (e_->api()->rewind(seq_, 0, prompt.front()) != 1) {
      needs_reset_ = true;
      *err = "rewinding after an incomplete snapshot: " + why;
      return -1;
    }
    needs_reset_ = false;
    e = failed_end ? store_->find(prompt, failed_end - 1, 0) : nullptr;
  }
  if (!e) {
    *how = "recomputed from the start after an incomplete snapshot: " + first_failure;
    return 0;
  }
  // Positional storage is now the restored sequence's, even where a discarded sequence happened to have
  // identical tokens. Its old recurrent checkpoints must not be paired with these new bytes.
  for (auto& ck : ck_) ck = {};
  disk_tip_ = e->disk_id;
  state_pos_ = pos;
  tokens_.assign(prompt.begin(), prompt.begin() + pos);
  rounds_.forget();
  *how = "loaded a snapshot from " + std::string(from_disk ? "disk" : "host memory") + " at token " +
         std::to_string(pos);
  if (!first_failure.empty()) *how += " after an incomplete snapshot: " + first_failure;
  return pos;
}

std::pair<int, int> Session::checkpoint_for(const std::vector<int32_t>& prompt, int limit) const {
  int best = -1, best_pos = 0;
  for (int c = 0; c < FLYX_CHECKPOINTS; ++c) {
    const auto& ck = ck_[c];
    const int pos = static_cast<int>(ck.prefix.size());
    if (!ck.valid || pos > limit || pos <= best_pos) continue;
    if (std::equal(ck.prefix.begin(), ck.prefix.end(), prompt.begin())) {
      best = c;
      best_pos = pos;
    }
  }
  return {best, best_pos};
}

void Session::park_for(const std::vector<int32_t>& prompt, std::string* why) {
  if (!store_ || prompt.empty()) return;
  const int shared = static_cast<int>(std::min(prompt.size() - 1, common_prefix(prompt)));
  // A deep edit may be outside the backend's rewind window. Until prepare tries it, only the matching
  // local checkpoint is guaranteed to survive; parking must leave a better stored one available.
  const int floor = state_pos_ <= shared ? state_pos_ : checkpoint_for(prompt, shared).second;
  const auto* restore = store_->find(prompt, prompt.size() - 1, floor);
  save_to(*store_, why, restore);
}

// Bring the device state to the longest usable prefix of `prompt`.  Returns
// the number of prompt tokens already in the state, or -1.
int Session::prepare(const std::vector<int32_t>& prompt, std::string* how, std::string* err) {
  const auto* api = e_->api();
  if (needs_reset_) {
    if (api->rewind(seq_, 0, prompt.front()) != 1) {
      *err = std::string("rewind: ") + (api->last_error() ? api->last_error() : "failed to reset sequence");
      return -1;
    }
    needs_reset_ = false;
  }
  const int n = static_cast<int>(prompt.size());
  const int common = static_cast<int>(common_prefix(prompt));
  const int L = std::min(common, n - 1);  // the last prompt token must run to produce the first output
  if (state_pos_ <= L) {
    // A stored snapshot may hold more of the prompt than the live state (any of it but the last token).
    if (auto got = load_snapshot(prompt, n - 1, state_pos_, how, err)) return *got;
    *how = state_pos_ == 0 ? (tokens_.empty() ? "cold start" : "nothing reusable") : "continued the live state";
    tokens_.resize(state_pos_);
    rounds_.cut(tokens_.size());
    return state_pos_;
  }
  const std::string at = " (prompt differs from the session at token " + std::to_string(common) + ")";
  int r = api->rewind(seq_, L, prompt[L]);
  if (r < 0) {
    needs_reset_ = true;
    state_pos_ = 0;
    tokens_.clear();
    rounds_.forget();
    cut_checkpoints(0);
    *err = std::string("rewind: ") + (api->last_error() ? api->last_error() : "rewind failed");
    return -1;
  }
  if (r == 1) {
    state_pos_ = L;
    cut_checkpoints(L);
    tokens_.resize(L);
    rounds_.cut(tokens_.size());
    // A stored snapshot may hold more of the prompt than the session shares with it (another conversation's).
    if (auto got = load_snapshot(prompt, n - 1, L, how, err)) {
      if (*got > 0) *how += at;
      return *got;
    }
    *how = "rolled back to token " + std::to_string(L) + at;
    return L;
  }
  // Outside the rollback window: newest usable checkpoint, else from scratch.
  const auto [best, best_pos] = checkpoint_for(prompt, L);
  state_pos_ = 0;
  // A stored snapshot may hold more of the prompt than any checkpoint here.
  if (auto got = load_snapshot(prompt, n - 1, best_pos, how, err)) {
    if (*got > 0) *how += at;
    return *got;
  }
  if (best >= 0 && api->restore(seq_, best, prompt[best_pos]) == best_pos) {
    state_pos_ = best_pos;
    ck_[best].stamp = ++ck_clock_;
    *how = "restored the checkpoint at token " + std::to_string(best_pos) + at;
  } else {
    *how = "recomputed from the start" + at + ", no checkpoint at or before it";
  }
  tokens_.assign(prompt.begin(), prompt.begin() + state_pos_);
  cut_checkpoints(state_pos_);
  rounds_.cut(tokens_.size());
  return state_pos_;
}

}  // namespace flyx
