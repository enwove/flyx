// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/session/session.h"

#include <algorithm>

#include "core/vision/media_span.h"

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
  if (const int rc = e_->api()->seq_new_in(e_->bmodel(), &e_->kv_layout(), &seq_); rc != 0) {  // where its K/V go
    const std::string seq_err = e_->api()->last_error();
    e_->release_host_kv();
    lease_.release();
    if (rc == FLYX_ERR_OOM) {
      std::string replan_err;
      if (e_->replan_for_floor_increase(&replan_err)) {
        lease_ = e_->ledger().reserve("session", std::to_string(next_id++), e_->session_bytes(), &why);
        if (!lease_) return refused("not enough GPU memory for another session: " + why);
        if (!e_->reserve_host_kv()) {
          lease_.release();
          return refused("not enough host memory for another session's K/V (--kv-ram)");
        }
        if (e_->api()->seq_new_in(e_->bmodel(), &e_->kv_layout(), &seq_) == 0) {
          init_error_.clear();
          return true;
        }
        const std::string retry_err = e_->api()->last_error();
        e_->release_host_kv();
        lease_.release();
        return refused("seq_new: " + retry_err);
      }
      if (!replan_err.empty()) return refused(replan_err);
    }
    return refused("seq_new: " + seq_err);
  }
  init_error_.clear();
  return true;
}

namespace {

// `literal` (null: everywhere) within the first `bytes` of a text: where an encoding of those bytes reads special
// tokens.
std::vector<TextSpan> spans_within(const std::vector<TextSpan>* literal, size_t bytes) {
  std::vector<TextSpan> out;
  if (!literal) {
    if (bytes) out.push_back({0, bytes});
    return out;
  }
  for (const TextSpan& s : *literal)
    if (s.begin < std::min(s.end, bytes)) out.push_back({s.begin, std::min(s.end, bytes)});
  return out;
}

// Whether `tokens` (a history's) begin with what `text` read with `literal` gives for the history's prompt: that
// prompt was read under the same rules, or under rules that differ only where no special token is spelled (encoding
// it again says so; also where how it was read is not known).
bool read_alike(const Tokenizer& tok, const std::vector<int32_t>& tokens, const PromptEncoding& was,
                std::string_view text, const std::vector<TextSpan>* literal) {
  if (was.bytes > text.size() || was.tokens > tokens.size()) return false;
  const std::vector<TextSpan> now = spans_within(literal, was.bytes);
  auto same = [](const TextSpan& a, const TextSpan& b) { return a.begin == b.begin && a.end == b.end; };
  if (was.known && std::equal(was.literal.begin(), was.literal.end(), now.begin(), now.end(), same)) return true;
  const std::vector<int32_t> again = tok.encode(text.substr(0, was.bytes), &now);
  return again.size() == was.tokens && std::equal(again.begin(), again.end(), tokens.begin());
}

}  // namespace

size_t Session::common_prefix(const std::vector<int32_t>& prompt,
                              const std::vector<interfaces::MediaSpan>* media) const {
  static const std::vector<interfaces::MediaSpan> kEmptyMedia;
  return vision::media_prefix_match(tokens_, media_, prompt, media ? *media : kEmptyMedia);
}

std::vector<int32_t> Session::tokenize(const std::string& text, const std::vector<TextSpan>* literal,
                                       std::string* how) const {
  const auto& tok = e_->tokenizer();
  // A history whose text this one continues keeps its tokens: this session's, else that of a conversation whose
  // checkpoints are in the store (so they still match). Text-only histories only; multimodal requests use
  // prepare_request()'s media-aware path.  And only where this text's rules read the history's prompt alike: a chat
  // template reads no special token in what a user wrote and a raw prompt reads them everywhere, so after a chat
  // prewarm a raw prompt of the same bytes is encoded as it would be in a fresh session.
  const std::vector<int32_t>* kept = nullptr;
  size_t h = 0;
  bool other_rules = false;
  auto keep = [&](const std::vector<int32_t>& tokens, const std::string& history_text, const PromptEncoding& was,
                  const char* said) {
    if (!read_alike(tok, tokens, was, text, literal)) {
      other_rules = true;
      return;
    }
    kept = &tokens;
    h = history_text.size();
    if (how) *how = said;
  };
  if (history_media_.empty() && !history_text_.empty() && text.size() >= history_text_.size() &&
      text.compare(0, history_text_.size(), history_text_) == 0) {
    keep(history_, history_text_, history_encoding_, "history kept its tokens");
  } else if (const SnapshotStore::History* st = store_ ? store_->continued_by(text) : nullptr) {
    keep(st->tokens, st->text, st->encoding, "a stored conversation's history kept its tokens");
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
  if (how)
    *how = other_rules            ? "the history's prompt was read under other special-token rules: re-encoded"
           : history_text_.empty() ? ""
                                   : "history text changed: re-encoded";
  return tok.encode(text, literal);
}

void Session::reset_after_error() {
  // The state is unknown after a failure: reset to 0 now, or retry on the next prepare() if rewind fails.
  needs_reset_ = e_->api()->rewind(seq_, 0, 0) != 1;
  state_pos_ = 0;
  tokens_.clear();
  media_.clear();
  rounds_.forget();
  history_.clear();
  history_text_.clear();
  history_encoding_ = {};
  history_media_.clear();
  history_images_.clear();
  cut_checkpoints(0);
}

void Session::cut_checkpoints(int pos) {
  for (auto& ck : ck_)
    if (ck.valid && ck.prefix.size() > static_cast<size_t>(pos)) ck = {};
  if (disk_tip_) disk_tip_ = store_->ancestor(disk_tip_, static_cast<size_t>(pos));
}

void Session::advanced(const std::vector<int32_t>& prompt, int pos,
                       const std::vector<interfaces::MediaSpan>* media) {
  tokens_.assign(prompt.begin(), prompt.begin() + pos);
  if (media) media_ = vision::slice_media_prefix(*media, size_t(pos));
  else media_.clear();
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
  media_ = vision::slice_media_prefix(media_, size_t(pos));
  rounds_.cut(tokens_.size());
  state_pos_ = pos;
  cut_checkpoints(pos);
}

void Session::publish(const std::vector<int32_t>& prompt, const std::vector<int32_t>& reply, size_t published,
                      int32_t stop_token, const std::vector<interfaces::MediaSpan>* media,
                      const std::string* rendered_text,
                      const std::vector<interfaces::RenderedImage>* rendered_images, const PromptEncoding* encoding) {
  history_ = prompt;
  size_t prompt_bytes = 0;
  for (int32_t id : prompt) prompt_bytes += e_->tokenizer().token_bytes(id).size();
  history_encoding_ = {};
  if (encoding && encoding->known) history_encoding_ = {true, spans_within(&encoding->literal, prompt_bytes)};
  history_encoding_.bytes = prompt_bytes;
  history_encoding_.tokens = prompt.size();
  history_.insert(history_.end(), reply.begin(), reply.begin() + static_cast<std::ptrdiff_t>(published));
  if (stop_token >= 0 && published == reply.size()) history_.push_back(stop_token);
  if (media && !media->empty()) {
    history_media_ = *media;
    if (rendered_text && !rendered_text->empty()) {
      std::vector<int32_t> tail(reply.begin(), reply.begin() + static_cast<std::ptrdiff_t>(published));
      if (stop_token >= 0 && published == reply.size()) tail.push_back(stop_token);
      history_text_ = *rendered_text + e_->tokenizer().decode(tail);
    } else {
      history_text_ = e_->tokenizer().decode(history_);
    }
    if (rendered_images) history_images_ = *rendered_images;
    else history_images_.clear();
  } else {
    history_media_.clear();
    history_images_.clear();
    history_text_ = e_->tokenizer().decode(history_);
  }
}

bool Session::take_checkpoint(const std::vector<int32_t>& prompt, int pos, bool keep, Anchor anchor, bool share,
                              const std::vector<interfaces::MediaSpan>* media) {
  static const std::vector<interfaces::MediaSpan> kEmptyMedia;
  const auto& pm = media ? *media : media_;
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
  if (at < 0 || at > pos || vision::inside_media_span(pm, at)) {
    ck_[slot] = {};
    return false;
  }
  if (keep)
    for (auto& c : ck_) c.keep = false;
  if (anchor == Anchor::System)
    for (auto& c : ck_) c.system = false;
  ck_[slot] = {true, {prompt.begin(), prompt.begin() + at}, vision::slice_media_prefix(pm, size_t(at)),
               ++ck_clock_, keep, anchor == Anchor::System};
  if (anchor != Anchor::None && disk_cache() && at > 0) {
    std::string why;
    store_->put(e_->api(), seq_, slot, ck_[slot].prefix, &why,
                {.disk_tip = &disk_tip_, .staging_only = anchor == Anchor::Staging, .media = &ck_[slot].media});
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
  const auto& media = ck_[slot].media;
  const size_t n = prefix.size();
  if (n < kSnapshotWorth || store_->find(prefix, n, n - 1, &media)) return;
  uint64_t bytes = 0;
  if (spare_only && (e_->api()->snapshot_bytes(seq_, slot, 0, &bytes) != 0 ||
                     !store_->fits_without_eviction(bytes))) return;
  std::string why;
  store_->put(e_->api(), seq_, slot, prefix, &why, {.media = &media});
}

int Session::save_to(SnapshotStore& store, std::string* why, const SnapshotStore::Entry* restore) {
  // The checkpoints of the tokens this session holds now: the one kept through tool turns, and the newest.
  auto current = [&](int c) {
    const Ckpt& k = ck_[c];
    return k.valid && k.prefix.size() >= kSnapshotWorth &&
           vision::media_prefix_equal(k.prefix, k.media, tokens_, media_);
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
  auto history = history_text_.empty()
                     ? nullptr
                     : std::make_shared<const SnapshotStore::History>(
                           SnapshotStore::History{history_, history_text_, history_media_, history_images_,
                                                  history_encoding_});
  // Disk ranges build from the earlier checkpoint. In RAM the last insertion survives eviction,
  // so give the protected question priority when the budget can hold only one of the two.
  const bool disk = store.disk_enabled();
  for (int c : {disk ? kept : newest, disk ? newest : kept}) {
    if (c < 0 || (c == kept && kept == newest && longest > 0)) continue;
    std::string w;
    bool on_disk = false;
    uint64_t tip = current_tip ? store.ancestor(current_tip, ck_[c].prefix.size()) : 0;
    const bool ok = store.put(e_->api(), seq_, c, ck_[c].prefix, &w,
                              {.history = history, .on_disk = &on_disk, .disk_tip = &tip, .restore = restore,
                               .media = &ck_[c].media});
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
std::optional<int> Session::load_snapshot(const std::vector<int32_t>& prompt, int limit, int floor,
                                         const std::vector<interfaces::MediaSpan>& media, std::string* how,
                                         std::string* err) {
  if (!store_ || limit <= floor) return std::nullopt;
  const SnapshotStore::Entry* e = store_->find(prompt, size_t(limit), size_t(floor), &media);
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
    media_.clear();
    rounds_.forget();
    cut_checkpoints(0);
    if (e_->api()->rewind(seq_, 0, prompt.front()) != 1) {
      needs_reset_ = true;
      *err = "rewinding after an incomplete snapshot: " + why;
      return -1;
    }
    needs_reset_ = false;
    e = failed_end ? store_->find(prompt, failed_end - 1, 0, &media) : nullptr;
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
  media_ = e->media;
  rounds_.forget();
  *how = "loaded a snapshot from " + std::string(from_disk ? "disk" : "host memory") + " at token " +
         std::to_string(pos);
  if (!first_failure.empty()) *how += " after an incomplete snapshot: " + first_failure;
  return pos;
}

std::pair<int, int> Session::checkpoint_for(const std::vector<int32_t>& prompt, int limit,
                                            const std::vector<interfaces::MediaSpan>& media) const {
  int best = -1, best_pos = 0;
  for (int c = 0; c < FLYX_CHECKPOINTS; ++c) {
    const auto& ck = ck_[c];
    const int pos = static_cast<int>(ck.prefix.size());
    if (!ck.valid || pos > limit || pos <= best_pos) continue;
    if (vision::media_prefix_equal(ck.prefix, ck.media, prompt, media)) {
      best = c;
      best_pos = pos;
    }
  }
  return {best, best_pos};
}

void Session::park_for(const std::vector<int32_t>& prompt, std::string* why,
                       const std::vector<interfaces::MediaSpan>* media) {
  if (!store_ || prompt.empty()) return;
  static const std::vector<interfaces::MediaSpan> kEmptyMedia;
  const auto& pm = media ? *media : kEmptyMedia;
  const int shared = static_cast<int>(std::min(prompt.size() - 1, common_prefix(prompt, &pm)));
  // A deep edit may be outside the backend's rewind window. Until prepare tries it, only the matching
  // local checkpoint is guaranteed to survive; parking must leave a better stored one available.
  const int floor = state_pos_ <= shared ? state_pos_ : checkpoint_for(prompt, shared, pm).second;
  const auto* restore = store_->find(prompt, prompt.size() - 1, floor, &pm);
  save_to(*store_, why, restore);
}

// Bring the device state to the longest usable prefix of `prompt`.  Returns
// the number of prompt tokens already in the state, or -1.
int Session::prepare(const std::vector<int32_t>& prompt, std::string* how, std::string* err,
                     const std::vector<interfaces::MediaSpan>* media) {
  static const std::vector<interfaces::MediaSpan> kEmptyMedia;
  const auto& pm = media ? *media : kEmptyMedia;
  const auto* api = e_->api();
  if (needs_reset_) {
    if (api->rewind(seq_, 0, prompt.front()) != 1) {
      *err = std::string("rewind: ") + (api->last_error() ? api->last_error() : "failed to reset sequence");
      return -1;
    }
    needs_reset_ = false;
  }
  const int n = static_cast<int>(prompt.size());
  const int common = static_cast<int>(common_prefix(prompt, &pm));
  const int L = std::min(common, n - 1);  // the last prompt token must run to produce the first output
  if (state_pos_ <= L) {
    // A stored snapshot may hold more of the prompt than the live state (any of it but the last token).
    if (auto got = load_snapshot(prompt, n - 1, state_pos_, pm, how, err)) return *got;
    *how = state_pos_ == 0 ? (tokens_.empty() ? "cold start" : "nothing reusable") : "continued the live state";
    tokens_.resize(state_pos_);
    media_ = vision::slice_media_prefix(pm, size_t(state_pos_));
    rounds_.cut(tokens_.size());
    return state_pos_;
  }
  const std::string at = " (prompt differs from the session at token " + std::to_string(common) + ")";
  int r = api->rewind(seq_, L, prompt[L]);
  if (r < 0) {
    needs_reset_ = true;
    state_pos_ = 0;
    tokens_.clear();
    media_.clear();
    rounds_.forget();
    cut_checkpoints(0);
    *err = std::string("rewind: ") + (api->last_error() ? api->last_error() : "rewind failed");
    return -1;
  }
  if (r == 1) {
    state_pos_ = L;
    cut_checkpoints(L);
    tokens_.resize(L);
    media_ = vision::slice_media_prefix(pm, size_t(L));
    rounds_.cut(tokens_.size());
    // A stored snapshot may hold more of the prompt than the session shares with it (another conversation's).
    if (auto got = load_snapshot(prompt, n - 1, L, pm, how, err)) {
      if (*got > 0) *how += at;
      return *got;
    }
    *how = "rolled back to token " + std::to_string(L) + at;
    return L;
  }
  // Outside the rollback window: newest usable checkpoint, else from scratch.
  const auto [best, best_pos] = checkpoint_for(prompt, L, pm);
  state_pos_ = 0;
  // A stored snapshot may hold more of the prompt than any checkpoint here.
  if (auto got = load_snapshot(prompt, n - 1, best_pos, pm, how, err)) {
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
  media_ = vision::slice_media_prefix(pm, size_t(state_pos_));
  cut_checkpoints(state_pos_);
  rounds_.cut(tokens_.size());
  return state_pos_;
}

}  // namespace flyx
