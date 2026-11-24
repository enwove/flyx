// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/session/prefix_cache.h"

#include <algorithm>

namespace flyx {
namespace {

size_t common_text(const std::string& a, const std::string& b) {
  size_t n = std::min(a.size(), b.size()), i = 0;
  while (i < n && a[i] == b[i]) ++i;
  return i;
}

bool same_image(const interfaces::RenderedImage& a, const interfaces::RenderedImage& b) {
  return a.image && b.image && a.image->sha256 == b.image->sha256 && a.image->detail == b.image->detail;
}

// Common text prefix clamped before the first differing rendered image (so identical text with different images
// never routes to the same resident session or claims a false shared history).
size_t common_rendered(const interfaces::RenderedChat& prompt, const std::string& history_text,
                       const std::vector<interfaces::RenderedImage>& history_images) {
  size_t c = common_text(prompt.text, history_text);
  if (prompt.images.empty() && history_images.empty()) return c;
  // Non-native Jinja path uses npos sentinel offsets: require ordered image prefix equality or clamp to 0.
  for (const auto& img : prompt.images)
    if (img.byte_offset == std::string::npos) {
      if (history_images.size() > prompt.images.size()) return 0;
      for (size_t i = 0; i < history_images.size(); ++i)
        if (!same_image(prompt.images[i], history_images[i])) return 0;
      return c;
    }
  for (const auto& img : history_images)
    if (img.byte_offset == std::string::npos) return 0;

  size_t i = 0;
  while (i < prompt.images.size() && i < history_images.size()) {
    const auto& pa = prompt.images[i];
    const auto& hb = history_images[i];
    if (pa.byte_offset >= c && hb.byte_offset >= c) break;
    if (pa.byte_end > c || hb.byte_end > c || pa.byte_offset != hb.byte_offset || pa.byte_end != hb.byte_end ||
        !same_image(pa, hb))
      return std::min(c, std::min(pa.byte_offset, hb.byte_offset));
    ++i;
  }
  if (i < prompt.images.size() && prompt.images[i].byte_offset < c) return prompt.images[i].byte_offset;
  if (i < history_images.size() && history_images[i].byte_offset < c) return history_images[i].byte_offset;
  return c;
}

// Shared text shorter than this is the template's opening or a few words in
// common, not a prefix that ties a request to a session: about 64 tokens,
// one prefill alignment of the CUDA backend, below which no checkpoint can
// hold it and which takes some 7 ms to compute again.
constexpr size_t kSharedWorth = 256;

}  // namespace

std::vector<PrefixCache::SlotInfo> PrefixCache::slots() const {
  std::vector<const Slot*> order;
  for (const auto& s : slots_) order.push_back(&s);
  std::sort(order.begin(), order.end(), [](const Slot* a, const Slot* b) { return a->used > b->used; });
  std::vector<SlotInfo> out;
  for (const Slot* s : order) out.push_back({s->id, static_cast<int>(s->s->resident_tokens())});
  return out;
}

Session* PrefixCache::acquire(const interfaces::RenderedChat& prompt, Choice* choice, std::string* err,
                              uint64_t resume_assignment) {
  *choice = {};
  // The session chosen, unless it is running another request.
  auto hand_out = [&](int i, std::string reason) -> Session* {
    Slot& slot = slots_[size_t(i)];
    if (slot.s->in_use()) {
      choice->busy = true;
      return nullptr;
    }
    slot.used = ++clock_;
    choice->assignment = slot.used;
    choice->id = slot.id;
    choice->reason = std::move(reason);
    return slot.s.get();
  };
  // Best match by shared text with each session's published history.
  int best = -1, sibling_slot = -1;
  size_t best_len = 0;
  for (size_t i = 0; i < slots_.size(); ++i) {
    // A replaced prewarm may have completed passes without publishing any conversation history yet.
    if (slots_[i].used == resume_assignment)
      return hand_out(static_cast<int>(i), "session " + std::to_string(slots_[i].id) + " (continues its prewarm)");
    const auto& history = slots_[i].s->history_text();
    const size_t c = common_rendered(prompt, history, slots_[i].s->history_images());
    if (best < 0 || c > best_len) {
      best = static_cast<int>(i);
      sibling_slot = best;
      best_len = c;
    } else if (c == best_len) {
      // A whole published history is a continuation even when a longer branch shares exactly that many bytes.
      if (c > 0 && c == history.size() && c < slots_[best].s->history_text().size()) best = static_cast<int>(i);
      const auto& sibling = slots_[sibling_slot];
      if (!slots_[i].s->in_use() && (sibling.s->in_use() || slots_[i].used < sibling.used))
        sibling_slot = static_cast<int>(i);
    }
  }
  // A request continues a session when it shares that session's history up to
  // the end of its first user message, or at least half of the history after the system instructions.
  // This also recognizes a follow-up when a new question made the template drop the
  // earlier turns' thinking (the text then differs from there on, and the
  // session's kept checkpoint is where it resumes).  Two conversations that
  // only share a system prompt (an agent and its sub-agent) get separate
  // sessions, so neither evicts the other's state while slots are free.
  if (best >= 0) {
    const auto& h = slots_[best].s->history_text();
    const bool same_start = prompt.first_user_end > 0 && best_len >= prompt.first_user_end;
    const size_t root = prompt.system_end;
    const bool shared_turns = best_len > root && best_len - root >= h.size() - best_len;
    if (!h.empty() && (same_start || best_len == h.size() || shared_turns))
      return hand_out(best, "session " + std::to_string(slots_[best].id) + " (continues its history)");
  }
  if (static_cast<int>(slots_.size()) < max_) {
    auto s = std::make_unique<Session>(e_);
    std::string e;
    if (s->init(&e)) {
      if (best >= 0 && prompt.system_end > 0 && best_len >= prompt.system_end)
        slots_[best].s->share_system_checkpoint();
      s->set_store(store_);
      slots_.push_back({std::move(s), ++clock_, next_id_++});
      choice->id = slots_.back().id;
      choice->assignment = slots_.back().used;
      choice->reason = "new session " + std::to_string(choice->id);
      return slots_.back().s.get();
    }
    if (slots_.empty()) {
      *err = e;
      return nullptr;
    }
    // Out of memory for another session: fall through and reuse one.
  }
  // Every slot is taken: one conversation gives up its place.  A sibling of
  // the request if there is one, so that unrelated conversations stay; else
  // the least recently used, an idle one before one that is running.
  const bool sibling = best >= 0 && best_len >= kSharedWorth;
  int victim = sibling ? sibling_slot : -1;  // among equally close siblings, the least recently used idle one
  for (bool idle_only : {true, false}) {
    if (victim >= 0) break;
    for (size_t i = 0; i < slots_.size(); ++i) {
      if (idle_only && slots_[i].s->in_use()) continue;
      if (victim < 0 || slots_[i].used < slots_[size_t(victim)].used) victim = static_cast<int>(i);
    }
  }
  std::string reason = "session " + std::to_string(slots_[size_t(victim)].id) + " reused (" +
                       (sibling ? "shares a prefix" : "least recently used") + ", all " +
                       std::to_string(slots_.size()) + " slots taken)";
  Session* s = hand_out(victim, std::move(reason));
  choice->park = s && store_ && store_->keeps() && s->resident_tokens() > 0;
  return s;
}

void PrefixCache::clear() {
  if (store_ && store_->keeps())
    for (auto& slot : slots_)
      if (!slot.s->in_use() && slot.s->resident_tokens() > 0) {
        std::string kept;
        slot.s->save_to(*store_, &kept);
      }
  slots_.clear();
}

}  // namespace flyx
