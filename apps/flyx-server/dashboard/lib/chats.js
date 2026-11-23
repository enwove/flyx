// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The Chat page's conversations, kept in this browser (localStorage) and nowhere else: the newest first, at most
// MOST of them and about ROOM characters of JSON, the oldest let go first.  The page works without storage: then
// nothing is kept.
const KEY = "flyx.chats", MOST = 100, ROOM = 3000000;

const sound = (c) => c && typeof c.id === "string" && Array.isArray(c.msgs) && c.msgs.length > 0;

// [{id, at (ms), title, msgs}], newest first.
export function loadChats() {
  try {
    const list = JSON.parse(localStorage.getItem(KEY) || "[]");
    return Array.isArray(list) ? list.filter(sound) : [];
  } catch (e) {
    return [];
  }
}

// Stores `list` and gives back what of it was kept: all of it, unless there are too many or the browser has no room.
export function storeChats(list) {
  let kept = list.slice(0, MOST);
  for (;;) {
    const text = JSON.stringify(kept);
    if (text.length <= ROOM || kept.length <= 1) {
      try {
        if (kept.length) localStorage.setItem(KEY, text); else localStorage.removeItem(KEY);
        return kept;
      } catch (e) {
        if (kept.length <= 1) return kept;  // no room for even one, or no storage at all: it lives as long as the page
      }
    }
    kept = kept.slice(0, kept.length - Math.max(1, kept.length >> 2));
  }
}

// A message as it is kept: what was said and the reply's own figures, not what only a reply being written has.
export function keptMessage(m) {
  if (m.role === "user") return { role: "user", content: m.content };
  const out = { role: "assistant", content: m.content || "", thinking: m.thinking || "", model: m.model };
  for (const k of ["id", "usage", "elapsed", "clientTtft", "error"]) if (m[k] !== undefined && m[k] !== null) out[k] = m[k];
  if (m.done) out.done = true; else if (!m.error) out.stopped = true;  // one cut short (by Stop, or by the page closing) reads as stopped
  return out;
}

export const newChatId = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
