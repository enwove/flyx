// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The server's API as the page calls it, with the API key if one was given, and the viewer's own settings.

// Per-viewer conveniences only (the page, the language, the theme, the sidebar): the page works without storage.
export const saved = {
  get(k) { try { return localStorage.getItem("flyx." + k); } catch (e) { return null; } },
  set(k, v) {
    try { if (v === null) localStorage.removeItem("flyx." + k); else localStorage.setItem("flyx." + k, v); } catch (e) { /* none */ }
  },
};

// The API key: from the address's fragment (flyx serve opens the page so), else as last given.
let key = saved.get("key") || "";
export function consumeAddressKey(previous = "") {
  const match = /(?:^#|&)key=(.*)$/.exec(location.hash);
  if (!match) return false;
  try { key = decodeURIComponent(match[1]); } catch { key = match[1]; }  // older servers appended the raw key
  saved.set("key", key || null);
  const page = location.hash.slice(0, match.index) || previous;
  window.history.replaceState(null, "", location.pathname + location.search + page);
  return true;
}
consumeAddressKey();
export const apiKey = () => key;
export function setKey(k) {
  key = k.trim();
  saved.set("key", key || null);
}
export const headers = (extra) => Object.assign(key ? { Authorization: "Bearer " + key } : {}, extra || {});

// The message of an error the server answered with, else the status.
const message = (r, j) => (j && j.error && j.error.message) || "HTTP " + r.status;

// A GET whose answer is JSON; an error answer throws with the server's message.
export async function getJson(path) {
  const r = await fetch(path, { headers: headers(), cache: "no-store" });
  const j = await r.json().catch(() => null);
  if (!r.ok) throw new Error(message(r, j));
  return j;
}

// Any other call: {ok, status, json, error}; error is the server's message, or says the server did not answer.
export async function call(method, path, body) {
  let r;
  try {
    r = await fetch(path, {
      method, headers: headers(body === undefined ? {} : { "Content-Type": "application/json" }),
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (e) {
    return { ok: false, status: 0, json: null, error: "The server does not answer." };
  }
  const j = await r.json().catch(() => null);
  return { ok: r.ok, status: r.status, json: j, error: r.ok ? null : message(r, j) };
}
