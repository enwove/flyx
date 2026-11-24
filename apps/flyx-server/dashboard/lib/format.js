// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Figures as the page writes them: grouped digits, bytes in GB / GiB, durations from milliseconds to hours.

export const fmt = (v, d = 0) => (v === null || v === undefined || !isFinite(v)) ? "-" :
  Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
export const gib = (b) => b ? (b / 1073741824).toFixed(b >= 1e11 ? 0 : 1) : "-";
export const gb = (b) => (b / 1e9).toFixed(1);
// An amount of memory with its unit: MiB under a GiB (a process without its model often holds MBs), else GiB.
export const mem = (b) => !isFinite(b) ? "-" : b < 1073741824 ? fmt(b / 1048576) + " MiB" : gib(b) + " GiB";
export const ktok = (n) => !isFinite(n) ? "-" : n >= 10000 ? (n / 1000).toFixed(n >= 100000 ? 0 : 1) + "k" : fmt(n);
export function secs(s) {
  if (s === null || s === undefined || !isFinite(s)) return "-";
  if (s < 1) return Math.round(s * 1000) + " ms";
  if (s < 10) return s.toFixed(2) + " s";
  if (s < 100) return s.toFixed(1) + " s";
  if (s < 3600) return Math.floor(s / 60) + "m " + Math.round(s % 60) + "s";
  return Math.floor(s / 3600) + "h " + Math.round((s % 3600) / 60) + "m";
}
export const clock = (t) => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
// Why a served model's context is shorter than the one wanted (telemetry's model.context_limit): which memory cut it
// and what would give more, in the page's English, and whether the server calls it short.  null: nothing cut it.
export function contextLimit(model) {
  const l = model && model.context_limit;
  if (!l || !model.context) return null;
  const gb2 = (b) => (b / 1e9).toFixed(2);
  const head = "Limited context window " + fmt(model.context) + " / " + fmt(l.wanted) + ".";
  const cause = l.by === "host"
    ? "Not enough free RAM (" + gb2(l.left) + " GB available for KV). Free RAM or raise \"KV in host RAM\" for a longer context."
    : l.closable
      ? "Not enough VRAM left (" + gb2(l.others) + " GB used by other apps). Close other apps and reload for a longer context."
      : "Not enough VRAM left. Disable speculative decoding or use a smaller quantization for a longer context.";
  return { parts: [head, cause], text: head + " " + cause, short: !!l.short };
}
// A meter's colour: near full, then full.
export const level = (f) => f > 0.95 ? "hot" : f > 0.85 ? "warn" : "";
// "NVIDIA GeForce RTX 5070 Ti" -> "RTX 5070 Ti".
export const shortDevice = (d) => d ? d.replace(/^NVIDIA (GeForce )?/, "") : "";
export const baseName = (p) => String(p).split(/[\\/]/).pop();
// A literal argument in a POSIX shell snippet.
export const shQuote = (s) => "'" + s.replace(/'/g, "'\\''") + "'";
// PowerShell treats curly apostrophes as quotes too.
export const psQuote = (s) => "'" + s.replace(/['\u2018-\u201b]/g, (c) => "'" + c) + "'";
// A figure under 100 with one decimal, else whole: the collapsed rail's.
export const short = (v) => v === undefined || v === null || !isFinite(v) ? "-" : v >= 100 ? fmt(v) : fmt(v, 1);
// A count as a list shows it: 1.2K, 56.8K, 3.4M.
export const compact = (n) => n === null || n === undefined || !isFinite(n) ? "-" : n >= 1e9 ? (n / 1e9).toFixed(1) + "B" :
  n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(1) + "K" : String(n);
// How long ago, from an ISO time or Unix seconds: "5m ago", "6h ago", "12d ago", "3mo ago", "2y ago"; "" for none.
export function ago(when) {
  const at = typeof when === "number" ? when * 1000 : Date.parse(when);
  if (!isFinite(at)) return "";
  const s = Math.max(0, (Date.now() - at) / 1000), day = 86400;
  return s < 3600 ? Math.max(1, Math.round(s / 60)) + "m ago" : s < day ? Math.round(s / 3600) + "h ago" :
         s < 60 * day ? Math.round(s / day) + "d ago" : s < 730 * day ? Math.round(s / day / 30) + "mo ago" : Math.round(s / day / 365) + "y ago";
}
