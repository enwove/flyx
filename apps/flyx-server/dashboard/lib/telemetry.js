// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// GET /v1/telemetry once a second (apps/flyx-server/telemetry.cpp), and what the page keeps of it: the last answer,
// and ten minutes of the server's per-second counts (tokens out, prompt tokens computed, rounds verified, their drafts
// and the drafts accepted).  The page draws from `live` on every change.
import { headers } from "./api.js";

export const ttftSeconds = (r) => Number.isFinite(r.ttft_ms) && r.ttft_ms >= 0 ? r.ttft_ms / 1000 : null;
// A finished request's prompt tokens computed (not reused) a second, up to its first token; null when none were.
export function prefillRate(r) {
  const computed = (r.prompt_tokens || 0) - (r.reused_tokens || 0);
  return computed > 0 && r.prefill_ms > 0 ? computed / r.prefill_ms * 1000 : null;
}
// Of a finished request's prompt, the share the session already held, in percent; null without a prompt.
export const cacheHit = (r) => r.prompt_tokens > 0 ? Math.min(1, (r.reused_tokens || 0) / r.prompt_tokens) * 100 : null;
// Of the drafts rounds verified, the share accepted, in percent; null where none were drafted.
export const acceptance = (accepted, drafted) => drafted > 0 ? Math.min(1, (accepted || 0) / drafted) * 100 : null;
// Decode and prefill throughput figures: of now, of the running request, or of the last finished request.
export function speeds(tm, phase, recent) {
  const state = tm.state || {}, rates = tm.rates || {}, last = recent[0] || {};
  const decoding = phase === "decode", prefilling = phase === "prefill", running = decoding || prefilling;
  return {
    decode: decoding ? rates.decode_tok_s : last.decode_tok_s,
    prefill: running ? state.prefill_tok_s : prefillRate(last),
    decodeWhen: decoding ? "now" : recent.length ? "last request" : "",
    prefillWhen: prefilling ? "now" : decoding ? "this request" : recent.length ? "last request" : "",
  };
}

// The GPU's and the host's memory: all in use, flyx's part of it, the rest (other programs, the system), and the
// whole; null without figures.  flyx's GPU part is its engine's ledger and, where the system counts a process's
// share (Windows), its own CUDA runtime beside it; elsewhere that runtime falls in the rest.  Its host part is the
// server's process.ram: its resident memory with the page-locked memory for the GPU that no process figure counts
// (the K/V past the GPU's part); a server before it gave the resident memory alone.
export function memoryOf(t) {
  const d = t.device, mem = t.memory || {}, sys = t.system || {}, proc = t.process || {};
  const split = (used, total, own) => ({ used, total, own: Math.min(own, used), others: Math.max(0, used - own) });
  return {
    vram: d && d.mem_total ? split(d.mem_used || 0, d.mem_total, (mem.ledger_used || 0) + (mem.runtime || 0)) : null,
    ram: sys.ram_total ? split(sys.ram_total - (sys.ram_available || 0), sys.ram_total, proc.ram || proc.rss || 0)
                       : null,
  };
}

export const live = {
  last: null,             // the last answer, with _cpu (the process's CPU %, of one processor) and _sys_cpu (the
                          // machine's, of all its processors), each from two answers
  bySecond: new Map(),    // whole second -> {out, pre, rounds, drafted, accepted}: the server's counts of the second
  notice: null,           // {msg, withKey}: the server is away, or wants its key
  offline: false,
};
let failing = 0, prevCpu = null, prevSys = null;
const listeners = new Set();
const changed = () => { for (const f of listeners) f(); };
export function onLive(f) {
  listeners.add(f);
  return () => listeners.delete(f);
}

function away(msg) {
  if (++failing < 2) return;  // one miss is a blip
  live.offline = true;
  live.notice = { msg, withKey: false };
  changed();
}

export async function poll() {
  let r;
  try {
    r = await fetch("/v1/telemetry?recent=40", { headers: headers(), cache: "no-store" });
  } catch (e) {
    return away("The server does not answer: stopped, or restarting. Retrying.");
  }
  if (r.status === 401) { live.notice = { msg: "The server wants its API key.", withKey: true }; changed(); return; }
  if (!r.ok) return away("GET /v1/telemetry answered HTTP " + r.status + ". Retrying.");
  const t = await r.json().catch(() => null);
  if (!t || t.schema_version !== 1) return away("The server's telemetry is not a version this page reads.");
  failing = 0;
  live.offline = false;
  live.notice = null;
  const proc = t.process || {};
  if (prevCpu && proc.cpu_s !== undefined && t.sampled_at > prevCpu.t)
    t._cpu = (proc.cpu_s - prevCpu.cpu) / (t.sampled_at - prevCpu.t) * 100;
  if (proc.cpu_s !== undefined) prevCpu = { t: t.sampled_at, cpu: proc.cpu_s };
  const sys = t.system || {};
  if (prevSys && sys.cpu_s > 0 && sys.cpus > 0 && t.sampled_at > prevSys.t)
    t._sys_cpu = Math.max(0, Math.min(100, (sys.cpu_s - prevSys.cpu) / (t.sampled_at - prevSys.t) / sys.cpus * 100));
  if (sys.cpu_s > 0) prevSys = { t: t.sampled_at, cpu: sys.cpu_s };
  // The server counts each of the last 60 whole seconds: a reply between two polls shows too.  A count a server
  // before it did not give stays null.
  const rates = t.rates || {}, base = Math.floor(t.sampled_at);
  const counts = { out: rates.output_per_second, pre: rates.prefill_per_second, rounds: rates.rounds_per_second,
                   drafted: rates.drafted_per_second, accepted: rates.accepted_per_second };
  const n = (counts.out || []).length;
  for (let i = 0; i < n; ++i) {
    const sec = {};
    for (const [k, xs] of Object.entries(counts)) sec[k] = xs && xs.length === n ? xs[i] : null;
    live.bySecond.set(base - n + i, sec);
  }
  for (const k of live.bySecond.keys()) if (k < base - 600) live.bySecond.delete(k);
  // A model loaded without a drafter says "none": the page reads that as no drafter at all.
  if (t.model && (t.model.drafter === "none" || t.model.drafter === "")) t.model.drafter = null;
  live.last = t;
  changed();
}

// What the server is doing, as one word the page shows: a load first, then no model, then the engine's phase.
export function phaseOf(t) {
  if (!t) return "connecting";
  const model = t.model || {}, state = t.state || {}, L = t.load || { state: "none" };
  return L.state === "fetching" ? "fetching" : L.state === "loading" ? "loading" : !model.id ? "none" : state.phase || "idle";
}
export const phaseNames = { none: "No model", fetching: "Fetching", unloaded: "Unloaded", loading: "Loading", idle: "Idle",
                            prefill: "Prefill", decode: "Decoding", connecting: "Connecting" };
export const phaseTone = (phase) => phase === "decode" || phase === "prefill" ? "ok" : phase === "loading" || phase === "fetching" ? "warn" :
                                    phase === "none" || phase === "unloaded" ? "bad" : "";
export const busyLoad = (L) => !!L && (L.state === "fetching" || L.state === "loading");
