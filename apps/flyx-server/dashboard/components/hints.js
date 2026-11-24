// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The hint over the pages: what the server's state asks of the user next, one at a time.
// Each rule reads the telemetry and the pages this browser has opened, and gives a hint or nothing; the first rule
// that gives one is shown: a load that failed, no model, a context memory cut short, then a client to connect.  A
// request that failed is no hint: its client was told, and nothing on the server is to change (the Overview's strip
// and Requests say it).  A hint is hidden on the page that already says it: the Model page's card says a load's state
// and a short context, the Overview says a load's state too, with the next step beside "Waiting for a request" (pendingConnect).
// A hint closed stays closed (per browser); the state hints (a failed load, no model) come back with the state.
import { html, useState } from "../lib/html.js";
import { saved } from "../lib/api.js";
import { t } from "../lib/i18n.js";
import { Icon } from "../lib/ui.js";
import { live, busyLoad, poll } from "../lib/telemetry.js";
import { loadWant } from "../lib/actions.js";
import { fmt } from "../lib/format.js";
import { isVramError } from "./vram_apps.js";

// The pages this browser has opened, and the hints closed in it.
function savedSet(name) {
  try { const v = JSON.parse(saved.get(name) || "[]"); return new Set(Array.isArray(v) ? v : []); } catch { return new Set(); }
}
const addTo = (name, k) => { const s = savedSet(name); if (!s.has(k)) { s.add(k); saved.set(name, JSON.stringify([...s].slice(-50))); } };
export const markVisited = (tab) => addTo("visited", tab);
const visited = (tab) => savedSet("visited").has(tab);
const closed = (key) => savedSet("hints_closed").has(key);

const agents = html`<span class="hint-marks" aria-hidden="true">${["claude", "codex", "opencode", "pi"].map((id) => html`
  <svg class=${"agent-logo agent-" + id}><use href=${"#i-agent-" + id}/></svg>`)}</span>`;

// Each rule: (c) => a hint or null.  A hint: {key, tone (ok, warn, bad), title, text (a string, or sentences each
// translated on its own), actions [{label, run, primary}],
// hideOn (pages), closable (default true)}.
const RULES = [
  // A load that failed; with a fallback, the model served before is back.
  ["failed", (c) => {
    if (!c.L.error || c.busy) return null;
    const fell = c.L.fallback, name = c.L.target || "the model";
    const acts = isVramError(c.L.error)
      ? [{ label: "Manage VRAM", primary: true, run: (n) => n.openMem && n.openMem("vram") },
         { label: "View details", run: (n) => n.showModel() },
         { label: "View logs", run: (n) => n.go("logs") }]
      : [{ label: "View details", primary: true, run: (n) => n.showModel() },
         { label: "View logs", run: (n) => n.go("logs") }];
    return { key: "failed:" + c.L.error, tone: fell ? "warn" : "bad", hideOn: fell ? ["model"] : ["model", "overview"],
      title: fell ? "Failed to load " + name + "; reverted to " + fell : "Failed to load " + name,
      text: c.L.error.split(/(?<=\.)\s+/), actions: acts };
  }],
  // Nothing served: the model library, or the model loaded last.
  ["no-model", (c) => {
    if (c.model.id || c.busy || c.L.error) return null;
    const last = c.L.remembered;
    return { key: "no-model", tone: "warn", hideOn: ["model", "overview"], closable: false,
      title: "No model loaded",
      text: last ? "Requests will fail until a model is loaded. Reload " + last.model + " or choose another model."
                 : "Requests will fail until a model is loaded. Choose one from Hugging Face or a local GGUF file.",
      actions: last ? [{ label: "Reload", primary: true, run: () => loadWant(last) },
                       { label: "Browse models", run: (n) => n.explore() }]
                    : [{ label: "Browse models", primary: true, run: (n) => n.explore() }] };
  }],
  // Memory cut the served model's context short (the server says when): agents fill it fast, so every page but the
  // Model page (its card says it) says it, with what gives more.
  ["short-context", (c) => {
    const l = c.served && c.model.context_limit;
    if (!l || !l.short) return null;
    const gb = (b) => (b / 1e9).toFixed(1) + " GB";
    const fix = l.by === "host" ? "Free system memory or raise the host KV cache limit, then reload the model."
      : l.closable ? "Other processes are using " + gb(l.others) + " of GPU memory. Close them and reload the model."
      : "Disable speculative decoding or use a smaller quantization to free GPU memory.";
    const acts = l.by === "host"
      ? [{ label: "Manage RAM", primary: true, run: (n) => n.openMem && n.openMem("ram") },
         { label: "View details", run: (n) => n.showModel() }]
      : l.closable
        ? [{ label: "Manage VRAM", primary: true, run: (n) => n.openMem && n.openMem("vram") },
           { label: "View details", run: (n) => n.showModel() }]
        : [{ label: "View details", primary: true, run: (n) => n.showModel() }];
    return { key: "short:" + c.model.id + ":" + c.model.context, tone: "warn", hideOn: ["model"],
      title: "Context limited to " + fmt(c.model.context) + " tokens (model supports " + fmt(l.wanted) + ")",
      text: ["Coding agents may run out of context in long sessions.", fix],
      actions: acts };
  }],
  // A model served, and Integration never opened: connect a client.
  ["connect", (c) => {
    if (!c.served || c.visited("integration")) return null;
    return { key: "connect", tone: "ok", hideOn: ["integration", "overview"], marks: true,
      title: c.model.id + " is ready", text: "Connect Claude Code, Codex, OpenCode, or any OpenAI- or Anthropic-compatible client.",
      actions: [{ label: "Set up a client", primary: true, arrow: true, run: (n) => n.go("integration") }] };
  }],
];

// The Overview's strip offers Integration beside "Waiting for a request" while the hint would.
export function pendingConnect(tm) {
  const model = (tm && tm.model) || {};
  return !!(model.id && model.loaded && !busyLoad(tm.load) && !visited("integration") && !closed("connect"));
}

function context(tm, tab) {
  const L = tm.load || { state: "none" }, model = tm.model || {}, busy = busyLoad(L);
  return { tm, tab, L, model, busy, served: !!(model.id && model.loaded && !busy),
           visited };
}

export function Hint({ tab, go, explore, showModel, onOpenMem }) {
  const [, again] = useState(0);
  const tm = live.last;
  if (!tm || live.notice) return "";
  let hint = null;
  const c = context(tm, tab);
  for (const [, rule] of RULES) {
    const h = rule(c);
    if (!h || closed(h.key)) continue;
    hint = h;
    break;
  }
  if (hint && hint.hideOn && hint.hideOn.includes(tab)) hint = null;
  if (!hint) return "";
  const nav = { go, explore, showModel, openMem: onOpenMem };
  const close = () => { addTo("hints_closed", hint.key); again((x) => x + 1); poll(); };
  return html`
    <div class=${"hint " + hint.tone} role="status" id="hint" data-hint=${hint.key.split(":")[0]}>
      <div class="hint-text"><b>${t(hint.title)}</b>${hint.text ? html`<span>${[].concat(hint.text).map((s) => t(s)).join(" ").replace(/([。！？]) /g, "$1")}</span>` : ""}</div>
      ${hint.marks ? agents : ""}
      <div class="hint-acts">${hint.actions.map((a) => html`
        <button class=${"btn sm" + (a.primary ? " primary" : "")} onClick=${() => a.run(nav)}>${t(a.label)}${a.arrow ? " →" : ""}</button>`)}</div>
      ${hint.closable === false ? "" : html`
        <button class="iconbtn" title=${t("Close")} aria-label=${t("Close")} onClick=${close}><${Icon} name="close"/></button>`}
    </div>`;
}
