// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The Model page, in two views behind tabs, since a model once loaded is mostly watched and tuned, seldom changed.
// "Current model": what the server serves (telemetry's "load") and, while it serves one, the settings a load takes
// (context, K/V, drafting) beside the command line that would load the same.  "Model library": where models come from, as a list and the one
// thing chosen in it: Hugging Face (GET /v1/hub/search, a repository's files from GET /v1/hub/files) or this machine
// (the cache and the model directories, GET /v1/hub/cached; any folder, GET /v1/hub/dir); one box loads the file
// chosen (POST /v1/load).  Fields an older server may not send are read defensively.
import { html, useState, useReducer, useEffect, useRef } from "../lib/html.js";
import { saved, getJson, call } from "../lib/api.js";
import { t, tErr } from "../lib/i18n.js";
import { fmt, gib, gb, secs, shortDevice, baseName, psQuote, shQuote, compact, ago, contextLimit } from "../lib/format.js";
import { Icon, CodeSnippet, CopyButton, confirmBox, defaultShell, ShellPicker } from "../lib/ui.js";
import { live, poll, busyLoad } from "../lib/telemetry.js";
import { unloadServed, reloadServed, stopFetch } from "../lib/actions.js";
import { VramApps, isVramError, parseVramNeeded } from "../components/vram_apps.js";

const hostOf = (u) => String(u || "").replace(/^https?:\/\//i, "").replace(/\/.*$/, "") || "huggingface.co";
const safeOrigin = (u) => /^https?:\/\//i.test(String(u || "")) ? String(u).replace(/\/+$/, "") : "https://huggingface.co";
function stripHubPrefix(v, extraOrigin) {
  const s = v.trim();
  const prefixes = ["https://huggingface.co/", "http://huggingface.co/", "huggingface.co/",
                    "https://hf.co/", "hf.co/", "https://hf-mirror.com/", "hf-mirror.com/"];
  if (extraOrigin) {
    const clean = safeOrigin(extraOrigin);
    prefixes.unshift(clean + "/", hostOf(clean) + "/");
  }
  const low = s.toLowerCase();
  for (const p of prefixes) if (low.startsWith(p.toLowerCase())) return s.slice(p.length);
  return s;
}
// "org/name[:quant|file][@rev]", not a path: a relative path of two parts reads the same unless it has ':' or lacks a model extension.
const hubName = (v, extraOrigin) => {
  const s = stripHubPrefix(v, extraOrigin);
  if (!/^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+(:[A-Za-z0-9._-]+)?(@[A-Za-z0-9._-]+)?$/.test(s)) return false;
  return s.includes(":") || !/\.(gguf|flyx)$/i.test(s);
};
const repoOf = (v, extraOrigin) => stripHubPrefix(v, extraOrigin).split("@")[0].split(":")[0];
const quantOf = (v) => { const m = /:([^@]+)/.exec(v); return m ? m[1] : null; };
const revOf = (v, extraOrigin) => {
  const s = stripHubPrefix(v, extraOrigin);
  const at = s.indexOf("@");
  return at >= 0 ? s.slice(at + 1) : "";
};
const repoSpec = (id, rev) => rev && rev !== "main" ? id + "@" + rev : id;

const EMPTY_HUB_VIEW = { sel: "", quant: "", rev: "", pinned: "", detail: null, note: null, draft: "", mmproj: "" };
function hubViewReducer(state, action) {
  switch (action.type) {
    case "clear": return EMPTY_HUB_VIEW;
    case "unpin": return state.pinned ? { ...state, pinned: "" } : state;
    case "load": {
      const same = state.sel === action.sel && state.rev === (action.rev || "");
      const pinned = action.pinned === undefined ? (same ? state.pinned : "") : (action.pinned ? action.sel : "");
      return {
        sel: action.sel,
        quant: action.quant || "",
        rev: action.rev || "",
        pinned,
        detail: same ? state.detail : null,
        note: { tone: "info", text: "Asking Hugging Face about " + repoSpec(action.sel, action.rev) + " ..." },
        draft: same ? state.draft : "",
        mmproj: same ? state.mmproj : "",
      };
    }
    case "ok":
      return {
        ...state,
        detail: { repo: action.sel, rev: action.rev || "", j: action.j },
        note: action.j.from_cache ? { tone: "info", text: "From the local cache: Hugging Face was not reached." } : null,
      };
    case "error":
      return { ...state, detail: null, note: { tone: "error", text: action.error } };
    case "draft": return { ...state, draft: action.value };
    case "mmproj": return { ...state, mmproj: action.value };
    case "clearCompanions": return { ...state, draft: "", mmproj: "" };
    default: return state;
  }
}

function describe(m, winPaths = null) {
  const repo = m.repo || (m.source === "dir" ? null : repoOf(m.name));
  const quant = m.quant || (m.source !== "dir" ? quantOf(m.name) : null);
  return { repo, quant, file: m.file || (m.path ? baseName(m.path, winPaths) : m.source === "dir" ? baseName(m.name, winPaths) : null) };
}

// Drafters the server found or was given, from the status (or, from an older server, its notes).
function draftersOf(L) {
  if (Array.isArray(L.drafters)) return L.drafters;
  const out = [];
  for (const n of L.notes || []) { const m = /^the MTP layer from (\S+)/.exec(n); if (m) out.push({ kind: "mtp", from: m[1] }); }
  return out;
}

// The memory a new model may use: the GPU's, less what other programs hold (flyx's own model goes when it switches).
let gpuTotal = Number(saved.get("gpu_total")) || 0;
function deviceReserve(total, m) {
  if (m && typeof m.device_reserve === "number" && m.device_reserve >= 0) return m.device_reserve;
  return Math.max(384 * 1048576, total / 40);
}
function freeForModel() {
  const d = live.last && live.last.device, m = live.last && live.last.memory;
  if (!d || !d.mem_total) return gpuTotal || 0;
  // When a model is loaded, ledger_budget is the engine's real VRAM ceiling (already excluding other programs and the
  // device reserve, without miscounting flyx's own CUDA context / driver pool as other programs).
  if (m && m.ledger_budget) return Math.min(d.mem_total, m.ledger_budget + deviceReserve(d.mem_total, m));
  return Math.max(0, d.mem_total - (d.mem_used || 0));
}
function FitTag({ size }) {
  const d = live.last && live.last.device, m = live.last && live.last.memory;
  const room = freeForModel();
  if (!room || !size) return html`<span class="tag2">?</span>`;
  // Untied embeddings (~4% on 27B) stay in host RAM; ~0.85 GB covers workspace and one minimal session beyond reserve.
  const res = deviceReserve(d && d.mem_total ? d.mem_total : room, m);
  const budget = m && m.ledger_budget ? m.ledger_budget : Math.max(0, room - res);
  const estWeights = size * 0.96;
  const fits = estWeights + 0.85e9 < budget;
  const budgetTxt = budget ? gib(budget) : "0.0";
  const resTxt = res ? gib(res) : "0.0";
  const title = "Estimate: " + gb(estWeights) + " GB of weights + 0.85 GB workspace/session against " + budgetTxt +
                " GiB available VRAM budget (after " + resTxt + " GiB reserve). Long contexts need more; the engine's bill shows once it loads.";
  return html`<span class=${"tag2 " + (fits ? "ok" : "bad")} title=${t(title)}>${t(fits ? "fits" : "too big")}</span>`;
}

const kDrafterLabels = { mtp: "MTP", dflash2: "DFlash2", dflash: "DFlash", dspark: "DSpark", dspine: "DSpine", dv: "DV" };
const drafterLabel = (k) => kDrafterLabels[k] || k;

// What is served and its state, its facts, a fetch's or a load's progress, an error.  With nothing
// served it is the view's whole content: `again` loads the model the server remembers, `browse` shows the library.
function Serving({ tm, again, retry, browse, winPaths = null }) {
  const L = tm.load || { state: "none" }, model = tm.model || {}, p = L.progress || {};
  const loadedWant = model.id && L.loaded ? L.loaded : null, busy = busyLoad(L);
  const [msg, setMsg] = useState("");
  const [vramOpen, setVramOpen] = useState(false);
  const activeDraft = model.drafter && model.drafter !== "none" ? model.drafter : "";
  const degraded = model.loaded && !busy ? ((Array.isArray(L.degraded) && L.degraded.length ? L.degraded : model.degraded) || []) : [];
  const drafters = model.loaded && !busy ? draftersOf(L).slice() : [];
  if (model.loaded && !busy && activeDraft && /mtp/.test(activeDraft) && !drafters.some((d) => d.kind === "mtp"))
    drafters.unshift({ kind: "mtp", from: "model file" });
  const draftTip = drafters.map((d) => drafterLabel(d.kind) + ": " + d.from + (d.size ? " · " + gb(d.size) + " GB" : "")).join("\n");
  const visionTip = model.loaded && !busy && model.vision
    ? (L.vision_projector && L.vision_projector.from ? L.vision_projector.from : t("model file")) +
      (L.vision_projector && L.vision_projector.size ? " · " + gb(L.vision_projector.size) + " GB" : "")
    : "";
  const facts = [];
  if (model.loaded && !busy) {
    if (model.device) facts.push(["GPU", shortDevice(model.device)]);
    facts.push(["Context", fmt(model.context)]);
    if (loadedWant) facts.push(["KV cache", loadedWant.kv === "16" ? "16-bit" : loadedWant.kv === "fp8" ? "fp8" : "auto"]);
    facts.push(["Sessions", fmt(model.max_sessions)]);
    facts.push(["Drafting", activeDraft ? activeDraft + (model.draft_weights ? " · " + model.draft_weights : "") + ", up to " + model.max_width : "off", draftTip]);
    if (loadedWant && loadedWant.lora) facts.push(["LoRA", baseName(trimTrailingSep(loadedWant.lora, winPaths), winPaths), loadedWant.lora]);
    facts.push(["Vision", model.vision ? t(normVisionDev(model.vision_device)) + " · " + normVisionQuant(model.vision_quant) : "off", visionTip]);
    if (model.load_s) facts.push(["Load time", secs(model.load_s)]);
    if (tm.memory && tm.memory.ledger_used) facts.push(["Memory", gib(tm.memory.ledger_used) + " GiB"]);
  }
  let sub = "";
  if (L.state === "fetching" && !p.total) sub = t("asking Hugging Face which files it needs");
  else if (L.state === "fetching") {
    const left = p.bytes_per_s > 0 ? (p.total - p.done) / p.bytes_per_s : null;
    sub = (p.file || "") + "  " + gb(p.done || 0) + " of " + gb(p.total || 0) + " GB" +
      (p.bytes_per_s ? " · " + fmt(p.bytes_per_s / 1e6) + " MB/s" : "") + (left !== null && !p.checking ? " · " + secs(left) + " left" : "") +
      (p.checking ? " · checking its sha256" : "") + (model.id ? " · " + model.id + " keeps serving meanwhile" : "");
  } else if (L.state === "loading") sub = t("loading onto the GPU · " + secs(L.elapsed_s));
  const limit = model.loaded && !busy ? contextLimit(model) : null;  // memory cut the context: why, and what gives more
  const ctxLimit = model.context_limit;
  const ctxMemKind = !limit || !ctxLimit ? ""
    : ctxLimit.by === "host" ? "ram"
    : ctxLimit.by === "gpu" && ctxLimit.closable ? "vram" : "";
  const vramErr = !busy && isVramError(L.error);
  const showMemPanel = vramErr || ((!!ctxMemKind || degraded.length > 0) && vramOpen);
  const panelKind = vramErr ? "vram" : (ctxMemKind || "vram");
  const stateName = { none: "Nothing loaded", fetching: "Fetching", loading: "Loading", ready: "Serving", failed: "Failed" }[L.state] || L.state;
  const tone = L.state === "ready" ? "ok" : L.state === "failed" ? "bad" : busy ? "warn" : "";
  const idle = !model.id && !busy;
  return html`
    <div class=${"srv" + (idle ? " idle" : "")}>
      <div class=${"srv-lamp " + tone}><${Icon} name="model"/></div>
      <div class="srv-what">
        <span class=${"badge " + tone} id="ld-state"><span class="dot"></span>${t(stateName)}</span>
        <div class="srv-name" id="ld-now">${busy ? L.target : model.id ? model.id : t("No model loaded")}</div>
        <div class=${"srv-src" + (idle ? " hint" : "")} id="ld-src">${busy ? "" : loadedWant && loadedWant.model !== model.id ? loadedWant.model :
          model.id ? "" : t("Pick one in the model library: from Hugging Face, or one on this machine.")}</div>
      </div>
      <div class="srv-acts">
        <button class="btn" id="ld-cancel" hidden=${L.state !== "fetching"} onClick=${stopFetch}>${t("Stop fetching")}</button>
        <button class="btn" id="ld-reload" hidden=${!model.id || busy || !loadedWant}
                onClick=${reloadServed}><${Icon} name="reload"/>${t("Reload model")}</button>
        <button class="btn danger" id="ld-unload" hidden=${!model.id || busy}
                onClick=${async () => setMsg((await unloadServed()) || "")}><${Icon} name="eject"/>${t("Unload")}</button>
      </div>
      ${idle && again ? html`<div class="srv-last" id="ld-last">${t("Loaded last")} <b>${again.model}</b></div>` : ""}
      ${idle ? html`<div class="srv-go">
        ${again ? html`<button class="btn primary" id="ld-again" title=${again.model} onClick=${again.load}>
          <${Icon} name="play"/>${t("Load the last model again")}</button>` : ""}
        <button class=${"btn" + (again ? "" : " primary")} id="ld-browse" onClick=${browse}>${t("Explore the model library")}</button></div>` : ""}
      <dl class="stats" id="ld-facts" hidden=${!facts.length}>${facts.map(([k, v, tip]) => html`<div title=${tip || undefined}><dt>${t(k)}</dt><dd>${t(v)}</dd></div>`)}</dl>
      <div class="banner warn" id="ld-degraded" hidden=${!degraded.length}>
        <div class="grow2">
          <b>${t("Loaded with VRAM adjustments")}</b>
          ${degraded.map((d) => html`<div class="deg-item">${t(d.detail || "")}</div>`)}
          <div class="deg-now">${t("Active now:")} ${t("Drafting")} ${activeDraft ? activeDraft + (model.draft_weights ? " (" + model.draft_weights + ")" : "") : t("off")} · ${t("Vision")} ${model.vision ? t(normVisionDev(model.vision_device)) + " (" + normVisionQuant(model.vision_quant) + ")" : t("off")}</div>
        </div>
        <div class="deg-acts">
          <button class="btn sm" id="ld-deg-vram" onClick=${() => setVramOpen((v) => !v)}>${t(vramOpen ? "Hide apps" : "Manage VRAM")}</button>
          ${loadedWant ? html`<button class="btn sm primary" id="ld-deg-restore" onClick=${reloadServed}>${t("Restore full settings")}</button>` : ""}
        </div>
      </div>
      <div class=${"banner " + (limit && limit.short ? "warn" : "hint")} id="ld-ctx" hidden=${!limit}>
        <div class="grow2">${limit ? limit.parts.map((s) => t(s)).join(" ").replace(/([。！？]) /g, "$1") : ""}</div>
        ${ctxMemKind ? html`<button class="btn sm" id="ld-vram-toggle" onClick=${() => setVramOpen((v) => !v)}>
          ${t(vramOpen ? "Hide apps" : ctxMemKind === "ram" ? "Manage RAM" : "Manage VRAM")}</button>` : ""}</div>
      <div class=${"bar" + (L.state === "loading" ? " busy" : "")} id="ld-prog" hidden=${!busy}>
        <i style=${{ width: L.state === "loading" ? "" : (p.total ? 100 * p.done / p.total : 0) + "%" }}></i></div>
      <div class="prog" id="ld-sub">${sub}</div>
      <div class="banner bad" id="ld-err" hidden=${!L.error && !msg}>${L.error ? tErr(L.error) : t(msg)}</div>
      <${VramApps} open=${showMemPanel} kind=${panelKind} id="ld-vram-apps"
                   needBytes=${vramErr ? parseVramNeeded(L.error) : (ctxLimit && ctxLimit.need_more > 0 && panelKind === (ctxLimit.by === "host" ? "ram" : "vram") ? ctxLimit.need_more : 0)}
                   onRetry=${vramErr ? retry : (model.id && loadedWant ? reloadServed : null)}
                   retryLabel=${vramErr ? "Retry loading" : "Reload model"}
                   onClose=${!vramErr ? () => setVramOpen(false) : null}/>
    </div>`;
}

// A model the library can load by itself: not a drafter, a vision projector or a LoRA adapter (they go with one).
const standalone = (m) => m.kind !== "drafter" && m.kind !== "mmproj" && m.kind !== "lora";
// The drafter kinds one drafter file can be (--drafter's words for them).  A drafter's kind is the server's word for
// it (its file's own fields, GET /v1/hub/drafter), never its name's; unknown, the loader reads it.
const kFileDrafterKinds = ["mtp", "dflash2", "dflash", "dspark", "dspine"];
const pickedDrafter = (path, kind) => ({
  spec: true, draftSource: "custom", savedDraft: null, drafter: path, drafterKind: kFileDrafterKinds.includes(kind) ? kind : "", draftPicked: true,
});
const normVisionDev = (v) => (v === "host" || v === "cpu") ? "host" : v === "gpu" ? "gpu" : v === "auto" ? "auto" : "disk";
const normVisionQuant = (v) => (v === "exact" || v === "bf16") ? "exact" : (v === "nvfp4" || v === "fp4") ? "nvfp4" : "q8";
const normDraftWeights = (v) => (v === "nvfp4" || v === "fp4") ? "nvfp4" : (v === "q8" || v === "q8_0") ? "q8" : (v === "exact" || v === "bf16") ? "exact" : "";

// The settings of a want as the form shows them, for comparing with what is loaded.
function shown(w) {
  const vOn = w.vision !== false;
  const dw = normDraftWeights(w.draft_weights);
  return { ctx: w.ctx || "auto", kv: w.kv === "16" ? "16-bit" : w.kv === "fp8" ? "fp8" : "auto",
           kv_ram: typeof w.kv_ram === "number" ? w.kv_ram + " GB" : "auto", kv_gpu: typeof w.kv_gpu === "number" ? fmt(w.kv_gpu) : "as many as fit",
           vram_reserve: typeof w.vram_reserve === "number" ? w.vram_reserve + " GB" : "auto",
           ram_reserve: typeof w.ram_reserve === "number" ? w.ram_reserve + " GB" : "auto",
           name: w.name || "the file's name",
           spec: w.speculative === false ? "off" : "on", drafter: [w.drafter || "default", ...(w.drafters || [])],
           width: w.width === null || w.width === undefined ? "auto" : String(w.width),
           head: w.draft_head === "full" ? "full head" : "1-bit pick",
           weights: dw === "nvfp4" ? "NVFP4" : dw === "q8" ? "Q8_0" : dw === "exact" ? "exact" : "auto",
           lookup: w.lookup !== false ? "on" : "off", trees: w.draft_trees === true ? "on" : w.draft_trees === false ? "off" : "default",
           lora: w.lora || "none",
           vision: vOn ? "on" : "off", mmproj: vOn ? (w.mmproj || "auto") : "auto",
           vision_device: normVisionDev(w.vision_device), vision_quant: normVisionQuant(w.vision_quant),
           vision_max_tokens: w.vision_max_tokens > 0 ? String(w.vision_max_tokens) : "auto" };
}

function draftingOf(w) {
  const files = w.drafters || [], mode = w.drafter || "", own = files.length === 1 ? files[0] : "";
  // The single custom-file control also selects its kind. Keep configurations it cannot express as a separate choice.
  const custom = w.speculative !== false && files.length === 1 && own === own.trim() && (!mode || kFileDrafterKinds.includes(mode));
  const savedDraft = !custom && (mode || files.length) ? { drafter: mode, drafters: files } : null;
  return { draftSource: savedDraft ? "saved" : custom ? "custom" : "default", savedDraft, drafter: own, drafterKind: custom ? mode : "" };
}

// The form's fields from a want ({model, ctx, kv, drafters, ...}).
function formOf(w) {
  const vOn = w.vision !== false;
  return { ctx: w.ctx || "", kv: w.kv === "16" || w.kv === "fp8" ? w.kv : "", kvram: typeof w.kv_ram === "number" ? String(w.kv_ram) : "",
           kvgpu: typeof w.kv_gpu === "number" ? String(w.kv_gpu) : "",
           vramres: typeof w.vram_reserve === "number" ? String(w.vram_reserve) : "",
           ramres: typeof w.ram_reserve === "number" ? String(w.ram_reserve) : "",
           name: w.name || "", lora: w.lora || "", loraPicked: false,
           spec: w.speculative !== false, ...draftingOf(w), draftPicked: false,
           width: w.width === null || w.width === undefined ? "" : String(w.width), trees: w.draft_trees === true ? "on" : w.draft_trees === false ? "off" : "", lookup: w.lookup !== false,
           head: w.draft_head === "full" ? "full" : "", weights: normDraftWeights(w.draft_weights),
           vision: vOn, visionSource: vOn && w.mmproj ? "custom" : "default", mmproj: vOn ? (w.mmproj || "") : "", mmprojPicked: false,
           vision_device: normVisionDev(w.vision_device), vision_quant: normVisionQuant(w.vision_quant),
           vision_max_tokens: w.vision_max_tokens > 0 ? String(w.vision_max_tokens) : "" };
}
const EMPTY_FORM = formOf({});

// What the panel asks of POST /v1/load.
function loadBody(chosen, f, hasLoaded = false) {
  const ctxText = String(f.ctx).trim().toLowerCase(), ctxTokens = parseInt(ctxText, 10);
  const ctx = ctxText === "max" ? "max" : ctxTokens > 0 ? ctxTokens : null;  // "max": the model's own, or the longest that fits
  const kvRam = parseFloat(f.kvram), kvGpu = parseInt(f.kvgpu, 10), name = f.name.trim(), lora = f.lora.trim();
  const vramRes = parseFloat(f.vramres), ramRes = parseFloat(f.ramres);
  const blankReserve = hasLoaded ? "auto" : null;
  const own = f.draftSource === "custom" && f.spec ? f.drafter.trim() : "";
  const adv = { width: f.width ? parseInt(f.width, 10) : null, draft_trees: f.trees === "on" ? true : f.trees === "off" ? false : null, lookup: f.lookup,
                draft_head: f.head || null, draft_weights: f.weights || null };
  const draft = f.draftSource === "saved" ? f.savedDraft : {
    drafter: own ? f.drafterKind : "", drafters: own ? [own] : [],
  };
  const mmproj = f.vision && f.visionSource === "custom" && f.mmproj.trim() ? f.mmproj.trim() : null;
  const vmaxRaw = String(f.vision_max_tokens).trim();
  const vmax = vmaxRaw ? parseInt(vmaxRaw, 10) : NaN;
  const vis = { vision: f.vision, mmproj, vision_device: normVisionDev(f.vision_device),
                vision_quant: normVisionQuant(f.vision_quant),
                vision_max_tokens: Number.isFinite(vmax) && vmax > 0 ? vmax : null };
  // kv "auto": the engine's default, not the server's --kv (null would keep that)
  return Object.assign({ model: chosen, ctx, kv: f.kv || "auto", kv_ram: kvRam >= 0 ? kvRam : "auto",
                         kv_gpu: kvGpu >= 0 ? kvGpu : null,
                         vram_reserve: vramRes >= 0 ? vramRes : blankReserve, ram_reserve: ramRes >= 0 ? ramRes : blankReserve,
                         name: name || null, lora: lora || null, speculative: f.spec },
                       draft, adv, vis);
}

// The command line that loads the same: what a user can put in a script.
function commandOf(b, shell) {
  const ps = shell === "powershell";
  // Windows PowerShell 5.1 drops literal double quotes and can consume a trailing backslash
  // when joining native arguments that contain spaces. Pass the native command line in these cases.
  const strings = [b.model, b.name, b.mmproj, b.lora, ...b.drafters];
  const native = ps && strings.some((s) => s && (s.includes('"') || (/\s/.test(s) && s.endsWith("\\"))));
  const nativeQuote = (s) => '"' + s.replace(/(\\*)"/g, '$1$1\\"').replace(/\\+$/, '$&$&') + '"';
  const q = native ? nativeQuote : ps ? psQuote : shQuote;
  const parts = [b.model ? q(b.model) : "MODEL"];
  if (b.ctx) parts.push("--ctx " + b.ctx);
  if (b.kv !== "auto") parts.push("--kv " + b.kv);
  if (b.kv_ram !== "auto") parts.push("--kv-ram " + b.kv_ram);
  if (b.kv_gpu !== null) parts.push("--kv-gpu " + b.kv_gpu);
  if (b.vram_reserve !== undefined && b.vram_reserve !== null && b.vram_reserve !== "auto") parts.push("--vram-reserve " + b.vram_reserve);
  if (b.ram_reserve !== undefined && b.ram_reserve !== null && b.ram_reserve !== "auto") parts.push("--ram-reserve " + b.ram_reserve);
  if (!b.speculative) parts.push("--no-spec");
  if (b.drafter) parts.push("--drafter " + b.drafter);
  for (const d of b.drafters) parts.push("--drafter-file " + q(d));
  if (b.width !== null) parts.push("--width " + b.width);
  if (b.draft_trees === true) parts.push("--draft-trees");
  if (b.draft_trees === false) parts.push("--no-draft-trees");
  if (!b.lookup) parts.push("--no-lookup");
  if (b.draft_head) parts.push("--backend-opt draft_head=" + b.draft_head);
  if (b.draft_weights) parts.push("--backend-opt draft_weights=" + b.draft_weights);
  if (b.lora) parts.push("--lora " + q(b.lora));
  if (b.vision === false) parts.push("--no-vision");
  if (b.mmproj) parts.push("--mmproj " + q(b.mmproj));
  if (b.vision !== false && b.vision_device && b.vision_device !== "disk") parts.push("--vision-device " + b.vision_device);
  if (b.vision !== false && b.vision_quant && b.vision_quant !== "q8") parts.push("--vision-quant " + b.vision_quant);
  if (b.vision !== false && b.vision_max_tokens) parts.push("--vision-max-tokens " + b.vision_max_tokens);
  if (b.name) parts.push("--name " + q(b.name));
  const port = location.port || (location.protocol === "https:" ? "443" : "80");
  if (port !== "8080") parts.push("--port " + port);
  if (native) return "Start-Process flyx -NoNewWindow -Wait -ArgumentList " + psQuote("serve " + parts.join(" "));
  const continuation = ps ? " `\n    " : " \\\n    ";
  let s = "flyx serve " + parts[0];
  for (const p of parts.slice(1)) s += (s.length - s.lastIndexOf("\n") + p.length > 60 ? continuation : " ") + p;
  return s;
}

// A labelled field of the form: `k` names it in the comparison with what is loaded.
const Field = ({ k, label, was, hint, wide, children }) => html`
  <label class=${"field" + (was ? " changed" : "")} data-k=${k} style=${wide ? "grid-column: 1 / -1" : ""}>
    <span>${t(label)}<span class="was">${was ? t("was " + t(was)) : ""}</span></span>${children}${hint ? html`<em>${t(hint)}</em>` : ""}</label>`;

// The orders GET /v1/hub/search lists the Hub's repositories in.
const SORTS = [["trending", "Trending"], ["downloads", "Most downloaded"], ["likes", "Most liked"], ["updated", "Recently updated"],
               ["created", "Newest"]];
// The folders kept in this browser's list of places (a convenience of the viewer's: the server knows only its own
// --models-dir).
function savedFolders() {
  try { const v = JSON.parse(saved.get("folders") || "[]"); return Array.isArray(v) ? v.filter((x) => typeof x === "string") : []; }
  catch (e) { return []; }
}
const isWinPath = (p) => /^[a-zA-Z]:(?:[\\/]|$)|^\\\\|\\/.test(String(p || ""));
const trimTrailingSep = (p, winPaths = null) => {
  const s = String(p || "");
  if (winPaths === false) return /^\/+$/.test(s) ? "/" : s.replace(/\/+$/, "");
  return s === "/" ? "/" : /^[a-zA-Z]:[\\/]+$/.test(s) ? s.slice(0, 3) : s.replace(/[\\/]+$/, "");
};
const folderOf = (m, winPaths = null) => {
  const raw = String(m.path || m.name || "");
  const i = winPaths === false ? raw.lastIndexOf("/") : Math.max(raw.lastIndexOf("/"), raw.lastIndexOf("\\"));
  if (i < 0) return "";
  if (i === 0) return "/";
  if (winPaths !== false && i === 2 && /^[a-zA-Z]:/.test(raw)) return raw.slice(0, 3);
  return raw.slice(0, i);
};
const normPath = (p, winPaths = null) => {
  const s = String(p || "");
  if (!s) return "";
  const win = winPaths === null ? isWinPath(s) : (winPaths && isWinPath(s));
  if (win) {
    const fwd = s.replace(/\\/g, "/");
    if (/^[a-zA-Z]:\/+$/.test(fwd)) return fwd.slice(0, 2).toLowerCase() + "/";
    return fwd.replace(/\/+$/, "").toLowerCase();
  }
  if (/^\/+$/.test(s)) return "/";
  return s.replace(/\/+$/, "");
};
const samePath = (a, b, winPaths = null) => normPath(a, winPaths) === normPath(b, winPaths);
const pathUnder = (p, dir, winPaths = null) => {
  const a = normPath(p, winPaths), b = normPath(dir, winPaths);
  if (!b) return false;
  if (a === b) return true;
  return b.endsWith("/") ? a.startsWith(b) : a.startsWith(b + "/");
};
const relUnder = (mf, base, winPaths = null) => {
  if (!pathUnder(mf, base, winPaths)) return "";
  const a = normPath(mf, winPaths), b = normPath(base, winPaths);
  if (a === b) return "";
  const raw = (winPaths === false ? String(mf || "") : String(mf || "").replace(/\\/g, "/")).replace(/\/+$/, "");
  const baseLen = b.endsWith("/") ? b.length : b.length + 1;
  return raw.slice(baseLen).replace(/^\/+/, "");
};
const whole = (g) => !(g.split && g.complete === false);
const sized = (b) => b >= 1e8 ? gb(b) + " GB" : Math.max(1, Math.round(b / 1e6)) + " MB";
// An answer of the server's newer endpoints, or why there is none: an older server does not know them (404 without JSON).
async function ask(path, what) {
  const r = await call("GET", path);
  if (r.ok) return { j: r.json };
  return { error: r.status === 404 && !r.json ? "This server cannot " + what + " yet: update FLYX." : r.error };
}

export function ModelPage({ active, libraryAsked, currentAsked }) {
  const tm = live.last || {}, L = tm.load || { state: "none" }, model = tm.model || {};
  const loadedWant = model.id && L.loaded ? L.loaded : null, busy = busyLoad(L);
  const [tab, setTabState] = useState(() => saved.get("models_tab") === "local" ? "local" : "hub");
  // Hugging Face: what was searched for, the repositories found, pagination, and the one shown
  const [q, setQ] = useState("");
  const [sort, setSort] = useState("trending");
  const [arch, setArch] = useState("");
  const [page, setPage] = useState(0);
  const [cursors, setCursors] = useState([""]);
  const [found, setFound] = useState(null);
  const [foundMsg, setFoundMsg] = useState("");
  const [hubView, rawDispatchHub] = useReducer(hubViewReducer, EMPTY_HUB_VIEW);
  const hubViewRef = useRef(hubView);
  hubViewRef.current = hubView;
  const dispatchHub = (action) => {
    hubViewRef.current = hubViewReducer(hubViewRef.current, action);
    rawDispatchHub(action);
  };
  const [hubCfg, setHubCfg] = useState(null);  // GET /v1/hub/config: {endpoint, default_endpoint, source, offline}
  const [epOpen, setEpOpen] = useState(false);
  const [epText, setEpText] = useState("");
  const [epMsg, setEpMsg] = useState("");
  const [epSaving, setEpSaving] = useState(false);
  const [verifiedOf, setVerifiedOf] = useState({});  // a repository's drafters for its base models (GET /v1/hub/files)
  const [baseOpen, setBaseOpen] = useState(false);  // the drafters for the model's base model, shown
  const [madeOpen, setMadeOpen] = useState(false);  // the drafters made for the model, shown
  // this machine: the models the server lists, the folders kept here, the folder shown (`loc`; "" for where to start)
  const [catalog, setCatalog] = useState(null);
  const [catalogErr, setCatalogErr] = useState("");
  const [folders, setFoldersState] = useState(savedFolders);
  const [loc, setLoc] = useState("");
  const [dir, setDir] = useState(null);  // GET /v1/hub/dir of `loc`
  const [dirMsg, setDirMsg] = useState("");
  const [pathText, setPathText] = useState("");
  const [localQ, setLocalQ] = useState("");
  const [picked, setPicked] = useState("");  // the model of the list shown at the right; "": the folder at `loc`
  const [localDraft, setLocalDraft] = useState("");
  const [localMmproj, setLocalMmproj] = useState("");
  const [listings, setListings] = useState({});  // a folder the server does not list by itself -> {models} of GET /v1/hub/dir
  // the model to load, and how
  const [target, setTarget] = useState("");
  const [f, setF] = useState(EMPTY_FORM);
  // A drafter typed in: its kind as its file's fields say it (GET /v1/hub/drafter), asked once typing stops.
  useEffect(() => {
    const name = f.drafter.trim();
    if (f.draftSource !== "custom" || !name || f.drafterKind) return;
    let live = true;
    const timer = setTimeout(async () => {
      const a = await ask("/v1/hub/drafter?name=" + encodeURIComponent(name), "read a drafter");
      if (live && !a.error && a.j.kind)
        setF((cur) => cur.drafter.trim() === name && !cur.drafterKind ? Object.assign({}, cur, { drafterKind: a.j.kind }) : cur);
    }, 400);
    return () => { live = false; clearTimeout(timer); };
  }, [f.draftSource, f.drafter, f.drafterKind]);
  // The view shown: "current" (the model served, the load settings) or "library".  The page opens on the model while
  // one is served or coming or its load failed, else on the library; a load started from the library shows the model
  // again.
  const [view, setView] = useState("current");
  const viewed = useRef(false);
  useEffect(() => { if (libraryAsked) setView("library"); }, [libraryAsked]);  // another page's way here
  useEffect(() => { if (currentAsked) setView("current"); }, [currentAsked]);  // and to the model's card
  const [advOpen, setAdvOpen] = useState(false);
  const [stab, setStab] = useState("general");
  // LoRA adapters on the Hub for the model's base (GET /v1/hub/loras): the base asked for ("": the model card's), the
  // list, and one repository opened (GET /v1/hub/lora: its adapters, each loadable or why not)
  const [loraBase, setLoraBase] = useState("");
  const [loraBaseDraft, setLoraBaseDraft] = useState("");
  const [loraHub, setLoraHub] = useState(null);
  const [loraHubMsg, setLoraHubMsg] = useState("");
  const [loraRepo, setLoraRepo] = useState(null);
  const loraSeq = useRef(0);
  const [shell, setShell] = useState(defaultShell);
  const [msg, setMsg] = useState("");
  const filled = useRef(false), sawState = useRef(""), sawLoaded = useRef("");
  const searchSeq = useRef(0), repoSeq = useRef(0), dirSeq = useRef(0), placed = useRef(false), hubPickedOnce = useRef(false);
  const lastAttempt = useRef(null);
  const set = (k) => (e) => setF(Object.assign({}, f, { [k]: e.target.type === "checkbox" ? e.target.checked : e.target.value }));
  const setTab = (k) => { setTabState(k); saved.set("models_tab", k); setMsg(""); };
  // `next`: the list, or what makes it of the list as it is by then (an answer that took long to come).
  const setFolders = (next) => setFoldersState((old) => {
    const list = typeof next === "function" ? next(old) : next;
    saved.set("folders", list.length ? JSON.stringify(list) : null);
    return list;
  });
  const [picking, setPicking] = useState(false);  // the desktop's folder dialog is open, on the server's machine

  const winPaths = catalog && typeof catalog.win_paths === "boolean" ? catalog.win_paths : null;
  const nPath = (p) => normPath(p, winPaths);
  const eqPath = (a, b) => samePath(a, b, winPaths);
  const underPath = (p, d) => pathUnder(p, d, winPaths);

  const hubEndpoint = (hubCfg && hubCfg.endpoint) || (found && found.endpoint) || "https://huggingface.co";
  const hubDefaultEndpoint = (hubCfg && hubCfg.default_endpoint) || "https://huggingface.co";
  const isCustomEndpoint = hubEndpoint.replace(/\/+$/, "").toLowerCase() !== hubDefaultEndpoint.replace(/\/+$/, "").toLowerCase();
  const isHub = (v) => hubName(v, hubEndpoint);
  const repoNameOf = (v) => repoOf(v, hubEndpoint);

  function clearHubDetail() {
    repoSeq.current++;
    dispatchHub({ type: "clear" });
    setTarget("");
  }

  async function refreshHubConfig() {
    const r = await call("GET", "/v1/hub/config");
    if (r.ok && r.json) setHubCfg(r.json);
  }
  async function applyEndpoint(raw) {
    if (epSaving) return;
    setEpSaving(true);
    setEpMsg("");
    const r = await call("POST", "/v1/hub/config", { endpoint: raw });
    setEpSaving(false);
    if (!r.ok) {
      setEpMsg(r.status === 404 && !r.json ? "This server cannot change the Hub endpoint yet: update FLYX." : r.error);
      return;
    }
    if (r.json) {
      setHubCfg(r.json);
      setEpText(r.json.endpoint || "");
    }
    search(q, sort, arch, 0, "");
    if (hubView.sel) showRepo(hubView.sel, hubView.quant, hubView.rev);
  }
  async function findLoras(base, modelName) {
    const n = ++loraSeq.current;
    setLoraHubMsg("Asking Hugging Face ...");
    try {
      const q = base ? "base=" + encodeURIComponent(base) : "model=" + encodeURIComponent(modelName || "");
      const d = await getJson("/v1/hub/loras?" + q + "&limit=30");
      if (n !== loraSeq.current) return;
      setLoraHub(d);
      setLoraHubMsg("");
      if (!base) setLoraBaseDraft(d.base || "");  // the model's base, or nothing for one whose base is not known
    } catch (e) {
      if (n === loraSeq.current) { setLoraHub(null); setLoraHubMsg(e.message); }
    }
  }
  async function openLoraRepo(id) {
    if (loraRepo && loraRepo.id === id) return setLoraRepo(null);
    setLoraRepo({ id, loading: true });
    try {
      const d = await getJson("/v1/hub/lora?repo=" + encodeURIComponent(id));
      setLoraRepo((cur) => cur && cur.id === id ? Object.assign({ id }, d) : cur);
    } catch (e) {
      setLoraRepo((cur) => cur && cur.id === id ? { id, err: e.message } : cur);
    }
  }
  async function refreshCached() {
    try { setCatalog(await getJson("/v1/hub/cached")); setCatalogErr(""); } catch (e) { setCatalogErr(e.message); }
  }
  // ---- Hugging Face ----
  async function search(text, by, archFilter = "", pageIdx = 0, cursorStr = "") {
    const n = ++searchSeq.current;
    setFoundMsg("Asking Hugging Face ...");
    const word = hubName(text, hubEndpoint) ? repoOf(text, hubEndpoint) : text.trim();  // a name with its quantization or revision: its repository
    let path = "/v1/hub/search?limit=20&sort=" + by + "&q=" + encodeURIComponent(word);
    if (!word && archFilter) path += "&arch=" + encodeURIComponent(archFilter);
    if (cursorStr) path += "&cursor=" + encodeURIComponent(cursorStr);
    const a = await ask(path, "search Hugging Face");
    if (n !== searchSeq.current) return;
    if (a.error) {
      setFound(null);
      setFoundMsg(a.error);
      const cur = hubViewRef.current;
      const pinned = !!(cur && cur.pinned && cur.pinned === cur.sel);
      if (!hubName(text, hubEndpoint) && !pinned) clearHubDetail();
      return;
    }
    setFound(a.j);
    setFoundMsg("");
    setPage(pageIdx);
    setCursors((old) => {
      const next = pageIdx === 0 ? [""] : old.slice(0, pageIdx + 1);
      if (a.j.next_cursor) next[pageIdx + 1] = a.j.next_cursor;
      return next;
    });
  }
  function goHubPage(nextPage) {
    if (nextPage < 0) return;
    const cur = nextPage === 0 ? "" : cursors[nextPage];
    if (nextPage > 0 && !cur) return;
    dispatchHub({ type: "unpin" });
    const listEl = document.getElementById("hf-list");
    if (listEl) listEl.scrollTop = 0;
    search(q, sort, arch, nextPage, cur);
  }
  // A repository's files and what the Hub says of it.  `quant`: the file to choose once they are known; `rev`: optional @revision.
  async function showRepo(id, quant = "", rev = "", pinned = undefined) {
    const n = ++repoSeq.current;
    const wantQuant = quant || "";
    const wantRev = rev && rev !== "main" ? rev : "";
    dispatchHub({ type: "load", sel: id, quant: wantQuant, rev: wantRev, pinned });
    setTarget("");
    setMsg("");
    const a = await ask("/v1/hub/files?repo=" + encodeURIComponent(repoSpec(id, wantRev)), "list a repository");
    if (n !== repoSeq.current) return;
    if (a.error) { dispatchHub({ type: "error", error: a.error }); return; }
    dispatchHub({ type: "ok", sel: id, rev: wantRev, j: a.j });
    const files = (a.j.ggufs || []).filter(whole);
    const hit = wantQuant ? files.find((g) => String(g.quant).toLowerCase() === wantQuant.toLowerCase()) : (a.j.ggufs || []).length === 1 ? files[0] : null;
    if (hit) setTarget(hit.name);
  }
  // The search box also takes a repository's whole name (org/name, org/name:QUANT[@REV], its address): shown at once.
  function named(text) {
    const v = text.trim();
    if (!hubName(v, hubEndpoint)) return false;
    const clean = stripHubPrefix(v, hubEndpoint);
    showRepo(repoOf(v, hubEndpoint), quantOf(clean.split("@")[0]) || "", revOf(v, hubEndpoint), false);
    return true;
  }
  useEffect(() => {
    if (!active || view !== "library" || tab !== "hub") return;
    refreshHubConfig();
    const h = setTimeout(() => {
      search(q, sort, arch, 0, "");
      if (hubName(q, hubEndpoint)) named(q);
    }, found || foundMsg ? 350 : 0);
    return () => clearTimeout(h);
  }, [active, view, tab, q, sort, arch]);
  // The repository shown: the served model's or first found on initial load, and the top match whenever the displayed list no longer contains the selected repo.
  useEffect(() => {
    if (view !== "library" || tab !== "hub" || !found) return;
    const qText = (found.query || "").trim();
    const direct = qText && hubName(qText, hubEndpoint) ? repoOf(qText, hubEndpoint) : "";
    const pinned = !!hubView.pinned && hubView.pinned === hubView.sel;
    if (!found.models.length) {
      if ((!direct || direct !== hubView.sel) && !pinned) clearHubDetail();
      return;
    }
    const served = !qText && !arch && loadedWant && isHub(loadedWant.model) ? repoNameOf(loadedWant.model) : "";
    if (!hubView.sel) {
      const servedHere = served && (!hubPickedOnce.current || found.models.some((m) => m.id === served));
      const first = servedHere ? served : (found.models[0] && found.models[0].id);
      if (first) {
        hubPickedOnce.current = true;
        showRepo(first, "", servedHere ? revOf(loadedWant.model, hubEndpoint) : "", false);
      }
      return;
    }
    if (found.models.some((m) => m.id === hubView.sel)) return;
    if (direct && direct === hubView.sel) return;
    if (pinned) return;
    showRepo(found.models[0].id, "", "", false);
  }, [found, tab, view]);

  // ---- this machine ----
  async function openPath(p) {
    const n = ++dirSeq.current;
    setLoc(p);
    setPathText(p);
    setLocalDraft("");
    setLocalMmproj("");
    setMsg("");
    // The folder Hugging Face downloads to is shown by its repositories, as the server lists them, not by its files as
    // they lie in it.
    if (catalog && p && eqPath(p, catalog.cache)) { setDir(null); setDirMsg(""); return; }
    setDirMsg("Looking...");
    const a = await ask("/v1/hub/dir?path=" + encodeURIComponent(p), "browse folders");
    if (n !== dirSeq.current) return;
    if (a.error) { setDir(null); setDirMsg(a.error); return; }
    setDir(a.j);
    setDirMsg("");
    setLoc(a.j.path);
    setPathText((cur) => cur === p ? a.j.path : cur);
    if (a.j.kind === "file" && a.j.models.length === 1 && standalone(a.j.models[0])) setTarget(a.j.models[0].name);
  }
  const kept = folders.some((x) => eqPath(x, loc));
  const keep = () => setFolders(kept ? folders.filter((x) => !eqPath(x, loc)) : [...folders, loc]);
  // A folder for the list, chosen in the desktop's own folder dialog.  The server shows it, on its machine: a page
  // cannot learn from the browser's where a folder is.  Where none can be shown (the browser is on another machine,
  // there is no desktop), the folders are browsed in the page instead.
  async function addFolder() {
    if (picking) return;
    setPicking(true);
    setMsg("");
    const r = await call("POST", "/v1/hub/dir", { title: t("Choose a folder of model files"), start: loc });
    setPicking(false);
    if (r.ok) {
      const j = r.json;
      if (!j || !j.path) return;  // the dialog was closed without a choice
      setFolders((old) => known.has(nPath(j.path)) || old.some((x) => eqPath(x, j.path)) ? old : [...old, j.path]);
      dirSeq.current++;
      setLoc(j.path);
      setPathText(j.path);
      setDir(j);
      setPicked("");
      setListings((old) => Object.assign({}, old, { [j.path]: { models: j.models } }));
      setDirMsg("");
      return;
    }
    setPicked("");  // the folders, at the right, in place of a model
    await openPath("");
    setDirMsg(r.status === 409 ? r.error : "The system's folder dialog cannot be shown from here: type the folder's path, or browse to it.");
  }
  // The first folder shown: the served model's when it is a file, else a configured --models-dir when the cache has
  // none, else the Hugging Face cache.
  useEffect(() => {
    if (placed.current || view !== "library" || tab !== "local" || !catalog) return;
    placed.current = true;
    const cfgDirs = (Array.isArray(catalog.dirs) ? catalog.dirs : []).filter(Boolean);
    const hasSep = (s) => winPaths === false ? s.includes("/") : /[\\/]/.test(s);
    const served = loadedWant && !isHub(loadedWant.model) && hasSep(loadedWant.model) ? folderOf({ name: loadedWant.model }, winPaths) : "";
    const firstHub = catalog.models.find((m) => m.source === "hub" && standalone(m));
    const firstDir = catalog.models.find((m) => m.source === "dir" && standalone(m));
    const dirLoc = firstDir ? (firstDir.root || cfgDirs.find((d) => underPath(folderOf(firstDir, winPaths), d)) || folderOf(firstDir, winPaths)) : (cfgDirs[0] || "");
    const initialLoc = served || (!firstHub && dirLoc) || catalog.cache;
    openPath(initialLoc);
    const first = loadedWant ? loadedWant.model : ((firstHub || firstDir) || {}).name;
    if (first) { setPicked(first); setTarget(first); }
  }, [catalog, tab, view]);

  useEffect(() => { if (active) refreshCached(); }, [active]);
  useEffect(() => {
    if (tm.device && tm.device.mem_total && tm.device.mem_total !== gpuTotal) {
      gpuTotal = tm.device.mem_total;
      saved.set("gpu_total", String(gpuTotal));
    }
    // Decided on the first answer this draw was made from: `model` is that draw's, and telemetry may have come since.
    if (!viewed.current && "model" in tm) {
      viewed.current = true;
      if (!model.id && !busy && !L.error) setView("library");  // a load that failed shows why first
    }
    // The settings start from what is loaded, else what the server remembers; and follow a model loaded since, from
    // this page or from elsewhere.
    const now = loadedWant ? JSON.stringify(loadedWant) : "";
    if (!filled.current && (L.loaded || L.remembered)) { setF(formOf(L.loaded || L.remembered)); filled.current = true; }
    else if (now && now !== sawLoaded.current && !busy) setF(formOf(loadedWant));
    if (!busy) sawLoaded.current = now;
    if (sawState.current && busyLoad({ state: sawState.current }) && !busy) refreshCached();
    sawState.current = L.state;
  }, [live.last]);

  const entryOf = (name) => catalog && catalog.models.find((m) => m.name === name);
  async function removeCached(m) {
    const d = describe(m, winPaths);
    const ok = await confirmBox("Delete " + (d.quant || d.file || m.name) + "?",
      [(d.file || m.name) + " (" + gb(m.size) + " GB) is removed from the Hugging Face cache, freeing its space.",
       "It can be downloaded again by loading the same name."], "Delete", true);
    if (!ok) return;
    const r = await call("DELETE", "/v1/hub/cached?name=" + encodeURIComponent(m.name));
    if (!r.ok) { setMsg(r.status === 404 && !r.json ? "This server cannot delete files yet." : r.error); return; }
    if (target === m.name) setTarget("");
    if (picked === m.name) setPicked("");
    refreshCached();
    if (hubView.detail) showRepo(hubView.detail.repo, "", hubView.detail.rev);
  }

  // Every file the cache holds of a repository, deleted: `list`, its models as the server lists them.
  async function removeRepo(name, list) {
    const ok = await confirmBox("Delete " + name + "?",
      [list.length + (list.length === 1 ? " file" : " files") + " (" + sized(sum(list)) + ") are removed from the Hugging Face cache, freeing their space.",
       "It can be downloaded again by loading the same name."], "Delete", true);
    if (!ok) return;
    let failed = "";
    for (const m of list) {
      const r = await call("DELETE", "/v1/hub/cached?name=" + encodeURIComponent(m.name));
      if (!r.ok) failed = r.status === 404 && !r.json ? "This server cannot delete files yet." : r.error;
      else if (target === m.name) setTarget("");
    }
    if (failed) await confirmBox("Not everything was deleted", [failed], "OK");
    refreshCached();
    if (hubView.detail) showRepo(hubView.detail.repo, "", hubView.detail.rev);
  }

  // ---- the settings a load takes: those shown with the model served (a change is compared with what it has); with
  // none served there are none to show, and a load takes the defaults ----
  const use = model.id ? f : EMPTY_FORM;
  const diffShown = (a, b) => Object.keys(b).filter((k) => k === "drafter"
    ? a[k].length !== b[k].length || a[k].some((value, i) => value !== b[k][i]) : a[k] !== b[k]);
  const before = loadedWant ? shown(loadedWant) : null, now = shown(loadBody(loadedWant ? loadedWant.model : "", use, Boolean(loadedWant)));
  const changed = before ? diffShown(before, now) : [];
  const was = (k) => changed.includes(k) ? (k === "drafter" ? before[k].join(" · ") : before[k]) : null;
  const advChanged = ["width", "head", "weights", "lookup", "trees"].filter((k) => was(k) !== null);
  useEffect(() => { if (advChanged.length) setAdvOpen(true); }, [advChanged.length]);
  // Companions (custom drafter, vision projector) and a custom API id belong to the model served: do not carry them
  // onto a different model unless the user edited or picked them for the next load.
  const keepDraft = changed.includes("drafter") || !!use.draftPicked;
  const keepMmproj = changed.includes("mmproj") || !!use.mmprojPicked;
  const keepName = changed.includes("name");
  const keepLora = changed.includes("lora") || !!use.loraPicked;
  function settingsFor(name, overrides) {
    const next = Object.assign({}, use);
    if (loadedWant && name && name !== loadedWant.model) {
      if (!keepDraft) { next.draftSource = "default"; next.savedDraft = null; next.drafter = ""; next.drafterKind = ""; }
      if (!keepMmproj) { next.visionSource = "default"; next.mmproj = ""; }
      if (!keepName) next.name = "";
      if (!keepLora) next.lora = "";
    }
    if (overrides && overrides.drafter) Object.assign(next, pickedDrafter(overrides.drafter, overrides.drafterKind));
    if (overrides && overrides.mmproj) {
      next.vision = true;
      next.visionSource = "custom";
      next.mmproj = overrides.mmproj;
      next.mmprojPicked = true;
    }
    return next;
  }
  const body = (name, overrides) => loadBody(name, settingsFor(name, overrides), Boolean(loadedWant));

  // Loads `name` with the settings; or `want`, a load the server remembers, as it was.
  async function loadModel(name, want, overrides) {
    if (!name) { setMsg("Choose a model first."); return; }
    const same = !!loadedWant && name === loadedWant.model;
    const st = tm.state || {}, served = model.id;
    if (served) {
      const lines = [];
      const n = (st.running || 0) + (st.queued || 0);
      if (n) lines.push(n + (n === 1 ? " request is" : " requests are") + " running or waiting: " + (n === 1 ? "it is" : "they are") + " stopped.");
      lines.push(served + " is unloaded first; its GPU memory comes back before the new one loads.");
      if (isHub(name) && !entryOf(name)) lines.push("The files are fetched first, while " + served + " keeps answering.");
      lines.push("If the new one fails to load, no model is served until another is loaded.");
      if (!await confirmBox(same ? "Reload " + served + " with these settings?" : "Switch to " + (describe(entryOf(name) || { name }, winPaths).quant || name) + "?",
                            lines, same ? "Reload" : "Switch")) return;
    }
    setMsg("");
    const reqBody = want || body(name, overrides);
    lastAttempt.current = reqBody;
    const r = await call("POST", "/v1/load", reqBody);
    if (!r.ok) { setMsg(r.error); return; }
    dispatchHub({ type: "clearCompanions" });
    setLocalDraft("");
    setLocalMmproj("");
    setView("current");  // where the fetch's and the load's progress show
    document.getElementById("main").scrollTo({ top: 0 });
    poll();
  }

  // The default drafter, as the settings say it: what the server loaded with the model served, when that drafts with
  // its default; else what a model brings along.
  function defaultDrafter() {
    const activeDraft = model.drafter && model.drafter !== "none" ? model.drafter : "";
    if (loadedWant && loadedWant.speculative !== false && !loadedWant.drafter && !loadedWant.drafters?.length) {
      const list = draftersOf(L).map((d) => drafterLabel(d.kind) + " · " +
        (d.from === "model file" ? t("the model file's own layer") : t("from " + d.from + (d.size ? " (" + gb(d.size) + " GB)" : ""))));
      if (!list.length && activeDraft)
        list.push(activeDraft.split("+").map(drafterLabel).join(" + ") + " · " + t("the model file's own layer"));
      return list.length ? { tag: "in use", tone: "ok", lines: list } : { tag: "none", tone: "", lines: [t("This model loaded no drafter: plain decoding.")] };
    }
    return { tag: "", tone: "", lines: [t("The model's own MTP layer, if it has one.")] };
  }
  const dflt = defaultDrafter(), own = f.drafter.trim();
  const ownCached = !!own && !!catalog && catalog.models.some((x) => x.kind === "drafter" && (x.name === own || x.path === own));
  const specOff = !f.spec, custom = f.draftSource === "custom";
  const savedDraft = f.draftSource === "saved" ? f.savedDraft : draftingOf(loadedWant || L.remembered || {}).savedDraft;
  const pickDraft = (source) => { if (!specOff) setF(Object.assign({}, f, { draftSource: source, savedDraft, draftPicked: false })); };
  // A drafter named by the page (one of the cache, one published for a repository's model) goes to the settings.
  const draftWith = (name, kind) => {
    setF((current) => Object.assign({}, current, pickedDrafter(name, kind)));
    setView("current");
    setStab("spec");
    setTimeout(() => { const row = document.getElementById("sd-custom"); if (row) row.scrollIntoView({ behavior: "smooth", block: "center" }); }, 60);
  };
  const projectWith = (path) => {
    setF((current) => Object.assign({}, current, { vision: true, visionSource: "custom", mmproj: path, mmprojPicked: true }));
    setView("current");
    setStab("vision");
    setTimeout(() => { const row = document.getElementById("vs-custom"); if (row) row.scrollIntoView({ behavior: "smooth", block: "center" }); }, 60);
  };
  const loraWith = (path) => {
    setF((current) => Object.assign({}, current, { lora: path, loraPicked: true }));
    setView("current");
    setStab("lora");
    setTimeout(() => { const row = document.getElementById("mf-lora"); if (row) row.scrollIntoView({ behavior: "smooth", block: "center" }); }, 60);
  };
  const applyCompanions = (draft, mmprojSpec) => {
    setF((current) => Object.assign({}, current,
      pickedDrafter(draft.spec, draft.kind),
      mmprojSpec ? { vision: true, visionSource: "custom", mmproj: mmprojSpec, mmprojPicked: true } : {}));
    setView("current");
    setStab("spec");
  };
  const visOff = !f.vision, visCustom = f.visionSource === "custom", mmOwn = f.mmproj.trim();
  const pickVision = (source) => { if (!visOff) setF(Object.assign({}, f, { visionSource: source, mmprojPicked: false })); };
  const visDefault = () => {
    if (loadedWant && loadedWant.vision !== false && !loadedWant.mmproj) {
      if (model.vision && L.vision_projector && L.vision_projector.from)
        return { tag: "in use", tone: "ok", line: t("from " + L.vision_projector.from + (L.vision_projector.size ? " (" + gb(L.vision_projector.size) + " GB)" : "")) };
      if (model.vision) return { tag: "in use", tone: "ok", line: t("Loaded from the model file.") };
      return { tag: "none", tone: "", line: t("No vision projector loaded: text-only.") };
    }
    return { tag: "", tone: "", line: t("Auto-loads mmproj from Hugging Face if available.") };
  };
  const vdflt = visDefault();
  const genChg = ["ctx", "kv", "kv_ram", "kv_gpu", "vram_reserve", "ram_reserve", "name"].filter((k) => changed.includes(k)).length;
  const specChg = ["spec", "drafter", "width", "head", "weights", "lookup", "trees"].filter((k) => changed.includes(k)).length;
  const loraChg = changed.includes("lora") ? 1 : 0;
  const visChg = ["vision", "mmproj", "vision_device", "vision_quant", "vision_max_tokens"].filter((k) => changed.includes(k)).length;

  function settingsView(cmdModel) {
    const activeDraft = model.drafter && model.drafter !== "none" ? model.drafter : "";
    const specLoaded = !!activeDraft, visLoaded = !!model.vision;
    // The drafters verified for the model the settings are for: a Hub repository's (GET /v1/hub/files), the served
    // model's (what its load found), a file of this machine's (the library's entry for it).
    const servedLocal = !cmdRepo && loadedWant && cmdModel === loadedWant.model;
    const entry = !cmdRepo && !servedLocal
      ? [...(catalog ? catalog.models : []), ...(dir ? dir.models : [])].find((m) => m.name === cmdModel) : null;
    const verified = cmdRepo ? verifiedOf[cmdRepoKey] || []
      : servedLocal ? L.suggested || []
      : (entry && entry.known_drafters) || [];
    const ownVerified = custom && verified.some((k) => k.repo === own);
    const madeFor = verified.filter((k) => k.exact), inferred = verified.filter((k) => !k.exact);
    const ownInferred = custom && inferred.some((k) => k.repo === own), ownMade = custom && madeFor.some((k) => k.repo === own);
    // A verified drafter (made for the model, or for its base model) as a choice: its kind, its speed here, its page.
    const drafterRow = (k, radio) => {
      const sel = custom && own === k.repo;
      return html`<div class=${"sdrow pick" + (sel ? " sel" : "")} data-verified=${k.repo}
                       onClick=${(e) => { e.stopPropagation(); if (!specOff) setF(Object.assign({}, f, { draftSource: "custom", savedDraft: null,
                                          drafter: k.repo, drafterKind: k.kind, draftPicked: true })); }}>
        <input type="radio" name=${radio} aria-label=${k.repo} checked=${sel} disabled=${specOff}/>
        <div class="grow">
          <div class="dhead"><b>${drafterLabel(k.kind)}</b><span class="tag2 ok">${t("verified")}</span>
            <span class="tag2" title=${t("Your own runs with it on this machine, on average (N/A: none yet).")}>${t("speed")}: ${k.tok_s ? Math.round(k.tok_s) + " tok/s" : "N/A"}</span></div>
          <a class="ddesc" href=${safeOrigin(hubEndpoint) + "/" + k.repo} target="_blank" rel="noopener noreferrer"
             onClick=${(e) => e.stopPropagation()}>${k.repo}</a>
        </div>
      </div>`;
    };
    const activeDrafters = model.loaded && !busy ? draftersOf(L).slice() : [];
    if (model.loaded && !busy && activeDraft && /mtp/.test(activeDraft) && !activeDrafters.some((d) => d.kind === "mtp"))
      activeDrafters.unshift({ kind: "mtp", from: "model file" });
    const specSum = !f.spec ? t("off") : f.draftSource === "custom" && own ? baseName(own, winPaths) : f.draftSource === "saved" ? t("last loaded") :
                    specLoaded ? t("default") : t("none");
    const visSum = !f.vision ? t("off") : visLoaded ? t(normVisionDev(f.vision_device)) + " · " + normVisionQuant(f.vision_quant) :
                   visCustom && mmOwn ? baseName(mmOwn, winPaths) : t("none");
    const loraName = (p) => baseName(trimTrailingSep(p, winPaths), winPaths);
    const loraOwn = (f.lora || "").trim(), loraLoaded = !!(loadedWant && loadedWant.lora);
    const adapters = catalog ? catalog.models.filter((m) => m.kind === "lora") : [];
    const adapterOf = (m) => m.source === "hub" ? m.name : m.path || m.name;
    return html`
      <div class="pane lset" id="mf-settings">
        <div class="lset-h">
          <div><h2>${t("Load settings")}</h2>
            <p class="lead">${t(loadedWant ? "What the next model loads with. The model being served takes a change when it is reloaded."
                                           : "What the next model loads with.")}</p></div>
          ${changed.length ? html`<span class="was">${t(changed.length + " changed")}</span>
            <button class="linkbtn" id="mf-reset" onClick=${() => { setF(formOf(loadedWant)); setMsg(""); }}>${t("Reset to what is loaded")}</button>
            <button class="btn primary" id="mf-reload" disabled=${busy} onClick=${() => loadModel(loadedWant.model)}>
              ${t("Reload with these settings")}</button>` : ""}
        </div>
        ${msg ? html`<div class="banner bad" id="mf-setmsg" style="margin-bottom: 12px">${t(msg)}</div>` : ""}
        <div class="lset-nav">
          <div class="seg lset-seg" role="tablist" id="lset-tabs">
            <button role="tab" data-stab="general" aria-selected=${String(stab === "general")} onClick=${() => setStab("general")}>
              <span>${t("Configuration")}</span>
              ${genChg ? html`<span class="chg">+${genChg}</span>` : ""}
            </button>
            <button role="tab" data-stab="spec" aria-selected=${String(stab === "spec")} onClick=${() => setStab("spec")}>
              <span class=${"dot" + (specLoaded && f.spec ? " on" : "")}></span>
              <span>${t("Speculative decoding")}</span>
              <span class="sum">${specSum}</span>
              ${specChg ? html`<span class="chg">+${specChg}</span>` : ""}
            </button>
            <button role="tab" data-stab="vision" aria-selected=${String(stab === "vision")} onClick=${() => setStab("vision")}>
              <span class=${"dot" + (visLoaded && f.vision ? " on" : "")}></span>
              <span>${t("Vision encoder")}</span>
              <span class="sum">${visSum}</span>
              ${visChg ? html`<span class="chg">+${visChg}</span>` : ""}
            </button>
            <button role="tab" data-stab="lora" aria-selected=${String(stab === "lora")} onClick=${() => setStab("lora")}>
              <span class=${"dot" + (loraLoaded && loraOwn ? " on" : "")}></span>
              <span>${t("LoRA")}</span>
              <span class="sum">${loraOwn ? loraName(loraOwn) : t("none")}</span>
              ${loraChg ? html`<span class="chg">+${loraChg}</span>` : ""}
            </button>
          </div>
        </div>
        <div class="lset-panel" id="lset-general" role="tabpanel" hidden=${stab !== "general"}>
          <div class="pane-h"><h2>${t("Configuration")}</h2></div>
          <div class="opts c3">
            <${Field} k="ctx" label="Context length" was=${was("ctx")} hint="Tokens, or empty (auto) for the model's full context, or the longest the GPU and host memory hold.">
              <input type="text" id="mf-ctx" inputmode="numeric" autocomplete="off" spellcheck="false" placeholder=${t("auto")}
                     value=${f.ctx} onInput=${set("ctx")}/><//>
            <${Field} k="kv" label="KV cache" was=${was("kv")} hint="fp8 holds twice the context.">
              <select id="mf-kv" value=${f.kv} onChange=${set("kv")}><option value="">${t("auto")}</option><option value="fp8">fp8</option><option value="16">${t("16-bit")}</option></select><//>
            <${Field} k="kv_ram" label="KV in host RAM" was=${was("kv_ram")} hint="GB, for context past the GPU.">
              <input type="number" id="mf-kvram" min="0" step="1" placeholder=${t("auto")} value=${f.kvram} onInput=${set("kvram")}/><//>
            <${Field} k="kv_gpu" label="KV on the GPU" was=${was("kv_gpu")} hint="Tokens, at most.">
              <input type="number" id="mf-kvgpu" min="0" step="1024" placeholder=${t("as many as fit")} value=${f.kvgpu} onInput=${set("kvgpu")}/><//>
            <${Field} k="vram_reserve" label="VRAM reserve" was=${was("vram_reserve")} hint="GB left free on the GPU (0: minimum allowed by the backend/OS; empty: auto).">
              <input type="number" id="mf-vramres" min="0" step="0.25" placeholder=${t("auto")} value=${f.vramres} onInput=${set("vramres")}/><//>
            <${Field} k="ram_reserve" label="RAM reserve" was=${was("ram_reserve")} hint="GB kept free in host RAM (0: use all RAM; empty: auto).">
              <input type="number" id="mf-ramres" min="0" step="0.5" placeholder=${t("auto")} value=${f.ramres} onInput=${set("ramres")}/><//>
            <${Field} k="name" label="API model id" was=${was("name")}>
              <input type="text" id="mf-name" placeholder=${t("the file's name")} spellcheck="false" autocomplete="off" value=${f.name} onInput=${set("name")}/><//>
          </div>
        </div>
        <div class="lset-panel" id="lset-lora" role="tabpanel" hidden=${stab !== "lora"}>
          <div class="pane-h"><h2>${t("LoRA adapter")}</h2></div>
          ${loraLoaded && !busy ? html`<div class="srv-dr in-panel" id="ld-lora-src">
            <div class="dr"><span class="tag2 ok">${t("in use")}</span><span class="k">${t("LoRA")}</span>
              <span class="f">${loadedWant.lora}</span></div></div>` : null}
          <div class="opts">
            <${Field} k="lora" label="LoRA adapter" was=${was("lora")} hint="A PEFT adapter's folder on the server's machine, or a Hugging Face name; empty: none.">
              <input type="text" id="mf-lora" placeholder=${t("none")} spellcheck="false" autocomplete="off" value=${f.lora}
                     onInput=${(e) => setF(Object.assign({}, f, { lora: e.target.value, loraPicked: false }))}/><//>
          </div>
          ${adapters.length ? html`<div class="sdopts" id="lora-list">
            <div class=${"sdrow pick" + (!loraOwn ? " sel" : "")} data-lora="" onClick=${() => setF(Object.assign({}, f, { lora: "", loraPicked: true }))}>
              <input type="radio" name="lora-src" aria-label=${t("None")} checked=${!loraOwn}/>
              <div class="grow"><div class="dhead"><b>${t("None")}</b></div><em class="sd-what">${t("The model alone.")}</em></div>
            </div>
            ${adapters.map((m) => html`<div class=${"sdrow pick" + (loraOwn === adapterOf(m) ? " sel" : "")} data-lora=${adapterOf(m)}
                 onClick=${() => setF(Object.assign({}, f, { lora: adapterOf(m), loraPicked: true }))}>
              <input type="radio" name="lora-src" aria-label=${loraName(m.name)} checked=${loraOwn === adapterOf(m)}/>
              <div class="grow"><div class="dhead"><b>${m.source === "hub" ? m.name : loraName(m.path || m.name)}</b>
                <span class="tag2">${m.size ? gb(m.size) + " GB" : ""}</span></div>
                <em class="sd-what">${m.path || m.name}</em></div>
            </div>`)}
          </div>` : html`<p class="note" id="lora-none">${t("No LoRA adapter in the library's folders; type its folder or Hugging Face name above.")}</p>`}
          <div class="pane-h lora-hub-h"><h3>${t("On Hugging Face")}</h3></div>
          <form class="lora-base" onSubmit=${(e) => { e.preventDefault(); const b = loraBaseDraft.trim(); setLoraRepo(null);
                                                       if (b === loraBase || (!loraBase && loraHub && b === loraHub.base)) findLoras(loraBase, cmdModel); else setLoraBase(b); }}>
            <label for="lora-base">${t("Adapters made for")}</label>
            <input type="text" id="lora-base" placeholder="org/name" spellcheck="false" autocomplete="off" value=${loraBaseDraft}
                   onInput=${(e) => setLoraBaseDraft(e.target.value)}/>
            <button class="btn" id="lora-find" type="submit">${t("Find")}</button>
            ${loraHub && loraHub.base && !loraBase && loraHub.from === "model card" ? html`<span class="note">${t("the model card's base")}</span>` : ""}
          </form>
          ${loraHubMsg ? html`<p class="note" id="lora-hub-msg">${t(loraHubMsg)}</p>` :
            loraHub && !loraHub.base ? html`<p class="note" id="lora-hub-msg">${t("The model's base is not known: type the Hugging Face model the adapters were trained on.")}</p>` :
            loraHub && !loraHub.models.length ? html`<p class="note" id="lora-hub-msg">${t("Hugging Face lists no LoRA adapter for this model.")}</p>` : ""}
          ${loraHub && loraHub.models.length ? html`<div class="lora-hub" id="lora-hub">
            ${loraHub.models.map((m) => html`<div class=${"lora-repo" + (loraRepo && loraRepo.id === m.id ? " open" : "")} data-repo=${m.id}>
              <button type="button" class="lora-repo-h" onClick=${() => openLoraRepo(m.id)}>
                <b>${m.id}</b>${m.cached ? html`<span class="tag2 ok">${t("downloaded")}</span>` : ""}${m.gated ? html`<span class="tag2 warn">${t("gated")}</span>` : ""}
                <span class="pm"><${Icon} name="heart"/>${compact(m.likes)}<${Icon} name="dl"/>${compact(m.downloads)}</span></button>
              ${loraRepo && loraRepo.id === m.id ? html`<div class="lora-ads">
                ${loraRepo.loading ? html`<p class="note">${t("Reading its adapters ...")}</p>` :
                  loraRepo.err ? html`<p class="note bad">${t(loraRepo.err)}</p>` :
                  !(loraRepo.adapters || []).length ? html`<p class="note">${t("No adapter_config.json in this repository.")}</p>` :
                  loraRepo.adapters.map((a) => html`<div class=${"lora-ad" + (a.why ? " no" : "") + (loraOwn === a.name ? " sel" : "")} data-name=${a.name}>
                    <div class="grow">
                      <div class="dhead"><b>${a.folder || t("(top folder)")}</b>
                        ${a.rank ? html`<span class="tag2">r${a.rank}</span>` : ""}${a.size ? html`<span class="tag2">${gb(a.size)} GB</span>` : ""}</div>
                      <em class="sd-what">${a.why ? t("Cannot load") + ": " + a.why :
                        (a.layers ? a.layers + " " + t("layers") + " · " : "") + (a.linears || []).join(", ") + (a.base ? " · " + t("trained on") + " " + a.base : "")}</em>
                    </div>
                    ${a.why ? "" : html`<button type="button" class="btn" disabled=${loraOwn === a.name}
                        onClick=${() => setF(Object.assign({}, f, { lora: a.name, loraPicked: true }))}>${loraOwn === a.name ? t("Chosen") : t("Use")}</button>`}
                  </div>`)}
              </div>` : ""}
            </div>`)}
          </div>` : ""}
        </div>
        <div class="lset-panel" id="lset-spec" role="tabpanel" hidden=${stab !== "spec"}>
          <div class="pane-h"><h2>${t("Speculative decoding")}</h2>
            <span class=${"sd-was" + (was("spec") !== null ? " on" : "")}>${was("spec") !== null ? t("was " + t(was("spec"))) : ""}</span>
            <label class="switch sd-switch" title=${t("Speculative decoding")}>
              <input type="checkbox" id="sd-on" aria-label=${t("Speculative decoding")} checked=${f.spec} onChange=${set("spec")}/><i></i></label></div>
          ${activeDrafters.length ? html`<div class="srv-dr in-panel" id="ld-drafters">${activeDrafters.map((d) => html`
            <div class="dr"><span class="tag2 ok">${t("in use")}</span><span class="k">${drafterLabel(d.kind)}</span>
              <span class="f">${(d.from === "model file" ? t("the model file's own layer") : d.from) + (d.size ? " · " + gb(d.size) + " GB" : "")}</span></div>`)}</div>` : null}
          <div class=${"sdopts" + (f.spec ? "" : " off")} data-k="drafter">
            ${savedDraft ? html`<div class=${"sdrow pick" + (f.draftSource === "saved" ? " sel" : "")} id="sd-saved" onClick=${() => pickDraft("saved")}>
              <input type="radio" name="sd-src" aria-label=${t("last loaded")} checked=${f.draftSource === "saved"} disabled=${specOff}/>
              <div class="grow"><div class="dhead"><b>${t("last loaded")}</b><span class="tag2">${savedDraft.drafter || t("default")}</span></div>
                ${savedDraft.drafters.map((file) => html`<span class="ddesc">${file}</span>`)}</div></div>` : ""}
            <div class=${"sdrow pick" + (f.draftSource === "default" ? " sel" : "")} id="sd-default" onClick=${() => pickDraft("default")}>
              <input type="radio" name="sd-src" aria-label=${t("Default")} checked=${f.draftSource === "default"} disabled=${specOff}/>
              <div class="grow">
                <div class="dhead"><b>${t("Default")}</b>${dflt.tag ? html`<span class=${"tag2 " + dflt.tone}>${t(dflt.tag)}</span>` : ""}</div>
                ${dflt.lines.map((l) => html`<em class="sd-what">${l}</em>`)}
              </div>
            </div>
            ${[[madeFor, "sd-made", ownMade, madeOpen, setMadeOpen, t("Drafters made for this model"), [],
                [...new Set(madeFor.map((k) => k.base))].join(", ")],
               [inferred, "sd-base", ownInferred, baseOpen, setBaseOpen, t("Inferred from the base model the model names"),
                [...new Set(inferred.map((k) => k.relation).filter(Boolean))], [...new Set(inferred.map((k) => k.base))].join(", ")]]
              .map(([list, id, picked, open, setOpen, title, tags, hint]) => list.length ? html`
              <div class=${"sdrow pick" + (picked ? " sel" : "")} id=${id} onClick=${() => setOpen(!open)}>
                <input type="radio" name="sd-src" aria-label=${title} checked=${picked} disabled=${specOff}/>
                <div class="grow">
                  <div class="dhead"><b>${title}</b>${tags.map((r) => html`<span class="tag2">${t(r)}</span>`)}
                    ${picked ? html`<span class="tag2 ok">${drafterLabel(f.drafterKind)}</span>` : ""}
                    <span class=${"sd-chev" + (open ? " open" : "")}><${Icon} name="chev"/></span></div>
                  <em class="sd-what">${hint}</em>
                  ${open ? html`<div class="sd-base-list">${list.map((k) => drafterRow(k, id + "-list"))}</div>` : ""}
                </div>
              </div>` : "")}
            <div class=${"sdrow pick" + (custom && !ownVerified ? " sel" : "")} id="sd-custom" onClick=${() => pickDraft("custom")}>
              <input type="radio" name="sd-src" aria-label=${t("Custom drafter")} checked=${custom && !ownVerified} disabled=${specOff}/>
              <div class="grow">
                <div class="dhead"><b>${t("Custom drafter")}</b>
                  ${custom && own ? html`<span class=${"tag2 " + (ownCached ? "ok" : "")}>${t(ownCached ? "downloaded" : isHub(own) ? "Hugging Face" : "file")}</span>` : ""}
                  <span class="was on">${was("drafter") !== null ? t("was " + t(was("drafter"))) : ""}</span></div>
                <p class="ddesc">${t("A drafter of your own, in place of the default: a Hugging Face name or a path on the server's machine.")}</p>
                <input type="text" id="sd-file" placeholder=${t("org/name, or a path")} spellcheck="false" autocomplete="off" disabled=${specOff}
                       value=${f.drafter} onFocus=${() => !custom && pickDraft("custom")}
                       onInput=${(e) => setF(Object.assign({}, f, { draftSource: "custom", drafter: e.target.value, drafterKind: "", draftPicked: false }))}/>
                ${custom ? html`<em>${t(!own ? "Give a Hugging Face name or a path; empty, the default is used." : ownCached ? "On this machine." :
                                         isHub(own) ? "Fetched from Hugging Face when the model loads." : "A file on the server's machine.")}</em>` : ""}
              </div>
            </div>
          </div>
          ${specOff ? html`<em class="sd-note">${t("Plain decoding: one token a step.")}</em>` : ""}
          <details class="adv" id="sd-adv" open=${advOpen} onToggle=${(e) => { e.stopPropagation(); setAdvOpen(e.target.open); }}>
            <summary><${Icon} name="chev"/>${t("Advanced")}<span class="was" id="sd-adv-was">${advChanged.length ? t(advChanged.length + " changed") : ""}</span></summary>
            <div class="opts c3" style="margin-top: 12px">
              <${Field} k="width" label="Drafts a round" was=${was("width")} hint="How many drafts the model checks a step.">
                <select id="sd-width" value=${f.width} onChange=${set("width")} disabled=${specOff}>
                  <option value="">${t("Auto (chosen each round)")}</option>
                  ${["1", "2", "3", "4", "5", "6", "7", "8", "12", "16"].map((n) => html`<option value=${n}>${n}</option>`)}
                </select><//>
              <${Field} k="head" label="Draft head" was=${was("head")} hint="How drafts choose their tokens.">
                <select id="sd-head" value=${f.head} onChange=${set("head")} disabled=${specOff}>
                  <option value="">${t("1-bit pick (default)")}</option><option value="full">${t("Full head")}</option></select><//>
              <${Field} k="weights" label="Drafter weights" was=${was("weights")} hint="Auto picks best precision when VRAM allows, and drops to NVFP4 when tight. Only drafts change, never output.">
                <select id="sd-weights" value=${f.weights} onChange=${set("weights")} disabled=${specOff}>
                  <option value="">${t("Auto (default)")}</option>
                  <option value="nvfp4">${t("NVFP4 (4-bit)")}</option>
                  <option value="q8">${t("Q8_0 (8-bit)")}</option>
                  <option value="exact">${t("Exact")}</option>
                </select><//>
              <${Field} k="lookup" label="Drafts from repeated text" was=${was("lookup")} hint="Where the reply repeats earlier text, that text is proposed too.">
                <label class="switch"><input type="checkbox" id="sd-lookup" checked=${f.lookup} onChange=${set("lookup")} disabled=${specOff}/><i></i>${t(f.lookup ? "on" : "off")}</label><//>
              <${Field} k="trees" label="Draft trees" was=${was("trees")} hint="The drafter proposes several branches at once, not one line.">
                <select id="sd-trees" value=${f.trees} onChange=${set("trees")} disabled=${specOff}>
                  <option value="">${t("default")}</option><option value="on">${t("on")}</option><option value="off">${t("off")}</option></select><//>
            </div>
          </details>
        </div>
        <div class="lset-panel" id="lset-vision" role="tabpanel" hidden=${stab !== "vision"}>
          <div class="pane-h"><h2>${t("Vision encoder")}</h2>
            <span class=${"sd-was" + (was("vision") !== null ? " on" : "")}>${was("vision") !== null ? t("was " + t(was("vision"))) : ""}</span>
            <label class="switch sd-switch" title=${t("Vision encoder")}>
              <input type="checkbox" id="vs-on" aria-label=${t("Vision encoder")} checked=${f.vision} onChange=${set("vision")}/><i></i></label></div>
          ${model.loaded && !busy && model.vision ? html`<div class="srv-dr in-panel" id="ld-vision-src">
            <div class="dr"><span class="tag2 ok">${t("in use")}</span><span class="k">${t("Vision")}</span>
              <span class="f">${(L.vision_projector && L.vision_projector.from ? L.vision_projector.from : t("model file")) +
                (L.vision_projector && L.vision_projector.size ? " · " + gb(L.vision_projector.size) + " GB" : "")}</span></div></div>` : null}
          <div class=${"sdopts" + (f.vision ? "" : " off")} data-k="mmproj">
            <div class=${"sdrow pick" + (!visCustom ? " sel" : "")} id="vs-default" onClick=${() => pickVision("default")}>
              <input type="radio" name="vs-src" aria-label=${t("Default")} checked=${!visCustom} disabled=${visOff}/>
              <div class="grow">
                <div class="dhead"><b>${t("Default")}</b>${vdflt.tag ? html`<span class=${"tag2 " + vdflt.tone}>${t(vdflt.tag)}</span>` : ""}</div>
                <em class="sd-what">${vdflt.line}</em>
              </div>
            </div>
            <div class=${"sdrow pick" + (visCustom ? " sel" : "")} id="vs-custom" onClick=${() => pickVision("custom")}>
              <input type="radio" name="vs-src" aria-label=${t("Custom projector")} checked=${visCustom} disabled=${visOff}/>
              <div class="grow">
                <div class="dhead"><b>${t("Custom projector")}</b>
                  ${visCustom && mmOwn ? html`<span class="tag2">${t(isHub(mmOwn) ? "Hugging Face" : "file")}</span>` : ""}
                  <span class="was on">${was("mmproj") !== null ? t("was " + t(was("mmproj"))) : ""}</span></div>
                <p class="ddesc">${t("Path to a companion mmproj-*.gguf file on the server's machine.")}</p>
                <input type="text" id="vs-file" placeholder=${t("path to mmproj-*.gguf")} spellcheck="false" autocomplete="off" disabled=${visOff}
                       value=${f.mmproj} onFocus=${() => !visCustom && pickVision("custom")}
                       onInput=${(e) => setF(Object.assign({}, f, { visionSource: "custom", mmproj: e.target.value, mmprojPicked: false }))}/>
              </div>
            </div>
          </div>
          ${visOff ? html`<em class="sd-note">${t("Vision disabled: text-only decoding.")}</em>` : ""}
          <div class=${"opts c3" + (visOff ? " off" : "")} style="margin-top: 14px">
            <${Field} k="vision_device" label="Vision storage" was=${was("vision_device")} hint="Disk keeps 0 VRAM and 0 locked RAM so KV capacity is untouched.">
              <select id="vs-device" value=${f.vision_device} onChange=${set("vision_device")} disabled=${visOff}>
                <option value="disk">${t("Disk (default · 0 VRAM, 0 RAM)")}</option>
                <option value="host">${t("Host RAM (0 VRAM)")}</option>
                <option value="gpu">${t("GPU (VRAM)")}</option>
                <option value="auto">${t("Auto (GPU if >= 8 GB free)")}</option>
              </select><//>
            <${Field} k="vision_quant" label="Vision precision" was=${was("vision_quant")} hint="Q8_0 matches BF16 quality (~46% smaller); pre-quantized files stay packed.">
              <select id="vs-quant" value=${f.vision_quant} onChange=${set("vision_quant")} disabled=${visOff}>
                <option value="q8">${t("Q8_0 (default)")}</option>
                <option value="exact">${t("Exact (file precision)")}</option>
                <option value="nvfp4">${t("NVFP4 (compact)")}</option>
              </select><//>
            <${Field} k="vision_max_tokens" label="Max tokens / image" was=${was("vision_max_tokens")} hint="Cap merged visual tokens per image (empty: auto).">
              <input type="number" id="vs-maxtok" min="16" step="1" placeholder=${t("auto")} disabled=${visOff}
                     value=${f.vision_max_tokens} onInput=${set("vision_max_tokens")}/><//>
          </div>
        </div>
        <div class="lset-cmd">
          <div class="pane-h"><h2>${t("Command line")}</h2><span class="aside"><${ShellPicker} shell=${shell} onChange=${setShell}/></span></div>
          <${CodeSnippet} id="mf-cmd" text=${commandOf(body(cmdModel), shell)}/>
        </div>
      </div>`;
  }

  // ---- the rows of what can be loaded, and the box that loads the one chosen ----
  // rows: {name, title, sub, tags: [[text, tone]], size, off (cannot be chosen), del (a cached model, to delete)}
  function rowsView(rows) {
    const serving = loadedWant ? loadedWant.model : null, remembered = L.remembered ? L.remembered.model : null;
    return rows.map((r) => html`
      <div class=${"mrow" + (r.name === target ? " sel" : "") + (r.off ? " off" : "")} data-name=${r.name} role="option" aria-selected=${String(r.name === target)}
           onClick=${r.off ? undefined : () => { setTarget(r.name === target ? "" : r.name); setMsg(""); }}>
        <span class="radio"></span>
        <div><div class="q">${r.title}</div>${r.sub ? html`<div class="fn">${r.sub}</div>` : ""}</div>
        <div class="tags">
          ${serving && r.name === serving ? html`<span class="tag2 live">${t("serving")}</span>`
            : remembered && r.name === remembered ? html`<span class="tag2">${t("last loaded")}</span>` : ""}
          ${r.tags.map(([text, tone]) => html`<span class=${"tag2 " + (tone || "")}>${t(text)}</span>`)}
        </div>
        <div class="sz">${r.size ? sized(r.size) : "-"}</div>
        <div class="fit">${r.off ? "" : html`<${FitTag} size=${r.size}/>`}</div>
        ${r.del ? html`<button class="iconbtn del" title=${t("Delete from the Hugging Face cache")} aria-label=${t("Delete")}
                               onClick=${(e) => { e.stopPropagation(); removeCached(r.del); }}><${Icon} name="trash"/></button>` : html`<span></span>`}
      </div>`);
  }
  // Companion files (at most one drafter + at most one vision projector) that can be combined with a model or applied together.
  function companionsView(id, comps, pickedDraft, setPickedDraft, pickedMm, setPickedMm) {
    const drafts = comps.filter((c) => c.kind !== "mmproj"), mmprojs = comps.filter((c) => c.kind === "mmproj");
    const renderRow = (c, sel, onToggle) => {
      const isMm = c.kind === "mmproj";
      const kindTag = isMm ? ["mmproj", "info"] : [drafterLabel(c.kind) || "drafter", "ok"];
      const sub = c.path && c.path !== c.file ? c.path : "";
      return html`
        <div class=${"mrow comp" + (sel ? " sel" : "")} data-comp=${c.key} data-kind=${c.kind} role="option" aria-selected=${String(sel)}
             onClick=${() => { onToggle(); setMsg(""); }}>
          <span class="check"><${Icon} name="check"/></span>
          <div class="cmain">
            <${Icon} name=${isMm ? "image" : "bolt"} cls=${isMm ? "k-mmproj" : "k-drafter"}/>
            <div><div class="q">${c.file}</div>${sub ? html`<div class="fn">${sub}</div>` : ""}</div>
          </div>
          <div class="tags">
            <span class=${"tag2 " + kindTag[1]}>${t(kindTag[0])}</span>
            ${c.default ? html`<span class="tag2 ok">${t("auto")}</span>` : ""}
            ${c.cached ? html`<span class="tag2 info">${t("downloaded")}</span>` : ""}
          </div>
          <div class="sz">${c.size ? sized(c.size) : "-"}</div>
          <div class="fit"></div>
          <span></span>
        </div>`;
    };
    return html`
      <div class="mrows comp-list" id=${id} role="listbox" aria-multiselectable="true">
        ${drafts.length ? html`
          <div class="mgroup-h">${t("Drafter")}<span>${t("pick up to one")}</span></div>
          ${drafts.map((c) => renderRow(c, pickedDraft === c.key, () => setPickedDraft(pickedDraft === c.key ? "" : c.key)))}` : ""}
        ${mmprojs.length ? html`
          <div class="mgroup-h">${t("Vision projector")}<span>${t("pick up to one")}</span></div>
          ${mmprojs.map((c) => renderRow(c, pickedMm === c.key, () => setPickedMm(pickedMm === c.key ? "" : c.key)))}` : ""}
      </div>`;
  }
  function companionBarState(rows, selDraft, selMm, clearPicked) {
    const hasMain = rows.some((r) => r.name === target && !r.off);
    if (hasMain) {
      const overrides = (selDraft || selMm) ? {
        drafter: selDraft ? selDraft.spec : null,
        drafterKind: selDraft?.kind,
        mmproj: selMm ? selMm.spec : null,
        drafterSize: selDraft ? (selDraft.size || 0) : 0,
        drafterCached: selDraft ? !!selDraft.cached : true,
        mmprojSize: selMm ? (selMm.size || 0) : 0,
        mmprojCached: selMm ? !!selMm.cached : true,
      } : null;
      return { action: null, overrides };
    }
    if (!selDraft && !selMm) return { action: null, overrides: null };
    const items = [selDraft, selMm].filter(Boolean);
    const allCached = items.every((c) => c.cached);
    const totalSz = items.reduce((n, c) => n + (c.size || 0), 0);
    const fetchSz = items.reduce((n, c) => n + (c.cached ? 0 : (c.size || 0)), 0);
    const sub = allCached ? sized(totalSz) + " on disk" : sized(fetchSz) + " to download first";
    if (selDraft && selMm) {
      return {
        action: { title: selDraft.file + " + " + selMm.file, sub, label: "Use selected companions",
                  disabled: !model.id || busy, onClick: () => { applyCompanions(selDraft, selMm.spec); clearPicked(); } },
        overrides: null,
      };
    }
    if (selMm) {
      return {
        action: { title: selMm.file, sub, label: "Use as projector", disabled: !model.id || busy,
                  onClick: () => { projectWith(selMm.spec); clearPicked(); } },
        overrides: null,
      };
    }
    return {
      action: { title: selDraft.file, sub, label: "Draft with it", disabled: !model.id || busy,
                onClick: () => { draftWith(selDraft.spec, selDraft.kind); clearPicked(); } },
      overrides: null,
    };
  }
  const localRow = (m) => {
    const d = describe(m, winPaths), tags = [];
    if (m.kind === "flyx") tags.push([".flyx"]);
    if (m.kind === "mmproj") tags.push(["mmproj", "info"]);
    if (m.mtp === "in-file") tags.push(["MTP inside", "ok"]); else if (m.mtp === "in-repo") tags.push(["+ MTP file", "ok"]);
    if (m.dflash2 === "in-repo") tags.push(["+ DFlash2", "ok"]);
    if (m.mmproj === "in-repo") tags.push(["+ Vision", "ok"]);
    if (m.current === false) tags.push(["older revision", "info"]);
    if (m.parts > 1) tags.push([m.parts + " parts"]);
    return { name: m.name, title: d.quant || d.file || m.name, sub: d.quant ? (d.file || m.name) : "", tags, size: m.size,
             del: m.source === "hub" ? m : null };
  };
  // `rows`: the ones shown, so a choice made elsewhere is not loaded from here.  `note`: what stands in the way.
  // `action` (optional): replaces loading with a companion action (e.g., attaching an mmproj or drafter).
  // `overrides` (optional): companion files picked alongside a main model row to load together in one shot.
  function loadBox(rows, note, action, overrides) {
    if (action) {
      return html`
        <div class="loadbox" id="mf-loadbox">
          <div class="what">
            ${action.title ? html`<div class="tn" id="mf-chosen" title=${action.title}>${action.title}</div>` : ""}
            <div class="ts" id="mf-chosen-s">${t(action.sub || note)}</div>
            ${msg ? html`<div class="ts bad" id="mf-msg">${t(msg)}</div>` : ""}
          </div>
          <button class="btn primary" id="mf-load" disabled=${!!action.disabled} onClick=${action.onClick}>${t(action.label)}</button>
        </div>`;
    }
    const row = rows.find((r) => r.name === target && !r.off), name = row ? row.name : "";
    const same = !!loadedWant && name === loadedWant.model, m = entryOf(name);
    const eff = settingsFor(name, overrides), effNow = shown(loadBody(name, eff, Boolean(loadedWant)));
    const effChanged = before ? diffShown(before, effNow) : [];
    let label = "Load model", disabled = !name || busy || !!note;
    if (busy) label = L.state === "fetching" ? "Fetching…" : "Loading…";
    else if (same) { label = effChanged.length ? "Reload with these settings" : "Loaded"; disabled = disabled || !effChanged.length; }
    else if (loadedWant && name) label = "Switch to this model";
    const mainOnDisk = !name ? true : (!!m || !isHub(name));
    const extraFetch = (overrides && overrides.drafter && !overrides.drafterCached ? overrides.drafterSize : 0) +
                       (overrides && overrides.mmproj && !overrides.mmprojCached ? overrides.mmprojSize : 0);
    const extraTotal = (overrides && overrides.drafter ? overrides.drafterSize : 0) +
                       (overrides && overrides.mmproj ? overrides.mmprojSize : 0);
    const fetchBytes = (mainOnDisk ? 0 : row.size) + extraFetch;
    const what = !name ? "" : same && !extraFetch ? "This is the model being served."
      : fetchBytes > 0 ? sized(fetchBytes) + " to download first"
      : sized(row.size + extraTotal) + " on disk";
    const drafting = !eff.spec ? "off" : eff.draftSource === "custom" && eff.drafter.trim() ? baseName(eff.drafter.trim(), winPaths) :
                     eff.draftSource === "saved" ? "last loaded" : "default";
    const visionBrief = !eff.vision ? "off" : (eff.visionSource === "custom" && eff.mmproj.trim() ? baseName(eff.mmproj.trim(), winPaths) + " · " : "") +
                        t(normVisionDev(eff.vision_device)) + " · " + normVisionQuant(eff.vision_quant);
    const brief = [["context", String(effNow.ctx)], ["KV", effNow.kv], ["drafting", drafting], ["vision", visionBrief]];
    return html`
      <div class="loadbox" id="mf-loadbox">
        <div class="what">
          ${name ? html`<div class="tn" id="mf-chosen" title=${name}>${name}</div>
                        <div class="ts" id="mf-chosen-s">${t(what)}<span class="with"> · ${brief.map(([k, v]) => t(k) + " " + t(v)).join(" · ")}${model.id ? " · " : ""}</span>
                          ${model.id ? html`<button class="linkbtn" id="mf-tosettings" onClick=${() => setView("current")}>${t("Change the settings")}</button>` : ""}</div>`
                 : html`<div class="ts" id="mf-chosen-s">${t(note || "Choose one of these to load.")}</div>`}
          ${msg ? html`<div class="ts bad" id="mf-msg">${t(msg)}</div>` : ""}
        </div>
        <button class="btn primary" id="mf-load" disabled=${disabled} onClick=${() => loadModel(name, null, overrides)}>${t(label)}</button>
      </div>`;
  }

  // ---- Hugging Face: the repositories found, the one shown ----
  function hubList() {
    const list = found ? found.models : [];
    const archs = found && Array.isArray(found.architectures) ? found.architectures : [];
    const showPager = !!found && !foundMsg && (page > 0 || !!found.next_cursor);
    return html`
      ${found && found.family && archs.length > 1 ? html`
        <div class="hub-archs pills" id="hf-archs" role="group">
          <button type="button" data-arch="" aria-pressed=${String(arch === "")} onClick=${() => { dispatchHub({ type: "unpin" }); setArch(""); }}>${t("All")}</button>
          ${archs.map((a) => html`
            <button type="button" key=${a} data-arch=${a} aria-pressed=${String(arch === a)} onClick=${() => { dispatchHub({ type: "unpin" }); setArch(a); }}>${a}</button>`)}
        </div>` : null}
      ${foundMsg ? html`
        <div class="hub-status">
          <div class="empty pad" id="hf-msg">${t(foundMsg)}</div>
          ${!foundMsg.startsWith("Asking ") ? html`
            <div class="hub-ephint"><button type="button" class="btn sm" id="hf-ep-help"
              onClick=${() => { setEpOpen(true); setEpText(hubEndpoint); setEpMsg(""); }}>
              <${Icon} name="globe"/>${t("Use a mirror")}</button></div>` : null}
        </div>` : null}
      ${found && !list.length && !foundMsg ? html`<div class="empty pad">${t("Nothing matches.")}</div>` : null}
      ${list.length ? html`<div class="hub-items">${list.map((m) => html`
        <button key=${m.id} class=${"prow" + (m.id === hubView.sel ? " sel" : "")} data-repo=${m.id} onClick=${() => showRepo(m.id, "", "", false)}>
          <span class="pn">${m.id.split("/")[1]}</span>
          <span class="pm"><${Icon} name="heart"/>${compact(m.likes)}<${Icon} name="dl"/>${compact(m.downloads)}</span>
          <span class="ps">${m.id.split("/")[0]}${m.architecture ? html` <span class=${"tag2" + (m.runs === false ? " bad" : m.drafter ? " ok" : "")}>${m.architecture}</span>` : m.drafter ? html` <span class="tag2 ok">${t("drafter")}</span>` : null}${m.cached ? html` <span class="tag2 info">${t("downloaded")}</span>` : null}${m.gated ? html` <span class="tag2">${t("gated")}</span>` : null}</span>
          <span class="pt">${t(ago(m.updated || m.created))}</span>
        </button>`)}</div>` : null}
      ${showPager ? html`
        <div class="hub-pager" id="hf-pager">
          <button class="btn sm" id="hf-prev" disabled=${page === 0} onClick=${() => goHubPage(page - 1)}>${t("Prev page")}</button>
          <span class="hub-page" id="hf-page">${t("Page " + (page + 1))}</span>
          <button class="btn sm" id="hf-next" disabled=${!found.next_cursor} onClick=${() => goHubPage(page + 1)}>${t("Next page")}</button>
        </div>` : null}`;
  }
  function hubDetail() {
    const repoSel = hubView.sel, repoQuant = hubView.quant, repoRev = hubView.rev;
    if (!repoSel) return html`<div class="hub-scroll"><div class="empty">${t("Pick a repository on the left to see its files.")}</div></div>${loadBox([], "")}`;
    const j = hubView.detail && hubView.detail.repo === repoSel && hubView.detail.rev === repoRev ? hubView.detail.j : null;
    const info = (j && j.info) || {};
    const spec = repoSpec(repoSel, repoRev);
    const revSuffix = repoRev ? "@" + repoRev : "";
    const rows = j ? (j.ggufs || []).map((g) => {
      const tags = [];
      if (g.split && g.parts > 1) tags.push([g.parts + " parts"]);
      if (!whole(g)) tags.push(["split, incomplete"]);
      if (g.cached) tags.push(["downloaded", "info"]);
      const m = entryOf(g.name);
      return { name: g.name, title: g.quant, sub: "", tags, size: g.size, off: !whole(g), del: m && m.source === "hub" ? m : null };
    }) : [];
    const rawComps = j && Array.isArray(j.companions) ? j.companions : [
      ...(j && j.mtp ? [{ name: repoSel + ":" + baseName(j.mtp.path) + revSuffix, path: j.mtp.path, file: baseName(j.mtp.path),
                          kind: "mtp", size: j.mtp.size, cached: !!j.mtp.cached, disk: null, default: true }] : []),
      ...(j && j.mmproj ? [{ name: repoSel + ":" + baseName(j.mmproj.path) + revSuffix, path: j.mmproj.path, file: baseName(j.mmproj.path),
                             kind: "mmproj", size: j.mmproj.size, cached: !!j.mmproj.cached, disk: null, default: true }] : []),
    ];
    const comps = rawComps.map((c) => ({ ...c, key: c.name, spec: c.disk || c.name,
      kind: c.kind === "mmproj" ? c.kind : (c.drafter_kind ?? c.kind) }));
    const hubDraft = hubView.draft, hubMmproj = hubView.mmproj;
    const setHubDraft = (v) => dispatchHub({ type: "draft", value: v });
    const setHubMmproj = (v) => dispatchHub({ type: "mmproj", value: v });
    const selDraft = comps.find((c) => c.kind !== "mmproj" && c.key === hubDraft) || null;
    const selMm = comps.find((c) => c.kind === "mmproj" && c.key === hubMmproj) || null;
    const barState = companionBarState(rows, selDraft, selMm, () => dispatchHub({ type: "clearCompanions" }));
    const compAction = barState.action || (j && j.drafter && !rows.length && !comps.length
      ? { title: spec, sub: ((entryOf(spec) || entryOf(repoSel)) ? sized(j.drafter.size) + " on disk" : sized(j.drafter.size) + " to download first"),
          label: "Draft with it", disabled: !model.id || busy, onClick: () => draftWith(spec, j.drafter.kind) }
      : null);
    const compOverrides = barState.overrides;
    const unsupported = j && j.runs === false;
    const note = unsupported ? "This build does not run this architecture."
      : j && j.drafter_repo && !rows.length ? "A drafter is not loaded on its own."
      : j && !rows.length && !j.drafter ? "Nothing here to load." : "";
    const chips = [];
    if (info.pipeline) chips.push([info.pipeline, ""]);
    if (info.architecture) chips.push([info.architecture, unsupported ? "bad" : j.runs ? "ok" : ""]);
    if (info.parameters) chips.push([compact(info.parameters) + " " + t("parameters"), ""]);
    if (info.context) chips.push([t("context") + " " + fmt(info.context), ""]);
    if (info.license) chips.push([info.license, ""]);
    if (info.gated) chips.push([t("gated"), "warn"]);
    if (comps.some((c) => c.kind === "mmproj")) chips.push([t("+ Vision"), "ok"]);
    if (comps.some((c) => c.kind === "mtp")) chips.push(["+ MTP", "ok"]);
    if (rows.length && comps.some((c) => c.kind === "dflash2")) chips.push(["+ DFlash2", "ok"]);
    const [org, short] = repoSel.split("/");
    return html`
      <div class="hub-scroll" id="hf-detail">
        <div class="dh">
          <div class="dh-icon"><${Icon} name="model"/></div>
          <div class="dh-t"><h3 id="hf-name">${short}</h3><span>${org}</span></div>
          <${CopyButton} get=${() => spec} label="Copy name"/>
          <a class="btn ghost sm" href=${safeOrigin((j && j.endpoint) || hubEndpoint) + "/" + repoSel} target="_blank" rel="noopener noreferrer">
            <${Icon} name="ext"/>${t("Open on Hugging Face")}</a>
        </div>
        ${chips.length ? html`<div class="dchips">${chips.map(([text, tone]) => html`<span class=${"tag2 " + tone}>${text}</span>`)}</div>` : ""}
        ${j && (info.updated || info.downloads || info.likes) ? html`
          <dl class="stats small">
            ${info.updated ? html`<div><dt>${t("Updated")}</dt><dd>${t(ago(info.updated))}</dd></div>` : ""}
            <div><dt>${t("Downloads")}</dt><dd>${compact(info.downloads)}</dd></div>
            <div><dt>${t("Likes")}</dt><dd>${compact(info.likes)}</dd></div>
            ${j.base_models && j.base_models.length ? html`<div><dt>${t("Base model")}</dt><dd>${j.base_models[0]}</dd></div>` : ""}
          </dl>` : ""}
        ${hubView.note ? (hubView.note.tone === "info"
          ? html`<p class="lead" id="mf-hubmsg">${t(hubView.note.text)}</p>`
          : html`<div class="banner bad" id="mf-hubmsg"><div class="grow2">${tErr(hubView.note.text)}</div>
              <button class="btn sm" id="hf-retry" onClick=${() => showRepo(repoSel, repoQuant, repoRev)}>${t("Retry")}</button></div>`) : ""}
        ${unsupported ? html`<div class="banner warn" id="hf-unsupported">${t("These files are of the " + info.architecture + " architecture, which this build does not run. It runs: " +
                                                                       (j.build_architectures || []).join(", ") + ".")}</div>` : ""}
        ${rows.length ? html`
          <div class="dsec">${t("Files")}<span>${t(rows.length + " quantizations")}</span></div>
          <div class="mrows" id="mf-files" role="listbox">${rowsView(rows)}</div>` : ""}
        ${j && j.drafter_repo ? html`<div class="banner hint" id="hf-drafter-repo"><div class="grow2">${t("This repository holds a drafter, not a standalone model: choose a file below to draft with it.")}</div></div>` : null}
        ${comps.length ? html`
          <div class="dsec">${t("Companion files")}<span>${t("pick up to one drafter and one vision projector")}</span></div>
          ${companionsView("hf-companions", comps, hubDraft, setHubDraft, hubMmproj, setHubMmproj)}` : ""}
        ${j && j.drafter ? html`<div class="banner hint" id="hf-drafter-only"><div class="grow2">${t(repoSel + " is a drafter (" + gb(j.drafter.size) + " GB), not a model: it drafts for one, and is not loaded on its own.")}</div></div>` : ""}
        ${j && !j.drafter && !j.drafter_repo && !rows.length ? html`<p class="lead">${t("No GGUF files in " + repoSel + ".")}</p>` : ""}
        ${j ? (j.known_drafters || []).map((k, i) => html`
          <div class="banner hint" id=${i ? null : "hf-drafters"}><div class="grow2">${t(k.exact ? "A drafter made for this model is published:" : "A drafter for its base model is published:")}
            ${" "}<a href=${safeOrigin((j && j.endpoint) || hubEndpoint) + "/" + k.repo} target="_blank" rel="noopener noreferrer"><code>${k.repo}</code></a>
            <span class="tag2 ok">${drafterLabel(k.kind)}</span> <span class="tag2 ok">${t("verified")}</span>
            ${k.exact ? "" : html`<span class="tag2">${k.base}${k.relation ? " · " + t(k.relation) : ""}</span>`}
            <span class="tag2" title=${t("Your own runs with it on this machine, on average (N/A: none yet).")}>${t("speed")}: ${k.tok_s ? Math.round(k.tok_s) + " tok/s" : "N/A"}</span></div>
            ${model.id ? html`<button class="btn sm" onClick=${() => draftWith(k.repo, k.kind)}>${t("Add to settings")}</button>` : ""}</div>`) : ""}
      </div>
      ${loadBox(rows, note, compAction, compOverrides)}`;
  }

  // ---- this machine: folders in a list, the models of the one shown ----
  const local = catalog ? catalog.models : [];
  const configDirs = (catalog && Array.isArray(catalog.dirs) ? catalog.dirs : []).filter(Boolean);
  const rootOf = (folder) => {
    let best = "";
    for (const d of configDirs) if (underPath(folder, d) && d.length > best.length) best = d;
    return best || folder;
  };
  const repos = new Map(), known = new Map(), knownRoots = new Map();  // repository -> its models; normalized folder -> models
  for (const m of local) {
    if (m.source === "hub") {
      const key = m.repo || repoOf(m.name);
      if (!repos.has(key)) repos.set(key, []);
      repos.get(key).push(m);
    } else {
      const root = m.root || rootOf(folderOf(m, winPaths));
      const nk = nPath(root);
      if (!known.has(nk)) { known.set(nk, []); knownRoots.set(nk, root); }
      known.get(nk).push(m);
    }
  }
  const knownAt = (p) => known.get(nPath(p)) || [];
  // The folders: the one Hugging Face downloads to, configured --models-dir roots, those with the server's other
  // models, those of the model served and the one remembered, those added here.
  const cacheDir = catalog ? catalog.cache : "", atCache = !!cacheDir && !!loc && eqPath(loc, cacheDir);
  const places = cacheDir ? [cacheDir] : [];
  const wanted = [loadedWant, L.remembered]
    .filter((w) => w && w.model && !isHub(w.model) && (winPaths === false ? w.model.includes("/") : /[\\/]/.test(w.model)))
    .map((w) => rootOf(folderOf({ name: w.model }, winPaths)));
  for (const x of [...configDirs, ...knownRoots.values(), ...wanted, ...folders]) if (!places.some((k) => eqPath(k, x))) places.push(x);
  const sum = (list) => list.reduce((n, m) => n + (m.size || 0), 0);
  const ql = localQ.trim().toLowerCase(), hit = (text) => !ql || text.toLowerCase().includes(ql);
  // A folder's models: the server's own list for the download folder (merged with any --models-dir entries when the
  // cache itself is configured as a models dir) and its model directories, else what it found there when asked.
  const hubModels = local.filter((x) => x.source === "hub");
  const hubPathSet = new Set(hubModels.map((m) => nPath(m.path || m.name)).filter(Boolean));
  const extraDirModels = (p) => knownAt(p).filter((m) => !hubPathSet.has(nPath(m.path || m.name)));
  const modelsOf = (p) => {
    if (eqPath(p, cacheDir)) return [...hubModels, ...extraDirModels(p)];
    const byDir = knownAt(p);
    return byDir.length ? byDir : (listings[p] && listings[p].models) || [];
  };
  useEffect(() => {
    if (!active || view !== "library" || tab !== "local" || !catalog) return;
    for (const folder of places) {
      if (eqPath(folder, cacheDir) || knownAt(folder).length || configDirs.some((d) => eqPath(d, folder)) || listings[folder]) continue;
      setListings((old) => Object.assign({}, old, { [folder]: { models: [], looking: true } }));
      ask("/v1/hub/dir?path=" + encodeURIComponent(folder), "browse folders").then((a) =>
        setListings((old) => Object.assign({}, old, { [folder]: a.error ? { models: [], error: a.error } : { models: a.j.models } })));
    }
  }, [active, view, tab, catalog, places.join("\n")]);
  const kindOfModel = (m) => m.kind === "drafter" ? "drafter" : m.kind === "mmproj" ? "mmproj" : m.kind === "lora" ? "LoRA" : m.kind === "flyx" ? ".flyx" : "GGUF";
  const iconOfModel = (m) => m.kind === "mmproj" ? "image" : m.kind === "drafter" ? "bolt" : m.kind === "lora" ? "plus" : "model";
  const kindCls = (m) => m.kind === "mmproj" ? "k-mmproj" : m.kind === "drafter" ? "k-drafter" : m.kind === "lora" ? "k-lora" : "k-model";
  // A model as the list names it: a repository's by its name and quantization, a file by its own name.
  const titleOf = (m) => {
    const d = describe(m, winPaths);
    return m.source === "hub" && d.repo ? d.repo.split("/")[1] + (d.quant ? " \u00b7 " + d.quant : "") : d.file || baseName(m.name, winPaths);
  };
  const pickModel = (m) => { setPicked(m.name); setTarget(standalone(m) ? m.name : ""); setMsg(""); };
  function localList() {
    if (catalogErr) return html`<div class="empty pad">${catalogErr}</div>`;
    if (!catalog) return html`<div class="empty pad">${t("Looking...")}</div>`;
    const act = (go) => (e) => { e.stopPropagation(); go(); };
    const onEnter = (go) => (e) => { if (e.key === "Enter" && e.target === e.currentTarget) go(); };
    const serving = loadedWant ? loadedWant.model : null;
    return html`
      ${places.map((p) => {
        const all = modelsOf(p), state = listings[p] || {};
        const list = all.filter((m) => hit(titleOf(m) + " " + m.name)).sort((x, y) => titleOf(x).localeCompare(titleOf(y)));
        const mine = !eqPath(p, cacheDir) && !configDirs.some((d) => eqPath(d, p)) && folders.some((x) => eqPath(x, p));
        const open = () => { setPicked(""); openPath(p); };
        const pTrim = trimTrailingSep(p, winPaths);
        return html`
          <div class=${"prow act folder" + (!picked && eqPath(loc, p) ? " sel" : "")} role="button" tabindex="0" data-place=${"dir:" + p}
               onClick=${open} onKeyDown=${onEnter(open)}>
            <span class="pn path"><${Icon} name="folder"/>${baseName(pTrim, winPaths) || p}</span>
            <span class="pm">${all.length ? sized(sum(all)) : ""}</span>
            <span class="ps path">${p}</span>
            <span class="pt">${all.length ? t(all.length + (all.length === 1 ? " file" : " files")) : ""}</span>
            ${mine ? html`<button class="iconbtn del" title=${t("Remove from the list")} aria-label=${t("Remove from the list")}
                                  onClick=${act(() => setFolders((old) => old.filter((x) => !eqPath(x, p))))}><${Icon} name="close"/></button>`
                   : html`<span class="del"></span>`}
          </div>
          ${list.map((m) => {
            const d = describe(m, winPaths), hubModel = m.source === "hub", mf = folderOf(m, winPaths);
            const under = hubModel ? (d.repo || "").split("/")[0] : relUnder(mf, p, winPaths);
            const go = () => pickModel(m);
            return html`
              <div class=${"prow act model" + (picked === m.name ? " sel" : "")} role="button" tabindex="0" data-model=${m.name} onClick=${go} onKeyDown=${onEnter(go)}>
                <span class="pn item"><${Icon} name=${iconOfModel(m)} cls=${kindCls(m)}/><span class="pn-t">${titleOf(m)}</span></span>
                <span class="pm">${m.size ? sized(m.size) : ""}</span>
                <span class="ps">${under}${serving && m.name === serving ? html` <span class="tag2 live">${t("serving")}</span>` : ""}</span>
                <span class="pt">${t(kindOfModel(m))}</span>
                ${hubModel ? html`<button class="iconbtn del" title=${t("Delete from the Hugging Face cache")} aria-label=${t("Delete")}
                                          onClick=${act(() => removeCached(m))}><${Icon} name="trash"/></button>` : html`<span class="del"></span>`}
              </div>`;
          })}
          ${list.length ? "" : html`<div class="pnone">${t(state.looking ? "Looking..." : state.error ? state.error : all.length ? "Nothing matches." : "No model files here.")}</div>`}`;
      })}
      <button class="prow add" id="mf-add" disabled=${picking} onClick=${addFolder}>
        <span class="pn path"><${Icon} name="plus"/>${t(picking ? "Choose the folder in the window that opened\u2026" : "Add a folder\u2026")}</span></button>`;
  }
  const openHubRepo = (repo, rev = "") => {
    setQ("");
    setArch("");
    setTab("hub");
    showRepo(repo, "", rev, true);
  };
  // One model of the list: what it is, where its file is, and the box that loads it.
  function modelDetail(m) {
    const d = describe(m, winPaths), hubModel = m.source === "hub", drafter = m.kind === "drafter", mmproj = m.kind === "mmproj", lora = m.kind === "lora";
    const isDrafterFile = drafter && /\.(gguf|flyx)$/i.test(m.path || m.file || "");
    const chips = [[drafter && m.drafter_kind ? drafterLabel(m.drafter_kind) : t(kindOfModel(m)), drafter ? "ok" : mmproj ? "info" : ""]];
    if (d.quant) chips.push([d.quant, ""]);
    if (m.mtp === "in-file") chips.push([t("MTP inside"), "ok"]); else if (m.mtp === "in-repo") chips.push([t("+ MTP file"), "ok"]);
    if (m.dflash2 === "in-repo") chips.push(["+ DFlash2", "ok"]);
    if (m.mmproj === "in-repo") chips.push([t("+ Vision"), "ok"]);
    if (m.parts > 1) chips.push([t(m.parts + " parts"), ""]);
    if (m.current === false) chips.push([t("older revision"), "info"]);
    if (loadedWant && m.name === loadedWant.model) chips.push([t("serving"), "live"]);
    const action = mmproj
      ? { title: m.path || m.name, sub: "A vision projector is not loaded on its own.", label: "Use as projector",
          disabled: !model.id || busy, onClick: () => projectWith(m.path || m.name) }
      : drafter
      ? { title: m.name, sub: "A drafter is not loaded on its own.", label: "Draft with it",
          disabled: !model.id || busy, onClick: () => draftWith(m.path || m.name, m.drafter_kind) }
      : lora
      ? { title: m.path || m.name, sub: "A LoRA adapter is not loaded on its own.", label: "Use with the model",
          disabled: !model.id || busy, onClick: () => loraWith(m.source === "hub" ? m.name : m.path || m.name) }
      : null;
    const pinnedRev = m.current === false && m.revision ? m.revision : "";
    return html`
      <div class="hub-scroll" id="lc-detail">
        <div class="dh">
          <div class="dh-icon"><${Icon} name=${iconOfModel(m)}/></div>
          <div class="dh-t"><h3 id="lc-name">${titleOf(m)}</h3><span>${hubModel ? (d.repo || "").split("/")[0] : folderOf(m, winPaths)}</span></div>
          ${hubModel && d.repo && (standalone(m) || (drafter && d.quant)) ? html`<button class="btn ghost sm" id="lc-quants" onClick=${() => openHubRepo(d.repo, pinnedRev)}>${t("All its quantizations")}</button>` : ""}
          ${hubModel ? html`<button class="btn ghost sm" id="lc-delete" onClick=${() => removeCached(m)}><${Icon} name="trash"/>${t("Delete")}</button>` : ""}
        </div>
        <div class="dchips">${chips.map(([text, tone]) => html`<span class=${"tag2 " + tone}>${text}</span>`)}${standalone(m) ? html`<${FitTag} size=${m.size}/>` : ""}</div>
        <dl class="stats small">
          <div><dt>${t("Size")}</dt><dd>${m.size ? sized(m.size) : "-"}</dd></div>
          ${m.modified ? html`<div><dt>${t("Modified")}</dt><dd>${t(ago(m.modified))}</dd></div>` : ""}
          ${hubModel && d.repo ? html`<div><dt>${t("Repository")}</dt><dd>${d.repo}</dd></div>` : ""}
          ${m.revision ? html`<div><dt>${t("Revision")}</dt><dd>${String(m.revision).slice(0, 8)}</dd></div>` : ""}
        </dl>
        <div class="dsec">${t((drafter && !isDrafterFile) || lora ? "Folder" : "File")}</div>
        <${CodeSnippet} id="lc-path" text=${m.path || m.name}/>
        ${drafter ? html`<div class="banner hint"><div class="grow2">${t(m.name + " is a drafter (" + gb(m.size) + " GB), not a model: it drafts for one, and is not loaded on its own.")}</div></div>` : ""}
        ${mmproj ? html`<div class="banner hint"><div class="grow2">${t(m.file + " is a vision projector (" + gb(m.size) + " GB), not a standalone model: attach it to a compatible VLM.")}</div></div>` : ""}
        ${lora ? html`<div class="banner hint"><div class="grow2">${t(m.file + " is a LoRA adapter (" + gb(m.size) + " GB), not a model: it runs beside the model it was trained for.")}</div></div>` : ""}
      </div>
      ${loadBox(standalone(m) ? [localRow(m)] : [], "", action)}`;
  }
  function localDetail() {
    const j = atCache ? null : dir, start = !!j && j.kind === "start";
    const go = () => openPath(pathText.trim());
    const locNorm = nPath(loc);
    const above = atCache
      ? (locNorm === "/" || (winPaths !== false && /^[a-z]:\/$/i.test(locNorm)) ? null : folderOf({ path: trimTrailingSep(loc, winPaths) }, winPaths) || null)
      : j ? j.parent : null;
    const bar = html`
      <div class="pathbar">
        <button class="iconbtn" id="mf-up" title=${t("The folder above")} aria-label=${t("The folder above")} disabled=${!above}
                onClick=${() => openPath(above)}><${Icon} name="up"/></button>
        <input type="text" id="mf-path" placeholder=${t("A folder or a model file on the server's machine")} spellcheck="false" autocomplete="off"
               aria-label=${t("A folder or a model file on the server's machine")} value=${pathText} onInput=${(e) => setPathText(e.target.value)}
               onKeyDown=${(e) => { if (e.key === "Enter") { e.preventDefault(); go(); } }}/>
        <button class="btn" id="mf-open" onClick=${go}>${t("Open")}</button>
        <button class="btn" id="mf-keep" hidden=${!j || j.kind !== "dir" || places.some((x) => eqPath(x, loc)) && !kept} aria-pressed=${String(kept)} onClick=${keep}>
          ${t(kept ? "Remove from the list" : "Keep in the list")}</button>
      </div>`;
    if (atCache) {
      // What Hugging Face downloaded, by repository: the files of each to load, the whole of it to delete.
      const groups = [...repos].filter(([name, list]) => hit(name) || list.some((m) => hit(m.name + " " + (m.file || ""))));
      const rows = [];
      const body = groups.map(([name, all]) => {
        const list = all.slice().sort((x, y) => (x.size || 0) - (y.size || 0));
        const mine = list.filter(standalone).map(localRow);
        const drafters = list.filter((m) => m.kind === "drafter");
        const sharedOldRev = list.length && list.every((m) => m.current === false) && list[0].revision &&
                             list.every((m) => m.revision === list[0].revision) ? list[0].revision : "";
        rows.push(...mine);
        return html`
          <div class="rgroup" data-repo=${name}>
            <div class="rgroup-h"><b>${name}</b><span>${sized(sum(list))}</span><span class="grow"></span>
              ${mine.length || drafters.some((d) => !!d.quant) ? html`<button class="btn ghost sm" onClick=${() => openHubRepo(name, sharedOldRev)}>${t("All its quantizations")}</button>` : ""}
              <button class="iconbtn del" title=${t("Delete from the Hugging Face cache")} aria-label=${t("Delete")}
                      onClick=${() => removeRepo(name, all)}><${Icon} name="trash"/></button></div>
            ${mine.length ? html`<div class="mrows" role="listbox">${rowsView(mine)}</div>` : ""}
            ${drafters.map((dr) => html`<div class="banner hint"><div class="grow2">${t(dr.name + " is a drafter (" + gb(dr.size) + " GB), not a model: it drafts for one, and is not loaded on its own.")}</div>
              ${model.id ? html`<button class="btn sm" onClick=${() => draftWith(dr.path || dr.name, dr.drafter_kind)}>${t("Draft with it")}</button>` : ""}</div>`)}
          </div>`;
      });
      const extraAll = extraDirModels(cacheDir);
      const extraDir = extraAll.filter((m) => hit(titleOf(m) + " " + m.name));
      const extraRows = extraDir.filter(standalone).map(localRow);
      rows.push(...extraRows);
      const extraBody = extraRows.length ? html`
        <div class="rgroup" data-repo="local">
          <div class="rgroup-h"><b>${cacheDir}</b><span>${sized(sum(extraDir))}</span><span class="grow"></span></div>
          <div class="mrows" role="listbox">${rowsView(extraRows)}</div>
        </div>` : null;
      return html`
        <div class="hub-scroll" id="lc-detail">
          ${bar}
          <p class="lead dnote" id="mf-cachenote">${t("What is downloaded from Hugging Face is kept here, a repository each.")}</p>
          <div id="mf-cached">${groups.length || extraBody ? html`${body}${extraBody}` : html`<div class="empty" style="margin-top: 16px">${t(repos.size || extraAll.length ? "Nothing matches." : "Nothing downloaded yet.")}</div>`}</div>
        </div>
        ${loadBox(rows, rows.length ? "" : "Nothing here to load.")}`;
    }
    // Any other folder: the folders in it, the model files in it and under it.
    const stripLeadSep = (s) => winPaths === false ? s.replace(/^\/+/, "") : s.replace(/^[\\/]+/, "");
    const allModels = j ? j.models.filter(standalone) : [];
    const rows = allModels.map((m) => {
      const r = localRow(m), rel = j.kind === "dir" && m.path.length > j.path.length ? stripLeadSep(m.path.slice(j.path.length)) : m.file;
      return Object.assign(r, { sub: r.title === rel ? "" : rel });
    }).filter((r) => hit(r.title + " " + r.sub));
    const folderMmproj = j ? j.models.filter((m) => m.kind === "mmproj" && hit(m.file + " " + (m.path || ""))) : [];
    const folderDrafters = j ? j.models.filter((m) => m.kind === "drafter" && hit(m.file + " " + (m.path || ""))) : [];
    const singleMmproj = j && j.kind === "file" && folderMmproj.length === 1 ? folderMmproj[0] : null;
    const singleDrafter = j && j.kind === "file" && folderDrafters.length === 1 ? folderDrafters[0] : null;
    const relOf = (m) => j && j.kind === "dir" && m.path.length > j.path.length ? stripLeadSep(m.path.slice(j.path.length)) : m.file;
    const localComps = [
      ...folderDrafters.map((m) => ({
        key: m.path || m.name, spec: m.path || m.name, file: m.file, path: relOf(m),
        kind: m.drafter_kind || "", size: m.size, cached: true, default: false,
      })),
      ...folderMmproj.map((m) => ({
        key: m.path || m.name, spec: m.path || m.name, file: m.file, path: relOf(m),
        kind: "mmproj", size: m.size, cached: true, default: false,
      })),
    ];
    const selLocalDraft = localComps.find((c) => c.kind !== "mmproj" && c.key === localDraft) || null;
    const selLocalMm = localComps.find((c) => c.kind === "mmproj" && c.key === localMmproj) || null;
    const localBar = singleMmproj
      ? { action: { title: singleMmproj.path || singleMmproj.name, sub: "A vision projector is not loaded on its own.",
                    label: "Use as projector", disabled: !model.id || busy, onClick: () => projectWith(singleMmproj.path || singleMmproj.name) },
          overrides: null }
      : singleDrafter
      ? { action: { title: singleDrafter.path || singleDrafter.name, sub: "A drafter is not loaded on its own.",
                    label: "Draft with it", disabled: !model.id || busy, onClick: () => draftWith(singleDrafter.path || singleDrafter.name, singleDrafter.drafter_kind) },
          overrides: null }
      : companionBarState(rows, selLocalDraft, selLocalMm, () => { setLocalDraft(""); setLocalMmproj(""); });
    const loadNote = start ? "Open a folder to see its model files." : "";
    return html`
      <div class="hub-scroll" id="lc-detail">
        ${bar}
        ${dirMsg ? html`<p class="lead" id="mf-dirmsg">${t(dirMsg)}</p>` : ""}
        ${j && j.dirs.length ? html`
          <div class="dsec">${t(start ? "Start from" : "Folders")}</div>
          <div class="fchips" id="mf-dirs">${j.dirs.map((d) => html`
            <button class="chip" title=${d.path} onClick=${() => openPath(d.path)}><${Icon} name="folder"/>${d.name}</button>`)}</div>` : ""}
        ${j && !start ? html`
          <div class="dsec">${t("Model files")}<span>${t(j.kind === "dir" ? "here and two folders down" : "")}</span></div>
          ${rows.length ? html`<div class="mrows" id="mf-cached" role="listbox">${rowsView(rows)}</div>`
            : singleMmproj || singleDrafter ? ""
            : html`<div class="empty">${t(allModels.length ? "Nothing matches." : "No .gguf or .flyx model files here.")}</div>`}
          ${singleDrafter ? html`<div class="banner hint" style="margin-top: 10px"><div class="grow2">${t(singleDrafter.file + " is a drafter (" + gb(singleDrafter.size) + " GB), not a model: it drafts for one, and is not loaded on its own.")}</div></div>` : ""}
          ${singleMmproj ? html`<div class="banner hint" style="margin-top: 10px"><div class="grow2">${t(singleMmproj.file + " is a vision projector (" + gb(singleMmproj.size) + " GB), not a standalone model: attach it to a compatible VLM.")}</div></div>` : ""}
          ${!singleDrafter && !singleMmproj && localComps.length ? html`
            <div class="dsec">${t("Companion files")}<span>${t("pick up to one drafter and one vision projector")}</span></div>
            ${companionsView("lc-companions", localComps, localDraft, setLocalDraft, localMmproj, setLocalMmproj)}` : ""}` : ""}
      </div>
      ${loadBox(rows, loadNote, localBar.action, localBar.overrides)}`;
  }

  const hubTab = tab === "hub";
  const pickedModel = picked ? places.flatMap(modelsOf).find((m) => m.name === picked) : null;
  const hubDetailMatch = hubView.detail && hubView.detail.repo === hubView.sel && hubView.detail.rev === hubView.rev ? hubView.detail.j : null;
  const shownRows = hubTab ? (hubDetailMatch ? (hubDetailMatch.ggufs || []).filter(whole).map((g) => g.name) : [])
                  : pickedModel ? [pickedModel.name]
                  : atCache ? modelsOf(cacheDir).filter(standalone).map((m) => m.name)
                  : dir ? dir.models.filter(standalone).map((m) => m.name) : [];
  const cmdModel = shownRows.includes(target) ? target : loadedWant ? loadedWant.model : L.remembered ? L.remembered.model : "";
  // The Hub's adapters, asked for when the LoRA tab is shown, again for another model or another base.  A base typed
  // by hand is the model's then shown: another model's is its own again.
  useEffect(() => setLoraBase(""), [cmdModel]);
  const loraKey = view === "current" && stab === "lora" ? (loraBase || "model:" + cmdModel) : "";
  useEffect(() => { if (loraKey) findLoras(loraBase, cmdModel); }, [loraKey]);
  // The drafters verified for the model the settings are for, when it is a Hub repository's: offered beside the default.
  const cmdRepo = cmdModel && isHub(cmdModel) ? repoNameOf(cmdModel) : "";
  const cmdRev = cmdRepo ? revOf(cmdModel, hubEndpoint) : "";
  const cmdRepoKey = cmdRepo ? repoSpec(cmdRepo, cmdRev) : "";
  useEffect(() => {
    if (!cmdRepoKey || verifiedOf[cmdRepoKey]) return;
    const wantRev = cmdRev && cmdRev !== "main" ? cmdRev : "";
    if (hubView.detail && hubView.detail.repo === cmdRepo && (hubView.detail.rev || "") === wantRev) {
      setVerifiedOf((v) => Object.assign({}, v, { [cmdRepoKey]: hubView.detail.j.known_drafters || [] }));
      return;
    }
    let live = true;
    ask("/v1/hub/files?repo=" + encodeURIComponent(cmdRepoKey), "list a repository").then((a) => {
      if (live && !a.error) setVerifiedOf((v) => Object.assign({}, v, { [cmdRepoKey]: a.j.known_drafters || [] }));
    });
    return () => { live = false; };
  }, [cmdRepoKey, hubView.detail]);
  const again = L.remembered ? { model: L.remembered.model, load: () => loadModel(L.remembered.model, L.remembered) } : null;
  const retryWant = L.target
    ? (lastAttempt.current && lastAttempt.current.model === L.target ? lastAttempt.current
       : L.remembered && L.remembered.model === L.target ? L.remembered
       : { model: L.target })
    : L.remembered || null;
  const retry = retryWant ? () => loadModel(retryWant.model, retryWant) : null;
  const current = view === "current";
  const stateTone = L.state === "ready" ? "ok" : L.state === "failed" ? "bad" : busy ? "warn" : "";
  const stateWord = L.state === "fetching" ? t("Fetching") + (L.progress && L.progress.total ? " " + fmt(100 * L.progress.done / L.progress.total) + "%" : "") :
                    L.state === "loading" ? t("Loading") : L.state === "failed" ? t("Failed") : model.id ? model.id : "";
  return html`
    <div class="tabs mtabs" role="tablist" id="mv-tabs">
      <button role="tab" data-view="current" aria-selected=${String(current)} onClick=${() => setView("current")}>
        ${t("Current model")}${stateWord ? html`<span class=${"badge " + stateTone} id="mv-state"><span class="dot"></span>${stateWord}</span>` : ""}</button>
      <button role="tab" data-view="library" aria-selected=${String(!current)} onClick=${() => setView("library")}>${t("Model library")}</button>
    </div>
    <div class="mview" id="mv-current" hidden=${!current}>
      <div class="sec"><div class="pane statbar"><${Serving} tm=${tm} again=${again} retry=${retry} browse=${() => setView("library")} winPaths=${winPaths}/></div></div>
      ${model.id ? html`<div class="sec">${settingsView(cmdModel)}</div>` : ""}
    </div>
    <div class="sec hubsec" id="mv-library" hidden=${current}>
      <div class="pane hub">
        <div class="hub-h">
          <div class="seg" role="tablist" id="src-seg">
            <button role="tab" data-src="hub" aria-selected=${String(hubTab)} onClick=${() => setTab("hub")}><${Icon} name="hf"/>Hugging Face</button>
            <button role="tab" data-src="local" aria-selected=${String(!hubTab)} onClick=${() => setTab("local")}>
              <${Icon} name="machine"/>${t("On this machine")}${local.length ? html`<span class="n" id="mf-count">${local.filter(standalone).length}</span>` : ""}</button>
          </div>
          <label class="search"><${Icon} name="search"/>
            ${hubTab ? html`<input type="text" id="hf-q" placeholder=${t("Search Hugging Face, or give a repository: org/name")} spellcheck="false" autocomplete="off"
                                   value=${q} onInput=${(e) => { dispatchHub({ type: "unpin" }); setQ(e.target.value); }}
                                   onKeyDown=${(e) => { if (e.key === "Enter") { e.preventDefault(); dispatchHub({ type: "unpin" }); if (!named(q)) search(q, sort, arch, 0, ""); } }}/>`
                     : html`<input type="text" id="mf-q" placeholder=${t("Filter by name or quantization")} spellcheck="false" autocomplete="off"
                                   value=${localQ} onInput=${(e) => setLocalQ(e.target.value)}/>`}</label>
          ${hubTab ? html`<select id="hf-sort" aria-label=${t("Order")} value=${sort} onChange=${(e) => { dispatchHub({ type: "unpin" }); setSort(e.target.value); }}>
            ${SORTS.map(([k, name]) => html`<option value=${k}>${t(name)}</option>`)}</select>` : null}
          ${hubTab ? html`<button type="button" class=${"btn epbtn" + (isCustomEndpoint ? " on" : "")} id="hf-ep-toggle"
                    aria-expanded=${String(epOpen)} title=${t("Hugging Face mirror or endpoint") + ": " + hubEndpoint}
                    onClick=${() => { setEpOpen((v) => !v); setEpText(hubEndpoint); setEpMsg(""); }}>
              <${Icon} name="globe"/><span>${hostOf(hubEndpoint)}</span></button>` : null}
        </div>
        ${hubTab && epOpen ? html`
          <div class="hub-epbar" id="hf-epbar">
            <span class="eplabel">${t("Hub mirror")}</span>
            <input type="text" id="hf-ep-input" spellcheck="false" autocomplete="off"
                   placeholder="https://huggingface.co"
                   aria-label=${t("Hugging Face mirror or endpoint")}
                   value=${epText} onInput=${(e) => { setEpText(e.target.value); setEpMsg(""); }}
                   onKeyDown=${(e) => { if (e.key === "Enter") { e.preventDefault(); applyEndpoint(epText); } }}/>
            <button type="button" class="btn sm primary" id="hf-ep-save" disabled=${epSaving}
                    onClick=${() => applyEndpoint(epText)}>${t("Apply")}</button>
            ${isCustomEndpoint || (hubCfg && hubCfg.source === "setting") ? html`
              <button type="button" class="btn sm" id="hf-ep-reset" disabled=${epSaving}
                      onClick=${() => applyEndpoint("")}>${t("Reset")}</button>` : null}
            <span class="ep-presets">
              <button type="button" class="chip" data-ep="https://huggingface.co" onClick=${() => applyEndpoint("https://huggingface.co")}>huggingface.co</button>
              <button type="button" class="chip" data-ep="https://hf-mirror.com" onClick=${() => applyEndpoint("https://hf-mirror.com")}>hf-mirror.com</button>
            </span>
            ${epMsg ? html`<span class="ep-msg bad" id="hf-ep-msg">${t(epMsg)}</span>`
                    : hubCfg && hubCfg.source === "HF_ENDPOINT" ? html`<span class="ep-msg" id="hf-ep-msg">${t("from HF_ENDPOINT")}</span>` : null}
          </div>` : null}
        <div class="hub-b">
          <div class="hub-list" id=${hubTab ? "hf-list" : "lc-list"}>${hubTab ? hubList() : localList()}</div>
          <div class="hub-detail">${hubTab ? hubDetail() : pickedModel ? modelDetail(pickedModel) : localDetail()}</div>
        </div>
      </div>
    </div>`;
}
