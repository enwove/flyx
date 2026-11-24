// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// On-demand GPU process viewer and safe local app closer (GET /v1/device/processes, POST /v1/device/close).
// Shown automatically below VRAM OOM load errors, via a toggle on GPU context-limit banners, and from Overview's Host
// card. Close buttons appear only when the server confirms the request came from this machine (`can_close`).
import { html, useState, useEffect, useRef } from "../lib/html.js";
import { call } from "../lib/api.js";
import { t } from "../lib/i18n.js";
import { gb } from "../lib/format.js";
import { Icon, confirmBox } from "../lib/ui.js";
import { poll } from "../lib/telemetry.js";

const INITIAL_LIMIT = 8;

function fmtVram(bytes) {
  if (!bytes || bytes <= 0) return "0 MB";
  if (bytes >= 1e9) return gb(bytes) + " GB";
  return Math.max(1, Math.round(bytes / 1e6)) + " MB";
}

function currentBrowserProc() {
  const ua = navigator.userAgent || "";
  if (/Edg\//i.test(ua)) return "msedge";
  if (/OPR\//i.test(ua) || /Opera/i.test(ua)) return "opera";
  if (/Firefox\//i.test(ua)) return "firefox";
  if (/Brave/i.test(ua)) return "brave";
  if (/Chrome\//i.test(ua)) return "chrome";
  return "";
}

// Parse how many bytes an OOM error asked to free (0 when not stated).
export function parseVramNeeded(err) {
  if (!err) return 0;
  const m = /free at least ([\d.]+) GB VRAM|([\d.]+) GB short/.exec(err);
  const v = m ? parseFloat(m[1] || m[2]) : 0;
  return Number.isFinite(v) && v > 0 ? Math.round(v * 1e9) : 0;
}

export function isVramError(err) {
  return !!err && /(?:^|:\s)Not enough VRAM/.test(err);
}

export function VramApps({ open, kind = "vram", needBytes = 0, onRetry, retryLabel = "Retry loading", onClose, id = "vram-apps", modal = false }) {
  const isRam = kind === "ram";
  const [snap, setSnap] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [busyId, setBusyId] = useState("");
  const [rowState, setRowState] = useState({});  // app id -> { kind: "still_running"|"no_window"|"error", msg? }
  const [showAll, setShowAll] = useState(false);
  const baseFree = useRef(null);
  const reqSeq = useRef(0);

  async function refresh() {
    const n = ++reqSeq.current;
    setLoading(true);
    setErr("");
    const url = isRam ? "/v1/device/processes?kind=ram" : "/v1/device/processes";
    const r = await call("GET", url);
    if (n !== reqSeq.current) return;
    setLoading(false);
    if (!r.ok) {
      setErr(r.status === 404 && !r.json
        ? (isRam ? "Update FLYX to view apps using RAM." : "Update FLYX to view apps using VRAM.")
        : (r.error || (isRam ? "Could not inspect running apps." : "Could not inspect GPU processes.")));
      return;
    }
    setSnap(r.json);
    if (baseFree.current === null && r.json && r.json.mem_free > 0) baseFree.current = r.json.mem_free;
  }

  useEffect(() => {
    baseFree.current = null;
    setSnap(null);
    setRowState({});
    setShowAll(false);
    if (!open) return;
    refresh();
  }, [open, kind]);

  if (!open) return "";

  const browserProc = currentBrowserProc();
  const isWin = /Windows/i.test(navigator.userAgent || "");
  const apps = (snap && snap.apps) || [];
  const visibleApps = showAll ? apps : apps.slice(0, INITIAL_LIMIT);
  const hiddenCount = apps.length - visibleApps.length;
  const freed = snap && baseFree.current !== null ? Math.max(0, (snap.mem_free || 0) - baseFree.current) : 0;
  const enoughFreed = needBytes > 0 && freed >= needBytes;

  async function closeApp(app, force) {
    const isBrowser = !!browserProc && (app.process || "").toLowerCase() === browserProc;
    if (force) {
      const ok = await confirmBox(
        "Force quit " + app.name + "?",
        ["Any unsaved work in " + app.name + " may be lost.",
         isRam ? "All processes of " + app.name + " will be ended to free memory."
               : "All processes of " + app.name + " will be ended to free GPU memory."],
        "Force quit", true);
      if (!ok) return;
    } else if (isBrowser) {
      const ok = await confirmBox(
        "Close " + app.name + "?",
        ["This dashboard appears to be open in " + app.name + ".",
         "Closing it may also close this page."],
        "Close");
      if (!ok) return;
    }
    setBusyId(app.id);
    const payload = isRam ? { id: app.id, force: !!force, kind: "ram" } : { id: app.id, force: !!force };
    const r = await call("POST", "/v1/device/close", payload);
    setBusyId("");
    if (!r.ok) {
      if (r.status === 404) {
        await refresh();
        poll();
        return;
      }
      setRowState((prev) => Object.assign({}, prev, { [app.id]: { kind: "error", msg: r.error || "Could not close this app." } }));
      return;
    }
    const j = r.json || {};
    if (j.snapshot) setSnap(j.snapshot);
    poll();
    if (j.status === "closed") {
      setRowState((prev) => {
        const next = Object.assign({}, prev);
        delete next[app.id];
        return next;
      });
    } else if (j.status === "no_window" || j.status === "still_running") {
      setRowState((prev) => Object.assign({}, prev, { [app.id]: { kind: j.status } }));
    }
  }

  return html`
    <div class=${"vapps" + (modal ? " in-modal" : "")} id=${id} data-kind=${kind}>
      <div class="vapps-h">
        <div class="vapps-t">
          <div class="vapps-title">
            <b>${t(isRam ? "Apps using RAM" : "Apps using VRAM")}</b>
            ${freed >= 16e6 ? html`<span class=${"tag2 " + (enoughFreed ? "ok" : "warn")}>
              ${t("+" + fmtVram(freed) + " freed")}</span>` : ""}
          </div>
          ${snap && snap.mem_total ? html`<div class="vapps-sum">
            ${t(fmtVram(snap.others_bytes) + " by other apps · " + fmtVram(snap.mem_free) + " free")}</div>` : ""}
        </div>
        <div class="vapps-acts">
          <button class="btn sm" id=${id + "-refresh"} disabled=${loading || !!busyId} onClick=${refresh}>
            <${Icon} name="reload"/>${t("Refresh")}</button>
          ${onRetry ? html`<button class=${"btn sm" + (enoughFreed || freed >= 64e6 ? " primary" : "")} id=${id + "-retry"}
            disabled=${!!busyId} onClick=${onRetry}>${t(retryLabel)}</button>` : ""}
          ${onClose ? html`<button class="iconbtn" title=${t("Close")} aria-label=${t("Close")} onClick=${onClose}>
            <${Icon} name="close"/></button>` : ""}
        </div>
      </div>
      ${err ? html`<div class="vapps-msg bad">${t(err)}</div>` : ""}
      ${!snap && loading ? html`<div class="vapps-msg">${t(isRam ? "Checking running apps..." : "Checking GPU processes...")}</div>` : ""}
      ${snap && !snap.supported ? html`<div class="vapps-msg">${t(isRam ? "Memory process details are unavailable on this system." : "GPU process details are unavailable on this system.")}</div>` : ""}
      ${snap && snap.supported ? html`
        <div class="vapps-list">
          ${!apps.length && !snap.system_bytes ? html`<div class="vapps-msg">${t(isRam ? "No other apps are using significant RAM." : "No other apps are using significant VRAM.")}</div>` : ""}
          ${visibleApps.map((app) => {
            const st = rowState[app.id];
            const isBrowser = !!browserProc && (app.process || "").toLowerCase() === browserProc;
            const busy = busyId === app.id;
            const needForce = !!(st && (st.kind === "still_running" || st.kind === "no_window"));
            const note = st && st.kind === "still_running" ? t("Still running (tray or unsaved prompt)")
                       : st && st.kind === "no_window" ? t("Background app (no window)")
                       : st && st.kind === "error" ? t(st.msg)
                       : app.reason === "runs_flyx" ? t("Runs FLYX")
                       : app.reason === "other_user" ? t("Other user")
                       : isBrowser ? t("Current browser")
                       : app.processes > 1 ? t(app.processes + " processes") : "";
            return html`
              <div class="vapps-row" data-app=${app.process || app.name}>
                <div class="vapps-name">
                  <span class="vn">${app.name}</span>
                  ${note ? html`<span class=${"vs" + (st && st.kind === "error" ? " bad" : needForce ? " warn" : "")}>${note}</span>` : ""}
                </div>
                <span class="vapps-mem">${fmtVram(app.bytes)}</span>
                <div class="vapps-act">
                  ${snap.can_close && app.closable ? html`
                    <button class=${"btn sm" + (needForce ? " danger" : "")} disabled=${busy || (!!busyId && !busy)}
                            onClick=${() => closeApp(app, needForce)}>
                      ${t(busy ? "Closing..." : needForce ? "Force quit" : "Close")}</button>`
                    : html`<span class="vapps-lock">${t(app.reason === "runs_flyx" ? "Protected" : !snap.can_close ? "" : "Protected")}</span>`}
                </div>
              </div>`;
          })}
          ${hiddenCount > 0 ? html`
            <button class="linkbtn vapps-more" onClick=${() => setShowAll(true)}>${t("Show " + hiddenCount + " more")}</button>` : ""}
          ${snap.system_bytes > 0 ? html`
            <div class="vapps-row sys">
              <div class="vapps-name">
                <span class="vn">${t(isRam ? (isWin ? "Windows system" : "System processes") : (isWin ? "Windows desktop & system" : "Desktop & system"))}</span>
                ${!isRam ? html`<span class="vs">${t("Shrinks when desktop apps close")}</span>` : ""}
              </div>
              <span class="vapps-mem">${fmtVram(snap.system_bytes)}</span>
              <div class="vapps-act"><span class="vapps-lock">${t("System")}</span></div>
            </div>` : ""}
        </div>
        ${!isRam && snap.idle_igpu ? html`<div class="vapps-tip">${t("Tip: Idle integrated GPU detected (" + snap.idle_igpu + "). Assigning browsers and chat apps to it in Windows Graphics settings keeps VRAM free.")}</div>` : ""}
      ` : ""}
    </div>`;
}

// Lightweight modal wrapper opened from the persistent sidebar VRAM/RAM rows, warning hints, or Overview's Host card.
export function VramModal({ open, kind = "vram", onClose, needBytes = 0, onRetry, retryLabel = "Reload model" }) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e) => {
      if (e.key === "Escape" && document.querySelectorAll(".mscrim").length <= 1) {
        e.stopPropagation();
        onClose();
      }
    };
    document.addEventListener("keydown", onKey, true);
    return () => document.removeEventListener("keydown", onKey, true);
  }, [open, onClose]);
  if (!open) return "";
  return html`
    <div class="mscrim vram-scrim" id="vram-modal" onClick=${(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div class="modal vram-modal" role="dialog" aria-modal="true" aria-label=${t(kind === "ram" ? "Apps using RAM" : "Apps using VRAM")}>
        <${VramApps} open=${open} kind=${kind} modal=${true} id="vram-modal-apps" needBytes=${needBytes}
                     onRetry=${onRetry} retryLabel=${retryLabel} onClose=${onClose}/>
      </div>
    </div>`;
}
