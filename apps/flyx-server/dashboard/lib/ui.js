// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Small parts every page uses: an icon from the sprite, a meter, a memory bar, a copy button, a confirmation over the
// page.
import { html, render, useState, useEffect, useRef } from "./html.js";
import { t } from "./i18n.js";
import { mem } from "./format.js";

// An icon of index.html's sprite (#i-NAME).
export const Icon = ({ name, cls }) => html`<svg class=${"i" + (cls ? " " + cls : "")}><use href=${"#i-" + name}/></svg>`;

// A bar filled to `frac` (0 to 1), coloured by `cls` (warn, hot).
export const Meter = ({ frac, cls }) =>
  html`<div class="bar"><i class=${cls || ""} style=${{ width: Math.max(0, Math.min(1, isFinite(frac) ? frac : 0)) * 100 + "%" }}></i></div>`;

// Memory as a bar of the whole: flyx's part, then the rest in use (telemetry's memoryOf()), the free part empty;
// under it, what each part holds (`bare`: in GiB without the unit, for a narrow column whose total names it).  A part
// too small to see is left out of the bar, not of the figures.
export function MemBar({ m, bare }) {
  const part = (b) => m && m.total ? Math.max(0, Math.min(1, b / m.total)) : 0;
  const own = part(m ? m.own : 0), others = part(m ? m.others : 0);
  const amount = bare ? (b) => (b / 1073741824).toFixed(1) : mem;
  const text = m ? "FLYX " + mem(m.own) + " · " + t("others") + " " + mem(m.others) + " · " + t("free") + " " +
                   mem(Math.max(0, m.total - m.used)) : "";
  return html`
    <div class="mbar" title=${text || undefined}>
      ${own > 0.002 ? html`<i class="own" style=${{ width: own * 100 + "%" }}></i>` : ""}
      ${others > 0.002 ? html`<i class="oth" style=${{ width: others * 100 + "%" }}></i>` : ""}</div>
    <div class="mleg">${m ? html`<span class="own">FLYX ${amount(m.own)}</span>
      <span class="oth">${t("others")} ${amount(m.others)}</span>` : ""}</div>`;
}

// A figure and its unit, the unit smaller; no unit after "-".
export const Val = ({ v, unit }) => html`${v}${unit && v !== "-" ? html`<small>${unit}</small>` : ""}`;

// A button that copies `get()` and says so for a moment.
export function CopyButton({ get, label, cls, title, icon = true }) {
  const [said, setSaid] = useState("");
  useEffect(() => { if (!said) return; const h = setTimeout(() => setSaid(""), 1400); return () => clearTimeout(h); }, [said]);
  const copy = async (e) => {
    e.stopPropagation();
    try { await navigator.clipboard.writeText(get()); setSaid("Copied"); } catch (x) { setSaid("Select it"); }
  };
  return html`<button class=${cls || "btn ghost sm"} title=${title || undefined} onClick=${copy}>${icon ? html`<${Icon} name="copy"/> ` : ""}${t(said || label || "Copy")}</button>`;
}

// The same string is shown and copied, including model names and paths with spaces.
export function CodeSnippet({ text, id }) {
  return html`<div class="snip"><pre id=${id}>${text}</pre><${CopyButton} cls="btn sm" icon=${false} get=${() => text}/></div>`;
}

export const defaultShell = () => /Windows/i.test(navigator.userAgent) ? "powershell" : "sh";

export function ShellPicker({ shell, onChange }) {
  return html`<div class="pills" role="group" aria-label=${t("Shell")}>
    ${[["powershell", "Windows PowerShell"], ["sh", "Bash / zsh"]].map(([id, label]) => html`
      <button data-shell=${id} aria-pressed=${String(shell === id)} onClick=${() => onChange(id)}>${label}</button>`)}
  </div>`;
}

// A request's status as a badge: its name, the error on hover.
const stName = { stop: "Done", tool_calls: "Tool calls", cancelled: "Cancelled", length: "Length", error: "Error" };
const stTone = { stop: "ok", tool_calls: "info", cancelled: "", length: "warn", error: "bad" };
export const statusName = (r) => t(r.cancel_reason === "superseded" ? "Replaced by a newer request" : stName[r.status] || r.status);
export const StatusBadge = ({ r }) => html`<span class=${"badge " + (stTone[r.status] || "")} title=${r.error || undefined}>${statusName(r)}</span>`;

function Confirm({ title, lines, ok, danger, done }) {
  const yes = useRef(null);
  useEffect(() => {
    yes.current && yes.current.focus();
    const key = (e) => { if (e.key === "Escape") { e.stopPropagation(); done(false); } };
    document.addEventListener("keydown", key, true);
    return () => document.removeEventListener("keydown", key, true);
  }, []);
  return html`
    <div class="mscrim" onClick=${(e) => { if (e.target === e.currentTarget) done(false); }}>
      <div class="modal" role="dialog" aria-modal="true">
        <h3>${t(title)}</h3>
        <ul>${lines.map((l) => html`<li>${t(l)}</li>`)}</ul>
        <div class="row">
          <button class="btn" onClick=${() => done(false)}>${t("Cancel")}</button>
          <button class=${"btn primary" + (danger ? " danger" : "")} ref=${yes} onClick=${() => done(true)}>${t(ok)}</button>
        </div>
      </div>
    </div>`;
}

// A small confirmation over the page (English strings, shown in the page's language): true on the main button.
export function confirmBox(title, lines, ok, danger) {
  return new Promise((resolve) => {
    const host = document.createElement("div");
    document.body.appendChild(host);
    const done = (v) => { render(null, host); host.remove(); resolve(v); };
    render(html`<${Confirm} title=${title} lines=${lines} ok=${ok} danger=${danger} done=${done}/>`, host);
  });
}
