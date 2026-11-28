// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The Logs page: what the server writes to standard error (GET /v1/logs), as a table that follows
// the newest line.  Only the lines after the last one seen are asked for.
import { html, memo, useState, useEffect, useRef, useLayoutEffect } from "../lib/html.js";
import { headers } from "../lib/api.js";
import { t, currentLang } from "../lib/i18n.js";
import { fmt } from "../lib/format.js";
import { Icon } from "../lib/ui.js";
import { live } from "../lib/telemetry.js";

const SHOWN = 4000;  // lines the page keeps; the server keeps 2000
// A line's level, from its words: the lines carry none.
const levelOf = (s) => /\b(error|failed|fatal)\b|did not load|cannot /i.test(s) ? "error" : /\bwarn(ing)?\b/i.test(s) ? "warn" : "";
// Who wrote a line: the server, the engine (its request summaries), or the rest (the backend, the CUDA runtime).
function sourceOf(text) {
  const m = /^([\w.-]+): ([\s\S]*)$/.exec(text);
  if (!m) return { who: "", body: text };
  return { who: m[1] === "flyx-server" ? "server" : m[1] === "flyx" ? "engine" : m[1], body: m[2] };
}
const lineOf = (l) => ({ ...l, ...sourceOf(l.text), lv: levelOf(l.text), low: l.text.toLowerCase() });

// The engine's request summaries are fields joined by " | ": the joins dimmed, the fields easy to tell apart.
const Message = ({ body }) => html`<td class="msg">${body.split(" | ").map((p, i) => html`${i ? html`<span class="sep">  |  </span>` : ""}${p}`)}</td>`;

// `lang` only tells the row to draw again in another language.
const Row = memo(({ l }) => {
  const d = new Date(l.t * 1000);
  return html`
    <tr class=${"logline" + (l.lv ? " " + l.lv : "")}>
      <td class="tm" title=${d.toLocaleString()}>${d.toTimeString().slice(0, 8)}</td>
      <td class="lv"><span class=${"lvb " + (l.lv || "info")}>${t(l.lv === "error" ? "Error" : l.lv === "warn" ? "Warning" : "Info")}</span></td>
      <td class="src">${l.who}</td>
      <${Message} body=${l.body}/>
    </tr>`;
});

export function Logs({ active }) {
  const [lines, setLines] = useState([]);
  const [note, setNote] = useState("");
  const [q, setQ] = useState("");
  const [lv, setLv] = useState("");
  const box = useRef(null), next = useRef(0), busy = useRef(false), seen = useRef(false), lastTop = useRef(0);
  const atBottom = useRef(true);  // the reader is at the newest line: new lines keep the view there

  async function fetchLogs() {
    if (busy.current) return;
    busy.current = true;
    try {
      const r = await fetch("/v1/logs?after=" + next.current + "&limit=2000", { headers: headers(), cache: "no-store" });
      if (r.status === 401 || r.status === 403)
        return setNote("The log is for this machine, with --api-key, or with --no-auth on a trusted LAN.");
      if (r.status === 404) return setNote("This server keeps no system log.");
      if (!r.ok) return;
      const j = await r.json();
      if (!j.enabled) return setNote("The system log is off: flyx serve was started with --no-system-log.");
      if (j.next < next.current) {  // a new server: its lines from the start
        next.current = 0;
        setLines([]);
        busy.current = false;
        return fetchLogs();
      }
      setNote(j.missed && seen.current ? "Older lines were dropped: the server keeps the last 2,000." : "");
      seen.current = true;
      if (j.lines.length) {
        next.current = j.next;
        setLines((old) => old.concat(j.lines.map(lineOf)).slice(-SHOWN));
      }
    } catch (e) { /* the server is away: the poll says so */ }
    finally { busy.current = false; }
  }
  // While the page is shown: once a second, with the telemetry.
  useEffect(() => { if (active) fetchLogs(); }, [active, live.last]);

  const ql = q.trim().toLowerCase();
  const shown = lines.filter((l) => (!lv || l.lv === lv) && (!ql || l.low.includes(ql)));
  // As a terminal: at the bottom, new lines keep the view at the newest; scrolled up, the view stays where the reader
  // is; back at the bottom, it follows again.  No switch.  Only a move up leaves the bottom: the table growing,
  // shrinking or cleared also scrolls the box, and is not the reader.
  useLayoutEffect(() => { if (atBottom.current && box.current) box.current.scrollTop = box.current.scrollHeight; });
  const scrolled = () => {
    const b = box.current, up = b.scrollTop < lastTop.current;
    lastTop.current = b.scrollTop;
    const atEnd = b.scrollHeight - b.scrollTop - b.clientHeight < 24;
    if (atEnd) atBottom.current = true;
    else if (up) atBottom.current = false;
  };
  const download = () => {
    const text = shown.map((l) => new Date(l.t * 1000).toISOString() + "  " + l.text).join("\n") + "\n";
    const a = document.createElement("a"), stamp = new Date().toISOString().replace(/[-:]/g, "").slice(0, 15);
    a.href = URL.createObjectURL(new Blob([text], { type: "text/plain" }));
    a.download = "flyx-" + stamp + ".log";
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };
  const levels = [["", "All"], ["error", "Errors"], ["warn", "Warnings"]];
  return html`
    <div class="pane syslog">
      <div class="lbar">
        <h2>${t("Logs")}</h2>
        <label class="search"><${Icon} name="search"/>
          <input type="text" id="log-q" placeholder=${t("Filter the log")} spellcheck="false" autocomplete="off"
                 value=${q} onInput=${(e) => setQ(e.target.value)}/></label>
        <div class="apis" role="tablist" id="log-levels">
          ${levels.map(([k, name]) => html`
            <button data-lv=${k} aria-selected=${String(lv === k)} onClick=${() => setLv(k)}>${t(name)}</button>`)}
        </div>
        <span class="aside" id="log-count" title=${t("Lines shown. The server keeps the last 2,000 lines, in memory.")}>${fmt(shown.length) + (shown.length !== lines.length ? " / " + fmt(lines.length) : "") + " "}<span>${t("lines")}</span></span>
        <button class="iconbtn" id="log-dl" title=${t("Download")} aria-label=${t("Download")} onClick=${download}><${Icon} name="dl"/></button>
      </div>
      <div class="reqnote" id="log-note" hidden=${!note}>${t(note)}</div>
      <div class="logbody" id="log-body" ref=${box} onScroll=${scrolled}>
        <table class="ltable">
          <colgroup><col style="width: 96px"/><col style="width: 86px"/><col style="width: 82px"/><col/></colgroup>
          <thead><tr><th>${t("Time")}</th><th>${t("Level")}</th><th>${t("Source")}</th><th>${t("Message")}</th></tr></thead>
          <tbody id="log-rows">${shown.map((l) => html`<${Row} key=${l.seq} l=${l} lang=${currentLang()}/>`)}</tbody>
        </table>
        <div class="logempty" id="log-empty" hidden=${lines.length > 0}>${t("Nothing logged yet.")}</div>
      </div>
    </div>`;
}
