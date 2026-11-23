// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The Requests page: the server's request log (GET /v1/requests) as a list, and a reader beside it
// that shows one request as a conversation, or its JSON.  Without the log, the list is telemetry's recent requests.
import { html, useState, useEffect, useRef, useLayoutEffect } from "../lib/html.js";
import { headers } from "../lib/api.js";
import { t } from "../lib/i18n.js";
import { fmt, secs, clock } from "../lib/format.js";
import { Icon, statusName } from "../lib/ui.js";
import { live, ttftSeconds } from "../lib/telemetry.js";
import { Conversation, figures, stepsOf } from "../components/conversation.js";

function lastUser(e) {
  const st = stepsOf(e).filter((x) => x.kind === "user");
  return st.length ? st[st.length - 1].text.replace(/\s+/g, " ").trim() : "";
}
const apiName = (api) => api === "completions" ? "text" : api;

function Reader({ e, tab, setTab, wide, setWide, close, step }) {
  const body = useRef(null), shownFor = useRef("");
  const f = figures(e);
  const [linked, setLinked] = useState(false);
  // A new request or tab starts at the top; the same one, drawn again, keeps where the reader was.
  useLayoutEffect(() => {
    if (shownFor.current !== e.id + tab && body.current) body.current.scrollTop = 0;
    shownFor.current = e.id + tab;
  });
  const copyLink = async () => {
    try { await navigator.clipboard.writeText(location.origin + location.pathname + location.search + "#requests:" + encodeURIComponent(e.id)); setLinked(true); } catch (x) { /* none */ }
    setTimeout(() => setLinked(false), 1200);
  };
  const raw = tab === "req" ? e.request : tab === "res" ? Object.fromEntries(Object.entries(e).filter(([k]) => k !== "request")) : null;
  return html`
    <div class="pane reader" id="reader" aria-label=${t("Request")}>
      <div class="lbar rbar">
        <button class="iconbtn" id="rd-prev" title=${t("Previous (k or ↑)")} aria-label=${t("Previous")} onClick=${() => step(-1)}><${Icon} name="up"/></button>
        <button class="iconbtn" id="rd-next" title=${t("Next (j or ↓)")} aria-label=${t("Next")} onClick=${() => step(1)}><${Icon} name="down"/></button>
        <div class="ttl"><b id="rd-title">${apiName(e.api) + "  ·  " + clock(e.finished_at)}</b></div>
        <dl class="facts" id="rd-facts">
          ${[["TTFT", secs(f.ttft)], ["Speed", f.tps ? fmt(f.tps, 1) + " tok/s" : "-"],
             ["Tokens", fmt(f.u.prompt_tokens) + " → " + fmt(f.u.output_tokens)]].map(([k, v]) => html`<div><dt>${t(k)}</dt><dd>${v}</dd></div>`)}
        </dl>
        <button class=${"iconbtn" + (linked ? " on" : "")} id="rd-link" title=${t("Copy a link to this request")} aria-label=${t("Copy a link")}
                onClick=${copyLink}><${Icon} name="link"/></button>
        <button class="iconbtn" id="rd-wide" title=${t("Fill the page")} aria-label=${t("Fill the page")} onClick=${() => setWide(!wide)}><${Icon} name="wide"/></button>
        <button class="iconbtn" id="rd-close" title=${t("Close (Esc)")} aria-label=${t("Close")} onClick=${close}><${Icon} name="close"/></button>
      </div>
      <div class="tabs rtabs" role="tablist" id="rd-tabs">
        ${[["conv", "Conversation"], ["req", "Request JSON"], ["res", "Response JSON"]].map(([k, name]) => html`
          <button role="tab" data-dt=${k} aria-selected=${String(tab === k)} onClick=${() => setTab(k)}>${t(name)}</button>`)}
      </div>
      <div class="rbody" id="rd-body" ref=${body}>
        ${tab === "conv" ? html`<${Conversation} e=${e}/>` : html`<pre class="raw">${JSON.stringify(raw, null, 2)}</pre>`}
      </div>
    </div>`;
}

export function Requests({ active, openId, select }) {
  const [log, setLog] = useState({ state: "unknown", requests: null });
  const [tab, setTab] = useState("conv");
  const [wide, setWide] = useState(false);
  const [q, setQ] = useState("");
  const [api, setApi] = useState("");
  const etag = useRef("");
  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    let timer;
    async function refresh() {
      try {
        const r = await fetch("/v1/requests?limit=50", {
          headers: headers(etag.current ? { "If-None-Match": etag.current } : {}),
          cache: "no-store", signal: controller.signal,
        });
        if (controller.signal.aborted || r.status === 304) return;
        if (r.status === 404 || r.status === 403 || r.status === 401) {
          etag.current = "";
          setLog({ state: r.status === 404 ? "absent" : "denied", requests: null });
        } else if (r.ok) {
          const j = await r.json();
          if (controller.signal.aborted) return;
          etag.current = r.headers.get("ETag") || "";
          setLog({ state: j.enabled ? "on" : "off", requests: j.enabled ? j.requests : null });
        }
      } catch (e) { /* keep what was shown */ }
      finally { if (!controller.signal.aborted) timer = setTimeout(refresh, 2000); }
    }
    refresh();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [active]);

  const ql = q.trim().toLowerCase();
  const list = (log.requests || []).filter((e) => (!api || e.api === api) &&
    (!ql || (lastUser(e) + " " + (e.content || "") + " " + e.id + " " + (e.tool_calls || []).map((c) => c.name).join(" ")).toLowerCase().includes(ql)));
  const open = openId && (log.requests || []).find((e) => e.id === openId);

  const show = (id) => {
    select(id);
    if (!id) setWide(false);
  };
  const step = (d) => {
    const i = list.findIndex((e) => e.id === openId);
    if (i >= 0 && list[i + d]) show(list[i + d].id);
  };
  useEffect(() => {
    const row = document.querySelector("#req-rows tr.open");
    if (row) row.scrollIntoView({ block: "nearest" });
  }, [openId]);
  // Esc closes the reader; j / k (or the arrows) move through the list.
  useEffect(() => {
    if (!active || !openId) return;
    const key = (e) => {
      const tag = (e.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea") return;
      if (e.key === "Escape") show(null);
      else if (e.key === "j" || e.key === "ArrowDown") { e.preventDefault(); step(1); }
      else if (e.key === "k" || e.key === "ArrowUp") { e.preventDefault(); step(-1); }
    };
    document.addEventListener("keydown", key);
    return () => document.removeEventListener("keydown", key);
  });

  const tm = live.last || {}, recent = tm.recent || [];
  const notes = { off: "Prompts and answers are not kept: the server runs with --no-request-log. Figures come from telemetry.",
                  absent: "This server keeps no request log (GET /v1/requests). Figures come from telemetry.",
                  denied: "The request log is readable from the server's machine, or with its API key." };
  const empty = (msg) => html`<tr><td colspan="7" class="empty">${t(msg)}</td></tr>`;
  let rows;
  if (log.requests) {
    rows = !list.length ? empty(log.requests.length ? "Nothing matches." : "No requests yet.") : list.map((e) => {
      const f = figures(e);
      const out = e.content ? e.content.replace(/\s+/g, " ") : (e.tool_calls || []).length ? "→ " + e.tool_calls.map((c) => c.name).join(", ") : e.error || "—";
      return html`
        <tr key=${e.id} data-id=${e.id} class=${e.id === openId ? "open" : ""} onClick=${() => show(e.id)}>
          <td class="mono" title=${statusName(e) + (e.error ? ": " + e.error : "")}><span class=${"sdot " + e.status}></span>${clock(e.finished_at)}</td>
          <td class="txt1">${lastUser(e) || "—"}</td>
          <td class="txt1 muted wide-only">${out}</td>
          <td class="n wide-only">${fmt(f.u.prompt_tokens) + " → " + fmt(f.u.output_tokens)}</td>
          <td class="n wide-only">${secs(f.ttft)}</td>
          <td class="n wide-only">${f.tps ? fmt(f.tps, 1) : "-"}</td>
          <td class="n">${apiName(e.api)}</td>
        </tr>`;
    });
  } else {
    rows = !recent.length ? empty("No requests yet.") : recent.map((r) => html`
      <tr><td class="mono" title=${statusName(r)}><span class=${"sdot " + r.status}></span>${clock(r.finished_at)}</td>
        <td class="txt1 muted">—</td><td class="txt1 muted wide-only">—</td>
        <td class="n wide-only">${fmt(r.prompt_tokens) + " → " + fmt(r.output_tokens)}</td>
        <td class="n wide-only">${secs(ttftSeconds(r))}</td>
        <td class="n wide-only">${r.decode_tok_s > 0 ? fmt(r.decode_tok_s, 1) : "-"}</td><td class="n">${r.label}</td></tr>`);
  }
  const aside = log.requests ? t(log.requests.length + " kept in memory") : t("the last " + recent.length + " of " + fmt((tm.totals || {}).requests));
  return html`
    <div class=${"logs" + (open ? " open" : "") + (open && wide ? " wide" : "")} id="logs">
      <div class="pane list">
        <div class="lbar">
          <h2>${t("Requests")}</h2>
          <label class="search"><${Icon} name="search"/>
            <input type="text" id="req-q" placeholder=${t("Search prompts and answers")} spellcheck="false" autocomplete="off"
                   value=${q} onInput=${(e) => setQ(e.target.value)}/></label>
          <div class="apis" role="tablist" id="req-apis">
            ${[["", "All"], ["chat", "Chat"], ["messages", "Messages"], ["responses", "Responses"]].map(([k, name]) => html`
              <button data-api=${k} aria-selected=${String(api === k)} onClick=${() => setApi(k)}>${t(name)}</button>`)}
          </div>
          <span class="aside wide-only" id="req-aside">${aside}</span>
        </div>
        <div class="reqnote" id="req-note" hidden=${!notes[log.state]}>${t(notes[log.state] || "")}</div>
        <div class="lscroll">
          <table>
            <colgroup><col style="width: 92px"/><col/><col class="wide-only"/><col class="wide-only" style="width: 120px"/>
              <col class="wide-only" style="width: 84px"/><col class="wide-only" style="width: 84px"/><col style="width: 86px"/></colgroup>
            <thead><tr><th>${t("Time")}</th><th>${t("Input")}</th><th class="wide-only">${t("Output")}</th><th class="n wide-only">${t("Tokens")}</th>
              <th class="n wide-only">${t("TTFT")}</th><th class="n wide-only">${t("Speed")}</th><th class="n">${t("API")}</th></tr></thead>
            <tbody id="req-rows">${rows}</tbody>
          </table>
        </div>
      </div>
      ${open ? html`<${Reader} e=${open} tab=${tab} setTab=${setTab} wide=${wide} setWide=${setWide} close=${() => show(null)} step=${step}/>`
             : html`<div class="pane reader" id="reader"></div>`}
    </div>`;
}
