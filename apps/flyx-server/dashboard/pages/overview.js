// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The Overview page: what runs now, throughput (decode and prefill) and speculative decoding over two minutes, the
// recent requests, the sessions on the GPU, and the engine, device and host.
import { html } from "../lib/html.js";
import { t, tErr } from "../lib/i18n.js";
import { fmt, gib, gb, ktok, secs, clock, level, shortDevice } from "../lib/format.js";
import { Meter, MemBar, StatusBadge, Icon } from "../lib/ui.js";
import { Chart } from "../lib/charts.js";
import { pendingConnect } from "../components/hints.js";
import { openRequest } from "../lib/actions.js";
import { live, phaseOf, phaseNames, phaseTone, ttftSeconds, prefillRate, cacheHit, acceptance, memoryOf } from "../lib/telemetry.js";

// What runs now: a sentence of parts (bold where it names a figure), what sits on the right, how far along it is.
function now(tm, phase) {
  const model = tm.model || {}, state = tm.state || {}, recent = tm.recent || [], L = tm.load || {};
  const prompt = state.prompt_tokens || 0, gen = state.generated || 0;
  const parts = [], side = [];
  const say = (s, bold) => parts.push(bold ? html`<b>${s}</b>` : t(s));
  let frac = 0;
  if (phase === "prefill" && prompt > 0) {
    frac = (state.prefilled || 0) / prompt;
    say("Reading the prompt: "); say(fmt(state.prefilled) + " / " + fmt(prompt), true); say(" tokens");
    if (state.reused_tokens) say(", " + fmt(state.reused_tokens) + " reused");
  } else if (phase === "decode") {
    const limit = state.max_new_tokens > 0 ? state.max_new_tokens : (model.context || 0) - prompt;
    frac = limit > 0 ? gen / limit : 0;
    say("Generating: "); say(fmt(gen), true); say(" tokens");
    if (state.max_new_tokens > 0) {
      say(" · "); say("Request output limit"); say(" "); say(fmt(state.max_new_tokens), true);
    }
    side.push(fmt(tm.rates.decode_tok_s, 1) + " tok/s");
  } else if (phase === "fetching") {
    const p = L.progress || {};
    frac = p.total ? p.done / p.total : 0;
    say(p.total ? "Fetching " : "Asking Hugging Face about "); say(p.total ? p.file : L.target, true);
    if (p.total) side.push(gb(p.done) + " of " + gb(p.total) + " GB");
    if (p.bytes_per_s) side.push(fmt(p.bytes_per_s / 1e6) + " MB/s");
  } else if (phase === "none") {
    if (L.state === "failed") { say("Load failed: "); say(tErr(L.error || ""), true); }
    else say("No model loaded. Choose one under Model.");
  } else if (phase === "loading") {
    say("Loading "); say(L.target || "the model", true);
  } else if (phase === "unloaded") {
    say("Unloaded after idling; the next request loads it again.");
  } else {
    say("Waiting for a request");
    const last = recent[0];
    if (last && last.status === "error") {  // said here, at the top: the request list is further down
      parts.push(html`<span class="now-err">${" · " + t("Last request failed") + (last.error ? ": " : ".")}${last.error || ""}</span>`);
      side.push(t("at " + clock(last.finished_at)));
    } else if (last) side.push(t("last one finished " + clock(last.finished_at)));
  }
  if (state.label) {
    say("  ·  " + state.label + ", session " + state.session);
    side.push(secs(state.elapsed_s));
  }
  if (state.queued) side.push(t(state.queued + " queued"));
  return { parts, side: side.join(" · "), frac };
}

// The two speeds over the throughput chart, decoding's and prefill's: of now, of the running request, or of the last.
function speeds(tm, phase, recent) {
  const state = tm.state || {}, last = recent[0] || {}, running = phase === "decode" || phase === "prefill";
  return [
    { v: phase === "decode" ? tm.rates.decode_tok_s : last.decode_tok_s, d: 1,
      what: phase === "decode" ? "decode tok/s, now" : recent.length ? "decode tok/s, last request" : "decode tok/s" },
    { v: running ? state.prefill_tok_s : prefillRate(last), d: 0,
      what: phase === "prefill" ? "prefill tok/s, now" : running ? "prefill tok/s, this request" :
            recent.length ? "prefill tok/s, last request" : "prefill tok/s" },
  ];
}

// Speculative decoding's two figures, tokens a round and the share of drafts accepted: of now (the recent window,
// matching decode tok/s), of the running request where a server reports no window, or of the last one.
function drafting(tm, phase, recent) {
  const state = tm.state || {}, rates = tm.rates || {}, last = recent[0] || {}, running = phase === "decode";
  const hasNow = running && Number.isFinite(rates.tokens_per_round) && rates.tokens_per_round > 0;
  const when = hasNow ? ", now" : running ? ", this request" : recent.length ? ", last request" : "";
  return [
    { v: hasNow ? rates.tokens_per_round : running ? state.tokens_per_round : last.tokens_per_round, d: 2, unit: "",
      what: "tok/round" + when },
    { v: hasNow ? (Number.isFinite(rates.acceptance) && rates.acceptance >= 0 ? rates.acceptance * 100 : null)
                : running ? acceptance(state.accepted, state.drafted) : acceptance(last.accepted_drafts, last.drafted),
      d: 0, unit: "%", what: "accepted" + when },
  ];
}
const figure = (v, d, unit) => v === null || v === undefined || !isFinite(v) ? "-" : fmt(v, d) + unit;

// A figure with a bar of its whole under it (the device's load and power).
const Gauge = ({ id, label, value, frac }) => html`
  <div class="memrow" id=${id}><div class="mk"><span>${t(label)}</span><span class="v">${value}</span></div>
    <div class="mbar">${frac > 0.002 ? html`<i class="own" style=${{ width: Math.min(1, frac) * 100 + "%" }}></i>` : ""}</div></div>`;

const Kv = ({ pairs }) => html`<dl class="kv">${pairs.map(([k, v]) => html`<dt>${t(k)}</dt><dd>${t(v)}</dd>`)}</dl>`;

export function Overview({ active, go, explore, showModel, onOpenMem }) {
  const tm = live.last;
  if (!tm) return html`<div class="now"><span class="badge">${t("Connecting")}</span><div class="what">-</div></div>`;
  const phase = phaseOf(tm), model = tm.model || {}, d = tm.device, recent = tm.recent || [], L = tm.load || {};
  const n = now(tm, phase);
  const tt = tm.totals || {};
  const sl = (tm.memory && tm.memory.sessions) || [];
  // Each second: tokens out, prompt tokens computed, and of the rounds verified, the tokens a round (the drafts
  // accepted and the token every round adds) and the share of their drafts accepted.  The chart's scale for tokens a
  // round reaches the widest drafter round's and what the second saw (rounds of drafts looked up in the text are
  // wider).
  const perSecond = active ? [...live.bySecond].sort((a, b) => a[0] - b[0]).map(([s, c]) => ({
    t: s, out: c.out, pre: c.pre, rounds: c.rounds,
    tpr: c.rounds > 0 ? 1 + (c.accepted || 0) / c.rounds : null, acc: c.rounds > 0 ? acceptance(c.accepted, c.drafted) : null,
  })) : [];
  const sys = tm.system || {}, memory = memoryOf(tm);
  const activeDraft = model.drafter && model.drafter !== "none" ? model.drafter : "";
  const draftSummary = activeDraft ? activeDraft + (model.draft_weights ? " · " + model.draft_weights : "") + ", up to " + model.max_width : "";
  return html`
    <div class="now">
      <span class=${"badge " + phaseTone(phase)} id="act-badge">${t(phaseNames[phase] || phase)}</span>
      <div class="what" id="act-what">${n.parts}</div>
      <div class="side-t">${n.side}${phase === "none" && L.state === "failed" ? html`
        <button class="btn sm" id="act-model" onClick=${showModel}>${t("View details")}</button>` : phase === "none" ? html`
        <button class="btn sm" id="act-explore" onClick=${explore}>${t("Explore the model library")}</button>` : phase === "idle" && recent[0] && recent[0].status === "error" ? html`
        <button class="btn sm" id="act-failed" onClick=${() => openRequest(go, recent[0].id)}>${t("View request")}</button>` : phase === "idle" && pendingConnect(tm) ? html`
        <button class="btn sm primary" id="act-connect" onClick=${() => go("integration")}>${t("Set up a client")} →</button>` : ""}</div>
      <div class="bar"><i style=${{ width: Math.max(0, Math.min(1, n.frac)) * 100 + "%" }}></i></div>
    </div>
    <div class="sec c-2-1">
      <div class="pane chart" id="ov-throughput">
        <div class="pane-h"><h2>${t("Throughput")}</h2><span class="legend"><span style="--c: var(--chart)">${t("decode, left axis")}</span>
          <span style="--c: var(--chart-2)">${t("prefill, right axis")}</span></span>
          <span class="aside">${t("last 2 minutes")}</span></div>
        <div class="fig">${speeds(tm, phase, recent).map((f) => html`<b>${fmt(f.v, f.d)}</b><span>${t(f.what)}</span>`)}</div>
        ${active ? html`<${Chart} samples=${perSecond}
                         series=${[{ key: "out", color: "--chart", fill: true }, { key: "pre", color: "--chart-2", axis: "right" }]}
                         opts=${{ span: 120, minTop: 10, right: { minTop: 100 } }}
                         describe=${(s) => t("decode") + " " + fmt(s.out) + (s.pre === null ? "" : " · " + t("prefill") + " " + fmt(s.pre)) + " tok"}
                         label=${t("tokens out and prompt tokens computed in each second of the last two minutes")}/>` : ""}
      </div>
      <div class="pane chart" id="ov-spec">
        <div class="pane-h"><h2>${t("Speculative decoding")}</h2>
          <span class="aside">${draftSummary ? t(draftSummary) :
                                model.loaded ? t("no drafter") : ""}</span></div>
        <div class="fig pair">${drafting(tm, phase, recent).map((f) => html`
          <div><b>${figure(f.v, f.d, f.unit)}</b><span>${t(f.what)}</span></div>`)}</div>
        <div class="legend"><span style="--c: var(--chart)">${t("tokens a round, left axis")}</span>
          <span style="--c: var(--chart-2)">${t("drafts accepted, right axis")}</span></div>
        ${active ? html`<${Chart} samples=${perSecond}
                         series=${[{ key: "tpr", color: "--chart", fill: true }, { key: "acc", color: "--chart-2", axis: "right" }]}
                         opts=${{ span: 120, minTop: Math.max(2, (model.max_width || 1) + 1), right: { top: 100, unit: "%" } }}
                         describe=${(s) => s.tpr === null ? t("no rounds") : fmt(s.tpr, 2) + " " + t("tok/round") +
                                           (s.acc === null ? "" : " · " + fmt(s.acc) + "% " + t("accepted")) +
                                           " · " + fmt(s.rounds) + " " + t("rounds")}
                         label=${t("tokens a round and drafts accepted in each second of the last two minutes")}/>` : ""}
      </div>
    </div>
    <div class="sec c-2-1">
      <div class="pane">
        <div class="pane-h"><h2>${t("Recent requests")}</h2>
          <span class="aside"><a href="#requests" id="ov-all" onClick=${(e) => { e.preventDefault(); go("requests"); }}>
            ${t("View all")}${tt.requests ? " (" + fmt(tt.requests) + ")" : ""}</a></span></div>
        <div class="tablewrap"><table class="flush">
          <thead><tr><th>${t("Finished")}</th><th>${t("API")}</th><th>${t("Status")}</th><th class="n">${t("Output")}</th>
            <th class="n">${t("Tok/s")}</th><th class="n">${t("TTFT")}</th><th class="n">${t("Cache hit")}</th>
            <th class="n">${t("Tok/round")}</th></tr></thead>
          <tbody id="ov-req">
            ${!recent.length ? html`<tr><td colspan="8" class="empty">${t("No requests yet.")}</td></tr>` : ""}
            ${recent.slice(0, 8).map((r) => html`
              <tr><td class="mono">${clock(r.finished_at)}</td><td>${r.label}</td><td><${StatusBadge} r=${r}/></td>
                <td class="n">${fmt(r.output_tokens)}</td><td class="n">${fmt(r.decode_tok_s, 1)}</td>
                <td class="n">${secs(ttftSeconds(r))}</td>
                <td class="n" title=${r.prompt_tokens > 0 ? t(fmt(r.reused_tokens) + " of " + fmt(r.prompt_tokens) + " prompt tokens reused") : undefined}>
                  ${cacheHit(r) === null ? "-" : fmt(cacheHit(r), 1) + "%"}</td>
                <td class="n">${fmt(r.tokens_per_round, 2)}</td></tr>`)}
          </tbody></table></div>
      </div>
      <div class="pane">
        <div class="pane-h"><h2>${t("Sessions")}</h2>
          <span class="aside">${model.max_sessions ? t(sl.length + " of " + model.max_sessions + " on the GPU") : ""}</span></div>
        <div id="sessions">
          ${!sl.length ? html`<div class="empty">${t("No conversation yet.")}</div>` : ""}
          ${sl.map((s) => {
            const cf = model.context ? s.tokens / model.context : 0;
            return html`<div class="fill"><span class="mono">#${s.id}</span><${Meter} frac=${cf} cls=${level(cf)}/>
              <span class="n">${fmt(s.tokens) + " / " + ktok(model.context)}</span></div>`;
          })}
        </div>
      </div>
    </div>
    <div class="sec c-3">
      <div class="pane"><div class="pane-h"><h2>${t("Engine")}</h2></div>
        <${Kv} pairs=${[
          ["Backend", model.backend || "-"],
          ["Context", model.context ? fmt(model.context) : "-"],
          ["Drafter", draftSummary || "none"],
          ["Chat format", model.chat_format || "-"],
          ["Sessions", model.max_sessions ? fmt(model.max_sessions) : "-"],
          ["Load time", model.load_s ? secs(model.load_s) : "-"],
        ]}/>
        <div class="sysnote" id="engine-mem">${(model.memory || "").split(" | ").join("\n")}</div></div>
      <div class="pane" id="ov-device"><div class="pane-h"><h2>${t("Device")}</h2>
          <span class="aside">${d && model.device ? shortDevice(model.device) : ""}</span></div>
        ${d ? html`
          <${Gauge} id="ov-util" label="Util" value=${figure(d.util, 0, "%")} frac=${d.util / 100}/>
          <${Gauge} id="ov-power" label="Power" value=${fmt(d.power_w) + " / " + fmt(d.power_limit_w) + " W"}
                    frac=${d.power_limit_w ? d.power_w / d.power_limit_w : 0}/>` : ""}
        <${Kv} pairs=${d ? [
          ["Temp, fan", fmt(d.temp_c) + "°C" + (d.fan !== null && d.fan !== undefined ? ", " + fmt(d.fan) + "%" : "")],
          ["SM clock", d.sm_clock_mhz ? fmt(d.sm_clock_mhz) + " MHz" + (d.clock_limits && d.clock_limits !== "idle" ? " (" + d.clock_limits + ")" : "") : "-"],
          ["PCIe", d.pcie_gen ? "gen " + d.pcie_gen + " x" + d.pcie_width : "-"],
        ] : [["GPU", model.loaded ? "no figures from this device" : "model not loaded"]]}/></div>
      <div class="pane" id="ov-host"><div class="pane-h"><h2>${t("Host")}</h2></div>
        ${[["VRAM", "vram", memory.vram], ["RAM", "ram", memory.ram]].map(([k, kind, m]) => {
          const act = !!(m && onOpenMem);
          const onKey = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onOpenMem(kind); } };
          return html`
          <div class=${"memrow" + (act ? " memrow-act" : "")} id=${"ov-" + kind}
               role=${act ? "button" : undefined} tabIndex=${act ? 0 : undefined}
               title=${act ? t(kind === "ram" ? "Inspect and close apps using RAM" : "Inspect and close apps using VRAM") : undefined}
               onClick=${act ? () => onOpenMem(kind) : undefined}
               onKeyDown=${act ? onKey : undefined}>
            <div class="mk"><span class="lk-lbl">${t(k)}${act ? html`<${Icon} name="chev" cls="lk-chev"/>` : ""}</span>
              <span class="v">${m ? gib(m.used) + " / " + gib(m.total) + " GiB" : "-"}</span></div>
            <${MemBar} m=${m}/></div>`;
        })}
        <${Kv} pairs=${[
          ["CPU, FLYX", tm._cpu === undefined ? "-" : tm._cpu > 100 ? fmt(tm._cpu / 100, 1) + " cores" : fmt(tm._cpu, 1) + "% of a core"],
          ["Cores", fmt(sys.cpus)],
          ["Uptime", secs(tm.server ? tm.server.uptime_s : null)],
        ]}/></div>
    </div>`;
}
