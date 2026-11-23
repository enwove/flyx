// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The sidebar: the pages, and the figures a user watches on every page.  Collapsed, a rail keeps the key figures.
import { html } from "../lib/html.js";
import { t } from "../lib/i18n.js";
import { fmt, gib, ktok, secs, level, shortDevice, short } from "../lib/format.js";
import { Icon, Meter, MemBar, Val } from "../lib/ui.js";
import { Spark } from "../lib/charts.js";
import { live, phaseOf, phaseNames, ttftSeconds, memoryOf } from "../lib/telemetry.js";

export const PAGES = [
  ["overview", "Overview", "overview"], ["model", "Model", "model"], ["requests", "Requests", "requests"],
  ["chat", "Chat", "chat"], ["logs", "Logs", "logs"], ["integration", "Integration", "integration"],
];

// The figures both views show, from the last telemetry.
function figures(tm) {
  const model = tm.model || {}, state = tm.state || {}, d = tm.device, recent = tm.recent || [];
  const phase = phaseOf(tm), busy = phase === "decode" || phase === "prefill", decoding = phase === "decode";
  const lastReq = recent[0] || {};
  const decode = decoding ? tm.rates.decode_tok_s : lastReq.decode_tok_s;
  const tpr = decoding ? (Number.isFinite(tm.rates && tm.rates.tokens_per_round) && tm.rates.tokens_per_round > 0
                          ? tm.rates.tokens_per_round : state.tokens_per_round)
                       : lastReq.tokens_per_round;
  const ttft = ttftSeconds(lastReq);
  // Context: the running request's session fill (matching Sessions), else the most recent session's.
  const sl = (tm.memory && tm.memory.sessions) || [];
  const activeSlot = busy && state.session ? sl.find((s) => s.id === state.session) : null;
  const liveUsed = phase === "prefill"
    ? (Number.isFinite(state.reused_tokens) && state.reused_tokens >= 0
       ? (state.prefilled || 0) : (activeSlot ? activeSlot.tokens : (state.prefilled || 0)))
    : (state.prompt_tokens || 0) + (state.generated || 0);
  const used = busy ? liveUsed : sl.length ? sl[0].tokens : null;
  // The GPU's and the machine's memory (flyx's part and the rest), and the processors' load: all of it, and flyx's
  // part (both of all the processors).
  const sys = tm.system || {}, { vram, ram } = memoryOf(tm);
  const cpu = { all: tm._sys_cpu === undefined ? null : tm._sys_cpu, own: tm._cpu === undefined || !sys.cpus ? null : tm._cpu / sys.cpus };
  return { model, state, d, recent, phase, busy, decoding, lastReq, decode, tpr, ttft, used, vram, ram, cpu,
           ops: (tm.rates && tm.rates.output_per_second) || [] };
}

function LiveCard({ f, tm }) {
  const { model, state, d, recent, decoding, lastReq, decode, tpr, ttft, used, vram, ram, cpu } = f;
  const fresh = (lastReq.prompt_tokens || 0) - (lastReq.reused_tokens || 0);
  const pct = (v) => fmt(v, v < 10 ? 1 : 0) + "%";
  return html`
    <div class="live" id="live" aria-live="off">
      <div class="live-head"><span class="caps">${t("Live")}</span></div>
      <div class="live-body">
        <div class="lhero">
          <div class="lk">${t("Decode")}<span class="lsub">${decoding ? t("now") : recent.length ? t("last request") : ""}</span></div>
          <div class="lbig" id="r-decode"><${Val} v=${fmt(decode, 1)} unit="tok/s"/></div>
          <${Spark} cls="spark" values=${f.ops.slice(-60)} label=${t("tokens out each second, the last minute")}/>
        </div>
        <div class="lpair">
          <div><div class="lk">${t("Tokens / round")}</div><div class="lbig">${fmt(tpr, 2)}</div>
            <div class="lsub">${model.drafter ? t(model.drafter + ", up to " + model.max_width) : t("no drafter")}</div></div>
          <div><div class="lk">${t("TTFT")}</div><div class="lbig">${secs(ttft)}</div>
            <div class="lsub">${recent.length ? t(fmt(fresh) + " new tokens") : ""}</div></div>
        </div>
        <div class="lrow" id="r-vram-row"><div class="lk">${t("VRAM")}<span class="v" id="r-vram">
            ${vram ? html`<${Val} v=${gib(vram.used) + " / " + gib(vram.total)} unit="GiB"/>` : "-"}</span></div>
          <${MemBar} m=${vram} bare/></div>
        <div class="lrow" id="r-ram-row"><div class="lk">${t("RAM")}<span class="v" id="r-ram">
            ${ram ? html`<${Val} v=${gib(ram.used) + " / " + gib(ram.total)} unit="GiB"/>` : "-"}</span></div>
          <${MemBar} m=${ram} bare/></div>
        <div class="lrow"><div class="lk">${t("Context")}<span class="v">
            ${model.context && used !== null ? ktok(used) + " / " + ktok(model.context) : model.context ? "0 / " + ktok(model.context) : "-"}</span></div>
          <${Meter} frac=${model.context && used !== null ? used / model.context : 0} cls=${model.context && used !== null ? level(used / model.context) : ""}/></div>
        <div class="lrow"><div class="lk">${t("Running / queued")}<span class="v">${fmt(state.running || 0) + " / " + fmt(state.queued || 0)}</span></div></div>
        <div class="lrow"><div class="lk">${t("GPU")}<span class="v">
          ${d && d.mem_total ? fmt(d.util) + "% · " + fmt(d.temp_c) + "°C · " + fmt(d.power_w) + " W" : "-"}</span></div></div>
        <div class="lrow"><div class="lk">${t("CPU")}<span class="v" id="r-cpu">
          ${cpu.all === null && cpu.own === null ? "-" : [cpu.all === null ? "" : pct(cpu.all), cpu.own === null ? "" : "FLYX " + pct(cpu.own)].filter(Boolean).join(" · ")}</span></div></div>
      </div>
      <div class="live-foot" id="uptime">${[shortDevice(model.device), t("up " + secs(tm.server ? tm.server.uptime_s : null))].filter(Boolean).join("  ·  ")}</div>
    </div>`;
}

// A memory bar of the rail: flyx's part, then the rest.
const RailMem = ({ m }) => html`<span class="rb mb">
  ${m && m.own > 0 ? html`<i class="own" style=${{ width: m.own / m.total * 100 + "%" }}></i>` : ""}
  ${m && m.others > 0 ? html`<i class="oth" style=${{ width: m.others / m.total * 100 + "%" }}></i>` : ""}</span>`;

function Rail({ f, open }) {
  const { model, d, phase, busy, decode, tpr, ttft, used, vram, ram, cpu } = f;
  const ctx = model.context && used !== null ? used / model.context : null;
  const load = cpu.all === null ? cpu.own : cpu.all;
  const memLine = (k, m) => m ? k + " " + gib(m.used) + " / " + gib(m.total) + " GiB (FLYX " + gib(m.own) + ")" : "";
  const state = busy ? "busy" : phase === "loading" || phase === "fetching" ? "load" : phase === "none" || phase === "unloaded" ? "off" : "";
  const title = [model.id, t(phaseNames[phase] || phase), memLine("VRAM", vram), memLine("RAM", ram),
                 model.context ? t("context") + " " + ktok(used || 0) + " / " + ktok(model.context) : ""].filter(Boolean).join("\n") +
                "\n" + t("Click to open the sidebar.");
  const key = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } };
  return html`
    <div class="rail" id="rail" title=${title} role="button" tabindex="0" onClick=${open} onKeyDown=${key}>
      <span class=${"rs " + state} id="rl-state" title=${t(phaseNames[phase] || phase)}></span>
      <div class="rm"><span class="rv big">${short(decode)}</span><span class="rl">tok/s</span>
        <${Spark} cls="rsp" values=${f.ops.slice(-30)}/></div>
      <div class="rm"><span class="rv">${fmt(tpr, 2)}</span><span class="rl">${t("tok/rnd")}</span></div>
      <div class="rm"><span class="rv">${ttft === null ? "-" : ttft < 1 ? Math.round(ttft * 1000) + "ms" : ttft.toFixed(1) + "s"}</span>
        <span class="rl">${t("TTFT")}</span></div>
      <div class="rm"><span class="rv">${vram ? (vram.used / 1073741824).toFixed(1) : "-"}</span><span class="rl">${t("VRAM")}</span>
        <${RailMem} m=${vram}/></div>
      <div class="rm"><span class="rv">${ram ? (ram.used / 1073741824).toFixed(1) : "-"}</span><span class="rl">${t("RAM")}</span>
        <${RailMem} m=${ram}/></div>
      <div class="rm"><span class="rv">${ctx === null ? "-" : fmt(100 * ctx) + "%"}</span><span class="rl">${t("context")}</span>
        <span class="rb"><i class=${ctx === null ? "" : level(ctx)} style=${{ width: (ctx || 0) * 100 + "%" }}></i></span></div>
      <div class="rm"><span class="rv">${d && d.mem_total ? fmt(d.util) + "%" : "-"}</span><span class="rl">${t("GPU")}</span></div>
      <div class="rm"><span class="rv">${load === null ? "-" : fmt(load) + "%"}</span><span class="rl">${t("CPU")}</span></div>
    </div>`;
}

export function Sidebar({ tab, go, collapsed, setCollapsed }) {
  const tm = live.last || {};
  const f = figures(tm);
  return html`
    <aside class="side">
      <nav role="tablist" aria-label="pages">
        ${PAGES.map(([id, name, icon], i) => html`
          <button class="navi" role="tab" data-tab=${id} data-tip=${t(name)} title=${t(name) + "  (" + (i + 1) + ")"}
                  aria-selected=${String(tab === id)} onClick=${() => go(id)}>
            <${Icon} name=${icon}/><span class="t">${t(name)}</span></button>`)}
      </nav>
      <div class="grow" id="live-dock">${live.last ? html`<${LiveCard} f=${f} tm=${tm}/>` : ""}</div>
      ${live.last ? html`<${Rail} f=${f} open=${() => setCollapsed(false)}/>` : ""}
      <div class="foot">
        <button class="navi" id="collapse" data-tip=${t("Expand")} aria-expanded=${String(!collapsed)}
                onClick=${() => setCollapsed(!collapsed)}><${Icon} name="panel"/><span class="t">${t("Collapse")}</span></button>
      </div>
    </aside>`;
}
