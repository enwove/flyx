// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The top bar: the logo and the page; the model being served with its quick actions; the theme and the language.
import { html, useState, useEffect } from "../lib/html.js";
import { t, LANGS, currentLang, setLang } from "../lib/i18n.js";
import { THEMES, themeChoice, setTheme, shownTheme } from "../lib/theme.js";
import { fmt, shortDevice } from "../lib/format.js";
import { Icon } from "../lib/ui.js";
import { live, phaseOf, phaseNames, phaseTone } from "../lib/telemetry.js";
import { unloadServed, reloadServed, loadWant, stopFetch } from "../lib/actions.js";

// A menu that closes on a click elsewhere or Escape.  `wrap` is the class of the element holding button and menu.
function useMenu(wrap) {
  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (!open) return;
    const click = (e) => { if (!e.target.closest("." + wrap)) setOpen(false); };
    const key = (e) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("click", click);
    document.addEventListener("keydown", key);
    return () => { document.removeEventListener("click", click); document.removeEventListener("keydown", key); };
  }, [open]);
  return [open, setOpen];
}

// The status of what is served, as the chip's badge says it: with the fetch's or the prefill's progress.
function stateText(t0, phase) {
  const L = (t0 && t0.load) || {}, state = (t0 && t0.state) || {};
  let st = t(phaseNames[phase] || phase);
  if (live.offline) return t("Offline");
  if (phase === "fetching" && L.progress && L.progress.total) st += " " + fmt(100 * L.progress.done / L.progress.total) + "%";
  if (phase === "prefill" && state.prompt_tokens) st += " " + fmt(100 * (state.prefilled || 0) / state.prompt_tokens) + "%";
  return st;
}

function ModelMenu({ go }) {
  const [open, setOpen] = useMenu("mwrap");
  const last = live.last, phase = phaseOf(last);
  const model = (last && last.model) || {}, L = (last && last.load) || {};
  const busy = phase === "fetching" || phase === "loading";
  const name = model.id || (L.target ? L.target : "No model");
  const ctx = model.context;
  const sub = model.id ? [shortDevice(model.device), model.backend, ctx ? "ctx " + fmt(ctx) : ""].filter(Boolean).join(" · ")
            : busy ? t(phaseNames[phase]) : L.remembered ? t("not loaded") : "";
  const act = (f) => () => { setOpen(false); f(); };
  return html`
    <div class="mwrap">
      <button class="mchip" id="mchip" title=${t("The model being served")} aria-haspopup="menu" aria-expanded=${String(open)}
              onClick=${() => setOpen(!open)}>
        <${Icon} name="model"/>
        <span class="mchip-name" id="r-model" title=${[model.device, model.backend].filter(Boolean).join(" · ")}>
          ${model.id || L.target || t("No model")}</span>
        <span class=${"badge " + (live.offline ? "bad" : phaseTone(phase))} id="r-state" hidden=${phase === "none" && !live.offline}>
          <span class="dot"></span><span id="r-state-text">${stateText(last, phase)}</span></span>
        <${Icon} name="down" cls="caret"/>
      </button>
      <div class="menu mmenu" id="mmenu" role="menu" hidden=${!open}>
        <div class="mm-head"><b class="mname" id="mm-name">${model.id || L.target || (L.remembered && L.remembered.model) || t(name)}</b>
          <span class="mm-sub" id="mm-sub">${sub}</span></div>
        <button role="menuitem" id="mm-load" hidden=${!!model.id || busy || !L.remembered}
                onClick=${act(() => loadWant(L.remembered))}><${Icon} name="play"/><span>${t("Load model")}</span></button>
        <button role="menuitem" id="mm-cancel" hidden=${L.state !== "fetching"}
                onClick=${act(stopFetch)}><${Icon} name="close"/><span>${t("Stop fetching")}</span></button>
        <button role="menuitem" id="mm-reload" hidden=${!model.id || busy || !L.loaded}
                onClick=${act(reloadServed)}><${Icon} name="reload"/><span>${t("Reload")}</span></button>
        <button role="menuitem" id="mm-unload" class="danger" hidden=${!model.id || busy}
                onClick=${act(unloadServed)}><${Icon} name="eject"/><span>${t("Unload")}</span></button>
        <div class="mm-sep"></div>
        <button role="menuitem" id="mm-settings" onClick=${act(() => go("model"))}><${Icon} name="gear"/><span>${t("Model settings")}</span></button>
        <button role="menuitem" id="mm-logs" onClick=${act(() => go("logs"))}><${Icon} name="logs"/><span>${t("Logs")}</span></button>
      </div>
    </div>`;
}

function LangMenu() {
  const [open, setOpen] = useMenu("langwrap");
  const lang = currentLang();
  const item = (code, name) => html`
    <button role="menuitemradio" data-lang=${code} aria-checked=${String(lang === code)}
            onClick=${() => { setOpen(false); setLang(code); }}><${Icon} name="check"/>${name}</button>`;
  return html`
    <div class="langwrap">
      <button class="iconbtn" id="lang" title=${t("Language")} aria-label=${t("Language")} aria-haspopup="menu"
              aria-expanded=${String(open)} onClick=${() => setOpen(!open)}><${Icon} name="globe"/></button>
      <div class="menu" id="lang-menu" role="menu" hidden=${!open}>
        ${item("en", "English")}
        ${Object.entries(LANGS).map(([code, L]) => item(code, L.name))}
      </div>
    </div>`;
}

const THEME_NAMES = { system: "Match system", light: "Light", dark: "Dark" };

// The theme as chosen in this browser; the button shows the one on the page.
function ThemeMenu() {
  const [open, setOpen] = useMenu("themewrap");
  const choice = themeChoice();
  return html`
    <div class="themewrap">
      <button class="iconbtn" id="theme" title=${t("Theme")} aria-label=${t("Theme")} aria-haspopup="menu"
              aria-expanded=${String(open)} onClick=${() => setOpen(!open)}><${Icon} name=${shownTheme() === "dark" ? "moon" : "sun"}/></button>
      <div class="menu" id="theme-menu" role="menu" hidden=${!open}>
        ${THEMES.map((id) => html`
          <button role="menuitemradio" data-choice=${id} aria-checked=${String(choice === id)}
                  onClick=${() => { setOpen(false); setTheme(id); }}><${Icon} name="check"/>${t(THEME_NAMES[id])}</button>`)}
      </div>
    </div>`;
}

export function TopBar({ title, go }) {
  return html`
    <header class="topbar">
      <a class="word" href="#overview" aria-label="FLYX" onClick=${(e) => { e.preventDefault(); go("overview"); }}>
        <img src=${"/dashboard/brand/flyx-lockup" + (shownTheme() === "dark" ? "-dark" : "") + ".svg"} alt="" width="102" height="24"/></a>
      <span class="sep">/</span><h1 class="crumb" id="page-title">${t(title)}</h1>
      <div class="right">
        <${ModelMenu} go=${go}/>
        <${ThemeMenu}/>
        <${LangMenu}/>
      </div>
    </header>`;
}
