// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The dashboard: the frame, which page is shown, and the telemetry every page reads.
// Every page stays mounted and is hidden when another is shown, so a chat or a filter survives a visit elsewhere.
// The whole tree draws again on each telemetry answer (once a second) and on a change of language or theme.
import { html, render, useState, useEffect } from "./lib/html.js";
import { apiKey, consumeAddressKey, saved, setKey } from "./lib/api.js";
import { t, onLang } from "./lib/i18n.js";
import { onTheme } from "./lib/theme.js";
import { live, onLive, poll, busyLoad } from "./lib/telemetry.js";
import { TopBar } from "./components/topbar.js";
import { Sidebar, PAGES } from "./components/sidebar.js";
import { Overview } from "./pages/overview.js";
import { ModelPage } from "./pages/model.js";
import { Requests } from "./pages/requests.js";
import { Chat } from "./pages/chat.js";
import { Logs } from "./pages/logs.js";
import { Integration } from "./pages/integration.js";
import { NetworkSettings } from "./pages/settings.js";
import { Hint, markVisited } from "./components/hints.js";
import { VramModal, isVramError, parseVramNeeded } from "./components/vram_apps.js";
import { reloadServed, loadWant } from "./lib/actions.js";

const TABS = PAGES.map((p) => p[0]);
const TITLES = Object.fromEntries(PAGES.map((p) => [p[0], p[1]]));
const typing = (e) => ["input", "textarea", "select"].includes((e.target.tagName || "").toLowerCase());

// The page the address names (#chat, #requests:ID), else the one shown last.
function pageName(name) {
  if (name === "docs" || name === "connect") return "integration";  // older bookmarks and saved tabs
  if (name && name.startsWith("requests:")) return "requests";
  return TABS.includes(name) ? name : null;
}
function requestFromAddress() {
  if (!location.hash.startsWith("#requests:")) return null;
  try { return decodeURIComponent(location.hash.slice("#requests:".length)) || null; } catch { return null; }
}
function setAddress(page, id) {
  const hash = page === "requests" && id ? "#requests:" + encodeURIComponent(id) : page === "overview" ? "" : "#" + page;
  window.history.replaceState(null, "", location.pathname + location.search + hash);
}
const named = pageName(location.hash.slice(1));

function Notice({ keyChanged }) {
  const [key, setKeyText] = useState("");
  const n = live.notice;
  if (!n) return html`<div class="notice"></div>`;
  return html`
    <div class="notice show">
      <span>${t(n.msg)}</span>
      ${n.withKey ? html`
        <input type="password" placeholder=${t("API key")} autocomplete="off" value=${key} onInput=${(e) => setKeyText(e.target.value)}/>
        <button class="btn" onClick=${() => { setKey(key); live.notice = null; keyChanged(); }}>${t("Use key")}</button>` : ""}
    </div>`;
}

function App() {
  const [, redraw] = useState(0);
  const [tab, setTab] = useState(named || pageName(saved.get("tab")) || "overview");
  const [openRequest, setOpenRequest] = useState(requestFromAddress);
  const [libraryAsked, setLibraryAsked] = useState(0);  // counts the times a page asked for the Model page's library
  const [currentAsked, setCurrentAsked] = useState(0);  // and for its card of the model (a load's state, its error)
  const [memKind, setMemKind] = useState("");           // "" | "vram" | "ram": modal showing apps using memory
  const [collapsed, setCollapsedState] = useState(document.documentElement.dataset.side === "collapsed");

  useEffect(() => {
    const again = () => redraw((x) => x + 1);
    const offs = [onLive(again), onLang(again), onTheme(again)];
    window.addEventListener("resize", again);
    return () => { offs.forEach((f) => f()); window.removeEventListener("resize", again); };
  }, []);

  const keyChanged = () => {
    redraw((x) => x + 1);
    poll();
  };
  const go = (to, id = openRequest) => {
    if (!TABS.includes(to)) to = "overview";
    setTab(to);
    saved.set("tab", to);
    setAddress(to, id);
    const main = document.getElementById("main");
    if (main) main.scrollTop = 0;
  };
  const selectRequest = (id) => { setOpenRequest(id); setAddress("requests", id); };
  const exploreLibrary = () => { setLibraryAsked((n) => n + 1); go("model"); };
  const showModel = () => { setCurrentAsked((n) => n + 1); go("model"); };
  useEffect(() => {
    const follow = (event) => {
      const old = new URL(event.oldURL).hash.split("&key=", 1)[0];
      if (consumeAddressKey(pageName(old.slice(1)) ? old : "")) {
        keyChanged();
        if (!location.hash) return;  // a key-only link keeps the page already shown here
      }
      const to = pageName(location.hash.slice(1)) || "overview";
      const id = requestFromAddress();
      if (to === "requests") setOpenRequest(id);
      go(to, id);
    };
    window.addEventListener("hashchange", follow);
    return () => window.removeEventListener("hashchange", follow);
  }, []);
  const setCollapsed = (c) => {
    if (c) document.documentElement.dataset.side = "collapsed"; else delete document.documentElement.dataset.side;
    saved.set("side", c ? "collapsed" : null);
    setCollapsedState(c);
    setTimeout(() => redraw((x) => x + 1), 200);  // the charts take the panel's new width once the sidebar has moved
  };

  // Number keys select a page, [ toggles the sidebar.
  useEffect(() => {
    const key = (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey || typing(e)) return;
      const n = parseInt(e.key, 10);
      if (n >= 1 && n <= TABS.length) go(TABS[n - 1]);
      if (e.key === "[") setCollapsed(document.documentElement.dataset.side !== "collapsed");
    };
    document.addEventListener("keydown", key);
    return () => document.removeEventListener("keydown", key);
  }, [openRequest]);

  // Nothing served and nothing coming, on the first answer: the Model page, unless the address names one.
  useEffect(() => {
    let first = true;
    return onLive(() => {
      if (!first || !live.last) return;
      first = false;
      const L = live.last.load;
      if (!pageName(location.hash.slice(1)) && !live.last.model.id && L && !busyLoad(L)) go("model");
    });
  }, []);

  useEffect(() => {
    document.documentElement.dataset.page = tab;
    markVisited(tab);
  }, [tab]);
  const tm = live.last, L = tm && tm.load;
  const busy = tm && tm.state && (tm.state.phase === "decode" || tm.state.phase === "prefill");
  document.title = !tm ? "FLYX"
    : tm.model && tm.model.id ? (busy ? "▸ " : "") + tm.model.id + " · FLYX"
    : L && busyLoad(L) ? (L.target || t(L.state === "fetching" ? "Fetching" : "Loading")) + " · FLYX"
    : t("No model loaded") + " · FLYX";

  const page = (id, body) => html`<section id=${"tab-" + id} hidden=${tab !== id}>${body}</section>`;
  const ctxLim = tm && tm.model && tm.model.context_limit;
  const ctxNeed = ctxLim && ctxLim.need_more > 0 && memKind === (ctxLim.by === "host" ? "ram" : "vram") ? ctxLim.need_more : 0;
  const modalNeed = memKind === "vram" && L && !busyLoad(L) && isVramError(L.error) ? parseVramNeeded(L.error) : ctxNeed;
  const modalRetry = tm && tm.model && tm.model.id && L && L.loaded
    ? () => { setMemKind(""); reloadServed(); }
    : L && L.state === "failed" && L.target
      ? () => { setMemKind(""); loadWant(L.remembered && L.remembered.model === L.target ? L.remembered : { model: L.target }); }
      : null;
  const modalRetryLabel = tm && tm.model && tm.model.id ? "Reload model" : "Retry loading";
  return html`
    <div class="app">
      <${TopBar} title=${TITLES[tab]} go=${go}/>
      <${Sidebar} tab=${tab} go=${go} collapsed=${collapsed} setCollapsed=${setCollapsed} onOpenMem=${setMemKind}/>
      <main id="main">
        <div class="page">
          <${Notice} keyChanged=${keyChanged}/>
          <${Hint} tab=${tab} go=${go} explore=${exploreLibrary} showModel=${showModel} onOpenMem=${setMemKind}/>
          ${page("overview", html`<${Overview} active=${tab === "overview"} go=${go} explore=${exploreLibrary} showModel=${showModel} onOpenMem=${setMemKind}/>`)}
          ${page("model", html`<${ModelPage} key=${apiKey()} active=${tab === "model"} libraryAsked=${libraryAsked} currentAsked=${currentAsked}/>`)}
          ${page("requests", html`<${Requests} key=${apiKey()} active=${tab === "requests"} openId=${openRequest} select=${selectRequest}/>`)}
          ${page("chat", html`<${Chat} active=${tab === "chat"} showModel=${showModel}/>`)}
          ${page("logs", html`<${Logs} key=${apiKey()} active=${tab === "logs"}/>`)}
          ${page("integration", html`<${Integration} active=${tab === "integration"}/>`)}
          ${page("settings", tab === "settings" && html`<${NetworkSettings} key=${JSON.stringify([apiKey(), tm?.server?.started_at, live.offline])}/>`)}
        </div>
      </main>
      <${VramModal} open=${!!memKind} kind=${memKind || "vram"} onClose=${() => setMemKind("")} needBytes=${modalNeed}
                    onRetry=${modalRetry} retryLabel=${modalRetryLabel}/>
    </div>`;
}

for (const k of ["live", "live_pos", "live_closed"]) saved.set(k, null);  // what older pages kept for their floats
render(html`<${App}/>`, document.getElementById("app"));
poll();
setInterval(poll, 1000);
