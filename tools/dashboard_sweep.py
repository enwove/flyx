# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The dashboard in a browser, in one language: every page and the states a click opens, and the text still in English.

    python tools/dashboard_sweep.py [--url http://127.0.0.1:8080] [--lang zh] [--dir apps/flyx-server/dashboard]
                                    [--shots DIR] [--browser msedge|chromium]

Needs Playwright (pip install playwright) and Microsoft Edge, or `playwright install chromium` with --browser chromium.
--dir serves the page's files from that directory (and Preact and htm from third_party/) in place of the server's
built ones, so a change to the page is seen without a rebuild (flyx serve --dashboard-dir does the same on the server).

It walks the same states twice, in English and in --lang: each page; a chat reply; a request opened, with each of its
tabs; the model page's two views: the model served with the load settings (advanced options, a custom drafter typed
in), and the library (Hugging Face, this machine and a folder of it, a file chosen); the model menu; the reload and unload
confirmations (cancelled); the collapsed sidebar.  A string shown in --lang exactly as it is in English, and made of
words, is reported, with where it showed.  Text kept as written by design (model output, prompts, code, names, log
lines, the API reference) is left out; words the languages keep as they are (GPU, TTFT, Chat, ...) are not reported.
It sends one short chat message and changes nothing else on the server.  Exit status 1 if a page script raised.
lib/i18n.js says how the page's languages work.
"""
from __future__ import annotations

import argparse
import os
import sys

from playwright.sync_api import sync_playwright

# Shown text and the placeholder / title / aria-label of what is in sight, but not what the page keeps as written.
COLLECT = r"""() => {
  const keep = "pre, code, script, style, .md, .cv-user, .cv-think > div, .cv-call pre, .logline td.msg, .logline td.src, "
             + ".logline td.tm, td.txt1, .mname, .mchip-name, .tmodel, .srv-name, .srv-src, .q, .fn, #mf-cmd, .rbody pre, "
             + "textarea, .mm-sub, #mm-name, #r-model, svg, .word, [data-lang], .path, .method, kbd, .doc .ep, "
             + ".doc table, #d-endpoints, #d-fields, #d-sampling, #d-flags, #d-telemetry, .cv-chips, .srv-dr .f, .tn, "
             + ".srv-last b, .ht, .pn, .ps, .dh-t, .dchips, .fchips, .stats.small dd, #mf-path";
  const out = new Set();
  const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let n; (n = w.nextNode());) {
    const p = n.parentElement, t = n.nodeValue.trim().replace(/\s+/g, " ");
    if (!t || !p || p.closest(keep) || !p.getClientRects().length || getComputedStyle(p).visibility === "hidden") continue;
    out.add("text: " + t);
  }
  for (const e of document.querySelectorAll("[placeholder], [title], [aria-label]")) {
    if (!e.getClientRects().length || e.closest(keep)) continue;
    for (const a of ["placeholder", "title", "aria-label"]) {
      const v = e.getAttribute(a);
      if (v) out.add(a + ": " + v.replace(/\s+/g, " ").trim());
    }
  }
  return [...out];
}"""

# Words every language keeps: names, units, protocol words.
KEPT = {"mtp", "dflash2", "gpu", "vram", "ttft", "kv", "api", "json", "hugging", "face", "tok/s", "token", "tokens", "prompt",
        "drafter", "drafters", "flyx", "fp8", "gguf", "cuda", "rtx", "gib", "gb", "mb", "ms", "mhz", "chat", "messages",
        "responses", "openai", "anthropic", "claude", "code", "curl", "python", "telemetry", "ok", "pcie", "sse", "docs",
        "jinja", "lora", "ctx", "nvidia", "geforce", "english", "http", "url", "cpu", "ram", "bearer", "post", "get", "delete",
        "sdk", "sdks", "enter", "gen", "backend", "host", "streaming", "error", "info", "auto", "exact", "source",
        "message", "pages", "sessions", "session", "logs", "model", "models", "total", "endpoints", "text"}


def wordy(s: str) -> bool:
    """Whether a reported string has a word a language would translate."""
    import re
    body = s.split(": ", 1)[1]
    if " " not in body.strip() and re.search(r"[\d_/.:-]", body):  # a name: a quantization, a repository, a version
        return False
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", body)]
    return any(w not in KEPT for w in words)


def walk(pg, lang: str, shots: str | None) -> dict[str, str]:
    """Every state, in `lang`: each string seen, and where it was first seen."""
    seen: dict[str, str] = {}

    def look(where: str) -> None:
        for s in pg.evaluate(COLLECT):
            seen.setdefault(s, where)

    def shot(name: str) -> None:
        if shots and lang != "en":
            pg.screenshot(path=f"{shots}/{lang}-{name}.png")

    def setlang(code: str) -> None:
        pg.click("#lang")
        pg.click(f"#lang-menu button[data-lang={code}]")
        pg.wait_for_timeout(600)

    def go(tab: str) -> None:
        pg.click(f".side nav .navi[data-tab={tab}]")
        pg.wait_for_timeout(1200)

    setlang(lang)
    go("chat"); look("chat")
    if lang == "en":  # one reply is enough for both passes: the English pass sent it
        pg.fill("#prompt", "Say hi in three words.")
        pg.keyboard.press("Enter")
        pg.wait_for_timeout(9000)
    look("chat, after a reply"); shot("chat")
    go("overview"); look("overview"); shot("overview")
    go("requests"); pg.wait_for_timeout(1500); look("requests")
    rows = pg.locator("#req-rows tr[data-id]")
    if rows.count():
        rows.first.click()
        pg.wait_for_timeout(800)
        look("a request, opened"); shot("requests")
        for t in pg.locator(".reader [role=tab]").all():
            t.click()
            pg.wait_for_timeout(400)
            look("a request, its tabs")
    go("model")
    pg.click('#mv-tabs [data-view="current"]'); pg.wait_for_timeout(400); look("model"); shot("model")
    if pg.locator("#mf-settings").count():  # the load settings: only while a model is served
        pg.click('#lset-tabs [data-stab="spec"]'); pg.wait_for_timeout(300)
        if pg.locator("#sd-adv").get_attribute("open") is None:
            pg.click("#sd-adv summary"); pg.wait_for_timeout(300)
        look("model, advanced")
        if pg.locator("#sd-file").is_enabled():  # not with speculative decoding off
            pg.fill("#sd-file", "org/drafter"); pg.wait_for_timeout(300); look("model, a custom drafter typed")
            pg.fill("#sd-file", ""); pg.wait_for_timeout(200)
        pg.click('#lset-tabs [data-stab="vision"]'); pg.wait_for_timeout(300)
        look("model, vision")
        pg.click('#lset-tabs [data-stab="general"]'); pg.wait_for_timeout(200)
    pg.click('#mv-tabs [data-view="library"]'); pg.wait_for_timeout(2500); look("model, its library"); shot("model-library")
    for t in reversed(pg.locator('button[role="tab"][data-src]').all()):  # this machine, then back to Hugging Face
        t.click()
        pg.wait_for_timeout(2500)
        look("model, " + (t.get_attribute("data-src") or ""))
        folder = pg.locator('#lc-list .prow[data-place^="dir:"]')  # not "Add a folder": that opens the system's dialog
        if folder.count():
            folder.first.click(); pg.wait_for_timeout(1200); look("model, a folder")
        rows = pg.locator(".hub-detail .mrow:not(.off)")
        if rows.count():
            rows.first.click(); pg.wait_for_timeout(300); look("model, a file chosen")
    pg.click('#mv-tabs [data-view="current"]')
    go("logs"); look("logs"); shot("logs")
    go("integration"); look("integration"); shot("integration")
    for agent in pg.locator("#d-agents button[data-agent]").all():
        agent.click()
        look("integration, " + agent.inner_text())
    pg.click("#mchip"); pg.wait_for_timeout(300); look("the model menu")
    if pg.locator("#mm-reload").is_visible():
        pg.click("#mm-reload"); pg.wait_for_timeout(500); look("the reload confirmation")
        pg.keyboard.press("Escape"); pg.wait_for_timeout(300)
        pg.click("#mchip"); pg.wait_for_timeout(300)
    if pg.locator("#mm-unload").is_visible():
        pg.click("#mm-unload"); pg.wait_for_timeout(500); look("the unload confirmation")
        pg.keyboard.press("Escape"); pg.wait_for_timeout(300)
    else:
        pg.keyboard.press("Escape")
    pg.keyboard.press("["); pg.wait_for_timeout(800); look("the sidebar, collapsed")
    pg.keyboard.press("["); pg.wait_for_timeout(300)
    return seen


VENDOR = {"preact.module.js": "preact/preact.module.js", "hooks.module.js": "preact/hooks.module.js", "htm.module.js": "htm/htm.module.js"}
TYPES = {
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".png": "image/png",
}


def serve(route, root: str) -> None:
    """A file of the page from `root`, as the server would answer it; the libraries from third_party/."""
    path = route.request.url.split("/dashboard", 1)[1].split("?")[0].lstrip("/") or "index.html"
    third_party = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party")
    src = os.path.join(third_party, VENDOR.get(path[7:], "-")) if path.startswith("vendor/") else os.path.join(root, path)
    if not os.path.isfile(src):
        route.fulfill(status=404, body="no such file: " + path)
        return
    with open(src, "rb") as f:
        route.fulfill(status=200, content_type=TYPES.get(os.path.splitext(src)[1], "application/octet-stream"),
                      body=f.read())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--lang", default="zh", help="a code of the page's language menu: zh, es, fr, de, ja, ko")
    ap.add_argument("--dir", help="serve the page's files from this directory instead of the server's built ones")
    ap.add_argument("--shots", help="a directory for a screenshot of each page, in --lang")
    ap.add_argument("--browser", default="msedge" if sys.platform == "win32" else "chromium",
                    help="msedge (Windows Edge) or chromium (Playwright's own)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    errors: list[str] = []
    with sync_playwright() as p:
        b = p.chromium.launch(channel=None if a.browser == "chromium" else a.browser, headless=True)
        pg = b.new_page(viewport={"width": 1500, "height": 950})
        pg.on("pageerror", lambda e: errors.append(str(e)))
        if a.dir:
            pg.route("**/dashboard/**", lambda r: serve(r, a.dir))
        pg.goto(a.url.rstrip("/") + "/dashboard/#overview")
        pg.reload()
        pg.wait_for_timeout(2500)
        english = walk(pg, "en", None)
        shown = walk(pg, a.lang, a.shots)
        pg.click("#lang"); pg.click("#lang-menu button[data-lang=en]")
        b.close()
    left = {s: where for s, where in shown.items() if s in english and wordy(s)}
    by: dict[str, list[str]] = {}
    for s, where in left.items():
        by.setdefault(where, []).append(s)
    for where, ss in by.items():
        print(f"== {where}")
        for s in ss:
            print("  ", s[:140])
    print(f"{len(left)} strings still in English in {a.lang} (of {len(shown)} seen); page errors: {errors or 'none'}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
