# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Check the dashboard's light and dark themes with Playwright against stubbed telemetry: with nothing chosen on a
light system, the light theme as it always was; the dark one from the top bar's menu, kept across a reload and set
before the first paint; the system's setting followed while nothing is chosen, as it changes too; a browser without
storage; the menu in every language.  In the dark theme the grounds are neutral greys with no green in them, the
text and the accents read on them (WCAG AA, the text AAA), and the logo and the charts take the dark colours.
Use --browser msedge for Edge; --shots DIR keeps screenshots."""
import argparse
from pathlib import Path
import re
import sys
import time

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from dashboard_sweep import serve

ORIGIN = "http://[::1]:18479"
GiB = 1 << 30
# The light theme's tokens ("Paper"), as they were before the dark theme came.
PAPER = {"--frame": "#f7f4ec", "--surface": "#fbf9f4", "--surface-2": "#efeadc", "--text": "#12352a",
         "--muted": "#4d5b54", "--border": "#ddd6c3", "--primary": "#12352a", "--accent": "#26704f", "--chart": "#26704f"}
GROUNDS = ("--frame", "--surface", "--surface-2", "--primary-soft", "--border")
# Before any page script: the theme when <body> starts, and the body's colour in the first frame drawn.
FIRST_PAINT = """new MutationObserver((_, seen) => {
  if (!document.body) return;
  seen.disconnect();
  window.themeAtBody = document.documentElement.dataset.theme;
  requestAnimationFrame(() => { window.firstFrameBg = getComputedStyle(document.body).backgroundColor; });
}).observe(document, { childList: true, subtree: true });"""


def telemetry():
    out = [0] * 40 + [118] * 20
    return {"schema_version": 1, "sampled_at": time.time(),
            "server": {"name": "flyx-server", "version": "test", "uptime_s": 100, "api_key_required": False},
            "model": {"id": "tiny", "loaded": True, "context": 32768, "device": "NVIDIA GeForce RTX 5070 Ti",
                      "backend": "cuda", "max_sessions": 2},
            "load": {"state": "ready"}, "state": {"phase": "idle", "queued": 0, "running": 0},
            "rates": {"decode_tok_s": 0, "output_per_second": out, "prefill_per_second": [0] * 60},
            "memory": {"sessions": [{"id": 1, "tokens": 2000}]},
            "device": {"name": "NVIDIA GeForce RTX 5070 Ti", "util": 40, "mem_used": 12 * GiB, "mem_total": 16 * GiB,
                       "temp_c": 55, "power_w": 120, "power_limit_w": 300},
            "recent": [{"finished_at": 1000.0, "label": "messages", "status": "stop", "session": 1, "prompt_tokens": 2000,
                        "reused_tokens": 1700, "output_tokens": 120, "ttft_ms": 230, "prefill_ms": 100,
                        "decode_ms": 1000, "decode_tok_s": 119.0, "tokens_per_round": 2.5}],
            "totals": {"requests": 1, "errors": 0}}


def rgb(css):
    """An sRGB triple from "#rrggbb" or "rgb(r, g, b)"."""
    css = css.strip()
    if css.startswith("#"):
        return tuple(int(css[i:i + 2], 16) for i in (1, 3, 5))
    return tuple(round(float(v)) for v in re.findall(r"[\d.]+", css)[:3])


def contrast(a, b):
    def lum(c):
        c = [v / 255 for v in c]
        c = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def grey(c):
    """A neutral grey: the channels close together, and green not above both of the others."""
    return max(c) - min(c) <= 12 and c[1] <= max(c[0], c[2])


def tokens(page, names):
    return page.evaluate("names => { const s = getComputedStyle(document.documentElement);"
                         " return Object.fromEntries(names.map((n) => [n, s.getPropertyValue(n).trim()])); }", list(names))


def background(page, selector):
    return rgb(page.locator(selector).first.evaluate("e => getComputedStyle(e).backgroundColor"))


def inked(page, selector, colour):
    """Whether the canvas has pixels of the colour (within a step or two), drawn opaque."""
    return page.locator(selector).first.evaluate("""(c, want) => {
        const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
        for (let i = 0; i < d.length; i += 4)
          if (d[i + 3] === 255 && Math.abs(d[i] - want[0]) <= 3 && Math.abs(d[i + 1] - want[1]) <= 3 && Math.abs(d[i + 2] - want[2]) <= 3)
            return true;
        return false; }""", list(colour))


def choose(page, choice):
    page.locator("#theme").click()
    page.locator(f'#theme-menu [data-choice="{choice}"]').click()
    expect(page.locator("#theme-menu")).to_be_hidden()


def chosen(page, name):
    """The menu marks `name` as the choice; the button closes it again."""
    page.locator("#theme").click()
    expect(page.locator("#theme-menu [aria-checked=true]")).to_have_text(name)
    page.locator("#theme").click()
    expect(page.locator("#theme-menu")).to_be_hidden()


def saved_theme(page):
    return page.evaluate("localStorage.getItem('flyx.theme')")


def open_page(browser, errors, system="light", lang="en", storage=True):
    ctx = browser.new_context(viewport={"width": 1440, "height": 900}, color_scheme=system, locale="en-US")
    page = ctx.new_page()
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route("**/dashboard/**", lambda route: serve(route, ROOT / "apps/flyx-server/dashboard"))
    page.route("**/v1/**", lambda route: route.fulfill(json=telemetry() if "/telemetry" in route.request.url else {}))
    if storage:
        page.add_init_script("try { localStorage.setItem('flyx.lang', '%s'); } catch (e) {}" % lang)
    else:  # a browser that keeps nothing for the page: every access to its storage throws
        page.add_init_script("Object.defineProperty(window, 'localStorage', "
                             "{ get() { throw new DOMException('denied', 'SecurityError'); } });")
    page.add_init_script(FIRST_PAINT)
    page.goto(ORIGIN + "/dashboard/#overview")
    page.locator("#ov-req tr").first.wait_for()
    return ctx, page


def check_light(page, shots):
    """Nothing chosen, a light system: the light theme, its tokens as they were."""
    html = page.locator("html")
    expect(html).to_have_attribute("data-theme", "light")
    assert page.evaluate("window.themeAtBody") == "light"
    assert tokens(page, PAPER) == PAPER, tokens(page, PAPER)
    assert background(page, "body") == rgb(PAPER["--frame"])
    assert page.evaluate("getComputedStyle(document.documentElement).colorScheme") == "light"
    expect(page.locator(".word img")).to_have_attribute("src", "/dashboard/brand/flyx-lockup.svg")
    expect(page.locator("#theme use")).to_have_attribute("href", "#i-sun")
    assert inked(page, "canvas.spark", rgb(PAPER["--chart"])), "the sparkline is not in the light theme's colour"
    assert saved_theme(page) is None
    chosen(page, "Match system")
    if shots:
        page.screenshot(path=str(Path(shots) / "theme-light.png"))


def check_dark(page, shots):
    """Dark chosen from the menu: neutral grey grounds, readable text and accents, the dark logo and chart colours."""
    choose(page, "dark")
    html = page.locator("html")
    expect(html).to_have_attribute("data-theme", "dark")
    assert saved_theme(page) == "dark"
    assert page.evaluate("getComputedStyle(document.documentElement).colorScheme") == "dark"
    expect(page.locator(".word img")).to_have_attribute("src", "/dashboard/brand/flyx-lockup-dark.svg")
    expect(page.locator("#theme use")).to_have_attribute("href", "#i-moon")

    tk = {k: rgb(v) for k, v in tokens(page, GROUNDS + ("--text", "--muted", "--faint", "--accent", "--success",
                                                        "--warning", "--danger", "--info", "--chart", "--chart-2",
                                                        "--primary", "--on-primary")).items()}
    for name in GROUNDS:
        assert grey(tk[name]), (name, tk[name], "is not a neutral grey")
    assert contrast(tk["--frame"], (0, 0, 0)) < 1.3, ("the frame is not dark", tk["--frame"])
    assert sum(tk["--frame"]) < sum(tk["--surface"]) < sum(tk["--surface-2"]), "the grounds do not rise frame, panel, track"
    for ground in ("--frame", "--surface", "--surface-2"):
        assert contrast(tk["--text"], tk[ground]) >= 7, ("--text", ground)
        for fg in ("--muted", "--faint", "--accent", "--success", "--warning", "--danger", "--info", "--chart-2"):
            assert contrast(tk[fg], tk[ground]) >= 4.5, (fg, ground, round(contrast(tk[fg], tk[ground]), 2))
    assert contrast(tk["--on-primary"], tk["--primary"]) >= 7

    # What the page draws: the frame, the panels, the chosen page in the sidebar, an open menu; none green.
    page.locator("#theme").click()
    drawn = {s: background(page, s) for s in ("body", "main", ".pane", ".navi[aria-selected=true]", "#theme-menu")}
    page.locator("#theme").click()
    for selector, c in drawn.items():
        assert grey(c), (selector, c, "is not a neutral grey")
    assert drawn["body"] == tk["--frame"] and drawn[".pane"] == tk["--surface"], drawn
    assert contrast(rgb(page.locator(".pane h2").first.evaluate("e => getComputedStyle(e).color")), drawn[".pane"]) >= 7
    assert inked(page, "canvas.spark", tk["--chart"]), "the sparkline was not drawn again in the dark theme's colour"
    if shots:
        page.screenshot(path=str(Path(shots) / "theme-dark.png"))

    # Kept across a reload, and on <html> before <body> starts: the first frame is already dark.
    page.reload()
    page.locator("#ov-req tr").first.wait_for()
    expect(html).to_have_attribute("data-theme", "dark")
    assert page.evaluate("window.themeAtBody") == "dark"
    assert rgb(page.evaluate("window.firstFrameBg")) == tk["--frame"], page.evaluate("window.firstFrameBg")
    chosen(page, "Dark")

    # Light again: the tokens are the light theme's once more.
    choose(page, "light")
    expect(html).to_have_attribute("data-theme", "light")
    assert saved_theme(page) == "light"
    assert tokens(page, PAPER) == PAPER


def check_system(browser, errors):
    """Nothing chosen: the system's setting, before the first paint and as it changes; a choice outranks it."""
    ctx, page = open_page(browser, errors, system="dark")
    html = page.locator("html")
    expect(html).to_have_attribute("data-theme", "dark")
    assert page.evaluate("window.themeAtBody") == "dark"
    assert saved_theme(page) is None
    page.emulate_media(color_scheme="light")
    expect(html).to_have_attribute("data-theme", "light")
    page.emulate_media(color_scheme="dark")
    expect(html).to_have_attribute("data-theme", "dark")
    choose(page, "light")  # a choice stays whatever the system says
    page.reload()
    page.locator("#ov-req tr").first.wait_for()
    expect(html).to_have_attribute("data-theme", "light")
    choose(page, "system")  # and the system's again, with nothing kept
    expect(html).to_have_attribute("data-theme", "dark")
    assert saved_theme(page) is None
    ctx.close()


def check_no_storage(browser, errors):
    """A browser that keeps nothing: the system's theme, and a choice that holds while the page is open."""
    ctx, page = open_page(browser, errors, system="dark", storage=False)
    html = page.locator("html")
    expect(html).to_have_attribute("data-theme", "dark")
    choose(page, "light")
    expect(html).to_have_attribute("data-theme", "light")
    chosen(page, "Light")
    ctx.close()


def check_languages(browser, errors):
    """The menu in each language: the language's words for the theme and its three choices."""
    english = ["Match system", "Light", "Dark"]
    for lang in ("en", "zh", "es", "fr", "de", "ja", "ko"):
        ctx, page = open_page(browser, errors, lang=lang)
        words = page.evaluate("async lang => (await import('/dashboard/i18n/' + lang + '.js')).default.dict",
                              lang) if lang != "en" else {}
        want = [words.get(w, w) for w in english]
        title = words.get("Theme", "Theme")
        if lang != "en":
            assert all(w not in english for w in want) and title != "Theme", (lang, want, title)
        expect(page.locator("#theme")).to_have_attribute("title", title)
        expect(page.locator("#theme")).to_have_attribute("aria-label", title)
        page.locator("#theme").click()
        expect(page.locator("#theme-menu button")).to_have_text(want)
        page.locator('#theme-menu [data-choice="dark"]').click()
        expect(page.locator("html")).to_have_attribute("data-theme", "dark")
        ctx.close()


def check(browser, shots):
    errors = []
    ctx, page = open_page(browser, errors)
    check_light(page, shots)
    check_dark(page, shots)
    ctx.close()
    check_system(browser, errors)
    check_no_storage(browser, errors)
    check_languages(browser, errors)
    assert not errors, errors


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--browser", choices=("chromium", "msedge"), default="chromium")
    cli.add_argument("--shots", help="a directory for screenshots")
    args = cli.parse_args()
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.browser, args=["--disable-gpu"])
        try:
            check(browser, args.shots)
        finally:
            browser.close()
    print("The dashboard kept its light theme, switched to neutral dark greys before the first paint and across a "
          "reload, followed the system's setting, ran without storage, and named the themes in all 7 languages.")
