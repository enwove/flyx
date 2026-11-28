# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The dashboard's frame at a phone's width and a desktop's, without a server. Requires Playwright. Use --browser
msedge for Microsoft Edge.

A phone (390 px): no page wider than the screen; the sidebar a drawer the menu button opens and a page chosen, a tap
beside it or Escape shuts, whole even where a wide screen collapsed it to its rail. Both widths: the chat, the request
log and the server log fill the window under the notice above them, the chat's input in sight."""
import argparse
from pathlib import Path
import sys

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_dashboard_model import Server, telemetry  # the Model page's stubbed server: its library, its folders

PAGES = ("overview", "model", "chat", "requests", "logs", "integration", "settings")

# Elements wider than the screen that no box scrolling across holds.
OVERFLOW = """() => { const out = [];
  for (const el of document.querySelectorAll('main *')) {
    const r = el.getBoundingClientRect();
    if (!r.width || r.right <= innerWidth + 1 || getComputedStyle(el).position === 'fixed') continue;
    let held = false;
    for (let p = el.parentElement; p && p.tagName !== 'MAIN'; p = p.parentElement) {
      const o = getComputedStyle(p).overflowX;
      if ((o === 'auto' || o === 'scroll' || o === 'hidden') && p.getBoundingClientRect().right <= innerWidth + 1) {
        held = true;
        break;
      }
    }
    if (!held) out.push(el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') + ' ' + Math.round(r.right));
  }
  return out.slice(0, 5); }"""


def frame(page, url, width):
    page.set_viewport_size({"width": width, "height": 844})
    for name in PAGES:
        page.goto(url + "#" + name)
        page.wait_for_timeout(400)
        shown = page.evaluate("[...document.querySelectorAll('.page > section')].filter((s) => !s.hidden).map((s) => s.id)")
        assert shown == ["tab-" + name], (name, width, shown)
        if width < 760:
            over = page.evaluate(OVERFLOW)
            assert not over, (name, width, over)
        if name in ("chat", "requests", "logs"):
            main = page.evaluate("[document.querySelector('main').scrollHeight, document.querySelector('main').clientHeight]")
            assert main[0] <= main[1] + 1, (name, width, "the page runs past the window", main)
        if name == "chat":
            box = page.locator("#prompt").bounding_box()
            assert box and box["y"] + box["height"] <= 844, (width, "the chat's input out of sight", box)


def drawer(page, url):
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(url + "#chat")
    page.wait_for_timeout(400)
    side, menu = page.locator("#side"), page.locator("#menu")
    assert menu.is_visible() and not side.is_visible(), "a phone: the menu button, the sidebar shut"
    menu.click()
    assert side.is_visible() and menu.get_attribute("aria-expanded") == "true", "the menu opens the drawer"
    side.locator(".navi").nth(2).click()  # Requests
    page.wait_for_timeout(300)
    assert not side.is_visible() and page.url.endswith("#requests"), "a page chosen shuts the drawer and shows the page"
    menu.click()
    page.locator(".scrim").click(position={"x": 370, "y": 400})
    page.wait_for_timeout(300)  # the drawer slides out, hidden at the end
    assert not side.is_visible(), "a tap beside the drawer shuts it"
    menu.click()
    page.keyboard.press("Escape")
    page.wait_for_timeout(300)
    assert not side.is_visible(), "Escape shuts it"
    # A wide screen's collapsed sidebar: the drawer still holds the whole of it, the pages' names in words.
    page.evaluate("document.documentElement.dataset.side = 'collapsed'")
    menu.click()
    assert side.locator(".navi .t").first.is_visible(), "a collapsed sidebar opens whole in the drawer"
    page.keyboard.press("Escape")
    page.set_viewport_size({"width": 1400, "height": 844})
    assert not menu.is_visible() and side.is_visible(), "a wide screen: no menu button, the sidebar in place"


def check(page):
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    # A model served: a notice above every page (the model ready), what pushed full-height pages past the window.
    server = Server(page, telemetry({"model": "acme/Tiny-27B-GGUF:Q4_K_M-with-a-long-name"}))
    network = {"lan_access": False, "listen_host": "127.0.0.1", "port": 8080, "host_from_cli": False,
               "serving_lan": False, "restart_required": False, "remote_manage": False, "remote_manage_effective": False,
               "no_auth_from_cli": False, "api_key_set": False, "api_urls": ["http://127.0.0.1:8080/v1"],
               "settings_error": None}
    page.route("**/v1/settings/network", lambda route: route.fulfill(json=network))
    page.route("**/v1/requests**", lambda route: route.fulfill(json={"enabled": True, "requests": []}))
    page.route("**/v1/logs**", lambda route: route.fulfill(json={"enabled": True, "lines": []}))
    assert server
    url = "http://[::1]:18479/dashboard/"
    frame(page, url, 390)
    frame(page, url, 1400)
    drawer(page, url)
    assert not errors, errors


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--browser", choices=("chromium", "msedge"), default="chromium")
    args = cli.parse_args()
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.browser)
        try:
            check(browser.new_page(locale="en-US"))
        finally:
            browser.close()
    print("The dashboard fit a phone's width (the sidebar a drawer) and its full-height pages fit the window.")
