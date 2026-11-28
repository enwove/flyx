# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Check the Overview page and the sidebar with Playwright against stubbed telemetry: the throughput chart's two axes
(tokens out left, prompt tokens computed right) and the decode and prefill speeds over it in each phase; speculative
decoding (tokens a round and the share of drafts accepted, the request's and each second's); the recent requests'
cache hit and the count beside "View all", no row of figures since start; the device's load and power as bars; and
VRAM and RAM as bars of flyx's part and the rest, in the Host pane, the open sidebar and the collapsed rail.  Use --browser msedge for Edge;
--shots DIR keeps screenshots."""
import argparse
import copy
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


def telemetry():
    # A decode plateau, then a prefill of 6,000 tokens over two seconds, then decoding again.
    out = [0] * 30 + [118] * 12 + [0, 0] + [121] * 16
    pre = [0] * 42 + [3200, 2800] + [0] * 16
    # Its rounds: 3 drafts each; 1.5 accepted a round in the first stretch (2.5 tokens a round, 50%), 2 in the second
    # (3 tokens a round, 67%).
    rounds = [0] * 30 + [40] * 12 + [0, 0] + [40] * 16
    drafted = [3 * r for r in rounds]
    accepted = [0] * 30 + [60] * 12 + [0, 0] + [80] * 16
    recent = [{"finished_at": 1000.0, "label": "messages", "status": "stop", "session": 1, "prompt_tokens": 2000,
               "reused_tokens": 1700, "output_tokens": 120, "queue_ms": 1, "ttft_ms": 230, "prefill_ms": 100,
               "decode_ms": 1000, "decode_tok_s": 119.0, "tokens_per_round": 2.5, "tool_calls": 0, "grammar_fixes": 0,
               "drafted": 141, "accepted_drafts": 70},
              {"finished_at": 990.0, "label": "chat", "status": "cancelled", "session": 0, "prompt_tokens": 0,
               "reused_tokens": 0, "output_tokens": 0, "queue_ms": 0, "ttft_ms": None, "prefill_ms": 0,
               "decode_ms": 0, "decode_tok_s": None, "tokens_per_round": 0, "tool_calls": 0, "grammar_fixes": 0}]
    return {"schema_version": 1, "sampled_at": time.time(),
            "server": {"name": "flyx-server", "version": "test", "uptime_s": 100, "api_key_required": False},
            "model": {"id": "tiny", "loaded": True, "context": 32768, "device": "NVIDIA GeForce RTX 5070 Ti",
                      "backend": "cuda", "drafter": "mtp", "max_width": 3, "chat_format": "qwen35", "max_sessions": 2,
                      "memory": "weights 10 GB", "load_s": 3},
            "load": {"state": "ready"},
            "state": {"phase": "idle", "queued": 0, "running": 0},
            "rates": {"decode_tok_s": 0, "output_per_second": out, "prefill_per_second": pre, "rounds_per_second": rounds,
                      "drafted_per_second": drafted, "accepted_per_second": accepted},
            "memory": {"ledger_used": 10 * GiB, "ledger_budget": 13 * GiB, "runtime": GiB // 2,
                       "sessions": [{"id": 1, "tokens": 2120}]},
            "device": {"name": "NVIDIA GeForce RTX 5070 Ti", "util": 40, "mem_util": 30, "mem_used": 12 * GiB,
                       "mem_total": 16 * GiB, "temp_c": 55, "power_w": 120, "power_limit_w": 300, "sm_clock_mhz": 2800,
                       "pcie_gen": 5, "pcie_width": 16, "fan": 30, "mem_clock_mhz": 14000, "clock_limits": ""},
            "process": {"cpu_s": 10, "rss": GiB, "ram": 3 * GiB},  # the K/V's host memory outside the resident set
            "system": {"ram_total": 32 * GiB, "ram_available": 20 * GiB, "cpus": 16, "cpu_s": 1000},
            "totals": {"requests": 12, "errors": 0, "cancelled": 1, "prompt_tokens": 2000, "reused_tokens": 1700,
                       "output_tokens": 120, "tool_calls": 0, "grammar_fixes": 0},
            "recent": recent}


def check(page, shots):
    tm = {"now": telemetry()}
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))

    def api(route):
        if "/telemetry" in route.request.url:
            body = copy.deepcopy(tm["now"])
            body["sampled_at"] = time.time()
            return route.fulfill(json=body)
        return route.fulfill(json={})

    page.route("**/dashboard/**", lambda route: serve(route, ROOT / "apps/flyx-server/dashboard"))
    page.route("**/v1/**", api)
    page.add_init_script("try { localStorage.setItem('flyx.lang', 'en'); } catch (e) {}")
    page.goto(ORIGIN + "/dashboard/#overview")
    page.locator("#ov-req tr").first.wait_for()

    # No row of figures since start: the request count went beside "View all".
    assert page.locator(".kpi, .kpis").count() == 0
    expect(page.locator("#ov-all")).to_have_text("View all (12)")
    heads = page.locator("#ov-req").locator("xpath=..").locator("thead th").all_inner_texts()
    assert heads == ["Finished", "API", "Status", "Output", "Tok/s", "TTFT", "Cache hit", "Tok/round"], heads
    first = page.locator("#ov-req tr").nth(0).locator("td")
    assert first.nth(6).inner_text() == "85.0%"
    assert first.nth(6).get_attribute("title") == "1,700 of 2,000 prompt tokens reused"
    assert page.locator("#ov-req tr").nth(1).locator("td").nth(6).inner_text() == "-"

    # Idle: both speeds are the last request's (prefill: 300 tokens computed in 100 ms).
    fig = page.locator("#ov-throughput .fig")
    expect(fig).to_have_text("119.0decode tok/s, last request3,000prefill tok/s, last request")
    expect(page.locator("#r-decode")).to_have_text("119.0tok/s")
    expect(page.locator("#r-prefill")).to_have_text("3,000tok/s")
    # Another model loaded since (a switch): the requests stay listed, but "the last request"'s figures beside its
    # drafter are of none: the served model has run none yet.
    tm["now"]["model"]["loaded_at"] = 1500.0
    expect(fig).not_to_contain_text("119.0")
    expect(fig).not_to_contain_text("last request")
    expect(page.locator("#ov-spec .fig")).not_to_contain_text("2.50")
    expect(page.locator("#r-decode")).not_to_contain_text("119.0")
    expect(page.locator("#ov-req tr")).to_have_count(2)
    tm["now"]["model"]["loaded_at"] = 900.0  # loaded before both: they are its own
    expect(fig).to_have_text("119.0decode tok/s, last request3,000prefill tok/s, last request")
    tm["now"]["state"] = {"phase": "prefill", "queued": 0, "running": 1, "label": "messages", "session": 1,
                          "prompt_tokens": 9000, "reused_tokens": 1000, "prefilled": 5000, "generated": 0,
                          "prefill_tok_s": 5123.4, "decode_tok_s": None, "elapsed_s": 0.8, "max_new_tokens": 0}
    tm["now"]["memory"]["sessions"] = [{"id": 1, "tokens": 5000}]
    expect(fig).to_have_text("119.0decode tok/s, last request5,123prefill tok/s, now")
    expect(page.locator("#r-prefill")).to_have_text("5,123tok/s")
    expect(page.locator("#sessions .n")).to_have_text("5,000 / 32.8k")
    expect(page.locator("#live .lrow", has_text="Context").locator(".v")).to_have_text("5,000 / 32.8k")
    spec = page.locator("#ov-spec .fig")
    expect(spec).to_have_text("2.50tok/round, last request50%accepted, last request")
    expect(page.locator("#ov-spec .aside")).to_have_text("mtp, up to 3")
    tm["now"]["state"].update(phase="decode", prefilled=9000, generated=40, prefill_tok_s=4000.0, decode_tok_s=150.2,
                              tokens_per_round=3.1, drafted=30, accepted=19)
    tm["now"]["memory"]["sessions"] = [{"id": 1, "tokens": 9040}]
    tm["now"]["rates"]["decode_tok_s"] = 150.2
    expect(fig).to_have_text("150.2decode tok/s, now4,000prefill tok/s, this request")
    expect(page.locator("#r-decode")).to_have_text("150.2tok/s")
    expect(page.locator("#r-prefill")).to_have_text("4,000tok/s")
    expect(page.locator("#sessions .n")).to_have_text("9,040 / 32.8k")
    expect(page.locator("#live .lrow", has_text="Context").locator(".v")).to_have_text("9,040 / 32.8k")
    # Older server without windowed rates falls back to this request's cumulative figures.
    expect(spec).to_have_text("3.10tok/round, this request63%accepted, this request")
    # With windowed rates (matching decode tok/s), both panes report the recent window.
    tm["now"]["rates"]["tokens_per_round"] = 15.0
    tm["now"]["rates"]["acceptance"] = 0.93
    expect(spec).to_have_text("15.00tok/round, now93%accepted, now")

    # Speculative decoding's chart: tokens a round to 4 (the widest round's 3 drafts and its own token) on the left,
    # the share accepted on the right; a second without rounds is no point.
    sc = page.locator("#ov-spec canvas")
    inked = sc.evaluate("""c => { const g = c.getContext('2d'), w = c.width, h = c.height;
        const d = g.getImageData(w - Math.round(36 * devicePixelRatio), 0, Math.round(36 * devicePixelRatio), h).data;
        let n = 0; for (let i = 3; i < d.length; i += 4) n += d[i] > 0; return n; }""")
    assert inked > 50, "no right axis labels on the speculation chart: %d pixels" % inked
    sbox, stip = sc.bounding_box(), page.locator("#ov-spec .tip")
    seen = set()
    for i in range(120):
        page.mouse.move(sbox["x"] + sbox["width"] * (0.98 - i * 0.004), sbox["y"] + sbox["height"] / 2)
        seen.add(re.sub(r"^\d\d:\d\d:\d\d\s+", "", stip.inner_text()))  # less the second's time
    assert "3.00 tok/round · 67% accepted · 40 rounds" in seen and "2.50 tok/round · 50% accepted · 40 rounds" in seen, seen
    assert "no rounds" in seen, seen
    page.mouse.move(0, 0)
    assert page.locator(".pane h2", has_text="GPU").count() == 0, "the GPU chart is gone"

    # The device's load and power as bars, its name in the pane's head.
    expect(page.locator("#ov-device .aside")).to_have_text("RTX 5070 Ti")
    expect(page.locator("#ov-util .mk .v")).to_have_text("40%")
    expect(page.locator("#ov-power .mk .v")).to_have_text("120 / 300 W")

    # The chart draws the right axis (labels right of the plot) and both series in its tooltip.
    canvas = page.locator("#ov-throughput canvas")
    inked = canvas.evaluate("""c => { const g = c.getContext('2d'), w = c.width, h = c.height;
        const d = g.getImageData(w - Math.round(36 * devicePixelRatio), 0, Math.round(36 * devicePixelRatio), h).data;
        let n = 0; for (let i = 3; i < d.length; i += 4) n += d[i] > 0; return n; }""")
    assert inked > 50, "no right axis labels drawn: %d pixels" % inked
    box = canvas.bounding_box()
    tip = page.locator("#ov-throughput .tip")
    spike = lambda: "prefill 3,200" in tip.inner_text() or "prefill 2,800" in tip.inner_text()
    for i in range(80):  # the prefill seconds sit about 17 s before now, near the right
        page.mouse.move(box["x"] + box["width"] * (0.98 - i * 0.005), box["y"] + box["height"] / 2)
        if spike():
            break
    assert "decode" in tip.inner_text() and spike(), tip.inner_text()
    page.mouse.move(0, 0)

    # Host: VRAM (flyx: the ledger and its runtime, 10.5 of 12 GiB in use) and RAM (flyx: its resident 3 GiB).
    vram, ram = page.locator("#ov-vram"), page.locator("#ov-ram")
    expect(vram.locator(".mk .v")).to_have_text("12.0 / 16.0 GiB")
    expect(vram.locator(".mleg").locator("span")).to_have_text(["FLYX 10.5 GiB", "others 1.5 GiB"])
    expect(ram.locator(".mleg").locator("span")).to_have_text(["FLYX 3.0 GiB", "others 9.0 GiB"])

    def widths(bar):
        return bar.evaluate("""b => { const all = b.getBoundingClientRect().width;
            return [...b.children].map(i => [i.className, Math.round(i.getBoundingClientRect().width / all * 1000) / 10]); }""")

    def parts(bar, want):  # each part's share of the bar, in percent (within a pixel of a narrow bar)
        ok = lambda got: [c for c, _ in got] == [c for c, _ in want] and all(abs(g - w) < 1.5 for (_, g), (_, w) in zip(got, want))
        for _ in range(20):  # the parts move to a new width over 0.4 s
            got = widths(bar)
            if ok(got):
                return
            page.wait_for_timeout(100)
        assert ok(got), got

    parts(page.locator("#ov-util .mbar"), [["own", 40.0]])
    parts(page.locator("#ov-power .mbar"), [["own", 40.0]])
    parts(vram.locator(".mbar"), [["own", 65.6], ["oth", 9.4]])
    parts(ram.locator(".mbar"), [["own", 9.4], ["oth", 28.1]])
    assert "FLYX 10.5 GiB · others 1.5 GiB · free 4.0 GiB" == vram.locator(".mbar").get_attribute("title")
    device = page.locator(".sec.c-3 .pane").nth(1).locator("dt").all_inner_texts()
    assert "Memory" not in device, device
    # Where the system counts no process's share, flyx's GPU part is its ledger; the runtime falls in the rest.
    tm["now"]["memory"]["runtime"] = None
    expect(vram.locator(".mleg").locator("span")).to_have_text(["FLYX 10.0 GiB", "others 2.0 GiB"])
    tm["now"]["memory"]["runtime"] = GiB // 2

    # The open sidebar: the same bars and figures.
    expect(page.locator("#r-vram-row .mleg").locator("span")).to_have_text(["FLYX 10.5", "others 1.5"])
    expect(page.locator("#r-ram-row .mleg").locator("span")).to_have_text(["FLYX 3.0", "others 9.0"])
    parts(page.locator("#r-vram-row .mbar"), [["own", 65.6], ["oth", 9.4]])
    for row in ("#r-vram-row", "#r-ram-row"):
        assert page.locator(row + " .mleg").evaluate("e => e.scrollWidth <= e.clientWidth"), row + ": legend cut off"
    if shots:
        page.screenshot(path=str(Path(shots) / "overview-light.png"), full_page=True)

    # The collapsed rail: two parts in each memory bar.
    page.locator("#collapse").click()
    rail = page.locator("#rail .rb.mb")
    expect(rail).to_have_count(2)
    assert [rail.nth(i).locator("i").count() for i in range(2)] == [2, 2]
    assert "VRAM 12.0 / 16.0 GiB (FLYX 10.5)" in page.locator("#rail").get_attribute("title")
    if shots:
        page.screenshot(path=str(Path(shots) / "overview-rail.png"))
    page.locator("#rail").click()
    expect(page).to_have_title("▸ tiny · FLYX")

    # In another language.
    page.evaluate("localStorage.setItem('flyx.lang', 'zh')")
    page.add_init_script("try { localStorage.setItem('flyx.lang', 'zh'); } catch (e) {}")
    page.reload()
    page.locator("#ov-req tr").first.wait_for()
    expect(page.locator("#ov-all")).to_have_text("查看全部 (12)")
    assert "缓存命中" in page.locator("#ov-req").locator("xpath=..").locator("thead").inner_text()
    expect(fig).to_have_text("150.2Decoding tok/s，当前4,000Prefill tok/s，本条")
    expect(spec).to_have_text("15.00接受长度，当前93%接受率，当前")
    expect(vram.locator(".mleg").locator("span")).to_have_text(["FLYX 10.5 GiB", "其他 1.5 GiB"])
    expect(page).to_have_title("▸ tiny · FLYX")
    if shots:
        page.screenshot(path=str(Path(shots) / "overview-zh.png"), full_page=True)
    assert not errors, errors


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--browser", choices=("chromium", "msedge"), default="chromium")
    cli.add_argument("--shots", help="a directory for screenshots")
    args = cli.parse_args()
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.browser)
        try:
            check(browser.new_page(viewport={"width": 1500, "height": 950}), args.shots)
        finally:
            browser.close()
    print("The Overview showed both speeds, speculative decoding, the cache hit, the device's load and power, and VRAM "
          "and RAM as flyx's part and the rest.")
