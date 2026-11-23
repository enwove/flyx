# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Check concurrent request figures without a server. Requires Playwright. Use --browser msedge for Microsoft Edge."""
import argparse
from pathlib import Path
import sys

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from dashboard_sweep import serve


def check(page):
    # Both HTTP handlers finished just after the title. The main job had already finished on the engine.
    main = {"queue_ms": 30, "prefill_ms": 170, "decode_ms": 460, "decode_tok_s": 50, "tokens_per_round": 2}
    title = {"queue_ms": 2, "prefill_ms": 7, "decode_ms": 350, "decode_tok_s": 20, "tokens_per_round": 1}
    recent = [dict(main, label="messages", finished_at=10.85, ttft_ms=232),
              dict(title, label="messages", finished_at=11.02),
              dict(title, label="messages", finished_at=10.25, ttft_ms=None)]
    tm = {"schema_version": 1, "model": {"id": "tiny"}, "load": {"state": "ready"}, "state": {"phase": "idle"},
          "memory": {"sessions": []}, "rates": {}, "recent": recent, "totals": {}}

    def entry(key, finish, ttft, timing, cached, output):
        return {"id": key, "api": "messages", "model": "tiny", "started_at": 10, "finished_at": finish,
                "status": "length", "request": {}, "ttft_ms": ttft, "timing": timing,
                "usage": {"prompt_tokens": 2000, "cached_tokens": cached, "output_tokens": output}}

    rows = [entry("main", 11.04, 240, main, 1700, 24), entry("title", 11.03, 16, title, 0, 8)]
    rows[0]["client_hints"] = {"request_class": "main", "context_compacted": "manual"}
    rows[0]["request"] = {"thinking": {"type": "enabled"}, "output_config": {"effort": "max", "format": {"type": "json_schema"}}}
    rows[1]["client_hints"] = {"request_class": "auxiliary"}
    rows[1]["request"] = {"thinking": {"type": "disabled"}}
    rows.append(dict(rows[0], id="summary", client_hints={"request_class": "compaction", "compaction": "reactive"},
                     api="chat", request={"reasoning_effort": "high", "enable_thinking": False}))
    old = entry("older", 11.06, 75, None, 0, 3)
    old["api"] = "responses"
    old["request"] = {"reasoning": {"effort": "none"}}
    del old["timing"]  # Older logs carry observed TTFT but no engine figures.
    rows.append(old)
    rows.append(dict(old, id="replaced", status="cancelled", cancel_reason="superseded", request=None, ttft_ms=None))
    page.route("**/dashboard/**", lambda route: serve(route, ROOT / "apps/flyx-server/dashboard"))
    log = {"enabled": True, "requests": rows}
    polls = []
    tag = '"initial"'

    def api(route):
        if "/requests" not in route.request.url:
            return route.fulfill(json=tm if "/telemetry" in route.request.url else {})
        headers = route.request.headers
        polls.append((headers.get("authorization"), headers.get("if-none-match")))
        if headers.get("authorization") == "Bearer denied":
            return route.fulfill(status=403, json={})
        if headers.get("if-none-match") == tag:
            return route.fulfill(status=304, headers={"ETag": tag})
        return route.fulfill(json=log, headers={"ETag": tag})

    page.route("**/v1/**", api)
    page.goto("http://[::1]:18479/dashboard/#requests:main")
    facts = page.locator("#rd-facts")
    assert "240 ms" in facts.inner_text() and "50.0 tok/s" in facts.inner_text()
    chips = page.locator(".cv-chips")
    assert "First turn after compaction · Manual compaction" in chips.inner_text()
    assert "effort max" in chips.inner_text() and "JSON answer" in chips.inner_text()
    assert "thinking off" not in chips.inner_text()
    detail = " ".join(page.locator(".cv-meta").all_inner_texts())
    assert "Prompt reused 1,700 / 2,000" in detail
    for text in ("Queue time 30 ms", "Prefill time 170 ms", "Decode time 460 ms"):
        assert text in detail
    page.locator('tr[data-id="title"]').click()
    assert "16 ms" in facts.inner_text() and "20.0 tok/s" in facts.inner_text()
    assert "Claude Code · Auxiliary request" in chips.inner_text() and "thinking off" in chips.inner_text()
    page.locator('tr[data-id="summary"]').click()
    assert "Conversation summary" in chips.inner_text() and "Prompt exceeded the context" in chips.inner_text()
    assert "thinking off" in chips.inner_text() and "effort high" in chips.inner_text()
    page.locator('tr[data-id="older"]').click()
    assert "75 ms" in facts.inner_text() and "tok/s" not in facts.inner_text()
    assert "Claude Code" not in chips.inner_text() and "thinking off" in chips.inner_text()
    assert "effort none" not in chips.inner_text()
    page.locator('tr[data-id="replaced"]').click()
    page.locator('[data-dt="req"]').click()
    assert page.locator("#rd-body pre.raw").inner_text() == "null"
    def wait_log(status):
        return page.expect_response(lambda r: "/v1/requests" in r.url and r.status == status)

    with wait_log(304):
        pass  # The next poll revalidates the rows already on screen.
    assert polls[-1] == (None, tag) and page.locator('tr[data-id="main"]').count() == 1
    page.evaluate("location.hash = 'overview'")
    page.locator("#ov-req").wait_for(state="visible")
    with wait_log(304):
        page.go_back()
    assert page.locator('tr[data-id="main"]').is_visible()
    with wait_log(403):
        page.evaluate("location.hash = 'requests&key=denied'")
    page.locator('tr[data-id="main"]').wait_for(state="detached")
    assert polls[-1] == ("Bearer denied", None)
    with wait_log(200):
        page.evaluate("location.hash = 'requests&key=authorized'")
    assert polls[-1] == ("Bearer authorized", None)
    page.locator('tr[data-id="main"]').wait_for()
    tag = '"disabled"'
    log.update(enabled=False, requests=[])
    page.goto("http://[::1]:18479/dashboard/#requests")
    page.reload()
    for i, expected in enumerate(("232 ms", "-", "-")):
        assert page.locator("#logs tbody tr").nth(i).locator("td").nth(4).inner_text() == expected
    page.goto("http://[::1]:18479/dashboard/#overview")
    for i, expected in enumerate(("232 ms", "-", "-")):
        assert page.locator("#ov-req tr").nth(i).locator("td").nth(5).inner_text() == expected
    assert page.locator(".lk", has_text="TTFT").locator("+ .lbig").inner_text() == "232 ms"
    recent[0]["ttft_ms"] = None
    page.reload()
    assert page.locator(".lk", has_text="TTFT").locator("+ .lbig").inner_text() == "-"


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--browser", choices=("chromium", "msedge"), default="chromium")
    args = cli.parse_args()
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.browser)
        try:
            check(browser.new_page())
        finally:
            browser.close()
    print("The Requests view used each request's own timing and cache usage.")
