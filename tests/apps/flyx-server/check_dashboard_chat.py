# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Exercise the Chat reader against overlapping completions and incomplete streams, and the history the page keeps in
the browser. Requires Playwright."""
import argparse
import json
from pathlib import Path
import sys
import time

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from dashboard_sweep import serve


def check(page):
    model = {"id": "tiny", "context": 4096}
    recent = []
    history = []
    replies = []
    tm = {"schema_version": 1, "model": model, "load": {"state": "ready"}, "state": {"phase": "idle"},
          "memory": {"sessions": []}, "rates": {}, "recent": recent, "totals": {}}

    def api(route):
        if route.request.method == "POST" and "/chat/completions" in route.request.url:
            history.append(route.request.post_data_json["messages"])
            route.fulfill(content_type="text/event-stream", body=replies.pop(0))
        else:
            route.fulfill(json=tm if "/telemetry" in route.request.url else {})

    page.route("**/dashboard/**", lambda route: serve(route, ROOT / "apps/flyx-server/dashboard"))
    page.route("**/v1/**", api)
    page.goto("http://[::1]:18479/dashboard/#chat")
    meta = page.locator("#msgs .cv-meta").last

    def send(text, data):
        replies.append(data)
        page.locator("#prompt").fill(text)
        page.locator("#send").click()

    def chunk(body):
        return "data: " + json.dumps(dict(id="chatcmpl-own", **body)) + "\n\n"

    first = chunk({"choices": [{"delta": {"content": "seven"}}]})
    finish = chunk({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    usage = chunk({"choices": [], "usage": {"completion_tokens": 7}})
    done = "data: [DONE]\n\n"
    own = {"id": "chatcmpl-own", "label": "chat", "finished_at": time.time() - 1, "output_tokens": 7,
           "queue_ms": 1500, "prefill_ms": 5500, "decode_ms": 234, "decode_tok_s": 37.7, "tokens_per_round": 2.25}
    recent.extend([dict(own, id="chatcmpl-other", finished_at=time.time(), output_tokens=499, decode_tok_s=99), own])
    send("First", first + finish + usage + done)
    expect(meta).to_contain_text("7 tokens")
    expect(meta).to_contain_text("37.7 tok/s")
    expect(meta).to_contain_text("2.25 tok/round")
    assert "TTFT 7.00 s" not in meta.inner_text() and "499 tokens" not in meta.inner_text()
    send("Second", first + chunk({"error": {"message": "fixture queue timeout"}}) + done)
    expect(meta).to_contain_text("fixture queue timeout")
    expect(page.locator("#msgs .md").last).to_have_text("seven")
    send("Third", first)
    expect(meta).to_contain_text("Reply stream ended early.")
    assert [m["content"] for m in history[-1]] == ["First", "seven", "Third"]
    send("Fourth", (first + finish + usage + done).replace("\n", "\r\n"))
    expect(page.locator("#msgs .md").last).to_have_text("seven")
    expect(meta).to_contain_text("7 tokens")
    assert [m["content"] for m in history[-1]] == ["First", "seven", "Fourth"]

    # Hold the stream open while telemetry switches from another request to this completion.
    page.evaluate("""() => {
        const fetch = window.fetch.bind(window);
        window.fetch = (url, ...args) => {
            if (url !== '/v1/chat/completions') return fetch(url, ...args);
            const stream = new ReadableStream({start(controller) {
                const write = data => controller.enqueue(new TextEncoder().encode('data: ' + data + '\\n\\n'));
                write(JSON.stringify({id: 'chatcmpl-live', choices: [{delta: {role: 'assistant'}}]}));
                window.finishReply = () => { write('[DONE]'); controller.close(); };
            }});
            return Promise.resolve(new Response(stream, {headers: {'Content-Type': 'text/event-stream'}}));
        };
    }""")
    state = tm["state"] = {"id": "chatcmpl-other", "phase": "decode", "running": 2, "generated": 499,
                           "tokens_per_round": 3.1, "decode_tok_s": 82, "elapsed_s": 12}
    tm["rates"]["decode_tok_s"] = 99
    poll = "import('/dashboard/lib/telemetry.js').then(m => m.poll())"
    page.evaluate(poll)
    page.locator("#prompt").fill("Live")
    page.locator("#send").click()
    live = page.locator("#msgs .cv-live")
    expect(live).to_be_visible()
    expect(live).not_to_contain_text("499")
    state.update(id="chatcmpl-live", phase="prefill", prefilled=310, prompt_tokens=1400)
    page.evaluate(poll)
    expect(live).to_contain_text("310 / 1,400")
    state.update(phase="decode", generated=7, decode_tok_s=37.7)
    page.evaluate(poll)
    expect(live).to_contain_text("37.7 tok/s")
    expect(live).not_to_contain_text("99")
    page.evaluate("window.finishReply()")
    expect(live).to_have_count(0)


def check_history(page):
    """Conversations are kept by the browser: listed newest first, reopened and gone on with, deleted."""
    asked = []

    def api(route):
        if route.request.method == "POST" and "/chat/completions" in route.request.url:
            asked.append(route.request.post_data_json["messages"])
            said = "Reply to: " + asked[-1][-1]["content"]
            events = [{"id": "chatcmpl-1", "choices": [{"delta": {"content": said}}]},
                      {"id": "chatcmpl-1", "choices": [], "usage": {"completion_tokens": 7}}]
            route.fulfill(content_type="text/event-stream", body="".join("data: " + json.dumps(e) + "\n\n" for e in events) + "data: [DONE]\n\n")
        else:
            route.fulfill(json={"schema_version": 1, "model": {"id": "tiny", "context": 4096}, "load": {"state": "ready"},
                                "state": {"phase": "idle"}, "memory": {"sessions": []}, "rates": {}, "recent": [], "totals": {}}
                          if "/telemetry" in route.request.url else {})

    def say(text):
        page.locator("#prompt").fill(text)
        page.keyboard.press("Enter")
        expect(page.locator("#msgs .md").last).to_have_text("Reply to: " + text)
        expect(page.locator("#send")).to_be_enabled()

    def history():
        page.locator("#chat-history").click()
        expect(page.locator("#chat-hist")).to_be_visible()

    page.route("**/dashboard/**", lambda route: serve(route, ROOT / "apps/flyx-server/dashboard"))
    page.route("**/v1/**", api)
    page.goto("http://[::1]:18479/dashboard/#chat")
    rows = page.locator("#chat-hist .hist-row")
    say("What is the capital of France?")
    say("And of Italy?")
    page.locator("#clear").click()
    say("Write a haiku about GPUs")
    history()
    expect(rows).to_have_count(2)
    expect(rows.nth(0).locator(".ht")).to_have_text("Write a haiku about GPUs")
    expect(rows.nth(0)).to_have_class("hist-row sel")
    expect(rows.nth(1).locator(".hm")).to_contain_text("4 messages")
    page.reload()  # kept by the browser; the page starts on a new conversation
    expect(page.locator("#chat-empty")).to_be_visible()
    history()
    rows.nth(1).click()
    expect(page.locator("#chat-hist")).to_be_hidden()
    expect(page.locator("#msgs .cv-user")).to_have_count(2)
    say("And of Spain?")
    assert [m["content"] for m in asked[-1] if m["role"] == "user"] == ["What is the capital of France?", "And of Italy?", "And of Spain?"]
    history()
    expect(rows.nth(0).locator(".ht")).to_have_text("What is the capital of France?")  # the one last talked in comes first
    rows.nth(1).hover()
    rows.nth(1).locator(".del").click()
    expect(rows).to_have_count(1)
    kept = json.loads(page.evaluate("localStorage.getItem('flyx.chats')"))
    assert len(kept) == 1 and len(kept[0]["msgs"]) == 6 and kept[0]["msgs"][1]["done"] is True, kept
    page.locator("#chat-hist-clear").click()
    page.locator(".modal").get_by_role("button", name="Delete").click()
    expect(page.locator("#chat-empty")).to_be_visible()
    assert page.evaluate("localStorage.getItem('flyx.chats')") is None


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--browser", choices=("chromium", "msedge"), default="chromium")
    args = cli.parse_args()
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.browser)
        try:
            check(browser.new_page())
            check_history(browser.new_page())
        finally:
            browser.close()
    print("Chat matched completion IDs, handled partial, failed and complete reply streams, and kept its history.")
