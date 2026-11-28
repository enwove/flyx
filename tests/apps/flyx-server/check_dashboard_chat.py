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


def check_vision_upload(page):
    """Chat image attachment: visible hint without mmproj, WebP/misnamed-WebP canvas transcoding,
    clipboardData.items paste fallback, drag-and-drop, unsupported format error, and multipart payload."""
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    model = {"id": "vlm", "loaded": True, "context": 8192, "vision": False}
    tm = {"schema_version": 1, "model": model, "load": {"state": "ready"}, "state": {"phase": "idle"},
          "memory": {"sessions": []}, "rates": {}, "recent": [], "totals": {}}
    sent = []

    def api(route):
        url = route.request.url
        if route.request.method == "POST" and "/chat/completions" in url:
            sent.append(route.request.post_data_json["messages"])
            events = [{"id": "chatcmpl-v", "choices": [{"delta": {"content": "looks good"}}]},
                      {"id": "chatcmpl-v", "choices": [], "usage": {"completion_tokens": 2}}]
            route.fulfill(content_type="text/event-stream",
                          body="".join("data: " + json.dumps(e) + "\n\n" for e in events) + "data: [DONE]\n\n")
        elif "/telemetry" in url:
            route.fulfill(json=tm)
        elif "/hub/cached" in url:
            route.fulfill(json={"cache": "", "dirs": [], "models": []})
        elif "/hub/search" in url:
            route.fulfill(json={"query": "", "models": []})
        else:
            route.fulfill(json={})

    page.route("**/dashboard/**", lambda route: serve(route, ROOT / "apps/flyx-server/dashboard"))
    page.route("**/v1/**", api)
    page.goto("http://[::1]:18479/dashboard/#chat")

    # Without a vision projector, the attach button stays visible in a muted state and explains what to load.
    btn = page.locator("#chat-img")
    expect(btn).to_be_visible()
    expect(btn).to_have_attribute("data-vision", "0")
    btn.click()
    err = page.locator("#chat-imgerr")
    expect(err).to_contain_text("Load a vision projector (mmproj) in Model settings to attach images.")
    page.locator("#chat-img-model").click()
    expect(page.locator("#tab-model")).to_be_visible()

    # Back to Chat: pasting via clipboardData.items (empty .files) and dropping files show the hint without attaching.
    page.locator('.side nav .navi[data-tab="chat"]').click()
    expect(page.locator("#tab-chat")).to_be_visible()
    page.locator("#clear").click()
    expect(err).to_have_count(0)
    page.evaluate("""() => {
        const file = new File([new Uint8Array([137, 80, 78, 71])], 'clip.png', {type: 'image/png'});
        const ev = new Event('paste', {bubbles: true, cancelable: true});
        Object.defineProperty(ev, 'clipboardData', {
            value: {files: [], items: [{kind: 'file', type: 'image/png', getAsFile: () => file}]}
        });
        document.getElementById('prompt').dispatchEvent(ev);
    }""")
    expect(err).to_contain_text("Load a vision projector (mmproj)")
    expect(page.locator("#chat-imgs")).to_have_count(0)
    page.locator("#clear").click()
    page.evaluate("""() => {
        const file = new File([new Uint8Array([137, 80, 78, 71])], 'drop.png', {type: 'image/png'});
        const ev = new Event('drop', {bubbles: true, cancelable: true});
        Object.defineProperty(ev, 'dataTransfer', {value: {types: ['Files'], files: [file]}});
        document.querySelector('#tab-chat .compose').dispatchEvent(ev);
    }""")
    expect(err).to_contain_text("Load a vision projector (mmproj)")
    expect(page.locator("#chat-imgs")).to_have_count(0)

    # Enable vision in telemetry: button activates, no-vision note clears, and vision status appears.
    model.update(vision=True, vision_device="disk", vision_quant="q8")
    page.evaluate("import('/dashboard/lib/telemetry.js').then(m => m.poll())")
    expect(btn).to_have_attribute("data-vision", "1")
    expect(err).to_have_count(0)
    expect(page.locator("#chatstat")).to_contain_text("disk")

    # Build genuine 2x2 PNG and WebP payloads inside the browser via canvas.
    png_b64, webp_b64 = page.evaluate("""async () => {
        const c = document.createElement('canvas');
        c.width = 2; c.height = 2;
        const ctx = c.getContext('2d');
        ctx.fillStyle = '#26704f';
        ctx.fillRect(0, 0, 2, 2);
        const toB64 = (mime) => new Promise((res) => c.toBlob((b) => {
            const r = new FileReader();
            r.onload = () => res(r.result.split(',')[1]);
            r.readAsDataURL(b);
        }, mime));
        return [await toB64('image/png'), await toB64('image/webp')];
    }""")
    import base64
    png_bytes = base64.b64decode(png_b64)
    webp_bytes = base64.b64decode(webp_b64)
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    assert webp_bytes[:4] == b"RIFF" and webp_bytes[8:12] == b"WEBP"

    # Attach native PNG, WebP, and a WebP disguised as .png (like Wikia downloads): all produce valid PNG data URIs.
    file_input = page.locator('#tab-chat input[type="file"]')
    file_input.set_input_files([
        {"name": "native.png", "mimeType": "image/png", "buffer": png_bytes},
        {"name": "photo.webp", "mimeType": "image/webp", "buffer": webp_bytes},
        {"name": "disguised.png", "mimeType": "image/png", "buffer": webp_bytes},
    ])
    expect(page.locator("#chat-imgs .compose-thumb")).to_have_count(3)
    expect(err).to_have_count(0)

    # Paste via clipboardData.items fallback (empty .files) adds a 4th thumbnail with no duplicate.
    page.evaluate("""(b64) => {
        const raw = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        const file = new File([raw], 'pasted.webp', {type: 'image/webp'});
        const ev = new Event('paste', {bubbles: true, cancelable: true});
        Object.defineProperty(ev, 'clipboardData', {
            value: {files: [], items: [{kind: 'file', type: 'image/webp', getAsFile: () => file}]}
        });
        document.getElementById('prompt').dispatchEvent(ev);
    }""", webp_b64)
    expect(page.locator("#chat-imgs .compose-thumb")).to_have_count(4)

    # Remove the last thumbnail; verify non-image file shows an error without losing valid attachments.
    page.locator("#chat-imgs .compose-thumb button").last.click()
    expect(page.locator("#chat-imgs .compose-thumb")).to_have_count(3)
    file_input.set_input_files([{"name": "notes.txt", "mimeType": "text/plain", "buffer": b"hello"}])
    expect(err).to_contain_text("Unsupported image format")
    expect(page.locator("#chat-imgs .compose-thumb")).to_have_count(3)

    # Send message and verify all 3 attached images were encoded as genuine PNG data URIs.
    page.locator("#prompt").fill("Describe these")
    page.locator("#send").click()
    expect(page.locator("#msgs .md").last).to_have_text("looks good")
    parts = sent[-1][-1]["content"]
    assert isinstance(parts, list) and len(parts) == 4, parts
    for i in range(3):
        assert parts[i]["type"] == "image_url", parts[i]
        url = parts[i]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,"), url[:40]
        raw = base64.b64decode(url.split(",", 1)[1])
        assert raw[:8] == b"\x89PNG\r\n\x1a\n", f"part {i} was not valid PNG bytes: {raw[:16]!r}"
    assert parts[3] == {"type": "text", "text": "Describe these"}, parts[3]
    expect(page.locator("#msgs .cv-imgs img")).to_have_count(3)
    assert not errors, errors


def check_thinking(page):
    """The Thinking box shows the server's default (--no-think, --chat-template-kwargs: telemetry's
    server.enable_thinking) and a request leaves enable_thinking to the server until the viewer picks a mode."""
    bodies = []
    server = {"name": "flyx-server", "enable_thinking": False}

    def api(route):
        if route.request.method == "POST" and "/chat/completions" in route.request.url:
            bodies.append(route.request.post_data_json)
            events = [{"id": "chatcmpl-1", "choices": [{"delta": {"content": "ok"}}]},
                      {"id": "chatcmpl-1", "choices": [], "usage": {"completion_tokens": 1}}]
            route.fulfill(content_type="text/event-stream",
                          body="".join("data: " + json.dumps(e) + "\n\n" for e in events) + "data: [DONE]\n\n")
        else:
            route.fulfill(json={"schema_version": 1, "server": server, "model": {"id": "tiny", "context": 4096},
                                "load": {"state": "ready"}, "state": {"phase": "idle"}, "memory": {"sessions": []},
                                "rates": {}, "recent": [], "totals": {}}
                          if "/telemetry" in route.request.url else {})

    def say(text):
        page.locator("#prompt").fill(text)
        page.keyboard.press("Enter")
        expect(page.locator("#msgs .md").last).to_have_text("ok")
        expect(page.locator("#send")).to_be_enabled()
        return bodies[-1]

    page.route("**/dashboard/**", lambda route: serve(route, ROOT / "apps/flyx-server/dashboard"))
    page.route("**/v1/**", api)
    page.goto("http://[::1]:18479/dashboard/#chat")
    box = page.locator("#think")
    expect(box).not_to_be_checked()  # a server started with --no-think
    assert "chat_template_kwargs" not in say("First"), bodies[-1]
    box.check()
    assert say("Second")["chat_template_kwargs"] == {"enable_thinking": True}, bodies[-1]
    box.uncheck()
    assert say("Third")["chat_template_kwargs"] == {"enable_thinking": False}, bodies[-1]
    # A server with no default of its own: the box checked (the template thinks), still nothing sent until picked.
    server["enable_thinking"] = None
    page.reload()
    expect(box).to_be_checked()
    assert "chat_template_kwargs" not in say("Fourth"), bodies[-1]
    # A server launched with --chat-template-kwargs '{"reasoning_effort":"none"}', whose telemetry says false for it
    # (chat_template_kwargs.py checks that it does): a fresh page's box unchecked, and still no override sent.
    server["enable_thinking"] = False
    page.reload()
    expect(box).not_to_be_checked()
    assert "chat_template_kwargs" not in say("Fifth"), bodies[-1]


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--browser", choices=("chromium", "msedge"), default="chromium")
    args = cli.parse_args()
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.browser)
        try:
            check(browser.new_page(locale="en-US"))
            check_history(browser.new_page(locale="en-US"))
            check_vision_upload(browser.new_page(locale="en-US"))
            check_thinking(browser.new_page(locale="en-US"))
        finally:
            browser.close()
    print("Chat matched completion IDs, handled partial, failed and complete reply streams, kept its history, attached "
          "images, and left thinking to the server until asked.")
