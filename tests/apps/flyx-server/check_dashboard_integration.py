# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Check the rendered client guides without a server, client credentials or a model.

Python 3.11+, pip install playwright pyyaml, then playwright install chromium.
Run this file with --browser msedge to use an installed Microsoft Edge.
"""
import argparse
import base64
from functools import cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import product
import json
import os
import re
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from threading import Thread
import tomllib
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import quote

from playwright.sync_api import sync_playwright
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from dashboard_sweep import serve

# Non-default address and model names exercise the literal arguments, JSON, YAML and TOML a user copies.
ORIGIN = "http://[::1]:18473"
MODEL = """lab/β "Local" & 'One' ‘demo’"""
KEY = """sample 'key' "with" spaces & symbols"""
LABELS = {"en": "Integration", "zh": "集成", "es": "Integración", "de": "Integration",
          "fr": "Intégration", "ja": "連携", "ko": "연동"}


def claude_settings(command, shell, keyed):
    if shell == "sh":
        if keyed:
            # The quoted heredoc goes to jq. The actual key is read from the environment at launch.
            assert ".env.ANTHROPIC_AUTH_TOKEN = env.FLYX_KEY" in command
            settings = json.loads(command.split("<<'FLYX_JSON'\n", 1)[1].split("\nFLYX_JSON\n", 1)[0])
        else:
            cli, flag, value = shlex.split(command)
            assert (cli, flag) == ("claude", "--settings")
            settings = json.loads(value)
    else:
        ps = shutil.which("powershell") or shutil.which("pwsh")
        if not ps:
            return  # Still check the rendered command on hosts without PowerShell.
        # Read the temporary file inside the launch, before the snippet's finally block cleans it up.
        probe = "$ProgressPreference = 'SilentlyContinue'\nfunction claude { [Convert]::ToBase64String([IO.File]::ReadAllBytes($args[1])) }\n"
        encoded = base64.b64encode((probe + command).encode("utf-16-le")).decode("ascii")
        env = {k: v for k, v in os.environ.items() if k.lower() != "psmodulepath"}
        raw = subprocess.check_output([ps, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                                      env={**env, "FLYX_KEY": KEY}, timeout=20)
        settings = json.loads(base64.b64decode(raw.strip()))
    assert settings["model"] == settings["env"]["ANTHROPIC_MODEL"] == MODEL
    assert settings["env"]["ANTHROPIC_BASE_URL"] == ORIGIN
    assert settings["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "18000"
    assert settings["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "9000"  # half the context, at most 128,000
    assert settings["env"]["CLAUDE_CODE_GATEWAY_HINT_HEADERS"] == "1"
    token = settings["env"].get("ANTHROPIC_AUTH_TOKEN")
    assert token == ((None if shell == "sh" else KEY) if keyed else "flyx")


def guide_prose(panel):
    # Read the individual sentences around inline <code> paths. Commands and client names stay literal.
    words = panel.locator("#d-agents h2, #d-quick h2, #agent-guide p, #agent-guide summary").evaluate_all(r"""elements =>
        elements.flatMap(el => [...el.childNodes].filter(n => n.nodeType === Node.TEXT_NODE)
          .map(n => n.textContent.trim()).filter(s => /\p{L}/u.test(s)))""")
    words.extend(label.split(" · ")[-1] for label in panel.locator("#agent-guide a").all_text_contents())
    words.extend(panel.locator("#agent-guide [role=group]").evaluate_all(
        "groups => groups.map(el => el.getAttribute('aria-label'))"))
    return set(words)


def cache_prose(panel):
    flags = ("--sessions N", "--host-cache GB", "--kv-disk DIR[:GB]")
    prose = set()
    for flag in flags:
        row = panel.locator("#d-flags tr").filter(has=panel.page.get_by_text(flag, exact=True))
        prose.add(row.locator("td").nth(1).text_content().strip())
    assert len(prose) == len(flags)
    return prose


# The API reference, a sentence at a time (lib/i18n.js tx): each line, cell and note, with the prose outside its code
# (empty where there is none: a lone CJK character such as 或 is prose, so one letter counts).
REFERENCE = "#d-endpoints li, #d-endpoints p.lead, #d-fields td, #d-fields p.lead, #d-flags td, #d-flags p.lead, #d-telemetry td"


def reference_prose(panel):
    panel.locator("#d-endpoints details").evaluate_all("ds => ds.forEach(d => d.open = true)")
    return panel.locator(REFERENCE).evaluate_all(r"""elements => elements.map(el => {
        const copy = el.cloneNode(true);
        copy.querySelectorAll('code').forEach(c => c.remove());
        return [el.textContent.trim(), /\p{L}/u.test(copy.textContent) ? copy.textContent.trim() : ''];
    })""")


def anthropic_example(snippet, keyed):
    client = Mock()
    client.messages.create.return_value.content = [
        SimpleNamespace(type="thinking", thinking="Reasoning"), SimpleNamespace(type="text", text="Hello")]
    constructor = Mock(return_value=client)
    printed = []
    with patch.dict(sys.modules, anthropic=SimpleNamespace(Anthropic=constructor)), patch.dict(os.environ, FLYX_KEY=KEY):
        exec(snippet, {"print": printed.append})
    constructor.assert_called_once_with(base_url=ORIGIN, api_key=KEY if keyed else "flyx")
    assert client.messages.create.call_args.kwargs["model"] == MODEL
    assert printed == ["Hello"]


@cache
def powershell_example(snippet, keyed):
    shells = [path for name in ("powershell", "pwsh") if (path := shutil.which(name))]
    if not shells:
        return
    received = []

    class Probe(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append((self.path, self.headers.get("Authorization"), json.loads(body.decode("utf-8"))))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"choices":[{"message":{"content":"probe passed"}}]}')

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Probe) as server:
        service = Thread(target=server.serve_forever, daemon=True)
        service.start()
        try:
            assert ORIGIN + "/v1/chat/completions" in snippet
            command = "$ProgressPreference = 'SilentlyContinue'\n" + snippet.replace(ORIGIN, f"http://127.0.0.1:{server.server_port}")
            encoded = base64.b64encode(command.encode("utf-16-le")).decode("ascii")
            env = {k: v for k, v in os.environ.items() if k.lower() != "psmodulepath"}
            for shell in shells:
                reply = subprocess.check_output([shell, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                                                env={**env, "FLYX_KEY": KEY}, stderr=subprocess.PIPE, timeout=20)
                assert b"probe passed" in reply
                path, header, body = received.pop()
                assert path == "/v1/chat/completions" and header == ("Bearer " + KEY if keyed else None)
                assert body == {"model": MODEL, "messages": [{"role": "user", "content": "Hello"}],
                                "chat_template_kwargs": {"enable_thinking": False}}
        finally:
            server.shutdown()
            service.join()


def guides(panel, keyed):
    prose = set()
    for agent in ("claude", "codex", "deepseek", "opencode", "pi", "other"):
        panel.locator(f"button[data-agent={agent}]").click()
        prose.update(guide_prose(panel))
        snippets = panel.locator("#agent-guide pre").all_text_contents()
        assert snippets and all(KEY not in value for value in snippets)
        if agent == "claude":
            for shell in ("powershell", "sh"):
                panel.locator(f"button[data-shell={shell}]").click()
                prose.update(guide_prose(panel))
                command = panel.locator("#agent-guide pre").inner_text()
                assert "--settings" in command and KEY not in command
                claude_settings(command, shell, keyed)
        elif agent == "codex":
            cfg = tomllib.loads(snippets[0])
            provider = cfg["model_providers"][cfg["model_provider"]]
            assert cfg["model"] == MODEL and cfg["model_context_window"] == 18000
            assert provider["base_url"] == ORIGIN + "/v1" and provider["wire_api"] == "responses"
            assert provider.get("env_key") == ("FLYX_KEY" if keyed else None) and snippets[1] == "codex --profile flyx"
            assert snippets[2] == "codex --profile flyx -c windows.sandbox=mxc"
        elif agent == "deepseek":
            route, selection = yaml.safe_load(snippets[0])
            provider = route["config"]["providers"]["flyx"]
            assert provider["baseURL"] == ORIGIN + "/v1" and provider["api"] == "openai-completions"
            assert provider["apiKeyEnv"] == ("FLYX_KEY" if keyed else "FLYX_LOCAL_KEY")
            assert provider["models"][0]["id"] == selection["config"]["model"] == MODEL
        elif agent == "opencode":
            cfg = json.loads(snippets[0])
            provider = cfg["provider"]["flyx"]
            assert cfg["model"] == cfg["small_model"] == "flyx/" + MODEL
            assert provider["models"][MODEL]["limit"]["context"] == 18000 and provider["options"]["baseURL"] == ORIGIN + "/v1"
            assert provider.get("env") == (["FLYX_KEY"] if keyed else None)
            assert provider["options"].get("apiKey") == (None if keyed else "flyx")
        elif agent == "other":
            # The text any other agent is handed: where the server is and what it serves, the key by its variable.
            text = snippets[0]
            assert ORIGIN + "/v1  (" in text and "Model ID: " + MODEL + "\n" in text and "Context window: 18000 tokens" in text
            assert ("FLYX_KEY" in text) == keyed and ('use "flyx"' in text) == (not keyed)
            assert panel.locator("#agent-guide a").count() == 0
        else:
            provider = json.loads(snippets[0])["providers"]["flyx"]
            assert provider["models"][0]["id"] == MODEL and provider["baseUrl"] == ORIGIN + "/v1"
            assert provider["authHeader"] and provider["apiKey"] == ("$FLYX_KEY" if keyed else "flyx")
            assert shlex.split(snippets[1])[-1] == provider["models"][0]["name"]
    for sdk in ("powershell", "curl", "stream", "python", "anthropic"):
        panel.locator(f"button[data-snip={sdk}]").click()
        snippet = panel.locator("#sn-main").inner_text()
        if sdk in ("curl", "stream"):
            assert json.loads(snippet.split("<<'JSON'\n")[1].removesuffix("\nJSON"))["model"] == MODEL
        elif sdk == "powershell":
            powershell_example(snippet, keyed)
        elif sdk == "anthropic":
            anthropic_example(snippet, keyed)
        else:
            compile(snippet, "<SDK example>", "exec")
    return prose


def follow_links(page):
    def go(fragment, tab):
        page.evaluate("(hash) => { location.hash = hash }", fragment)
        page.wait_for_function("(tab) => document.documentElement.dataset.page === tab", arg=tab)

    go("#chat", "chat")
    go("#docs", "integration")
    page.evaluate("history.back()")
    page.wait_for_function("document.documentElement.dataset.page === 'chat'")
    page.evaluate("history.forward()")
    page.wait_for_function("document.documentElement.dataset.page === 'integration'")
    go("#requests:request%2Fone", "requests")
    page.locator('#req-rows tr.open[data-id="request/one"]').wait_for()
    go("#requests:two", "requests")
    page.locator('#req-rows tr.open[data-id="two"]').wait_for()
    go("#key=" + quote(KEY, safe=""), "requests")
    page.wait_for_function("(key) => localStorage.getItem('flyx.key') === key", arg=KEY)
    assert page.url.endswith("#requests:two")
    go("#integration&key=next%26key", "integration")
    assert page.evaluate("localStorage.getItem('flyx.key')") == "next&key" and page.url.endswith("#integration")
    go("#requests:%ZZ", "requests")
    assert page.locator("#req-rows tr.open").count() == 0 and page.url.endswith("#requests")
    go("", "overview")


def check_launch(browser, telemetry, errors):
    """The agents the server starts itself, for each it lists: the project folder first (the desktop's dialog, kept
    for next time), then the start, naming the agent and that folder and nothing else."""
    sent = []
    folder = r"C:\work\project"

    def launch(route):
        sent.append(route.request.post_data_json)
        route.fulfill(json={"launched": True, "dir": folder})

    page = browser.new_page()
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.route("**/dashboard/**", lambda r: serve(r, ROOT / "apps/flyx-server/dashboard"))
    page.route("**/v1/**", lambda r: r.fulfill(json={
        **telemetry, "server": {"api_key_required": False, "cors": {}, "agent_launch": ["claude", "codex"]}}
        if "/v1/telemetry" in r.request.url else {}))
    page.route("**/v1/agents/launch", launch)
    page.route("**/v1/hub/dir", lambda r: r.fulfill(json={"path": folder, "kind": "dir", "dirs": [], "models": []}))
    page.add_init_script("localStorage.setItem('flyx.lang', 'en')")
    page.goto(ORIGIN + "/dashboard/#integration")
    panel = page.locator("#tab-integration")
    dialog = page.locator("#project-dialog")
    for agent in ("claude", "codex"):
        panel.locator(f"button[data-agent={agent}]").click()
        panel.locator("#agent-launch-go").click()
        dialog.wait_for()
        if agent == "claude":
            assert dialog.locator("#pdlg-go").is_disabled()  # no folder yet
            dialog.locator("#pdlg-choose").click()
        # one folder: the one chosen, then the one used last
        dialog.locator("#pdlg-dir").wait_for()
        assert dialog.locator("#pdlg-dir").get_attribute("title") == folder
        dialog.locator("#pdlg-go").click()
        dialog.wait_for(state="detached")
        panel.locator("#agent-launch .launch-note.ok").wait_for()
        assert sent[-1] == {"agent": agent, "dir": folder}, sent
    for agent in ("deepseek", "opencode", "pi", "other"):
        panel.locator(f"button[data-agent={agent}]").click()
        assert panel.locator("#agent-launch").count() == 0, agent
    page.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--browser", default="chromium", choices=("msedge", "chromium"))
    args = p.parse_args()
    errors = []
    telemetry = {"schema_version": 1, "model": {"id": MODEL, "context": 18000}, "load": {"state": "ready"},
                 "state": {"phase": "idle"}, "memory": {"sessions": []}, "rates": {}, "recent": []}
    log = {"enabled": True, "requests": [{"id": id_, "api": "messages", "finished_at": 1, "content": "hello"}
                                         for id_ in ("request/one", "two")]}

    def stub_api(route):
        url = route.request.url
        route.fulfill(json=telemetry if "/v1/telemetry" in url else log if "/v1/requests" in url else {})

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel=args.browser if args.browser == "msedge" else None, args=["--disable-gpu"])
        english = {}
        for lang, keyed in product(LABELS, (False, True)):
            # the keyed server is also one started with --cors: the page says which pages a browser lets call the API
            telemetry["server"] = {"api_key_required": keyed,
                                   "cors": {"loopback": not keyed, "any": False, "origins": ["https://chat.example.com"] if keyed else []}}
            page = browser.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.route("**/dashboard/**", lambda r: serve(r, ROOT / "apps/flyx-server/dashboard"))
            page.route("**/v1/**", stub_api)
            # A server can restart on the same port without its old key. Browsers keep the saved value.
            saved = {"flyx.tab": "docs", "flyx.lang": lang, "flyx.key": KEY if keyed or lang == "en" else ""}
            page.add_init_script("for (const [k, v] of Object.entries(" + json.dumps(saved) + ")) localStorage.setItem(k, v)")
            fragment = {"en": "#integration", "zh": "#docs", "es": "#connect"}.get(lang, "")
            page.goto(ORIGIN + "/dashboard/" + fragment)
            panel = page.locator("#tab-integration")
            panel.locator("#d-agents [data-agent=claude]").wait_for()
            assert page.locator(".side button[data-tab=integration]").inner_text() == LABELS[lang]
            panel.locator("#agent-guide pre").wait_for()
            # which pages a browser lets call the API: the server's --cors, in the page's language
            cors = panel.locator("#d-cors").inner_text()
            assert ("https://chat.example.com" in cors) == keyed and "--cors" in cors, (lang, keyed, cors)
            if lang == "en":
                english[("cors", keyed)] = cors
            else:
                assert cors != english[("cors", keyed)], (lang, cors)
            # the reference: every line with prose outside its code reads differently from the English, its code kept
            reference = reference_prose(panel)
            if lang == "en":
                english[("reference", keyed)] = reference
            else:
                assert len(reference) == len(english[("reference", keyed)]), lang
                for (text, words), (en_text, en_words) in zip(reference, english[("reference", keyed)]):
                    if re.fullmatch(r"[a-z0-9_]+(, [a-z0-9_]+)+", en_words):
                        assert text == en_text, (lang, en_text, text)  # a list of telemetry fields, as they are named
                    elif re.fullmatch(r"[a-z]+", en_words):
                        pass  # the API column: an API's name (chat, messages) or "all", translated or not
                    elif en_words:
                        assert text != en_text and words != en_words, (lang, en_text)  # "-2 to 2" may be "-2～2"
                    else:
                        assert text == en_text, (lang, en_text, text)
            prose = guides(panel, keyed) | cache_prose(panel)
            if lang == "en" and not keyed:
                follow_links(page)
            page.route("**/v1/telemetry*", lambda r: r.fulfill(json={
                **telemetry, "model": {"id": None}, "load": {"state": "none"}}))
            page.goto(ORIGIN + "/dashboard/#integration")
            page.reload()
            panel.locator("#agent-guide p").first.wait_for()
            assert panel.locator("#agent-guide pre").count() == 0
            prose.update(guide_prose(panel))
            if lang == "en":
                english[keyed] = prose
            else:
                dictionary = page.evaluate("async lang => (await import('/dashboard/i18n/' + lang + '.js')).default.dict", lang)
                for original in english[keyed]:
                    translated = dictionary.get(original)
                    assert translated and translated in prose, (lang, keyed, original)
                    assert translated != original or original == "Shell", (lang, original)
                assert prose == {dictionary[s] for s in english[keyed]}, (lang, keyed, prose & english[keyed])
            page.close()
        check_launch(browser, telemetry, errors)
        browser.close()
    assert not errors, errors
    print("PASS: Integration navigation, localization and rendered client configurations")


if __name__ == "__main__":
    main()
