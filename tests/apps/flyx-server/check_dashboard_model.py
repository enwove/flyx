# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Check the Model page with Playwright against a stubbed server: the model served with its load settings (what a
reload posts, the command line as real shells read it), a context memory cut short and why, the library (Hugging
Face; this machine's folders with their models), a server with nothing loaded, and the Overview's way to the library.
Use --browser msedge for Edge."""
import argparse
import base64
import copy
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import parse_qs, unquote, urlparse

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from dashboard_sweep import serve

ORIGIN = "http://[::1]:18479"
GB = 10**9
REPO = "acme/Tiny-27B-GGUF"
CACHE = "C:\\Users\\me\\.cache\\huggingface\\hub"
FOLDER = "D:\\models"
PICKED = "E:\\gguf"


def native_args(command, shell, tmp):
    # Let Python run the local "serve" fixture using the target shell's native argument passing.
    # A PowerShell function would see different arguments from a Windows executable.
    result = tmp / "args.json"
    result.unlink(missing_ok=True)
    if shell in ("powershell", "pwsh"):
        interpreter = "'" + sys.executable.replace("'", "''") + "'"
        if command.startswith("Start-Process "):
            command = command.replace("Start-Process flyx ", "Start-Process -FilePath " + interpreter + " ", 1)
        else:
            assert command.startswith("flyx serve ")
            command = "& " + interpreter + command[len("flyx"):]
        encoded = base64.b64encode(("$ProgressPreference = 'SilentlyContinue'\n" + command).encode("utf-16-le")).decode("ascii")
        argv = [shell, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded]
    else:
        assert command.startswith("flyx serve ")
        argv = [shell, "-c", shlex.quote(sys.executable) + command[len("flyx"):]]
    subprocess.run(argv, cwd=tmp, check=True, capture_output=True, timeout=20)
    return json.loads(result.read_text(encoding="utf8"))


def hub_model(repo, quant, size, mtp=None, current=True):
    file = repo.split("/")[1].replace("-GGUF", "") + "-" + quant + ".gguf"
    return {"name": repo + ":" + quant, "kind": "gguf", "source": "hub", "repo": repo, "quant": quant, "file": file,
            "path": CACHE + "\\" + file, "size": size, "modified": 1.7e9, "revision": "abc123", "current": current, "mtp": mtp}


def dir_model(folder, file, size, kind="gguf", quant=None):
    return {"name": folder + "\\" + file, "kind": kind, "source": "dir", "repo": None, "quant": quant, "file": file,
            "path": folder + "\\" + file, "size": size, "modified": 1.7e9, "revision": None, "current": True, "mtp": None}


CATALOG = {"cache": CACHE, "cache_from": "default", "dirs": [FOLDER], "models": [
    hub_model(REPO, "Q4_K_M", int(17.6 * GB), "in-file"), hub_model(REPO, "IQ3_XXS", int(11.8 * GB), "in-repo"),
    hub_model("acme/Tiny-4B-GGUF", "Q8_0", int(4.5 * GB), None, False),
    dir_model(FOLDER, "tiny-bf16.flyx", int(0.4 * GB), "flyx"), dir_model(FOLDER, "Other-12B-Q5_K_M.gguf", int(8.9 * GB), quant="Q5_K_M"),
    {"name": "acme/Tiny-27B-DFlash2", "kind": "drafter", "source": "hub", "repo": "acme/Tiny-27B-DFlash2", "size": int(1.9 * GB)}]}
FOUND = {"query": "Qwen3", "family": True, "endpoint": "https://huggingface.co", "models": [
    {"id": REPO, "downloads": 7383193, "likes": 6300, "updated": "2026-11-12T02:15:52.000Z", "created": None, "pipeline": None,
     "gated": False, "cached": True},
    {"id": "acme/Other-9B-GGUF", "downloads": 1200, "likes": 14, "updated": "2026-02-20T02:15:52.000Z", "created": None,
     "pipeline": "text-generation", "gated": True, "cached": False}]}


def repository(repo):
    """GET /v1/hub/files: REPO as this build runs it, any other of an architecture it does not."""
    runs = repo == REPO
    quants = [("IQ3_XXS", 11.8, True), ("Q4_K_M", 17.6, True), ("Q5_K_M", 19.4, False), ("Q6_K", 22.3, False), ("BF16", 54.7, False)]
    return {"repo": repo, "from_cache": False, "drafter": None, "base_models": ["acme/Tiny-27B"], "runs": runs,
            "known_drafters": ["acme/Tiny-27B-DFlash2"] if runs else [], "build_architectures": ["qwen35", "qwen3"],
            "mtp": {"path": "MTP/mtp.gguf", "size": int(0.6 * GB), "cached": True} if runs else None,
            "info": {"downloads": 7383193, "likes": 6300, "updated": "2026-11-12T02:15:52.000Z", "license": "apache-2.0", "pipeline": None,
                     "gated": False, "architecture": "qwen35" if runs else "gemma4", "context": 262144, "parameters": 27320697856},
            "ggufs": [{"name": repo + ":" + q, "quant": q, "size": int(s * GB), "split": False, "parts": 1, "complete": True, "cached": c and runs}
                      for q, s, c in quants]}


LISTING = {"path": PICKED, "parent": "E:\\", "kind": "dir", "dirs": [],
           "models": [dir_model(PICKED, "Added-9B-Q4_K_M.gguf", int(5.6 * GB), quant="Q4_K_M")]}
FOLDERS = {"": {"path": "", "parent": None, "kind": "start", "dirs": [{"name": "D:\\", "path": "D:\\"}], "models": []},
           "D:\\": {"path": "D:\\", "parent": None, "kind": "dir", "dirs": [{"name": "models", "path": FOLDER}], "models": []},
           FOLDER: {"path": FOLDER, "parent": "D:\\", "kind": "dir", "dirs": [], "models": [m for m in CATALOG["models"] if m["source"] == "dir"]},
           PICKED: LISTING}


def telemetry(want, remembered=None):
    """What the server says of itself: serving `want`, or nothing (then `remembered` is the load it would repeat)."""
    load = {"state": "ready", "loaded": want, "remembered": want} if want else {"state": "none"}
    if not want and remembered:
        load["remembered"] = remembered
    return {"schema_version": 1, "sampled_at": 1000.0, "state": {"phase": "idle"}, "rates": {}, "recent": [], "load": load,
            "model": {"id": "local", "loaded": True, "context": 8192, "max_sessions": 8} if want else {"id": None, "loaded": False},
            "device": {"name": "GPU", "mem_total": 32 * 2**30, "mem_used": 2**30}, "memory": {"sessions": []}}


class Server:
    """The stubbed server of one page: what it was asked to load, to delete and to show a folder dialog for."""

    def __init__(self, page, tm, dialog=(200, LISTING)):
        self.tm, self.dialog, self.posted, self.deleted, self.dialogs = tm, dialog, [], [], []
        page.route("**/dashboard/**", lambda r: serve(r, ROOT / "apps/flyx-server/dashboard"))
        page.route("**/v1/**", self.answer)

    def answer(self, route):
        url, method = route.request.url, route.request.method
        query = parse_qs(urlparse(url).query)
        if method == "POST" and url.endswith("/v1/load"):
            self.posted.append(route.request.post_data_json)
            return route.fulfill(json={})
        if method == "POST" and "/v1/hub/dir" in url:
            self.dialogs.append(route.request.post_data_json)
            return route.fulfill(status=self.dialog[0], json=self.dialog[1])
        if method == "DELETE":
            self.deleted.append(unquote(url.split("name=", 1)[1]))
            return route.fulfill(json={"deleted": self.deleted[-1], "freed": 1})
        if "/v1/telemetry" in url:
            return route.fulfill(json=self.tm)
        if "/v1/hub/cached" in url:
            return route.fulfill(json=CATALOG)
        if "/v1/hub/search" in url:
            return route.fulfill(json=FOUND)
        if "/v1/hub/files" in url:
            return route.fulfill(json=repository(query["repo"][0]))
        if "/v1/hub/dir" in url:
            path = query.get("path", [""])[0]
            return route.fulfill(json=FOLDERS[path]) if path in FOLDERS else route.fulfill(
                status=404, json={"error": {"message": path + ": no such folder or file on this machine"}})
        route.fulfill(json={})


def view(page, name):
    page.locator(f'#mv-tabs [data-view="{name}"]').click()
    expect(page.locator("#mv-" + name)).to_be_visible()


def command(page):
    page.locator('#mf-settings [data-shell="sh"]').click()
    return shlex.split(page.locator("#mf-cmd").inner_text().replace("\\\n", ""))


def confirm(page, button, action):
    """A load asked for and confirmed: it is posted, and the page shows the model's view, where its progress is."""
    page.locator(button).click()
    with page.expect_request(lambda r: r.method == "POST" and r.url.endswith("/v1/load")):
        page.locator(".modal").get_by_role("button", name=action).click()
    expect(page.locator("#mv-current")).to_be_visible()


def check_settings(page, server, want):
    """The model served: its settings as loaded, a change against them, what a reload posts."""
    expect(page.locator("#mv-current")).to_be_visible()
    expect(page.locator("#mv-state")).to_have_text("local")
    expect(page.locator("#ld-ctx")).to_be_hidden()  # the context it asked for: nothing to explain
    field, cmd = page.locator("#mf-kvgpu"), page.locator("#mf-cmd")
    expect(field).to_have_value("0")
    expect(cmd).to_contain_text("--kv-gpu 0")
    expect(page.locator("#mf-reload")).to_have_count(0)
    for value in ("1024", "", "0"):
        field.fill(value)
        # Change a second setting so we can resubmit the original placement as well.
        page.locator("#mf-name").fill("selected")
        if value:
            expect(cmd).to_contain_text("--kv-gpu " + value)
        else:
            expect(cmd).not_to_contain_text("--kv-gpu")
        if value != "0":
            expect(page.locator('[data-k="kv_gpu"] .was')).to_contain_text("0")
        confirm(page, "#mf-reload", "Reload")
        assert server.posted[-1]["kv_gpu"] == (int(value) if value else None) and server.posted[-1]["model"] == want["model"], server.posted[-1]
    page.locator("#mf-reset").click()
    expect(page.locator("#mf-name")).to_have_value("")
    expect(page.locator("#mf-reload")).to_have_count(0)


def check_context_limit(open_page, want):
    """A context memory cut under the model's own: the status card says where K/V lives, which memory cut it and what
    would give more; a short one as a warning, on the Overview too."""
    gpu = {"by": "gpu", "wanted": 262144, "short": True, "left": 820_000_000, "gpu_tokens": 8192,
           "others": 3_600_000_000, "closable": True, "runtime": 540_000_000}
    cases = [
        (gpu, "warn", "Context 8,192 / 262,144 (all KV in VRAM). 0.82 GB VRAM left after weights "
                      "(3.60 GB used by other apps). Close other apps and reload for a longer context."),
        (dict(gpu, others=100_000_000, closable=False), "warn",
         "Disable speculative decoding or use a smaller quantization for more VRAM."),
        ({"by": "host", "wanted": 262144, "short": False, "left": 7_300_000_000, "gpu_tokens": 4096,
          "others": None, "closable": False, "runtime": None}, "hint",
         "Context 8,192 / 262,144 (KV: 4,096 tokens in VRAM, 4,096 in RAM). RAM KV budget is 7.30 GB."),
        ({"by": "host", "wanted": 262144, "short": False, "left": 7_300_000_000, "gpu_tokens": 0,
          "others": None, "closable": False, "runtime": None}, "hint",
         "Context 8,192 / 262,144 (all KV in RAM). RAM KV budget is 7.30 GB."),
    ]
    for limit, tone, text in cases:
        tm = telemetry(want)
        tm["model"]["context_limit"] = limit
        page, _ = open_page(tm)
        note = page.locator("#ld-ctx")
        expect(note).to_be_visible()
        expect(note).to_contain_text(text)
        expect(note).to_have_class(re.compile(r"\b" + tone + r"\b"))
        page.locator('.side nav .navi[data-tab="overview"]').click()
        expect(page.locator("#engine-mem")).to_be_visible()
        expect(page.locator('#hint[data-hint="short"]')).to_have_count(1 if limit["short"] else 0)
        page.close()


def check_commands(page, want):
    """The command line shown, as each of this system's shells hands it to a program."""
    shells = ("powershell", "pwsh") if os.name == "nt" else ("bash", "zsh")
    with tempfile.TemporaryDirectory() as dirname:
        tmp = Path(dirname)
        (tmp / "serve").write_text(
            'import json, sys\nfrom pathlib import Path\n'
            'Path("args.json").write_text(json.dumps(sys.argv[1:]), encoding="utf8")\n', encoding="utf8")
        for name in ("Qwen ‘local’", 'Local "27B"', "A path ending in a slash\\"):
            page.locator("#mf-name").fill(name)
            expected = [want["model"], "--ctx", "8192", "--kv", "fp8", "--kv-ram", "4", "--kv-gpu", "0", "--name", name, "--port", "18479"]
            assert command(page) == ["flyx", "serve", *expected]
            for shell in shells:
                if shutil.which(shell):
                    page.locator('#mf-settings [data-shell="powershell"]' if os.name == "nt" else '#mf-settings [data-shell="sh"]').click()
                    assert native_args(page.locator("#mf-cmd").inner_text(), shell, tmp) == expected
    page.locator("#mf-reset").click()


def check_drafters(page, server, want):
    """Drafting as it was loaded (also what the one control cannot say), the default, a drafter of one's own."""
    files = ["./drafts/DFlash model.flyx", "./drafts/MTP layer.gguf"]
    want.update(width=4, draft_trees=True, lookup=False)

    def check_choice(spec, mode, paths):
        args = command(page)
        assert ("--no-spec" in args) == (not spec), args
        assert args[args.index("--width") + 1] == "4" and "--draft-trees" in args and "--no-lookup" in args, args
        assert [args[i + 1] for i, arg in enumerate(args) if arg == "--drafter"] == ([mode] if mode else []), args
        assert [args[i + 1] for i, arg in enumerate(args) if arg == "--drafter-file"] == paths, args
        confirm(page, "#mf-reload", "Reload")
        body = server.posted[-1]
        assert (body["speculative"], body["drafter"], body["drafters"]) == (spec, mode, paths), body

    for spec, mode, paths in [(True, "none", []), (True, "dflash2", []), (False, "all", files), (True, "all", files)]:
        want.update(speculative=spec, drafter=mode, drafters=paths)
        page.reload()
        expect(page.locator("#sd-saved input")).to_be_checked()
        expect(page.locator("#mf-reload")).to_have_count(0)
        page.locator("#mf-ctx").fill("4096")
        check_choice(spec, mode, paths)
    page.locator("#mf-reset").click()
    page.locator("#sd-default").click()
    check_choice(True, "", [])
    page.locator("#sd-file").fill("./another.gguf")
    check_choice(True, "mtp", ["./another.gguf"])
    page.locator("#mf-reset").click()
    expect(page.locator("#sd-saved input")).to_be_checked()


def check_hub(page, server, want):
    """The library's Hugging Face side: repositories found, one shown with its files, a file loaded."""
    view(page, "library")
    expect(page.locator("#hf-list .prow")).to_have_count(2)
    expect(page.locator("#hf-family")).to_contain_text("Qwen3")
    expect(page.locator("#hf-name")).to_have_text("Tiny-27B-GGUF")  # the first repository found is shown
    expect(page.locator("#mf-files .mrow")).to_have_count(5)
    expect(page.locator("#mf-load")).to_be_disabled()
    page.locator('#mf-files .mrow[data-name$=":Q5_K_M"]').click()
    expect(page.locator("#mf-load")).to_have_text("Switch to this model")
    expect(page.locator("#mf-chosen-s")).to_contain_text("19.4 GB to download first")
    confirm(page, "#mf-load", "Switch")
    body = server.posted[-1]  # another model, with the settings of the one served
    assert (body["model"], body["ctx"], body["kv"], body["kv_gpu"]) == (REPO + ":Q5_K_M", want["ctx"], want["kv"], want["kv_gpu"]), body
    # A repository of an architecture this build does not run is not loaded.
    view(page, "library")
    page.locator('#hf-list .prow[data-repo="acme/Other-9B-GGUF"]').click()
    expect(page.locator("#hf-unsupported")).to_contain_text("gemma4")
    page.locator('#mf-files .mrow[data-name$=":Q5_K_M"]').click()
    expect(page.locator("#mf-load")).to_be_disabled()
    # A name with its quantization, in the search box: that file of that repository.
    page.locator("#hf-q").fill(REPO + ":q6_k")
    page.keyboard.press("Enter")
    expect(page.locator("#mf-files .mrow.sel")).to_have_attribute("data-name", REPO + ":Q6_K")
    expect(page.locator("#hf-drafters")).to_contain_text("acme/Tiny-27B-DFlash2")
    page.locator("#hf-drafters .btn").click()  # the drafter published for it goes to the settings
    expect(page.locator("#mv-current")).to_be_visible()
    expect(page.locator("#sd-file")).to_have_value("acme/Tiny-27B-DFlash2")
    page.locator("#mf-reset").click()


def check_local(page, server):
    """The library's other side: folders, the models in each under it, one loaded, folders added and removed."""
    selector = lambda path: path.replace("\\", "\\\\")
    view(page, "library")
    page.locator('#src-seg [data-src="local"]').click()
    folders, models = page.locator("#lc-list .prow.folder"), page.locator("#lc-list .prow.model")
    expect(folders).to_have_count(3)  # where Hugging Face downloads to, the model directory, the served model's folder
    expect(folders.nth(0).locator(".ps")).to_have_text(CACHE)
    expect(folders.nth(1).locator(".ps")).to_have_text(FOLDER)
    expect(models).to_have_count(6)
    one = page.locator('#lc-list .prow.model[data-model="' + REPO + ':IQ3_XXS"]')
    expect(one.locator(".pn")).to_have_text("Tiny-27B-GGUF · IQ3_XXS")
    expect(one.locator(".pm")).to_have_text("11.8 GB")
    expect(one.locator(".pt")).to_have_text("GGUF")
    expect(page.locator('#lc-list .prow.model[data-model$="tiny-bf16.flyx"] .pt')).to_have_text(".flyx")
    expect(page.locator('#lc-list .prow.model[data-model="acme/Tiny-27B-DFlash2"] .pt')).to_have_text("drafter")
    one.click()
    expect(page.locator("#lc-name")).to_have_text("Tiny-27B-GGUF · IQ3_XXS")
    expect(page.locator("#lc-path")).to_contain_text("Tiny-27B-IQ3_XXS.gguf")
    confirm(page, "#mf-load", "Switch")
    assert server.posted[-1]["model"] == REPO + ":IQ3_XXS", server.posted[-1]
    view(page, "library")
    page.locator('#lc-list .prow.model[data-model$="tiny-bf16.flyx"]').click()
    confirm(page, "#mf-load", "Switch")
    assert server.posted[-1]["model"] == FOLDER + "\\tiny-bf16.flyx", server.posted[-1]
    view(page, "library")
    page.locator("#mf-q").fill("12b")
    expect(models).to_have_count(1)
    page.locator("#mf-q").fill("")
    # A drafter is not loaded; a folder's own row shows the folder: where Hugging Face downloads to, by repository.
    page.locator('#lc-list .prow.model[data-model="acme/Tiny-27B-DFlash2"]').click()
    expect(page.locator("#mf-load")).to_be_disabled()
    folders.nth(0).click()
    expect(page.locator("#mf-path")).to_have_value(CACHE)
    expect(page.locator("#mf-cached .rgroup")).to_have_count(3)
    # A folder added through the system's dialog (the server shows it), kept by the browser, then removed.
    page.locator("#mf-add").click()
    added = page.locator('#lc-list .prow.folder[data-place="dir:' + selector(PICKED) + '"]')
    expect(added).to_have_class("prow act folder sel")
    assert server.dialogs[-1]["title"] == "Choose a folder of model files", server.dialogs
    mine = page.locator('#lc-list .prow.model[data-model$="Added-9B-Q4_K_M.gguf"]')
    expect(mine.locator(".pm")).to_have_text("5.6 GB")
    page.reload()
    view(page, "library")
    page.locator('#src-seg [data-src="local"]').click()
    expect(mine).to_have_count(1)
    added.hover()
    added.locator("button.del").click()
    expect(added).to_have_count(0)
    expect(mine).to_have_count(0)
    expect(page.locator('#lc-list .prow.folder[data-place="dir:' + selector(FOLDER) + '"] button.del')).to_have_count(0)
    # Downloaded models are deleted from their rows; a repository's, all at once, from the folder's view.
    row = page.locator('#lc-list .prow.model[data-model="acme/Tiny-4B-GGUF:Q8_0"]')
    row.hover()
    row.locator("button.del").click()
    page.locator(".modal").get_by_role("button", name="Delete").click()
    expect(page.locator(".modal")).to_have_count(0)
    folders.nth(0).click()
    page.locator('.rgroup[data-repo="' + REPO + '"] .rgroup-h .del').click()
    expect(page.locator(".modal")).to_contain_text("2 files")
    page.locator(".modal").get_by_role("button", name="Delete").click()
    expect(page.locator(".modal")).to_have_count(0)
    page.wait_for_function("() => true")
    page.wait_for_timeout(600)  # its files go one after another
    assert server.deleted == ["acme/Tiny-4B-GGUF:Q8_0", REPO + ":Q4_K_M", REPO + ":IQ3_XXS"], server.deleted
    # Where no dialog can be shown, the folders are browsed in the page.
    server.dialog = (501, {"error": {"message": "no desktop to show a folder dialog on: no display"}})
    page.locator("#mf-add").click()
    expect(page.locator("#mf-dirmsg")).to_contain_text("cannot be shown from here")
    page.locator('#mf-dirs .chip').first.click()
    expect(page.locator("#mf-path")).to_have_value("D:\\")
    page.locator("#mf-keep").click()
    expect(page.locator('#lc-list .prow.folder[data-place="dir:D:\\\\"]')).to_have_count(1)
    page.locator("#mf-keep").click()
    expect(page.locator('#lc-list .prow.folder[data-place="dir:D:\\\\"]')).to_have_count(0)


def check_nothing_served(page, server, remembered):
    """No model: the library first; no settings to show, so a load takes the defaults; the last load as it was."""
    expect(page.locator("#mv-library")).to_be_visible()
    expect(page.locator("#mv-state")).to_have_count(0)
    page.locator('#mf-files .mrow[data-name$=":Q5_K_M"]').click()
    expect(page.locator("#mf-load")).to_have_text("Load model")
    expect(page.locator("#mf-tosettings")).to_have_count(0)
    expect(page.locator("#hf-drafters .btn")).to_have_count(0)
    with page.expect_request(lambda r: r.method == "POST" and r.url.endswith("/v1/load")):
        page.locator("#mf-load").click()
    expect(page.locator("#mv-current")).to_be_visible()
    body = server.posted[-1]
    # kv "auto": the engine's default, not the server's --kv (null would keep that)
    assert (body["model"], body["ctx"], body["kv"], body["kv_ram"], body["kv_gpu"], body["speculative"], body["drafter"], body["drafters"]) == \
        (REPO + ":Q5_K_M", None, "auto", "auto", None, True, "", []), body
    expect(page.locator("#mf-settings")).to_have_count(0)
    expect(page.locator("#ld-unload")).to_be_hidden()
    page.locator("#ld-browse").click()
    expect(page.locator("#mv-library")).to_be_visible()
    view(page, "current")
    with page.expect_request(lambda r: r.method == "POST" and r.url.endswith("/v1/load")):
        page.locator("#ld-again").click()
    assert server.posted[-1] == remembered, server.posted[-1]
    # The Overview says so too, with the way to the library, wherever the Model page was left.
    page.locator('.side nav .navi[data-tab="overview"]').click()
    expect(page.locator("#act-what")).to_contain_text("No model loaded")
    expect(page).to_have_title("No model loaded · FLYX")
    page.locator("#act-explore").click()
    expect(page.locator("#tab-model")).to_be_visible()
    expect(page.locator("#mv-library")).to_be_visible()


def check(browser):
    want = {"model": "./models/Local [27B] & O'Brien.gguf", "ctx": 8192, "kv": "fp8", "kv_gpu": 0, "kv_ram": 4}
    errors = []

    def open_page(tm, **dialog):
        page = browser.new_page(locale="en-US")
        page.on("pageerror", lambda error: errors.append(str(error)))
        server = Server(page, tm, **dialog)
        page.goto(ORIGIN + "/dashboard/#model")
        page.locator("#mv-tabs").wait_for()
        return page, server

    page, server = open_page(telemetry(want))
    check_settings(page, server, want)
    check_commands(page, want)
    check_drafters(page, server, want)
    page.close()
    check_context_limit(open_page, want)
    plain = dict(want, model="models/local.gguf")
    page, server = open_page(telemetry(plain))
    check_hub(page, server, plain)
    check_local(page, server)
    page.locator('.side nav .navi[data-tab="overview"]').click()
    expect(page.locator("#act-explore")).to_have_count(0)  # a model is served: nothing to send anyone to the library for
    page.close()
    remembered = copy.deepcopy(plain)
    page, server = open_page(telemetry(None, remembered))
    check_nothing_served(page, server, remembered)
    page.close()
    assert not errors, errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", choices=("chromium", "msedge"), default="chromium")
    args = parser.parse_args()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel=args.browser, args=["--disable-gpu"])
        try:
            check(browser)
        finally:
            browser.close()
    print("The Model page kept its load settings and shell arguments, and loaded from Hugging Face and from this machine's folders.")
