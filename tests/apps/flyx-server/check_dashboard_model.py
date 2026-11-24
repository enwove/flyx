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

    def __init__(self, page, tm, dialog=(200, LISTING), gpu_apps=None):
        self.tm, self.dialog, self.posted, self.deleted, self.dialogs = tm, dialog, [], [], []
        self.gpu_apps = copy.deepcopy(gpu_apps) if gpu_apps is not None else None
        self.closed = []
        page.route("**/dashboard/**", lambda r: serve(r, ROOT / "apps/flyx-server/dashboard"))
        page.route("**/v1/**", self.answer)

    def answer(self, route):
        url, method = route.request.url, route.request.method
        query = parse_qs(urlparse(url).query)
        if method == "POST" and url.endswith("/v1/load"):
            self.posted.append(route.request.post_data_json)
            return route.fulfill(json={})
        if method == "GET" and "/v1/device/processes" in url:
            kind = query.get("kind", ["vram"])[0]
            if self.gpu_apps is None:
                return route.fulfill(json={"kind": kind, "supported": True, "can_close": True, "mem_total": 16 * GB,
                                           "mem_used": 2 * GB, "mem_free": 14 * GB, "own_bytes": 0,
                                           "others_bytes": 2 * GB, "system_bytes": 0, "idle_igpu": None, "apps": []})
            data = self.gpu_apps.get(kind, self.gpu_apps) if "vram" in self.gpu_apps or "ram" in self.gpu_apps else self.gpu_apps
            return route.fulfill(json=data)
        if method == "POST" and "/v1/device/close" in url:
            body = route.request.post_data_json
            self.closed.append(body)
            app_id, force = body["id"], bool(body.get("force"))
            kind = body.get("kind", "vram")
            bucket = self.gpu_apps.get(kind, self.gpu_apps) if "vram" in self.gpu_apps or "ram" in self.gpu_apps else self.gpu_apps
            apps = bucket["apps"]
            target = next((a for a in apps if a["id"] == app_id), None)
            if not target:
                return route.fulfill(status=404, json={"error": {"message": "not found"}})
            if target.get("tray") and not force:
                return route.fulfill(json={"status": "still_running", "windows_signaled": 1, "name": target["name"],
                                           "snapshot": bucket})
            bucket["apps"] = [a for a in apps if a["id"] != app_id]
            bucket["mem_free"] += target["bytes"]
            bucket["others_bytes"] = max(0, bucket["others_bytes"] - target["bytes"])
            return route.fulfill(json={"status": "closed", "windows_signaled": 1, "name": target["name"],
                                       "snapshot": bucket})
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
        (gpu, "warn", "Limited context window 8,192 / 262,144. Not enough VRAM left "
                      "(3.60 GB used by other apps). Close other apps and reload for a longer context."),
        (dict(gpu, others=100_000_000, closable=False), "warn",
         "Limited context window 8,192 / 262,144. Not enough VRAM left. "
         "Disable speculative decoding or use a smaller quantization for a longer context."),
        ({"by": "host", "wanted": 262144, "short": False, "left": 7_300_000_000, "gpu_tokens": 4096,
          "others": None, "closable": False, "runtime": None}, "hint",
         "Limited context window 8,192 / 262,144. Not enough free RAM (7.30 GB available for KV)."),
        ({"by": "host", "wanted": 262144, "short": False, "left": 7_300_000_000, "gpu_tokens": 0,
          "others": None, "closable": False, "runtime": None}, "hint",
         "Limited context window 8,192 / 262,144. Not enough free RAM (7.30 GB available for KV)."),
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
        page.locator('#lset-tabs [data-stab="spec"]').click()
        expect(page.locator("#sd-saved input")).to_be_checked()
        expect(page.locator("#mf-reload")).to_have_count(0)
        page.locator('#lset-tabs [data-stab="general"]').click()
        page.locator("#mf-ctx").fill("4096")
        check_choice(spec, mode, paths)
    page.locator("#mf-reset").click()
    page.locator('#lset-tabs [data-stab="spec"]').click()
    page.locator("#sd-default").click()
    check_choice(True, "", [])
    page.locator("#sd-file").fill("./another.gguf")
    check_choice(True, "mtp", ["./another.gguf"])
    page.locator("#mf-reset").click()
    expect(page.locator("#sd-saved input")).to_be_checked()


def check_vision(page, server):
    """Vision encoder settings tab: custom mmproj, storage/precision/max-tokens, disable toggle."""
    page.locator('#lset-tabs [data-stab="vision"]').click()
    expect(page.locator("#vs-on")).to_be_checked()
    expect(page.locator("#vs-device")).to_have_value("disk")
    expect(page.locator("#vs-quant")).to_have_value("q8")
    page.locator("#vs-file").fill("./mmproj-test.gguf")
    page.locator("#vs-device").select_option("host")
    page.locator("#vs-quant").select_option("nvfp4")
    page.locator("#vs-maxtok").fill("1024")
    args = command(page)
    assert args[args.index("--mmproj") + 1] == "./mmproj-test.gguf", args
    assert args[args.index("--vision-device") + 1] == "host", args
    assert args[args.index("--vision-quant") + 1] == "nvfp4", args
    assert args[args.index("--vision-max-tokens") + 1] == "1024", args
    confirm(page, "#mf-reload", "Reload")
    body = server.posted[-1]
    assert (body["vision"], body["mmproj"], body["vision_device"], body["vision_quant"], body["vision_max_tokens"]) == (
        True, "./mmproj-test.gguf", "host", "nvfp4", 1024), body
    page.locator("#vs-on").uncheck(force=True)
    args = command(page)
    assert "--no-vision" in args and "--mmproj" not in args and "--vision-device" not in args, args
    page.locator("#mf-reset").click()
    expect(page.locator("#vs-on")).to_be_checked()


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


def check_vram_apps(open_page, want):
    """VRAM manager panel: auto-opens on OOM, safe two-step close (graceful -> force quit), protected & remote read-only."""
    snap = {
        "supported": True, "can_close": True, "device": "NVIDIA GeForce RTX 5070 Ti",
        "mem_total": int(17.09 * GB), "mem_used": int(4.12 * GB), "mem_free": int(12.97 * GB),
        "own_bytes": int(0.24 * GB), "others_bytes": int(3.88 * GB), "system_bytes": int(0.95 * GB),
        "idle_igpu": "AMD Radeon(TM) Graphics",
        "apps": [
            {"id": "antigrav", "name": "Antigravity", "process": "Antigravity", "bytes": int(0.45 * GB),
             "local_bytes": int(0.43 * GB), "processes": 2, "closable": False, "reason": "runs_flyx"},
            {"id": "discord", "name": "Discord", "process": "Discord", "bytes": int(0.26 * GB),
             "local_bytes": int(0.26 * GB), "processes": 1, "closable": True, "reason": None, "tray": True},
            {"id": "slack", "name": "Slack", "process": "Slack", "bytes": int(0.24 * GB),
             "local_bytes": int(0.20 * GB), "processes": 2, "closable": True, "reason": None},
        ],
    }
    tm = telemetry(None, remembered=want)
    tm["load"]["state"] = "failed"
    tm["load"]["target"] = want["model"]
    tm["load"]["error"] = (
        "Not enough VRAM: 3.88 / 17.09 GB used by other apps. "
        "Loading needs at least 13.24 GB (10.72 GB weights + 0.74 GB session + 1.78 GB runtime/reserve). "
        "Close other apps to free at least 0.20 GB VRAM, disable speculative decoding, or use a smaller quantization.")
    page, server = open_page(tm, gpu_apps=snap)
    panel = page.locator("#ld-vram-apps")
    expect(panel).to_be_visible()
    expect(panel).to_contain_text("Apps using VRAM")
    expect(panel).to_contain_text("Antigravity")
    expect(panel).to_contain_text("Runs FLYX")
    expect(panel).to_contain_text("Protected")
    expect(panel).to_contain_text("AMD Radeon(TM) Graphics")
    expect(panel.locator('.vapps-row[data-app="Antigravity"] button')).to_have_count(0)
    # Normal app closes on first click and highlights Retry loading once enough VRAM is freed.
    panel.locator('.vapps-row[data-app="Slack"] button').click()
    expect(panel.locator('.vapps-row[data-app="Slack"]')).to_have_count(0)
    expect(panel).to_contain_text("+240 MB freed")
    expect(panel.locator("#ld-vram-apps-retry")).to_have_class(re.compile(r"\bprimary\b"))
    assert server.closed[-1] == {"id": "slack", "force": False}
    # Tray-resident app transitions to Force quit and requires confirmation modal.
    panel.locator('.vapps-row[data-app="Discord"] button').click()
    expect(panel.locator('.vapps-row[data-app="Discord"]')).to_contain_text("Still running")
    expect(panel.locator('.vapps-row[data-app="Discord"] button')).to_have_text("Force quit")
    panel.locator('.vapps-row[data-app="Discord"] button').click()
    page.locator(".modal").get_by_role("button", name="Force quit").click()
    expect(panel.locator('.vapps-row[data-app="Discord"]')).to_have_count(0)
    assert server.closed[-1] == {"id": "discord", "force": True}
    # Retry loading posts the failed target model.
    with page.expect_request(lambda r: r.method == "POST" and r.url.endswith("/v1/load")):
        panel.locator("#ld-vram-apps-retry").click()
    assert server.posted[-1]["model"] == want["model"]
    page.close()

    # Remote viewer (can_close=False) sees apps read-only without close buttons.
    remote_snap = copy.deepcopy(snap)
    remote_snap["can_close"] = False
    page, _ = open_page(tm, gpu_apps=remote_snap)
    expect(page.locator("#ld-vram-apps .vapps-row button")).to_have_count(0)
    page.close()

    # Short-context GPU banner has a Manage VRAM toggle; sidebar VRAM & RAM rows open their matching modals.
    ram_snap = {
        "kind": "ram", "supported": True, "can_close": True, "device": "System RAM",
        "mem_total": int(34 * GB), "mem_used": int(24 * GB), "mem_free": int(10 * GB),
        "own_bytes": int(1 * GB), "others_bytes": int(23 * GB), "system_bytes": int(2 * GB),
        "idle_igpu": None,
        "apps": [{"id": "browser", "name": "Firefox", "process": "firefox", "bytes": int(3.2 * GB),
                  "local_bytes": int(3.2 * GB), "processes": 8, "closable": True, "reason": None}],
    }
    tm_ctx = telemetry(want)
    tm_ctx["system"] = {"ram_total": int(34 * GB), "ram_available": int(10 * GB), "cpus": 16}
    tm_ctx["model"]["context_limit"] = {"by": "gpu", "wanted": 262144, "short": True, "left": 830_000_000,
                                        "gpu_tokens": 0, "others": 3_600_000_000, "closable": True, "runtime": 500_000_000}
    page, server = open_page(tm_ctx, gpu_apps={"vram": snap, "ram": ram_snap})
    expect(page.locator("#ld-vram-apps")).to_have_count(0)
    page.locator("#ld-vram-toggle").click()
    expect(page.locator("#ld-vram-apps")).to_be_visible()
    # Persistent sidebar VRAM & RAM rows open their matching modals; Escape closes cleanly.
    page.locator("#r-vram-row").click()
    expect(page.locator("#vram-modal-apps")).to_have_attribute("data-kind", "vram")
    expect(page.locator("#vram-modal-apps")).to_contain_text("Discord")
    page.keyboard.press("Escape")
    expect(page.locator("#vram-modal")).to_have_count(0)
    page.locator("#r-ram-row").click()
    expect(page.locator("#vram-modal-apps")).to_have_attribute("data-kind", "ram")
    expect(page.locator("#vram-modal-apps")).to_contain_text("Apps using RAM")
    expect(page.locator("#vram-modal-apps")).to_contain_text("Firefox")
    page.locator('#vram-modal-apps .vapps-row[data-app="firefox"] button').click()
    assert server.closed[-1] == {"id": "browser", "force": False, "kind": "ram"}
    page.keyboard.press("Escape")
    # Global short-context warning banner on Overview offers Manage VRAM directly.
    page.locator('.side nav .navi[data-tab="overview"]').click()
    hint = page.locator('#hint[data-hint="short"]')
    expect(hint).to_be_visible()
    hint.get_by_role("button", name="Manage VRAM").click()
    expect(page.locator("#vram-modal-apps")).to_have_attribute("data-kind", "vram")
    page.close()

    # Host-RAM-limited context banner offers Manage RAM on both Model card and global Hint banner.
    tm_ram = telemetry(want)
    tm_ram["system"] = {"ram_total": int(34 * GB), "ram_available": int(10 * GB), "cpus": 16}
    tm_ram["model"]["context_limit"] = {"by": "host", "wanted": 262144, "short": True, "left": 4_200_000_000,
                                        "gpu_tokens": 0, "others": None, "closable": False, "runtime": None}
    page, _ = open_page(tm_ram, gpu_apps={"vram": snap, "ram": ram_snap})
    expect(page.locator("#ld-vram-toggle")).to_have_text("Manage RAM")
    page.locator("#ld-vram-toggle").click()
    expect(page.locator("#ld-vram-apps")).to_have_attribute("data-kind", "ram")
    expect(page.locator("#ld-vram-apps")).to_contain_text("Firefox")
    page.locator('.side nav .navi[data-tab="overview"]').click()
    hint = page.locator('#hint[data-hint="short"]')
    hint.get_by_role("button", name="Manage RAM").click()
    expect(page.locator("#vram-modal-apps")).to_have_attribute("data-kind", "ram")
    page.close()


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
    check_vision(page, server)
    page.close()
    check_context_limit(open_page, want)
    check_vram_apps(open_page, want)
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
