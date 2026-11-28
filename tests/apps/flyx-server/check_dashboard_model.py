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


def hub_model(repo, quant, size, mtp=None, current=True, mmproj=None):
    file = repo.split("/")[1].replace("-GGUF", "") + "-" + quant + ".gguf"
    return {"name": repo + ":" + quant, "kind": "gguf", "source": "hub", "repo": repo, "quant": quant, "file": file,
            "path": CACHE + "\\" + file, "size": size, "modified": 1.7e9, "revision": "abc123", "current": current,
            "mtp": mtp, "mmproj": mmproj}


def dir_model(folder, file, size, kind="gguf", quant=None):
    return {"name": folder + "\\" + file, "kind": kind, "source": "dir", "repo": None, "quant": quant, "file": file,
            "path": folder + "\\" + file, "size": size, "modified": 1.7e9, "revision": None, "current": True, "mtp": None}


LOCAL_DRAFTER = dict(dir_model(FOLDER, "helper.gguf", int(0.3 * GB), kind="drafter"), drafter_kind="dflash2")
CATALOG = {"cache": CACHE, "cache_from": "default", "dirs": [FOLDER], "models": [
    hub_model(REPO, "Q4_K_M", int(17.6 * GB), "in-file", mmproj="in-repo"),
    hub_model(REPO, "IQ3_XXS", int(11.8 * GB), "in-repo", mmproj="in-repo"),
    hub_model("acme/Tiny-4B-GGUF", "Q8_0", int(4.5 * GB), None, False),
    dir_model(FOLDER, "tiny-bf16.flyx", int(0.4 * GB), "flyx"),
    dir_model(FOLDER, "Other-12B-Q5_K_M.gguf", int(8.9 * GB), quant="Q5_K_M"),
    LOCAL_DRAFTER,
    dir_model(FOLDER, "mmproj-BF16.gguf", int(0.9 * GB), kind="mmproj"),
    {"name": "acme/Tiny-27B-DFlash2", "kind": "drafter", "drafter_kind": "dflash2", "source": "hub", "repo": "acme/Tiny-27B-DFlash2", "size": int(1.9 * GB)},
    {"name": "acme/Model-DFlash2-GGUF:BF16", "kind": "drafter", "drafter_kind": "dflash2", "source": "hub",
     "repo": "acme/Model-DFlash2-GGUF", "quant": "BF16", "file": "Model-DFlash2-Draft-BF16.gguf",
     "path": CACHE + "\\Model-DFlash2-Draft-BF16.gguf", "size": int(1.0 * GB), "modified": 1.7e9,
     "revision": "def456", "current": True, "mtp": None, "mmproj": None}]}
FOUND = {"query": "", "family": True, "architectures": ["qwen35", "qwen3", "inkling2_loop"], "arch": None,
         "endpoint": "https://huggingface.co", "next_cursor": "page2-tok", "models": [
    {"id": REPO, "downloads": 7383193, "likes": 6300, "updated": "2026-11-12T02:15:52.000Z", "created": None, "pipeline": None,
     "gated": False, "cached": True, "architecture": "qwen35", "runs": True},
    {"id": "acme/Inkling-2-Mini-GGUF", "downloads": 1900000, "likes": 1200, "updated": "2026-11-20T02:15:52.000Z", "created": None,
     "pipeline": "text-generation", "gated": False, "cached": False, "architecture": "inkling2_loop", "runs": True},
    {"id": "acme/Other-9B-GGUF", "downloads": 1200, "likes": 14, "updated": "2026-02-20T02:15:52.000Z", "created": None,
     "pipeline": "text-generation", "gated": True, "cached": False, "architecture": "gemma4", "runs": False}]}
FOUND_PAGE2 = {"query": "", "family": True, "architectures": ["qwen35", "qwen3", "inkling2_loop"], "arch": None,
               "endpoint": "https://huggingface.co", "next_cursor": None, "models": [
    {"id": "acme/Page2-Model-GGUF", "downloads": 500, "likes": 9, "updated": "2026-11-01T02:15:52.000Z", "created": None,
     "pipeline": "text-generation", "gated": False, "cached": False, "architecture": "qwen3", "runs": True}]}


# What GET /v1/hub/drafter says of the drafters the checks type in.
DRAFTER_KINDS = {"./another.gguf": "mtp", "./new-mtp.gguf": "mtp", "./drafts/Speedy": "dspark"}


def repository(repo_spec, helper_cached=False):
    """GET /v1/hub/files: REPO as this build runs it, a dflash GGUF drafter repo, a safetensors drafter repo, or unsupported."""
    repo, _, rev = repo_spec.partition("@")
    rev_sfx = f"@{rev}" if rev and rev != "main" else ""
    if repo == "acme/Model-DFlash2-GGUF":
        return {"repo": repo, "revision": rev or "main", "from_cache": False, "drafter": None, "drafter_repo": True,
                "base_models": ["acme/Tiny-27B"], "runs": True, "known_drafters": [],
                "build_architectures": ["qwen35", "qwen3"], "mtp": None, "mmproj": None, "ggufs": [],
                "companions": [{"name": repo + ":helper.gguf" + rev_sfx, "path": "helper.gguf",
                                "file": "helper.gguf", "kind": "dspark" if helper_cached else "drafter",
                                "drafter_kind": "dspark" if helper_cached else "",
                                "size": int(0.5 * GB), "cached": helper_cached, "disk": None, "default": True}],
                "info": {"downloads": 12, "likes": 1, "updated": "2026-11-24T02:15:52.000Z", "license": "apache-2.0",
                         "pipeline": "text-generation", "gated": False, "architecture": "dflash",
                         "context": 1048576, "parameters": 495000000}}
    if repo == "acme/Model-DFlash2":
        return {"repo": repo, "revision": rev or "main", "from_cache": False, "drafter": {"size": int(0.9 * GB), "kind": "dflash2"},
                "drafter_repo": False, "base_models": ["acme/Tiny-27B"], "runs": None, "known_drafters": [],
                "build_architectures": ["qwen35", "qwen3"], "mtp": None, "mmproj": None, "ggufs": [], "companions": [],
                "info": {"downloads": 500, "likes": 25, "updated": "2026-11-24T02:15:52.000Z", "license": "apache-2.0",
                         "pipeline": None, "gated": False, "architecture": None, "context": None, "parameters": None}}
    if repo == "acme/Speedy-Drafter":  # a DSpark drafter whose name does not say so: the server says its kind
        return {"repo": repo, "revision": rev or "main", "from_cache": False, "drafter": {"size": int(0.8 * GB), "kind": "dspark"},
                "drafter_repo": False, "base_models": ["acme/Tiny-27B"], "runs": None, "known_drafters": [],
                "build_architectures": ["qwen35", "qwen3"], "mtp": None, "mmproj": None, "ggufs": [], "companions": [],
                "info": {"downloads": 50, "likes": 2, "updated": "2026-11-24T02:15:52.000Z", "license": "mit",
                         "pipeline": None, "gated": False, "architecture": None, "context": None, "parameters": None}}
    runs = repo == REPO
    old_rev = runs and rev == "oldrev123"
    quants = ([("Q2_K", 9.8, True), ("Q4_K_M", 17.6, True)] if old_rev
              else [("IQ3_XXS", 11.8, True), ("Q4_K_M", 17.6, True), ("Q5_K_M", 19.4, False), ("Q6_K", 22.3, False), ("BF16", 54.7, False)])
    drafters = ([{"repo": "acme/OldRev-DFlash2", "kind": "dflash2", "base": "acme/Tiny-27B", "exact": True,
                  "relation": None, "tok_s": None}] if old_rev
                else [{"repo": "acme/Tiny-27B-DFlash2", "kind": "dflash2", "base": "acme/Tiny-27B", "exact": False,
                       "relation": "quantized", "tok_s": None}] if runs else [])
    return {"repo": repo, "revision": rev or "main", "from_cache": False, "drafter": None, "base_models": ["acme/Tiny-27B"], "runs": runs,
            "known_drafters": drafters, "build_architectures": ["qwen35", "qwen3"],
            "mtp": {"path": "MTP/mtp.gguf", "size": int(0.6 * GB), "cached": True} if runs else None,
            "mmproj": {"path": "mmproj-BF16.gguf", "size": int(0.9 * GB), "cached": True} if runs else None,
            "companions": [
                {"name": repo + ":mtp.gguf" + rev_sfx, "path": "MTP/mtp.gguf", "file": "mtp.gguf", "kind": "mtp",
                 "size": int(0.6 * GB), "cached": True, "disk": None, "default": True},
                {"name": repo + ":mmproj-BF16.gguf" + rev_sfx, "path": "mmproj-BF16.gguf", "file": "mmproj-BF16.gguf", "kind": "mmproj",
                 "size": int(0.9 * GB), "cached": True, "disk": None, "default": True},
                {"name": repo + ":mmproj-F16.gguf" + rev_sfx, "path": "mmproj-F16.gguf", "file": "mmproj-F16.gguf", "kind": "mmproj",
                 "size": int(0.9 * GB), "cached": False, "disk": None, "default": False},
            ] if runs else [],
            "info": {"downloads": 7383193, "likes": 6300, "updated": "2026-11-12T02:15:52.000Z", "license": "apache-2.0", "pipeline": None,
                     "gated": False, "architecture": "qwen35" if runs else "gemma4", "context": 262144, "parameters": 27320697856},
            "ggufs": [{"name": repo + ":" + q + rev_sfx, "quant": q, "size": int(s * GB), "split": False, "parts": 1, "complete": True, "cached": c and runs}
                      for q, s, c in quants]}


LISTING = {"path": PICKED, "parent": "E:\\", "kind": "dir", "dirs": [],
           "models": [dir_model(PICKED, "Added-9B-Q4_K_M.gguf", int(5.6 * GB), quant="Q4_K_M")]}
FOLDERS = {"": {"path": "", "parent": None, "kind": "start", "dirs": [{"name": "D:\\", "path": "D:\\"}], "models": []},
           "D:\\": {"path": "D:\\", "parent": None, "kind": "dir", "dirs": [{"name": "models", "path": FOLDER}], "models": []},
           FOLDER: {"path": FOLDER, "parent": "D:\\", "kind": "dir", "dirs": [], "models": [m for m in CATALOG["models"] if m["source"] == "dir"]},
           LOCAL_DRAFTER["path"]: {"path": LOCAL_DRAFTER["path"], "parent": FOLDER, "kind": "file", "dirs": [], "models": [LOCAL_DRAFTER]},
           PICKED: LISTING}


def telemetry(want, remembered=None):
    """What the server says of itself: serving `want`, or nothing (then `remembered` is the load it would repeat)."""
    load = {"state": "ready", "loaded": want, "remembered": want} if want else {"state": "none"}
    if not want and remembered:
        load["remembered"] = remembered
    return {"schema_version": 1, "sampled_at": 1000.0, "state": {"phase": "idle"}, "rates": {}, "recent": [], "load": load,
            "model": {"id": "local", "loaded": True, "context": 8192, "max_sessions": 8} if want else {"id": None, "loaded": False},
            "device": {"name": "GPU", "mem_total": 32 * 2**30, "mem_used": 2**30}, "memory": {"sessions": []}}


# The Hub's adapters for a base (GET /v1/hub/loras) and one repository's (GET /v1/hub/lora): one it runs, one it does not.
HUB_LORAS = [{"id": "acme/Tiny-27B-coder-lora", "downloads": 12000, "likes": 30, "updated": None, "gated": False, "cached": False},
             {"id": "acme/Tiny-27B-chat-lora", "downloads": 900, "likes": 2, "updated": None, "gated": True, "cached": True}]
HUB_LORA = {"repo": "acme/Tiny-27B-coder-lora", "revision": "main", "adapters": [
    {"folder": "", "name": "acme/Tiny-27B-coder-lora", "rank": 16, "scale": 2.0, "base": "acme/Tiny-27B", "targets": None,
     "linears": ["attn.q", "mlp.down"], "layers": 64, "size": int(0.23 * GB), "why": None},
    {"folder": "fp8", "name": "acme/Tiny-27B-coder-lora:fp8", "rank": 16, "scale": 2.0, "base": "acme/Tiny-27B",
     "targets": None, "linears": [], "layers": 0, "size": int(0.12 * GB),
     "why": "L0.q.lora_A.weight: dtype F8_E4M3 (F32, BF16 or F16)"}]}


class Server:
    """The stubbed server of one page: what it was asked to load, to delete and to show a folder dialog for."""

    def __init__(self, page, tm, dialog=(200, LISTING), gpu_apps=None, catalog=None, folders=None):
        self.tm, self.dialog, self.posted, self.deleted, self.dialogs = tm, dialog, [], [], []
        self.gpu_apps = copy.deepcopy(gpu_apps) if gpu_apps is not None else None
        self.catalog = catalog if catalog is not None else CATALOG
        self.folders = folders if folders is not None else FOLDERS
        self.closed = []
        self.hub_endpoint = "https://huggingface.co"
        self.hub_source = "default"
        self.hub_fail = False
        self.files_fail = None
        self.files_asked = []
        self.dir_asked = []
        self.hub_configs = []
        self.lora_asked = []
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
            return route.fulfill(json=self.catalog)
        if "/v1/hub/loras" in url:
            self.lora_asked.append({k: v[0] for k, v in query.items()})
            base = query.get("base", [""])[0]
            if not base and query.get("model", [""])[0].startswith(FOLDER):  # a file of a folder: its base not known
                return route.fulfill(json={"base": None, "from": None, "models": [], "next_cursor": None,
                                           "endpoint": self.hub_endpoint})
            return route.fulfill(json={"base": base or "acme/Tiny-27B", "from": "asked" if base else "model card",
                                       "models": [] if base == "acme/Other" else HUB_LORAS, "next_cursor": None,
                                       "endpoint": self.hub_endpoint})
        if "/v1/hub/lora" in url:
            return route.fulfill(json=HUB_LORA)
        if "/v1/hub/config" in url:
            if method == "POST":
                body = route.request.post_data_json or {}
                raw = (body.get("endpoint") or "").strip()
                self.hub_configs.append(raw)
                if raw.startswith("ftp://"):
                    return route.fulfill(status=400, json={"error": {"message": "endpoint: URL must use http:// or https://"}})
                if not raw or raw.rstrip("/") == "https://huggingface.co":
                    self.hub_endpoint, self.hub_source = "https://huggingface.co", "default"
                else:
                    norm = raw if raw.startswith(("http://", "https://")) else "https://" + raw
                    self.hub_endpoint, self.hub_source = norm.rstrip("/"), "setting"
            return route.fulfill(json={"endpoint": self.hub_endpoint, "default_endpoint": "https://huggingface.co",
                                       "source": self.hub_source, "offline": False})
        if "/v1/hub/search" in url:
            if self.hub_fail:
                return route.fulfill(status=502, json={"error": {"message": "cannot reach the Hub (timed out)"}})
            cursor = query.get("cursor", [""])[0]
            arch = query.get("arch", [""])[0]
            q_text = query.get("q", [""])[0]
            if cursor == "page2-tok":
                res = copy.deepcopy(FOUND_PAGE2)
                res["endpoint"] = self.hub_endpoint
                return route.fulfill(json=res)
            res = copy.deepcopy(FOUND)
            res["endpoint"] = self.hub_endpoint
            if q_text:
                res["query"], res["family"], res["next_cursor"] = q_text, False, None
                if q_text == "no-such-model-xyz":
                    res["models"] = []
                elif q_text == "inkling":
                    res["models"] = [m for m in res["models"] if "Inkling" in m["id"]]
            elif arch:
                res["arch"], res["next_cursor"] = arch, None
                res["models"] = [m for m in res["models"] if m.get("architecture") == arch]
            return route.fulfill(json=res)
        if "/v1/hub/drafter" in url:  # a drafter's kind, as its fields say (none read from its name)
            name = query.get("name", [""])[0]
            return route.fulfill(json={"name": name, "kind": DRAFTER_KINDS.get(name)})
        if "/v1/hub/files" in url:
            self.files_asked.append(query["repo"][0])
            if self.files_fail:
                status, msg = self.files_fail
                return route.fulfill(status=status, json={"error": {"message": msg}})
            res = repository(query["repo"][0], getattr(self, "helper_cached", False))
            res["endpoint"] = self.hub_endpoint
            return route.fulfill(json=res)
        if "/v1/hub/dir" in url:
            path = query.get("path", [""])[0]
            self.dir_asked.append(path)
            return route.fulfill(json=self.folders[path]) if path in self.folders else route.fulfill(
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
    if action == "Switch":
        expect(page.locator(".modal")).to_contain_text("no model is served until another is loaded")
        expect(page.locator(".modal")).not_to_contain_text("is loaded again")
    with page.expect_request(lambda r: r.method == "POST" and r.url.endswith("/v1/load")):
        page.locator(".modal").get_by_role("button", name=action).click()
    expect(page.locator("#mv-current")).to_be_visible()


def check_settings(page, server, want):
    """The model served: its settings as loaded, a change against them, what a reload posts, and active-status dots."""
    expect(page.locator("#mv-current")).to_be_visible()
    expect(page.locator("#mv-state")).to_have_text("local")
    expect(page.locator("#ld-ctx")).to_be_hidden()  # the context it asked for: nothing to explain
    # Model status card offers Reload model alongside Unload.
    expect(page.locator("#ld-reload")).to_be_visible()
    expect(page.locator("#ld-unload")).to_be_visible()
    confirm(page, "#ld-reload", "Reload")
    assert server.posted[-1] == want, server.posted[-1]
    # When neither a drafter nor a vision projector is loaded, both tab indicator dots stay gray (not .on) and show none.
    spec_tab = page.locator('#lset-tabs [data-stab="spec"]')
    vis_tab = page.locator('#lset-tabs [data-stab="vision"]')
    expect(spec_tab.locator(".dot")).not_to_have_class(re.compile(r"\bon\b"))
    expect(vis_tab.locator(".dot")).not_to_have_class(re.compile(r"\bon\b"))
    expect(spec_tab.locator(".sum")).to_have_text("none")
    expect(vis_tab.locator(".sum")).to_have_text("none")
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
    page.locator("#mf-vramres").fill("0")
    page.locator("#mf-ramres").fill("0")
    expect(cmd).to_contain_text("--vram-reserve 0")
    expect(cmd).to_contain_text("--ram-reserve 0")
    confirm(page, "#mf-reload", "Reload")
    assert server.posted[-1]["vram_reserve"] == 0 and server.posted[-1]["ram_reserve"] == 0, server.posted[-1]
    page.locator("#mf-vramres").fill("")
    page.locator("#mf-ramres").fill("")
    expect(cmd).not_to_contain_text("--vram-reserve")
    expect(cmd).not_to_contain_text("--ram-reserve")
    page.locator("#mf-reset").click()
    expect(page.locator("#mf-name")).to_have_value("")
    expect(page.locator("#mf-reload")).to_have_count(0)


def check_context_limit(open_page, want):
    """A context memory cut under the model's own: the status card says where K/V lives, which memory cut it and what
    would give more; a short one as a warning, on the Overview too."""
    gpu = {"by": "gpu", "wanted": 262144, "short": True, "left": 820_000_000, "need_more": 5_120_000_000,
           "gpu_tokens": 8192, "others": 3_600_000_000, "closable": True, "runtime": 540_000_000}
    cases = [
        (gpu, "warn", "Limited context window 8,192 / 262,144. Not enough VRAM left "
                      "(3.60 GB used by other apps). Close other apps and reload for a longer context. "
                      "Needs 5.12 GB more VRAM for max context."),
        (dict(gpu, others=100_000_000, closable=False, need_more=None), "warn",
         "Limited context window 8,192 / 262,144. Not enough VRAM left. "
         "Disable speculative decoding or use a smaller quantization for a longer context."),
        ({"by": "host", "wanted": 262144, "short": False, "left": 7_300_000_000, "need_more": 4_500_000_000,
          "gpu_tokens": 4096, "others": None, "closable": False, "runtime": None}, "hint",
         "Limited context window 8,192 / 262,144. Not enough free RAM (7.30 GB available for KV). "
         "Free RAM or raise \"KV in host RAM\" for a longer context. Needs 4.50 GB more RAM for max context."),
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


def check_commands(page, open_page, want):
    """The command line shown, as each of this system's shells hands it to a program."""
    shells = ("powershell", "pwsh") if os.name == "nt" else ("bash", "zsh")
    with tempfile.TemporaryDirectory() as dirname:
        tmp = Path(dirname)
        (tmp / "serve").write_text(
            'import json, sys\nfrom pathlib import Path\n'
            'Path("args.json").write_text(json.dumps(sys.argv[1:]), encoding="utf8")\n', encoding="utf8")
        for name in ("Qwen ‘local’", "foo$bar", "foo`bar", 'Local "27B"', "A path ending in a slash\\"):
            page.locator("#mf-name").fill(name)
            expected = [want["model"], "--ctx", "8192", "--kv", "fp8", "--kv-ram", "4", "--kv-gpu", "0", "--name", name, "--port", "18479"]
            assert command(page) == ["flyx", "serve", *expected]
            page.locator('#mf-settings [data-shell="powershell"]').click()
            ps_cmd = page.locator("#mf-cmd").inner_text()
            if name in ("foo$bar", "foo`bar"):
                assert f"--name '{name}'" in ps_cmd, ps_cmd
            elif name == "Qwen ‘local’":
                assert f'--name "{name}"' in ps_cmd, ps_cmd
            else:
                assert ps_cmd.startswith("Start-Process flyx "), ps_cmd
            for shell in shells:
                if shutil.which(shell):
                    page.locator('#mf-settings [data-shell="powershell"]' if os.name == "nt" else '#mf-settings [data-shell="sh"]').click()
                    assert native_args(page.locator("#mf-cmd").inner_text(), shell, tmp) == expected
    page.locator("#mf-reset").click()

    # Smart quoting: safe specs stay unquoted, space-containing Windows paths use double quotes (cmd.exe + PowerShell).
    for model_spec, want_ps, want_sh in (
        ("unsloth/Qwen3-GGUF:Q4_K_M", "flyx serve unsloth/Qwen3-GGUF:Q4_K_M", "flyx serve unsloth/Qwen3-GGUF:Q4_K_M"),
        (r"C:\models\foo.gguf", r"flyx serve C:\models\foo.gguf", r"flyx serve 'C:\models\foo.gguf'"),
        (r"C:\My Models\foo.gguf", r'flyx serve "C:\My Models\foo.gguf"', r"flyx serve 'C:\My Models\foo.gguf'"),
    ):
        p2, _ = open_page(telemetry({"model": model_spec}))
        p2.locator('#mf-settings [data-shell="powershell"]').click()
        ps_text = p2.locator("#mf-cmd").inner_text()
        assert ps_text.startswith(want_ps), ps_text
        p2.locator('#mf-settings [data-shell="sh"]').click()
        sh_text = p2.locator("#mf-cmd").inner_text()
        assert sh_text.startswith(want_sh), sh_text
        p2.close()


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
    expect(page.locator("#mf-cmd")).to_contain_text("--drafter mtp")  # the kind the server read
    check_choice(True, "mtp", ["./another.gguf"])
    # A drafter whose name says nothing of its kind: the server's word for it, DSpark's here.
    page.locator("#sd-file").fill("./drafts/Speedy")
    expect(page.locator("#mf-cmd")).to_contain_text("--drafter dspark")
    page.locator("#sd-file").fill("./unknown.flyx")  # nothing says what it is: no --drafter
    page.wait_for_timeout(800)
    assert "--drafter" not in command(page) and "./unknown.flyx" in command(page), command(page)
    page.locator("#mf-reset").click()
    expect(page.locator("#sd-saved input")).to_be_checked()
    # Draft trees: the drafter's default, on, or off (--no-draft-trees: a DSpark drafter's default trees off).
    page.locator('#lset-tabs [data-stab="spec"]').click()
    if not page.locator("#sd-adv").evaluate("e => e.open"):
        page.locator("#sd-adv > summary").click()
    expect(page.locator("#sd-trees")).to_be_visible()
    expect(page.locator("#sd-trees")).to_have_value("on")  # as loaded
    page.locator("#sd-trees").select_option("off")
    assert "--no-draft-trees" in command(page) and "--draft-trees" not in command(page), command(page)
    confirm(page, "#mf-reload", "Reload")
    assert server.posted[-1]["draft_trees"] is False, server.posted[-1]
    page.locator("#sd-trees").select_option("")
    assert "--draft-trees" not in command(page) and "--no-draft-trees" not in command(page), command(page)
    confirm(page, "#mf-reload", "Reload")
    assert server.posted[-1]["draft_trees"] is None, server.posted[-1]
    page.locator("#mf-reset").click()

    # A saved selection carries its header identity across restore. Editing the path drops that identity.
    want.update(drafter="dflash2", drafters=["./helper.gguf"])
    page.reload()
    page.locator('#lset-tabs [data-stab="spec"]').click()
    expect(page.locator("#sd-custom input[type=radio]")).to_be_checked()
    expect(page.locator("#sd-file")).to_have_value("./helper.gguf")
    page.locator("#sd-file").fill("./renamed.gguf")
    check_choice(True, "", ["./renamed.gguf"])
    page.locator("#mf-reset").click()
    expect(page.locator("#mf-cmd")).to_contain_text("--drafter dflash2")
    want.update(drafter="all", drafters=files)
    page.reload()


def check_degraded_and_draft_weights(open_page, want):
    """Drafter weight precision choices (auto/nvfp4/q8/exact) and VRAM adjustment banner + restore action."""
    base = dict(want, drafter="dflash2", drafters=["acme/Tiny-27B-DFlash2"])
    tm = telemetry(base)
    tm["model"]["drafter"] = "mtp"
    tm["model"]["draft_weights"] = "nvfp4"
    tm["model"]["max_width"] = 7
    tm["load"]["degraded"] = [
        {"kind": "drafter", "from": "dflash2", "to": "mtp",
         "detail": "DFlash2 drafter skipped due to tight VRAM; falling back to MTP"}
    ]
    tm["model"]["degraded"] = tm["load"]["degraded"]
    snap = {
        "supported": True, "can_close": True, "device": "NVIDIA GeForce RTX 5070 Ti",
        "mem_total": int(17.09 * GB), "mem_used": int(14.5 * GB), "mem_free": int(2.59 * GB),
        "own_bytes": int(10.8 * GB), "others_bytes": int(3.7 * GB), "system_bytes": int(0.8 * GB),
        "idle_igpu": None,
        "apps": [{"id": "slack", "name": "Slack", "process": "Slack", "bytes": int(0.4 * GB),
                  "local_bytes": int(0.4 * GB), "processes": 1, "closable": True, "reason": None}],
    }
    page, server = open_page(tm, gpu_apps=snap)
    # Degraded banner clearly explains what was degraded and what is actively running.
    deg = page.locator("#ld-degraded")
    expect(deg).to_be_visible()
    expect(deg).to_contain_text("Loaded with VRAM adjustments")
    expect(deg).to_contain_text("DFlash2 drafter skipped due to tight VRAM; falling back to MTP")
    expect(deg).to_contain_text("Active now:")
    expect(deg).to_contain_text("mtp (nvfp4)")
    # Manage VRAM opens the VRAM manager inline; Restore full settings reloads the preserved requested config.
    page.locator("#ld-deg-vram").click()
    expect(page.locator("#ld-vram-apps")).to_be_visible()
    confirm(page, "#ld-deg-restore", "Reload")
    assert server.posted[-1]["drafter"] == "dflash2" and server.posted[-1]["drafters"] == ["acme/Tiny-27B-DFlash2"], server.posted[-1]
    # Overview also shows the global degraded hint banner with Manage VRAM.
    page.locator('.side nav .navi[data-tab="overview"]').click()
    hint = page.locator('#hint[data-hint="degraded"]')
    expect(hint).to_be_visible()
    expect(hint).to_contain_text("Loaded with VRAM adjustments")
    hint.get_by_role("button", name="View details").click()
    expect(page.locator("#tab-model")).to_be_visible()
    # Advanced drafter weights dropdown supports auto (default), nvfp4, q8, and exact.
    page.locator('#lset-tabs [data-stab="spec"]').click()
    if page.locator("#sd-adv").get_attribute("open") is None:
        page.locator("#sd-adv summary").click()
    expect(page.locator("#sd-weights")).to_have_value("")
    for val in ("nvfp4", "q8", "exact"):
        page.locator("#sd-weights").select_option(val)
        assert "--backend-opt" in command(page) and f"draft_weights={val}" in command(page), command(page)
        confirm(page, "#mf-reload", "Reload")
        assert server.posted[-1]["draft_weights"] == val, server.posted[-1]
    page.close()


def check_vision(page, server, open_page, want):
    """Vision encoder settings tab: custom mmproj, storage/precision/max-tokens, disable toggle."""
    page.locator('#lset-tabs [data-stab="vision"]').click()
    expect(page.locator("#vs-on")).to_be_checked()
    expect(page.locator("#vs-device")).to_have_value("disk")
    expect(page.locator("#vs-quant")).to_have_value("q8")
    page.locator("#vs-file").fill("./mmproj-test.gguf")
    page.locator("#vs-device").select_option("host")
    page.locator("#vs-quant").select_option("nvfp4")
    page.locator("#vs-maxtok").fill("1024")
    assert page.locator("#vs-maxtok").evaluate("el => el.checkValidity()")
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
    confirm(page, "#mf-reload", "Reload")
    body = server.posted[-1]
    assert body["vision"] is False and body["mmproj"] is None, body
    page.locator("#mf-reset").click()
    expect(page.locator("#vs-on")).to_be_checked()
    # A model loaded with vision disabled (even if a stale mmproj remained in saved settings) shows no ghost changes.
    off_page, _ = open_page(telemetry(dict(want, vision=False, mmproj="./stale.gguf")))
    expect(off_page.locator("#mf-reload")).to_have_count(0)
    off_page.close()
    # A model that actually loaded a drafter and vision encoder lights both tab dots green, and toggling off grays them.
    tm_on = telemetry(dict(want, drafter="", drafters=[]))
    tm_on["model"]["drafter"] = "mtp"
    tm_on["model"]["max_width"] = 4
    tm_on["model"]["vision"] = True
    tm_on["load"]["vision_projector"] = {"from": "acme/Tiny-27B-GGUF/mmproj-BF16.gguf", "size": int(0.9 * GB)}
    on_page, _ = open_page(tm_on)
    # Active drafter and vision sources live in their settings tabs rather than inflating the Serving card.
    expect(on_page.locator(".srv #ld-drafters")).to_have_count(0)
    expect(on_page.locator(".srv #ld-vision-src")).to_have_count(0)
    spec_dot = on_page.locator('#lset-tabs [data-stab="spec"] .dot')
    vis_dot = on_page.locator('#lset-tabs [data-stab="vision"] .dot')
    expect(spec_dot).to_have_class(re.compile(r"\bon\b"))
    expect(vis_dot).to_have_class(re.compile(r"\bon\b"))
    expect(on_page.locator('#lset-tabs [data-stab="spec"] .sum')).to_have_text("default")
    expect(on_page.locator('#lset-tabs [data-stab="vision"] .sum')).to_have_text("disk · q8")
    on_page.locator('#lset-tabs [data-stab="spec"]').click()
    expect(on_page.locator("#lset-spec #ld-drafters")).to_be_visible()
    expect(on_page.locator("#lset-spec #ld-drafters")).to_contain_text("MTP")
    on_page.locator('#lset-tabs [data-stab="vision"]').click()
    expect(on_page.locator("#lset-vision #ld-vision-src")).to_be_visible()
    expect(on_page.locator("#lset-vision #ld-vision-src")).to_contain_text("mmproj-BF16.gguf")
    on_page.locator("#vs-on").uncheck(force=True)
    expect(vis_dot).not_to_have_class(re.compile(r"\bon\b"))
    on_page.close()


def check_hub(page, server, want):
    """The library's Hugging Face side: supported architectures, filter pills, pagination, files, a file loaded."""
    view(page, "library")
    expect(page.locator("#hf-list .prow")).to_have_count(3)
    expect(page.locator("#hf-family")).to_have_count(0)
    expect(page.locator("#hf-archs button")).to_have_count(4)
    expect(page.locator("#hf-archs")).to_contain_text("qwen35")
    expect(page.locator("#hf-archs")).to_contain_text("inkling2_loop")
    expect(page.locator("#hf-pager")).to_be_visible()
    expect(page.locator("#hf-prev")).to_be_disabled()
    expect(page.locator("#hf-page")).to_have_text("Page 1")
    expect(page.locator("#hf-next")).to_be_enabled()
    # Select a quantization first, then verify pagination reconciles detail & clears stale #mf-load target.
    expect(page.locator("#hf-name")).to_have_text("Tiny-27B-GGUF")
    page.locator('#mf-files .mrow[data-name$=":Q5_K_M"]').click()
    expect(page.locator("#mf-load")).to_be_enabled()
    page.locator("#hf-next").click()
    expect(page.locator("#hf-page")).to_have_text("Page 2")
    expect(page.locator("#hf-list .prow")).to_have_count(1)
    expect(page.locator('#hf-list .prow[data-repo="acme/Page2-Model-GGUF"]')).to_have_count(1)
    expect(page.locator("#hf-name")).to_have_text("Page2-Model-GGUF")
    expect(page.locator("#mf-load")).to_be_disabled()
    expect(page.locator("#hf-prev")).to_be_enabled()
    expect(page.locator("#hf-next")).to_be_disabled()
    page.locator("#hf-prev").click()
    expect(page.locator("#hf-page")).to_have_text("Page 1")
    expect(page.locator("#hf-list .prow")).to_have_count(3)
    expect(page.locator("#hf-name")).to_have_text("Tiny-27B-GGUF")
    # Architecture filter pills narrow the list, reconcile detail & clear stale #mf-load target, and reset back to All.
    page.locator('#mf-files .mrow[data-name$=":Q5_K_M"]').click()
    expect(page.locator("#mf-load")).to_be_enabled()
    page.locator('#hf-archs button[data-arch="inkling2_loop"]').click()
    expect(page.locator("#hf-list .prow")).to_have_count(1)
    expect(page.locator('#hf-list .prow[data-repo="acme/Inkling-2-Mini-GGUF"]')).to_have_count(1)
    expect(page.locator("#hf-name")).to_have_text("Inkling-2-Mini-GGUF")
    expect(page.locator("#mf-load")).to_be_disabled()
    page.locator('#hf-archs button[data-arch=""]').click()
    expect(page.locator("#hf-list .prow")).to_have_count(3)
    page.locator('#hf-list .prow[data-repo="acme/Tiny-27B-GGUF"]').click()
    expect(page.locator("#hf-name")).to_have_text("Tiny-27B-GGUF")  # the first repository found is shown
    expect(page.locator("#hf-detail .dchips")).to_contain_text("+ Vision")
    expect(page.locator("#hf-detail .dchips")).to_contain_text("+ MTP")
    expect(page.locator("#mf-files .mrow")).to_have_count(5)
    expect(page.locator("#hf-companions .mrow")).to_have_count(3)
    expect(page.locator('#hf-companions .mrow[data-kind="mtp"] .cmain svg use')).to_have_attribute("href", "#i-bolt")
    expect(page.locator('#hf-companions .mrow[data-kind="mmproj"]').first.locator(".cmain svg use")).to_have_attribute("href", "#i-image")
    expect(page.locator("#mf-load")).to_be_disabled()
    # Copy name copies org/name by default, or org/name:QUANT when a quantization row is selected.
    page.evaluate("() => { window.__copied = []; navigator.clipboard.writeText = (s) => { window.__copied.push(s); return Promise.resolve(); }; }")
    copy_btn = page.locator("#hf-detail .dh button")
    expect(copy_btn).to_have_attribute("title", REPO)
    copy_btn.click()
    assert page.evaluate("() => window.__copied.at(-1)") == REPO
    page.locator('#mf-files .mrow[data-name$=":Q5_K_M"]').click()
    expect(copy_btn).to_have_attribute("title", REPO + ":Q5_K_M")
    copy_btn.click()
    assert page.evaluate("() => window.__copied.at(-1)") == REPO + ":Q5_K_M"
    page.locator('#mf-files .mrow[data-name$=":Q5_K_M"]').click()  # restore deselected state
    # Selecting a companion in Hugging Face detail turns the bottom-right CTA into Use as projector / Draft with it.
    page.locator('#hf-companions .mrow[data-comp$=":mmproj-F16.gguf"]').click()
    expect(page.locator("#mf-load")).to_have_text("Use as projector")
    page.locator("#mf-load").click()
    expect(page.locator("#mv-current")).to_be_visible()
    expect(page.locator("#vs-file")).to_have_value(REPO + ":mmproj-F16.gguf")
    expect(page.locator("#vs-custom .tag2")).to_have_text("Hugging Face")
    page.locator("#mf-reset").click()
    view(page, "library")
    page.locator('#hf-companions .mrow[data-comp$=":mtp.gguf"]').click()
    expect(page.locator("#mf-load")).to_have_text("Draft with it")
    page.locator("#mf-load").click()
    expect(page.locator("#sd-file")).to_have_value(REPO + ":mtp.gguf")
    page.locator("#mf-reset").click()
    view(page, "library")
    # Selecting one drafter + one vision projector simultaneously applies both in one click.
    page.locator('#hf-companions .mrow[data-comp$=":mtp.gguf"]').click()
    page.locator('#hf-companions .mrow[data-comp$=":mmproj-F16.gguf"]').click()
    expect(page.locator('#hf-companions .mrow[data-comp$=":mtp.gguf"]')).to_have_attribute("aria-selected", "true")
    expect(page.locator('#hf-companions .mrow[data-comp$=":mmproj-F16.gguf"]')).to_have_attribute("aria-selected", "true")
    # Picking another mmproj swaps only the projector while keeping the drafter selected.
    page.locator('#hf-companions .mrow[data-comp$=":mmproj-BF16.gguf"]').click()
    expect(page.locator('#hf-companions .mrow[data-comp$=":mtp.gguf"]')).to_have_attribute("aria-selected", "true")
    expect(page.locator('#hf-companions .mrow[data-comp$=":mmproj-F16.gguf"]')).to_have_attribute("aria-selected", "false")
    expect(page.locator('#hf-companions .mrow[data-comp$=":mmproj-BF16.gguf"]')).to_have_attribute("aria-selected", "true")
    expect(page.locator("#mf-load")).to_have_text("Use selected companions")
    page.locator("#mf-load").click()
    expect(page.locator("#mv-current")).to_be_visible()
    expect(page.locator("#sd-file")).to_have_value(REPO + ":mtp.gguf")
    expect(page.locator("#mf-cmd")).to_contain_text("--drafter mtp")
    page.locator('#lset-tabs [data-stab="vision"]').click()
    expect(page.locator("#vs-file")).to_have_value(REPO + ":mmproj-BF16.gguf")
    page.locator("#mf-reset").click()
    view(page, "library")
    # Picking a quantization together with 1 drafter + 1 mmproj loads all three in one POST /v1/load.
    page.locator('#mf-files .mrow[data-name$=":Q5_K_M"]').click()
    page.locator('#hf-companions .mrow[data-comp$=":mtp.gguf"]').click()
    page.locator('#hf-companions .mrow[data-comp$=":mmproj-F16.gguf"]').click()
    expect(page.locator("#mf-load")).to_have_text("Switch to this model")
    expect(page.locator("#mf-chosen-s")).to_contain_text("20.3 GB to download first")
    expect(page.locator("#mf-chosen-s")).to_contain_text("mtp.gguf")
    expect(page.locator("#mf-chosen-s")).to_contain_text("mmproj-F16.gguf")
    confirm(page, "#mf-load", "Switch")
    body = server.posted[-1]
    assert (body["model"], body["drafter"], body["drafters"], body["mmproj"]) == (
        REPO + ":Q5_K_M", "mtp", [REPO + ":mtp.gguf"], REPO + ":mmproj-F16.gguf"), body
    # Selecting a quantization alone posts with default companions and exact quantization size.
    view(page, "library")
    page.locator('#mf-files .mrow[data-name$=":Q5_K_M"]').click()  # deselect previous Q5_K_M
    page.locator('#mf-files .mrow[data-name$=":Q5_K_M"]').click()
    expect(page.locator("#mf-load")).to_have_text("Switch to this model")
    expect(page.locator("#mf-chosen-s")).to_contain_text("19.4 GB to download first")
    confirm(page, "#mf-load", "Switch")
    body = server.posted[-1]  # another model, with the settings of the one served
    assert (body["model"], body["ctx"], body["kv"], body["kv_gpu"], body["drafters"], body["mmproj"]) == (
        REPO + ":Q5_K_M", want["ctx"], want["kv"], want["kv_gpu"], [], None), body
    # Hub mirror endpoint selector: validate bad URL, apply mirror, update links, and reset back to default.
    view(page, "library")
    expect(page.locator("#hf-ep-toggle")).to_contain_text("huggingface.co")
    page.locator("#hf-ep-toggle").click()
    expect(page.locator("#hf-epbar")).to_be_visible()
    page.locator("#hf-ep-input").fill("ftp://bad.example")
    page.locator("#hf-ep-save").click()
    expect(page.locator("#hf-ep-msg")).to_contain_text("http:// or https://")
    page.locator("#hf-ep-input").fill("hf-mirror.com")
    page.locator("#hf-ep-save").click()
    expect(page.locator("#hf-ep-toggle")).to_contain_text("hf-mirror.com")
    expect(page.locator("#hf-ep-toggle")).to_have_class(re.compile(r"\bon\b"))
    expect(page.locator('#hf-detail .dh a[href^="https://hf-mirror.com/"]')).to_have_count(1)
    expect(page.locator('#hf-detail a[href^="https://huggingface.co/"]')).to_have_count(0)  # every link: the mirror
    page.locator("#hf-ep-reset").click()
    expect(page.locator("#hf-ep-toggle")).to_contain_text("huggingface.co")
    expect(page.locator("#hf-ep-toggle")).not_to_have_class(re.compile(r"\bon\b"))
    expect(page.locator("#hf-list .prow")).to_have_count(3)
    page.locator("#hf-ep-toggle").click()
    expect(page.locator("#hf-epbar")).to_have_count(0)
    # When the Hub cannot be reached, the list shows a Use a mirror shortcut that opens the mirror bar.
    server.hub_fail = True
    page.locator("#hf-sort").select_option("downloads")
    expect(page.locator("#hf-msg")).to_contain_text("cannot reach the Hub")
    expect(page.locator("#hf-ep-help")).to_be_visible()
    page.locator("#hf-ep-help").click()
    expect(page.locator("#hf-epbar")).to_be_visible()
    server.hub_fail = False
    page.locator('#hf-epbar .chip[data-ep="https://hf-mirror.com"]').click()
    expect(page.locator("#hf-list .prow")).to_have_count(3)
    page.locator("#hf-ep-reset").click()
    page.locator("#hf-ep-toggle").click()
    page.locator("#hf-sort").select_option("trending")
    # A repository of an architecture this build does not run is not loaded.
    view(page, "library")
    page.locator('#hf-list .prow[data-repo="acme/Other-9B-GGUF"]').click()
    expect(page.locator("#hf-unsupported")).to_contain_text("gemma4")
    page.locator('#mf-files .mrow[data-name$=":Q5_K_M"]').click()
    expect(page.locator("#mf-load")).to_be_disabled()
    # A dflash GGUF drafter repo is never marked unsupported and lets the user draft with its GGUF directly.
    for mode in ("", "dspark"):
        server.helper_cached = bool(mode)
        view(page, "library")
        page.locator("#hf-q").fill("")
        page.locator("#hf-q").fill("acme/Model-DFlash2-GGUF")
        expect(page.locator("#hf-name")).to_have_text("Model-DFlash2-GGUF")
        expect(page.locator("#hf-drafter-repo")).to_be_visible()
        page.locator(f'#hf-companions .mrow[data-kind="{mode}"]').click()
        expect(page.locator("#mf-load")).to_have_text("Draft with it")
        page.locator("#mf-load").click()
        expect(page.locator("#sd-file")).to_have_value("acme/Model-DFlash2-GGUF:helper.gguf")
        confirm(page, "#mf-reload", "Reload")
        body = server.posted[-1]
        assert (body["drafter"], body["drafters"]) == (mode, ["acme/Model-DFlash2-GGUF:helper.gguf"]), body
        page.locator("#mf-reset").click()
    # Typing a safetensors DFlash2 repo org/name auto-opens it without Enter and offers Draft with it.
    view(page, "library")
    page.locator("#hf-q").fill("acme/Model-DFlash2")
    expect(page.locator("#hf-name")).to_have_text("Model-DFlash2")
    expect(page.locator("#hf-drafter-only")).to_be_visible()
    expect(page.locator("#mf-load")).to_have_text("Draft with it")
    # A DSpark drafter repo: drafted with as the kind the server says (--drafter dspark), whatever its name.
    page.locator("#hf-q").fill("acme/Speedy-Drafter")
    expect(page.locator("#hf-name")).to_have_text("Speedy-Drafter")
    expect(page.locator("#hf-drafter-only")).to_be_visible()
    page.locator("#mf-load").click()
    expect(page.locator("#sd-file")).to_have_value("acme/Speedy-Drafter")
    expect(page.locator("#mf-cmd")).to_contain_text("--drafter dspark")
    page.locator("#mf-reset").click()
    view(page, "library")
    page.locator("#hf-q").fill("")
    # A name with its quantization, in the search box: that file of that repository.
    page.locator("#hf-q").fill(REPO + ":q6_k")
    page.keyboard.press("Enter")
    expect(page.locator("#mf-files .mrow.sel")).to_have_attribute("data-name", REPO + ":Q6_K")
    expect(page.locator("#hf-drafters")).to_contain_text("acme/Tiny-27B-DFlash2")
    expect(page.locator("#hf-drafters")).to_contain_text("DFlash2")
    expect(page.locator("#hf-drafters")).to_contain_text("N/A")  # not run on this machine yet
    page.locator("#hf-drafters .btn").click()  # the drafter published for it goes to the settings
    expect(page.locator("#mv-current")).to_be_visible()
    expect(page.locator("#sd-file")).to_have_value("acme/Tiny-27B-DFlash2")
    # The settings offer it beside the default, under the drafters for its base model, chosen now, as its kind; the
    # custom drafter is not.
    page.locator('#lset-tabs [data-stab="spec"]').click()
    base = page.locator("#sd-base")
    expect(base).to_contain_text("acme/Tiny-27B")
    expect(base).to_contain_text("quantized")
    expect(base).to_have_class(re.compile(r"\bsel\b"))
    base.click()  # its drafters, shown
    verified = page.locator('.sdrow[data-verified="acme/Tiny-27B-DFlash2"]')
    expect(verified).to_have_class(re.compile(r"\bsel\b"))
    expect(verified).to_contain_text("N/A")
    expect(page.locator("#sd-custom")).not_to_have_class(re.compile(r"\bsel\b"))
    expect(page.locator("#mf-cmd")).to_contain_text("--drafter dflash2")
    page.locator("#sd-default").click()
    expect(verified).to_be_visible()
    verified.click()
    expect(page.locator("#mf-cmd")).to_contain_text(re.compile(r"--drafter-file '?acme/Tiny-27B-DFlash2"))
    page.locator("#mf-reset").click()
    # A 404 JSON or 502 error from /v1/hub/files surfaces the real error with a Retry button (not "update FLYX"),
    # clears any stale selected target, and Retry preserves the originally requested quantization.
    view(page, "library")
    server.files_fail = (404, "acme/Missing: repository not found")
    page.locator("#hf-q").fill("acme/Missing")
    page.keyboard.press("Enter")
    expect(page.locator("#mf-hubmsg.banner.bad")).to_contain_text("acme/Missing: repository not found")
    expect(page.locator("#mf-hubmsg")).not_to_contain_text("update FLYX")
    expect(page.locator("#mf-load")).to_be_disabled()
    server.files_fail = (502, "cannot reach the Hub (timed out)")
    page.locator("#hf-retry").click()
    expect(page.locator("#mf-hubmsg.banner.bad")).to_contain_text("cannot reach the Hub (timed out)")
    # Searching org/repo:Q5_K_M when /v1/hub/files fails once still selects Q5_K_M when Retry succeeds.
    page.locator("#hf-q").fill(REPO + ":Q5_K_M")
    page.keyboard.press("Enter")
    expect(page.locator("#mf-hubmsg.banner.bad")).to_contain_text("cannot reach the Hub")
    expect(page.locator("#mf-load")).to_be_disabled()
    server.files_fail = None
    page.locator("#hf-retry").click()
    expect(page.locator("#mf-files .mrow")).to_have_count(5)
    expect(page.locator("#mf-files .mrow.sel")).to_have_attribute("data-name", REPO + ":Q5_K_M")
    # An ordinary text search that returns 0 results clears the previously shown repository detail and target.
    page.locator("#hf-q").fill("no-such-model-xyz")
    page.keyboard.press("Enter")
    expect(page.locator("#hf-list .prow")).to_have_count(0)
    expect(page.locator("#hf-detail")).to_have_count(0)
    expect(page.locator(".hub-detail")).to_contain_text("Pick a repository on the left")
    expect(page.locator("#mf-load")).to_be_disabled()
    # A subsequent non-empty text search selects from the fresh results rather than staying empty.
    page.locator("#hf-q").fill("inkling")
    page.keyboard.press("Enter")
    expect(page.locator("#hf-list .prow")).to_have_count(1)
    expect(page.locator("#hf-name")).to_have_text("Inkling-2-Mini-GGUF")
    # Search failure clears stale repository detail and disables Load.
    server.hub_fail = True
    page.locator("#hf-q").fill("broken-search")
    page.keyboard.press("Enter")
    expect(page.locator("#hf-detail")).to_have_count(0)
    expect(page.locator("#mf-load")).to_be_disabled()
    server.hub_fail = False
    page.locator("#hf-q").fill("")
    expect(page.locator("#hf-list .prow")).to_have_count(3)
    expect(page.locator("#hf-detail")).to_have_count(1)
    # A zero-result architecture filter clears stale repository detail and disables Load.
    page.locator('#hf-archs [data-arch="qwen3"]').click()
    expect(page.locator("#hf-list .prow")).to_have_count(0)
    expect(page.locator("#hf-detail")).to_have_count(0)
    expect(page.locator("#mf-load")).to_be_disabled()
    page.locator('#hf-archs [data-arch=""]').click()
    expect(page.locator("#hf-list .prow")).to_have_count(3)
    # Pinned @revision is preserved across direct lookup, Retry after failure, and Hub endpoint change.
    server.files_fail = (502, "cannot reach the Hub (timed out)")
    page.locator("#hf-q").fill(REPO + ":Q5_K_M@v2")
    page.keyboard.press("Enter")
    expect(page.locator("#mf-hubmsg.banner.bad")).to_contain_text("cannot reach the Hub")
    assert server.files_asked[-1] == REPO + "@v2", server.files_asked
    server.files_fail = None
    page.locator("#hf-retry").click()
    expect(page.locator("#mf-files .mrow.sel")).to_have_attribute("data-name", REPO + ":Q5_K_M@v2")
    assert server.files_asked[-1] == REPO + "@v2", server.files_asked
    page.locator("#hf-ep-toggle").click()
    page.locator('#hf-epbar .chip[data-ep="https://hf-mirror.com"]').click()
    expect(page.locator("#hf-ep-toggle")).to_contain_text("hf-mirror.com")
    expect(page.locator("#mf-files .mrow.sel")).to_have_attribute("data-name", REPO + ":Q5_K_M@v2")
    assert server.files_asked[-1] == REPO + "@v2", server.files_asked
    page.locator("#hf-ep-reset").click()
    expect(page.locator("#hf-ep-toggle")).to_contain_text("huggingface.co")
    page.locator("#hf-ep-toggle").click()
    page.locator("#hf-q").fill("")
    expect(page.locator("#hf-list .prow")).to_have_count(3)


def check_local(page, server):
    """The library's other side: folders, the models in each under it, one loaded, folders added and removed."""
    selector = lambda path: path.replace("\\", "\\\\")
    view(page, "library")
    page.locator('#src-seg [data-src="local"]').click()
    folders, models = page.locator("#lc-list .prow.folder"), page.locator("#lc-list .prow.model")
    expect(folders).to_have_count(3)  # where Hugging Face downloads to, the model directory, the served model's folder
    expect(folders.nth(0).locator(".ps")).to_have_text(CACHE)
    expect(folders.nth(1).locator(".ps")).to_have_text(FOLDER)
    expect(models).to_have_count(9)  # #4 added a cached Hub GGUF drafter, #31 a local one
    one = page.locator('#lc-list .prow.model[data-model="' + REPO + ':IQ3_XXS"]')
    expect(one.locator(".pn")).to_have_text("Tiny-27B-GGUF · IQ3_XXS")
    expect(one.locator(".pm")).to_have_text("11.8 GB")
    expect(one.locator(".pt")).to_have_text("GGUF")
    expect(page.locator('#lc-list .prow.model[data-model$="tiny-bf16.flyx"] .pt')).to_have_text(".flyx")
    expect(page.locator('#lc-list .prow.model[data-model$="mmproj-BF16.gguf"] .pt')).to_have_text("mmproj")
    expect(page.locator('#lc-list .prow.model[data-model="acme/Tiny-27B-DFlash2"] .pt')).to_have_text("drafter")
    expect(page.locator('#lc-list .prow.model[data-model="acme/Model-DFlash2-GGUF:BF16"] .pt')).to_have_text("drafter")
    # Every file under a folder carries a kind-specific icon (model, vision projector, drafter).
    expect(page.locator('#lc-list .prow.model[data-model$="tiny-bf16.flyx"] .pn svg use')).to_have_attribute("href", "#i-model")
    expect(page.locator('#lc-list .prow.model[data-model$="mmproj-BF16.gguf"] .pn svg use')).to_have_attribute("href", "#i-image")
    expect(page.locator('#lc-list .prow.model[data-model="acme/Tiny-27B-DFlash2"] .pn svg use')).to_have_attribute("href", "#i-bolt")
    expect(page.locator('#lc-list .prow.model[data-model="acme/Model-DFlash2-GGUF:BF16"] .pn svg use')).to_have_attribute("href", "#i-bolt")
    one.click()
    expect(page.locator("#lc-name")).to_have_text("Tiny-27B-GGUF · IQ3_XXS")
    expect(page.locator("#lc-detail .dchips")).to_contain_text("+ Vision")
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
    # Drafters and vision projectors use the bottom-right primary button instead of a duplicate banner button.
    page.locator('#lc-list .prow.model[data-model="acme/Tiny-27B-DFlash2"]').click()
    expect(page.locator("#lc-detail .banner button")).to_have_count(0)
    expect(page.locator("#lc-detail .dsec")).to_have_text("Folder")
    expect(page.locator("#mf-load")).to_have_text("Draft with it")
    expect(page.locator("#mf-load")).to_be_enabled()
    # A cached Hub GGUF drafter shows File (not Folder), a DFlash2 chip, and Draft with it (never Load model).
    page.locator('#lc-list .prow.model[data-model="acme/Model-DFlash2-GGUF:BF16"]').click()
    expect(page.locator("#lc-detail .dchips")).to_contain_text("DFlash2")
    expect(page.locator("#lc-detail .dsec")).to_have_text("File")
    expect(page.locator("#lc-quants")).to_be_visible()
    expect(page.locator("#mf-load")).to_have_text("Draft with it")
    expect(page.locator("#mf-load")).to_be_enabled()
    page.locator('#lc-list .prow.model[data-model$="mmproj-BF16.gguf"]').click()
    expect(page.locator("#lc-detail")).to_contain_text("vision projector")
    expect(page.locator("#lc-detail .banner button")).to_have_count(0)
    expect(page.locator("#mf-load")).to_have_text("Use as projector")
    expect(page.locator("#mf-load")).to_be_enabled()
    page.locator("#mf-load").click()
    expect(page.locator("#mv-current")).to_be_visible()
    expect(page.locator("#vs-file")).to_have_value(FOLDER + "\\mmproj-BF16.gguf")
    page.locator("#mf-reset").click()
    view(page, "library")
    page.locator('#src-seg [data-src="local"]').click()
    folders.nth(1).click()
    page.locator("#mf-path").fill(LOCAL_DRAFTER["path"])
    page.locator("#mf-path").press("Enter")
    expect(page.locator("#mf-load")).to_have_text("Draft with it")
    page.locator("#mf-load").click()
    confirm(page, "#mf-reload", "Reload")
    body = server.posted[-1]
    assert (body["drafter"], body["drafters"]) == ("dflash2", [LOCAL_DRAFTER["path"]]), body
    page.locator("#mf-reset").click()
    view(page, "library")
    folders.nth(1).click()
    expect(page.locator("#lc-companions .mrow")).to_have_count(2)
    page.locator('#lc-companions .mrow[data-comp$="helper.gguf"]').click()
    page.locator('#mf-cached .mrow[data-name$="Other-12B-Q5_K_M.gguf"]').click()
    page.locator('#lc-companions .mrow[data-comp$="mmproj-BF16.gguf"]').click()
    confirm(page, "#mf-load", "Switch")
    assert (server.posted[-1]["drafter"], server.posted[-1]["drafters"]) == ("dflash2", [FOLDER + "\\helper.gguf"])
    assert (server.posted[-1]["model"], server.posted[-1]["mmproj"]) == (
        FOLDER + "\\Other-12B-Q5_K_M.gguf", FOLDER + "\\mmproj-BF16.gguf"), server.posted[-1]
    view(page, "library")
    page.locator('#src-seg [data-src="local"]').click()
    folders.nth(0).click()
    expect(page.locator("#mf-path")).to_have_value(CACHE)
    expect(page.locator("#mf-cached .rgroup")).to_have_count(4)
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


def check_switch_companions(open_page, want):
    """Switching to another model drops the served model's companions and API id unless picked or edited for the switch."""
    served = dict(want, model=FOLDER + "\\Qwen3.8-27B.gguf", name="served-alias",
                  drafter="dflash2", drafters=["acme/Tiny-27B-DFlash2"],
                  mmproj=FOLDER + "\\mmproj-Qwen3.8.gguf",
                  vision_device="host", vision_quant="nvfp4", vision_max_tokens=1024)
    page, server = open_page(telemetry(served))
    # Switching to a different local model clears stale mmproj, drafter, and API id while keeping general settings.
    view(page, "library")
    page.locator('#src-seg [data-src="local"]').click()
    page.locator('#lc-list .prow.model[data-model$="Other-12B-Q5_K_M.gguf"]').click()
    expect(page.locator("#mf-chosen-s")).to_contain_text("drafting default")
    expect(page.locator("#mf-chosen-s")).not_to_contain_text("mmproj-Qwen3.8.gguf")
    expect(page.locator("#mf-chosen-s")).not_to_contain_text("Tiny-27B-DFlash2")
    expect(page.locator("#mf-strange")).to_have_count(0)
    expect(page.locator("#mf-strange-vis")).to_have_count(0)
    confirm(page, "#mf-load", "Switch")
    body = server.posted[-1]
    assert body["model"] == FOLDER + "\\Other-12B-Q5_K_M.gguf", body
    assert (body["mmproj"], body["drafter"], body["drafters"], body["name"]) == (None, "", [], None), body
    assert (body["ctx"], body["kv"], body["kv_gpu"], body["kv_ram"], body["speculative"]) == (
        want["ctx"], want["kv"], want["kv_gpu"], want["kv_ram"], True), body
    assert (body["vision"], body["vision_device"], body["vision_quant"], body["vision_max_tokens"]) == (
        True, "host", "nvfp4", 1024), body
    # The command line for the chosen other model agrees: no stale companion or alias flags.
    args = command(page)
    assert args[2] == FOLDER + "\\Other-12B-Q5_K_M.gguf", args
    assert not ({"--mmproj", "--drafter", "--drafter-file", "--name"} & set(args)), args
    # Switching to a Hugging Face model likewise drops stale companions.
    view(page, "library")
    page.locator('#src-seg [data-src="hub"]').click()
    page.locator('#mf-files .mrow[data-name$=":Q5_K_M"]').click()
    confirm(page, "#mf-load", "Switch")
    body = server.posted[-1]
    assert body["model"] == REPO + ":Q5_K_M", body
    assert (body["mmproj"], body["drafter"], body["drafters"], body["name"]) == (None, "", [], None), body
    # Explicitly picking a new projector for the switch keeps that projector while still dropping stale drafter/name.
    view(page, "library")
    page.locator('#src-seg [data-src="local"]').click()
    page.locator('#lc-list .prow.model[data-model$="mmproj-BF16.gguf"]').click()
    page.locator("#mf-load").click()
    view(page, "library")
    page.locator('#lc-list .prow.model[data-model$="Other-12B-Q5_K_M.gguf"]').click()
    expect(page.locator("#mf-chosen-s")).to_contain_text("mmproj-BF16.gguf")
    expect(page.locator("#mf-chosen-s")).to_contain_text("drafting default")
    confirm(page, "#mf-load", "Switch")
    body = server.posted[-1]
    assert (body["mmproj"], body["drafter"], body["drafters"], body["name"]) == (
        FOLDER + "\\mmproj-BF16.gguf", "", [], None), body
    # Editing the drafter and API id keeps those edits while still dropping the untouched served mmproj.
    page.locator("#mf-reset").click()
    page.locator('#lset-tabs [data-stab="general"]').click()
    page.locator("#mf-name").fill("custom-alias")
    page.locator('#lset-tabs [data-stab="spec"]').click()
    page.locator("#sd-file").fill("./new-mtp.gguf")
    expect(page.locator("#mf-cmd")).to_contain_text("--drafter mtp")
    view(page, "library")
    page.locator('#lc-list .prow.model[data-model$="Other-12B-Q5_K_M.gguf"]').click()
    confirm(page, "#mf-load", "Switch")
    body = server.posted[-1]
    assert (body["mmproj"], body["drafter"], body["drafters"], body["name"]) == (
        None, "mtp", ["./new-mtp.gguf"], "custom-alias"), body  # its kind as its fields say (DRAFTER_KINDS)
    # Reloading the same model still preserves all of its own companions and API id.
    page.locator("#mf-reset").click()
    page.locator('#lset-tabs [data-stab="general"]').click()
    page.locator("#mf-ctx").fill("4096")
    confirm(page, "#mf-reload", "Reload")
    body = server.posted[-1]
    assert body["model"] == served["model"] and body["ctx"] == 4096, body
    assert (body["mmproj"], body["drafter"], body["drafters"], body["name"]) == (
        served["mmproj"], "dflash2", ["acme/Tiny-27B-DFlash2"], "served-alias"), body
    page.close()

    # Explicitly re-picking the same mmproj via "Use as projector" carries it onto a compatible switch target.
    same_mm = dict(served, mmproj=FOLDER + "\\mmproj-BF16.gguf")
    page, server = open_page(telemetry(same_mm))
    view(page, "library")
    page.locator('#src-seg [data-src="local"]').click()
    page.locator('#lc-list .prow.model[data-model$="mmproj-BF16.gguf"]').click()
    page.locator("#mf-load").click()
    view(page, "library")
    page.locator('#lc-list .prow.model[data-model$="Other-12B-Q5_K_M.gguf"]').click()
    confirm(page, "#mf-load", "Switch")
    body = server.posted[-1]
    assert (body["mmproj"], body["drafter"], body["drafters"]) == (FOLDER + "\\mmproj-BF16.gguf", "", []), body
    page.close()

    # A served model with a multi-file saved drafter config drops it on switch while keeping speculative=False.
    saved_draft = dict(want, model="models/local.gguf", speculative=False,
                       drafter="all", drafters=["./drafts/DFlash model.flyx", "./drafts/MTP layer.gguf"])
    page, server = open_page(telemetry(saved_draft))
    view(page, "library")
    page.locator('#mf-files .mrow[data-name$=":Q5_K_M"]').click()
    confirm(page, "#mf-load", "Switch")
    body = server.posted[-1]
    assert (body["speculative"], body["drafter"], body["drafters"]) == (False, "", []), body
    page.close()


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
    # kv "auto": the engine's default, not the server's --kv (null would keep that);
    # vram_reserve/ram_reserve stay null so CLI --vram-reserve / --ram-reserve are preserved on first load.
    assert (body["model"], body["ctx"], body["kv"], body["kv_ram"], body["kv_gpu"], body["speculative"], body["drafter"], body["drafters"]) == \
        (REPO + ":Q5_K_M", None, "auto", "auto", None, True, "", []), body
    assert body["vram_reserve"] is None and body["ram_reserve"] is None, body
    expect(page.locator("#mf-settings")).to_have_count(0)
    expect(page.locator("#ld-reload")).to_be_hidden()
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


def check_lora(open_page, want):
    """A LoRA adapter: listed with its folder's models (kind LoRA, never loaded alone), used with the model served (the
    setting filled in), what a reload posts and the command line says, and shown with the model once loaded."""
    adapter = dir_model(FOLDER, "my-lora", int(0.23 * GB), kind="lora")
    CATALOG["models"].append(adapter)
    FOLDERS[FOLDER]["models"].append(adapter)
    try:
        page, server = open_page(telemetry(want))
        view(page, "library")
        page.locator('#src-seg [data-src="local"]').click()
        row = page.locator('#lc-list .prow.model[data-model$="my-lora"]')
        expect(row.locator(".pt")).to_have_text("LoRA")
        expect(row.locator(".pn svg use")).to_have_attribute("href", "#i-plus")
        row.click()
        expect(page.locator("#lc-detail")).to_contain_text("my-lora is a LoRA adapter (0.2 GB), not a model")
        expect(page.locator("#lc-detail .banner button")).to_have_count(0)
        expect(page.locator("#mf-load")).to_have_text("Use with the model")
        page.locator("#mf-load").click()
        expect(page.locator("#mv-current")).to_be_visible()
        expect(page.locator("#lset-lora")).to_be_visible()  # its own tab, beside the vision encoder's
        lora_tab = page.locator('#lset-tabs [data-stab="lora"]')
        expect(lora_tab.locator(".sum")).to_have_text("my-lora")
        expect(lora_tab.locator(".chg")).to_have_text("+1")
        expect(lora_tab.locator(".dot")).not_to_have_class(re.compile(r"\bon\b"))
        expect(page.locator("#mf-lora")).to_have_value(adapter["path"])
        expect(page.locator('[data-k="lora"] .was')).to_contain_text("none")
        # The library's adapters are listed to pick from, and None.
        rows = page.locator("#lora-list .sdrow")
        expect(rows).to_have_count(2)
        expect(page.locator('#lora-list .sdrow.sel')).to_have_attribute("data-lora", adapter["path"])
        rows.nth(0).click()
        expect(page.locator("#mf-lora")).to_have_value("")
        expect(lora_tab.locator(".sum")).to_have_text("none")
        rows.nth(1).click()
        expect(page.locator("#mf-lora")).to_have_value(adapter["path"])
        args = command(page)
        assert args[args.index("--lora") + 1] == adapter["path"], args
        confirm(page, "#mf-reload", "Reload")
        assert server.posted[-1]["lora"] == adapter["path"] and server.posted[-1]["model"] == want["model"], server.posted[-1]
        # The Hub's adapters for the model's base (its model card's), a repository opened: the one it runs taken, the
        # one it does not with its reason and nothing to take it with.
        page.locator('#lset-tabs [data-stab="lora"]').click()
        expect(page.locator("#lora-base")).to_have_value("acme/Tiny-27B")
        assert server.lora_asked[-1].get("model") == want["model"], server.lora_asked
        repos = page.locator("#lora-hub .lora-repo")
        expect(repos).to_have_count(2)
        expect(repos.nth(1)).to_contain_text("gated")
        repos.nth(0).locator(".lora-repo-h").click()
        ads = page.locator(".lora-repo.open .lora-ad")
        expect(ads).to_have_count(2)
        expect(ads.nth(1)).to_have_class(re.compile(r"\bno\b"))
        expect(ads.nth(1)).to_contain_text("Cannot load: L0.q.lora_A.weight: dtype F8_E4M3")
        expect(ads.nth(1).locator("button")).to_have_count(0)
        expect(ads.nth(0)).to_contain_text("64 layers")
        ads.nth(0).get_by_role("button", name="Use").click()
        expect(page.locator("#mf-lora")).to_have_value("acme/Tiny-27B-coder-lora")
        expect(ads.nth(0).get_by_role("button")).to_have_text("Chosen")
        args = command(page)
        assert args[args.index("--lora") + 1] == "acme/Tiny-27B-coder-lora", args
        confirm(page, "#mf-reload", "Reload")
        assert server.posted[-1]["lora"] == "acme/Tiny-27B-coder-lora", server.posted[-1]
        # Another base asked for: the Hub's answer for it (none here).
        page.locator('#lset-tabs [data-stab="lora"]').click()
        page.locator("#lora-base").fill("acme/Other")
        page.locator("#lora-find").click()
        expect(page.locator("#lora-hub-msg")).to_have_text("Hugging Face lists no LoRA adapter for this model.")
        assert server.lora_asked[-1].get("base") == "acme/Other", server.lora_asked
        # Another model picked, a file whose base is not known: the base typed for the first is not kept, and the field
        # does not show it.
        view(page, "library")
        page.locator('#lc-list .prow.model[data-model$="Other-12B-Q5_K_M.gguf"]').click()
        view(page, "current")
        page.locator('#lset-tabs [data-stab="lora"]').click()
        expect(page.locator("#lora-hub-msg")).to_contain_text("The model's base is not known")
        expect(page.locator("#lora-base")).to_have_value("")
        assert "base" not in server.lora_asked[-1] and server.lora_asked[-1]["model"].endswith("Other-12B-Q5_K_M.gguf"), \
            server.lora_asked
        page.close()
        # Loaded with it: the model's facts name it, and a reload keeps it; emptied, a reload goes without.
        page, server = open_page(telemetry(dict(want, lora=adapter["path"])))
        expect(page.locator("#ld-facts")).to_contain_text("my-lora")
        page.locator('#lset-tabs [data-stab="lora"]').click()
        expect(page.locator('#lset-tabs [data-stab="lora"] .dot')).to_have_class(re.compile(r"\bon\b"))
        expect(page.locator("#ld-lora-src")).to_contain_text(adapter["path"])
        expect(page.locator("#mf-lora")).to_have_value(adapter["path"])
        assert "--lora" in command(page)
        page.locator("#mf-lora").fill("")
        expect(page.locator('[data-k="lora"] .was')).to_contain_text("my-lora")
        confirm(page, "#mf-reload", "Reload")
        assert server.posted[-1]["lora"] is None, server.posted[-1]
        # Another model of the library does not take the served model's adapter along.
        page.locator("#mf-reset").click()
        view(page, "library")
        page.locator('#src-seg [data-src="local"]').click()
        page.locator('#lc-list .prow.model[data-model$="Other-12B-Q5_K_M.gguf"]').click()
        confirm(page, "#mf-load", "Switch")
        assert server.posted[-1]["lora"] is None, server.posted[-1]
        page.close()
    finally:
        CATALOG["models"].remove(adapter)
        FOLDERS[FOLDER]["models"].remove(adapter)


def check_local_posix(open_page, want):
    """POSIX Local tab: case-sensitive folders (/Models vs /models), literal backslashes, root '/', --models-dir == cacheDir merging, and revision-pinned Hub jumps."""
    posix_cache = "/home/user/.cache/huggingface/hub"
    posix_catalog = {
        "cache": posix_cache,
        "cache_from": "default",
        "win_paths": False,
        "dirs": [posix_cache, "/Models", "/models", "/models\\", "/"],
        "models": [
            {"name": REPO + ":Q4_K_M", "kind": "gguf", "source": "hub", "repo": REPO, "quant": "Q4_K_M",
             "file": "Tiny-27B-Q4_K_M.gguf", "path": posix_cache + "/models--acme--Tiny-27B-GGUF/snapshots/abc/Tiny-27B-Q4_K_M.gguf",
             "size": int(17.6 * GB), "modified": 1.7e9, "revision": "abc", "current": True, "mtp": None, "mmproj": None},
            {"name": REPO + ":Q4_K_M@oldrev123", "kind": "gguf", "source": "hub", "repo": REPO, "quant": "Q4_K_M",
             "file": "Tiny-27B-Q4_K_M.gguf", "path": posix_cache + "/models--acme--Tiny-27B-GGUF/snapshots/oldrev123/Tiny-27B-Q4_K_M.gguf",
             "size": int(17.6 * GB), "modified": 1.6e9, "revision": "oldrev123", "current": False, "mtp": None, "mmproj": None},
            {"name": posix_cache + "/loose.gguf", "kind": "gguf", "source": "dir", "repo": None, "quant": "Q8_0",
             "file": "loose.gguf", "path": posix_cache + "/loose.gguf", "root": posix_cache,
             "size": int(2.1 * GB), "modified": 1.7e9, "revision": None, "current": True, "mtp": None},
            {"name": "/Models/Upper-7B.gguf", "kind": "gguf", "source": "dir", "repo": None, "quant": "Q4_K_M",
             "file": "Upper-7B.gguf", "path": "/Models/Upper-7B.gguf", "root": "/Models",
             "size": int(4.1 * GB), "modified": 1.7e9, "revision": None, "current": True, "mtp": None},
            {"name": "/models/lower-7B.gguf", "kind": "gguf", "source": "dir", "repo": None, "quant": "Q4_K_M",
             "file": "lower-7B.gguf", "path": "/models/lower-7B.gguf", "root": "/models",
             "size": int(4.2 * GB), "modified": 1.7e9, "revision": None, "current": True, "mtp": None},
            {"name": "/models\\/a\\b.gguf", "kind": "gguf", "source": "dir", "repo": None, "quant": "Q4_K_M",
             "file": "a\\b.gguf", "path": "/models\\/a\\b.gguf", "root": "/models\\",
             "size": int(4.3 * GB), "modified": 1.7e9, "revision": None, "current": True, "mtp": None},
            {"name": "/root-model.gguf", "kind": "gguf", "source": "dir", "repo": None, "quant": "Q4_K_M",
             "file": "root-model.gguf", "path": "/root-model.gguf", "root": "/",
             "size": int(3.0 * GB), "modified": 1.7e9, "revision": None, "current": True, "mtp": None},
        ],
    }
    posix_folders = {
        posix_cache: {"path": posix_cache, "parent": "/home/user/.cache/huggingface", "kind": "dir", "dirs": [],
                      "models": [posix_catalog["models"][2]]},
        "/Models": {"path": "/Models", "parent": "/", "kind": "dir", "dirs": [],
                    "models": [posix_catalog["models"][3]]},
        "/models": {"path": "/models", "parent": "/", "kind": "dir", "dirs": [],
                    "models": [posix_catalog["models"][4]]},
        "/models\\": {"path": "/models\\", "parent": "/", "kind": "dir", "dirs": [],
                      "models": [posix_catalog["models"][5]]},
        "/": {"path": "/", "parent": None, "kind": "dir", "dirs": [],
              "models": [posix_catalog["models"][6]]},
    }
    served = dict(want, model="/models/lower-7B.gguf")
    page, srv = open_page(telemetry(served), catalog=posix_catalog, folders=posix_folders)
    view(page, "library")
    page.locator('#src-seg [data-src="local"]').click()
    # All 5 POSIX folders stay distinct: cache, /Models, /models, /models\ (literal backslash), and root '/'.
    folders = page.locator("#lc-list .prow.folder")
    expect(folders).to_have_count(5)
    expect(page.locator('#lc-list .prow.folder[data-place="dir:' + posix_cache + '"]')).to_have_count(1)
    expect(page.locator('#lc-list .prow.folder[data-place="dir:/Models"]')).to_have_count(1)
    expect(page.locator('#lc-list .prow.folder[data-place="dir:/models"] .pn')).to_contain_text("models")
    bs_folder = page.locator(r'#lc-list .prow.folder[data-place="dir:/models\\"]')
    expect(bs_folder).to_have_count(1)
    expect(bs_folder.locator(".pn")).to_have_text("models\\")
    expect(page.locator('#lc-list .prow.folder[data-place="dir:/"]')).to_have_count(1)
    # Every model is visible, including /models\/a\b.gguf and both Hub cache revisions.
    expect(page.locator("#lc-list .prow.model")).to_have_count(7)
    expect(page.locator('#lc-list .prow.model[data-model="' + REPO + ':Q4_K_M"]')).to_have_count(1)
    expect(page.locator('#lc-list .prow.model[data-model="' + REPO + ':Q4_K_M@oldrev123"]')).to_have_count(1)
    expect(page.locator('#lc-list .prow.model[data-model="' + posix_cache + '/loose.gguf"]')).to_have_count(1)
    expect(page.locator('#lc-list .prow.model[data-model="/Models/Upper-7B.gguf"]')).to_have_count(1)
    expect(page.locator('#lc-list .prow.model[data-model="/models/lower-7B.gguf"]')).to_have_count(1)
    bs_model = page.locator(r'#lc-list .prow.model[data-model="/models\\/a\\b.gguf"]')
    expect(bs_model).to_have_count(1)
    expect(bs_model.locator(".pn-t")).to_have_text("a\\b.gguf")
    expect(page.locator('#lc-list .prow.model[data-model="/root-model.gguf"]')).to_have_count(1)
    # Clicking /models\ folder and pressing #mf-up navigates to '/' without stripping the literal backslash.
    bs_folder.click()
    expect(page.locator("#mf-path")).to_have_value("/models\\")
    expect(page.locator("#mf-cached")).to_contain_text("a\\b.gguf")
    page.locator("#mf-up").click()
    expect(page.locator("#mf-path")).to_have_value("/")
    expect(page.locator("#mf-cached")).to_contain_text("root-model.gguf")
    # Clicking the root-level model shows folder '/' cleanly without mangling it to empty string.
    page.locator('#lc-list .prow.model[data-model="/root-model.gguf"]').click()
    expect(page.locator("#lc-path")).to_contain_text("/root-model.gguf")
    # Clicking the cache folder lists both the Hub repo group and the loose local file in cacheDir.
    page.locator('#lc-list .prow.folder[data-place="dir:' + posix_cache + '"]').click()
    expect(page.locator("#mf-cached")).to_contain_text("Tiny-27B-GGUF")
    expect(page.locator("#mf-cached")).to_contain_text("loose.gguf")
    # Searching a named drafter repo in Hub, switching to Local, and clicking "All its quantizations" on an older
    # cached revision clears the stale search query, preserves @oldrev123 across the 350ms search debounce, and uses
    # its distinct quantizations and verified drafters instead of main's.
    page.locator('#src-seg [data-src="hub"]').click()
    page.locator("#hf-q").fill("acme/Model-DFlash2")
    page.keyboard.press("Enter")
    expect(page.locator("#hf-name")).to_have_text("Model-DFlash2")
    page.locator('#src-seg [data-src="local"]').click()
    page.locator('#lc-list .prow.model[data-model="' + REPO + ':Q4_K_M@oldrev123"]').click()
    page.locator("#lc-quants").click()
    expect(page.locator('#src-seg [data-src="hub"]')).to_have_attribute("aria-selected", "true")
    expect(page.locator("#hf-q")).to_have_value("")
    expect(page.locator("#hf-name")).to_have_text("Tiny-27B-GGUF")
    expect(page.locator('#mf-files .mrow[data-name="' + REPO + ':Q2_K@oldrev123"]')).to_have_count(1)
    expect(page.locator("#hf-drafters")).to_contain_text("acme/OldRev-DFlash2")
    expect(page.locator("#hf-drafters")).not_to_contain_text("acme/Tiny-27B-DFlash2")
    page.wait_for_timeout(600)  # survive past the 350ms Hub search debounce without reverting to main or list[0]
    expect(page.locator("#hf-name")).to_have_text("Tiny-27B-GGUF")
    expect(page.locator('#mf-files .mrow[data-name="' + REPO + ':Q2_K@oldrev123"]')).to_have_count(1)
    expect(page.locator("#hf-drafters")).to_contain_text("acme/OldRev-DFlash2")
    expect(page.locator("#hf-drafters")).not_to_contain_text("acme/Tiny-27B-DFlash2")
    assert srv.files_asked[-1] == REPO + "@oldrev123", srv.files_asked
    # Selecting the pinned-revision quantization populates verifiedOf keyed by repo@revision in Settings -> Speculation.
    page.locator('#mf-files .mrow[data-name="' + REPO + ':Q2_K@oldrev123"]').click()
    view(page, "current")
    page.locator('#lset-tabs [data-stab="spec"]').click()
    page.locator("#sd-made").click()
    expect(page.locator('[data-verified="acme/OldRev-DFlash2"]')).to_have_count(1)
    # Item 5: when /v1/hub/search fails (502) in the background after jumping from Local -> "All its quantizations"
    # on a cached older revision, the pinned revision detail, its quantizations, and its Load target remain usable.
    srv.hub_fail = True
    view(page, "library")
    page.locator('#src-seg [data-src="local"]').click()
    page.locator('#lc-list .prow.model[data-model="' + REPO + ':Q4_K_M@oldrev123"]').click()
    page.locator("#lc-quants").click()
    expect(page.locator('#src-seg [data-src="hub"]')).to_have_attribute("aria-selected", "true")
    expect(page.locator("#hf-msg")).to_contain_text("cannot reach the Hub")
    page.wait_for_timeout(600)  # past the 350ms search debounce while /v1/hub/search fails with 502
    expect(page.locator("#hf-name")).to_have_text("Tiny-27B-GGUF")
    expect(page.locator('#mf-files .mrow[data-name="' + REPO + ':Q2_K@oldrev123"]')).to_have_count(1)
    page.locator('#mf-files .mrow[data-name="' + REPO + ':Q2_K@oldrev123"]').click()
    expect(page.locator("#mf-load")).to_be_enabled()
    # Explicitly typing a new query unpins the cached revision so a failed search clears stale detail.
    page.locator("#hf-q").fill("nonexistent")
    page.wait_for_timeout(600)
    expect(page.locator("#hf-name")).to_have_count(0)
    page.close()

    # POSIX served model and LoRA with literal backslashes: Serving card LoRA label stays "a\b" and initial Local
    # directory fetch requests "/models" (never "/models/a").
    served_bs = dict(want, model="/models/a\\b.gguf", lora="/loras/a\\b")
    page2, srv2 = open_page(telemetry(served_bs), catalog=posix_catalog, folders=posix_folders)
    lora_fact = page2.locator("#ld-facts div", has=page2.locator("dt", has_text="LoRA")).locator("dd")
    expect(lora_fact).to_have_text("a\\b")
    view(page2, "library")
    page2.locator('#src-seg [data-src="local"]').click()
    expect(page2.locator("#lc-list .prow.folder")).to_have_count(5)
    assert "/models" in srv2.dir_asked and "/models/a" not in srv2.dir_asked, srv2.dir_asked
    page2.close()


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
    check_commands(page, open_page, want)
    check_drafters(page, server, want)
    check_vision(page, server, open_page, want)
    page.close()
    check_context_limit(open_page, want)
    check_degraded_and_draft_weights(open_page, want)
    check_vram_apps(open_page, want)
    check_switch_companions(open_page, want)
    check_lora(open_page, want)
    check_local_posix(open_page, want)
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
    print("The Model page kept its load settings and shell arguments, loaded from Hugging Face and from this machine's folders, "
          "and took a LoRA adapter.")
