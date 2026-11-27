# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""ctest's gpu_preflight: the GPU not already in use before the GPU tests start (every test labelled gpu needs it).

Tests beside another program's GPU memory fail for its reasons, not theirs: an engine load is refused ("Not enough
VRAM ... used by other apps") and the tests measuring the device's free memory (cuda_device_memory*) read the other
program's allocations as theirs.  So this fails at once, naming who holds the memory, when more than
FLYX_TEST_GPU_MAX_USED_MIB (2560 by default) is in use.  The GPU checked is the one the tests take: the first of
CUDA_VISIBLE_DEVICES, else nvidia-smi's GPU 0.  It fails too on a GPU the CUDA backend cannot run (compute capability
below 8.0: a T4, a V100), one failure that says why instead of every GPU test's.  Without nvidia-smi (or without a GPU
it can read) nothing is checked and the test is skipped (exit 77): the GPU tests then say for themselves what they
find.
"""
import csv
import os
import re
import shutil
import subprocess
import sys

SKIP = 77
LIMIT_VAR = "FLYX_TEST_GPU_MAX_USED_MIB"


def run(command):
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, str(e)
    return (done.stdout if done.returncode == 0 else None), (done.stdout + done.stderr).strip()


def skip(why):
    print(f"gpu_preflight: {why}: the GPU's memory is not checked")
    sys.exit(SKIP)


def windows_memory(used_before, read_used):
    """{pid: MiB} on the GPU from Windows' counters (nvidia-smi reports no process's memory under WDDM)."""
    out, _ = run(["typeperf", r"\GPU Process Memory(*)\Dedicated Usage", r"\GPU Adapter Memory(*)\Dedicated Usage",
                  "-sc", "1"])
    used = [used_before, read_used()]
    rows = [r for r in csv.reader((out or "").splitlines()) if len(r) > 1]
    if len(rows) < 2:
        return {}
    process, adapter = {}, {}
    for name, value in zip(rows[0][1:], rows[1][1:]):
        try:
            mib = float(value) / 2**20
        except ValueError:
            continue
        if m := re.search(r"Process Memory\(pid_(\d+)_luid_(\w+?_phys_\d+)\)", name):
            process.setdefault(m.group(2), {})[int(m.group(1))] = mib
        elif m := re.search(r"Adapter Memory\(luid_(\w+?_phys_\d+)\)", name):
            adapter[m.group(1)] = mib
    if not adapter:
        return {}
    # The counters name adapters by LUID, nvidia-smi by index: this GPU is the adapter using what nvidia-smi says,
    # before or after the sample (a model loading moves it by gigabytes in a second).
    luid = min(adapter, key=lambda a: min(abs(adapter[a] - u) for u in used if u is not None))
    return process.get(luid, {})


def windows_names():
    """{pid: executable name} of every process (a Toolhelp snapshot: tasklist takes seconds on a busy machine)."""
    import ctypes
    from ctypes import wintypes

    class Entry(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("usage", wintypes.DWORD), ("pid", wintypes.DWORD),
                    ("heap", ctypes.c_size_t), ("module", wintypes.DWORD), ("threads", wintypes.DWORD),
                    ("parent", wintypes.DWORD), ("priority", ctypes.c_long), ("flags", wintypes.DWORD),
                    ("exe", ctypes.c_wchar * 260)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    for step in (kernel.Process32FirstW, kernel.Process32NextW):
        step.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)  # TH32CS_SNAPPROCESS
    if snapshot in (None, wintypes.HANDLE(-1).value):
        return {}
    names, entry = {}, Entry()
    entry.size = ctypes.sizeof(Entry)
    more = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
    while more:
        names[entry.pid] = entry.exe
        more = kernel.Process32NextW(snapshot, ctypes.byref(entry))
    kernel.CloseHandle(snapshot)
    return names


def main():
    limit = os.environ.get(LIMIT_VAR, "2560")
    if not limit.isdigit():
        print(f"gpu_preflight: {LIMIT_VAR}={limit!r} is not a number of MiB")
        return 1
    limit = int(limit)
    smi = shutil.which("nvidia-smi")
    if not smi:
        skip("no nvidia-smi on PATH")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    pick = []
    if visible is not None:
        first = visible.split(",")[0].strip()
        if not first or first.startswith("-"):
            skip(f"CUDA_VISIBLE_DEVICES={visible!r} names no GPU")
        pick = ["-i", first]
    out, said = run([smi, "--query-gpu=index,uuid,name,memory.used,memory.total,compute_cap",
                     "--format=csv,noheader,nounits", *pick])
    if not out or not out.strip():
        skip(f"nvidia-smi found no GPU ({said or 'no output'})")
    index, uuid, name, used, total, cap = [f.strip() for f in out.strip().splitlines()[0].split(",")]
    # The CUDA backend needs compute capability 8.0 or newer (CMakeLists.txt, the backend's probe): on an older GPU
    # every GPU test would fail one by one, the kernel tests aborting with no kernel image for the device.
    major = cap.split(".")[0]
    if major.isdigit() and int(major) < 8:
        print(f"gpu_preflight: GPU {index} ({name}) has compute capability {cap}; flyx's CUDA backend needs 8.0 or "
              "newer, so the GPU tests cannot run here")
        return 1
    if not used.isdigit():
        skip(f"nvidia-smi reports no memory in use for GPU {index} ({used})")
    used, gpu = int(used), f"GPU {index} ({name})"

    def read_used():
        now, _ = run([smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", uuid])
        return int(now.strip()) if now and now.strip().isdigit() else None
    if used <= limit:
        print(f"gpu_preflight: {gpu}: {used} of {total} MiB in use, the limit {limit} MiB ({LIMIT_VAR}): ok")
        return 0

    print(f"gpu_preflight: {gpu}: {used} of {total} MiB already in use before the GPU tests, more than "
          f"{limit} MiB ({LIMIT_VAR}).\nBeside it engine loads would be refused (\"Not enough VRAM ... used by "
          "other apps\") and the tests measuring free memory (cuda_device_memory*) would fail.")
    apps, _ = run([smi, "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    procs = {}
    for line in (apps or "").splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) >= 4 and fields[0] == uuid and fields[1].isdigit():
            mib = fields[-1]
            procs[int(fields[1])] = [", ".join(fields[2:-1]), float(mib) if mib.isdigit() else None]
    source, compositor = "nvidia-smi", []
    if sys.platform == "win32" and not any(m for _, m in procs.values()):
        counted = windows_memory(used, read_used)
        if counted:
            source = "Windows' GPU Process Memory counter, read after the check"
            for pid, mib in counted.items():
                procs.setdefault(pid, ["", None])[1] = mib
            # Names nvidia-smi could not read ([Insufficient Permissions]) from the process list.
            names = windows_names()
            for pid, p in procs.items():
                if not p[0] or p[0].startswith("["):
                    p[0] = names.get(pid, p[0] or "?")
            # The desktop compositor's counter takes in the surfaces it shares with every other program (it reads
            # gigabytes on an idle GPU): it says nothing about who holds the memory.
            compositor = [(pid, p) for pid, p in procs.items() if re.search(r"(^|[\\/])dwm\.exe$", p[0], re.I)]
            for pid, _ in compositor:
                del procs[pid]
    held = sorted(procs.items(), key=lambda p: -(p[1][1] or 0))
    if any(mib is not None for _, mib in procs.values()):
        held = [p for p in held if p[1][1] is not None]
    shown = [p for p in held if p[1][1] is None or p[1][1] >= 16][:12]
    if shown:
        print(f"In use on {gpu}, MiB ({source}):")
        for pid, (exe, mib) in shown:
            print(f"  {'?' if mib is None else round(mib):>7}  pid {pid:<7} {exe}")
        rest = [mib or 0 for _, (_, mib) in held[len(shown):]]
        if rest:
            print(f"  ... and {len(rest)} more, {round(max(rest))} MiB or less each")
        for pid, (exe, mib) in compositor:
            print(f"  (pid {pid} {exe}, the desktop's compositor: {round(mib or 0)} MiB counted, the surfaces it "
                  "shares with other programs included)")
    else:
        print("nvidia-smi names no process on it (WSL and containers do not see the host's processes).")
    print(f"Stop what holds it or wait for it, or allow more: {LIMIT_VAR}=<MiB> (docs/testing.md, section 2).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
