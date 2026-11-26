# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Check the Windows release ZIP with independent readers."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


def main(shell):
    script = Path(__file__).resolve().parents[2] / "packaging/create_release_zip.ps1"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "release package"
        (root / "lib/flyx").mkdir(parents=True)
        payload = b"backend fixture\x00"
        (root / "lib/flyx/backend.dll").write_bytes(payload)
        (root / "empty").mkdir()
        archive = Path(tmp) / "release.zip"
        powershell = [shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass"]
        subprocess.run([*powershell, "-File", script, "-Package", root, "-Destination", archive], check=True)
        with zipfile.ZipFile(archive) as z:
            assert all("\\" not in name for name in z.namelist()), z.namelist()
            assert z.read("release package/lib/flyx/backend.dll") == payload
            assert z.getinfo("release package/empty/").is_dir()
        subprocess.run([*powershell, "-Command",
                        "Expand-Archive -LiteralPath release.zip -DestinationPath expanded -ErrorAction Stop"],
                       cwd=tmp, check=True)
        assert (Path(tmp) / "expanded/release package/lib/flyx/backend.dll").read_bytes() == payload
        if unzip := shutil.which("unzip"):
            subprocess.run([unzip, "-q", archive, "-d", Path(tmp) / "out"], check=True)
            assert (Path(tmp) / "out/release package/lib/flyx/backend.dll").read_bytes() == payload


if __name__ == "__main__":
    main(sys.argv[1])
