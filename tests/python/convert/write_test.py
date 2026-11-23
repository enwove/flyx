# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""A failed write keeps existing model files and removes unfinished outputs."""
from contextlib import contextmanager
import errno
import os
from pathlib import Path
import stat
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python" / "convert"))
from flyx_convert import fileformat, Writer


@contextmanager
def full_disk(path, mode):
    with open(path, mode) as stream:
        class PartialWrite:
            def write(self, data):
                stream.write(data[:16])
                raise OSError(errno.ENOSPC, "disk full")
        yield PartialWrite()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="flyx write ") as tmp:
        directory = Path(tmp)
        target = directory / "existing.flyx"
        writer = Writer()
        writer.add_blob("first", b"already converted")
        writer.write(target)
        original = target.read_bytes()
        writer.add_blob("next", b"new model data")
        with patch.object(fileformat, "open", full_disk, create=True):
            for path in (target, directory / "fresh.flyx"):
                try:
                    writer.write(path)
                except OSError as error:
                    assert error.errno == errno.ENOSPC
                else:
                    raise AssertionError("expected the write to fail")
        with patch.object(os, "replace", side_effect=PermissionError("destination is in use")):
            try:
                writer.write(target)
            except PermissionError:
                pass
            else:
                raise AssertionError("expected replacement to fail")
        assert list(directory.iterdir()) == [target] and target.read_bytes() == original
        if os.name == "posix":
            target.chmod(0o640)
            link = directory / "latest.flyx"
            link.symlink_to(target.name)
            writer.write(link)
            assert link.is_symlink() and stat.S_IMODE(target.stat().st_mode) == 0o640
        else:
            writer.write(target)
        assert target.read_bytes() != original
