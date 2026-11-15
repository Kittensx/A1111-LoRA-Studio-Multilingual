#!/usr/bin/env python3
"""Verify the committed Git archive, not merely the working directory.

Run inside a Git checkout after committing your changes:
    python -B tools/verify_git_archive.py

Requires Git and Python's standard library only. Never writes to the checkout.
"""
from __future__ import annotations

import io
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = (
    ("runtime SHA-256", "verify"),
    ("English/Russian documents", "docs-audit"),
    ("contextual help", "help-audit"),
)


def main():
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                           text=True, capture_output=True, check=True)
    if dirty.stdout.strip():
        print("ERROR: commit or discard working-tree changes before verifying the Git archive.", file=sys.stderr)
        return 2
    archived = subprocess.run(["git", "archive", "--format=zip", "HEAD"], cwd=ROOT,
                              capture_output=True, check=True)
    with tempfile.TemporaryDirectory(prefix="lora-studio-git-archive-") as temp:
        destination = Path(temp)
        with zipfile.ZipFile(io.BytesIO(archived.stdout)) as archive:
            for info in archive.infolist():
                name = PurePosixPath(info.filename)
                if (name.is_absolute() or ".." in name.parts or "\\" in info.filename
                        or stat.S_ISLNK(info.external_attr >> 16)):
                    raise ValueError(f"Unsafe archived Git entry: {info.filename}")
            archive.extractall(destination)
        for label, command in COMMANDS:
            result = subprocess.run([sys.executable, "-B", "run.py", command],
                                    cwd=destination, check=False)
            if result.returncode:
                print(f"ERROR: Git archive failed {label} verification.", file=sys.stderr)
                return result.returncode
    print("Git archive verified: runtime assets, documentation, contextual help.")
    print("Verification used committed HEAD, not local uncommitted edits.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, subprocess.CalledProcessError, OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"ERROR: Git archive verification could not finish: {error}", file=sys.stderr)
        raise SystemExit(1)
