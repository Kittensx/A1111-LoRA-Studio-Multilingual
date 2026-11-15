"""Portable input validation for the archived v10-v15 build scripts.

These scripts reconstruct old candidate packages from an independently prepared
source checkout, backup snapshot and output package. They are NOT the installer.
They must never write to the present, published project tree by accident.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def resolve_build_paths(version, *, needs_backup=False, previous=None, source_paths=(), backup_paths=()):
    parser = argparse.ArgumentParser(
        description=f"Reproduce historical LoRA Studio v{version} in a separate output package."
    )
    parser.add_argument("--source", default=os.environ.get("LORA_STUDIO_SOURCE"),
                        help="A1111 source checkout (or LORA_STUDIO_SOURCE)")
    parser.add_argument("--output", default=os.environ.get("LORA_STUDIO_OUTPUT"),
                        help="Independent writable copy of the previous release (or LORA_STUDIO_OUTPUT)")
    if previous is not None:
        parser.add_argument("--base", default=os.environ.get("LORA_STUDIO_BASE") or str(REPO.parent / previous),
                            help=f"Original v{version-1} package, default sibling directory (or LORA_STUDIO_BASE)")
    if needs_backup:
        parser.add_argument("--backup", "--before", dest="backup", default=os.environ.get("LORA_STUDIO_BACKUP"),
                            help="Original pre-change backup tree (or LORA_STUDIO_BACKUP)")
    args = parser.parse_args()
    for key in (("source", "output", "backup") if needs_backup else ("source", "output")):
        if getattr(args, key, None) is None:
            parser.error(f"--{key} is required (or configure the corresponding LORA_STUDIO_* environment variable)")
    for key in ("source", "output", "backup", "base"):
        value = getattr(args, key, None)
        if value is not None:
            setattr(args, key, Path(value).expanduser().resolve())
    if not args.source.is_dir():
        parser.error(f"A1111 source directory not found: {args.source}")
    if not args.output.is_dir() or not (args.output / "manifest.json").is_file():
        parser.error("--output must be an existing separate copy of a prior package containing manifest.json")
    protected = [REPO.resolve(), args.source]
    for key in ("base", "backup"):
        value = getattr(args, key, None)
        if value is not None:
            if not value.is_dir():
                parser.error(f"--{key} directory not found: {value}")
            protected.append(value)
    for root in protected:
        if args.output == root or root in args.output.parents or args.output in root.parents:
            parser.error(f"Output must be isolated from the repository and input trees: {args.output} overlaps {root}")
    for rel in source_paths:
        if not (args.source / rel).is_file():
            parser.error(f"Required source file is missing: {args.source / rel}")
    for rel in backup_paths:
        if not (args.backup / rel).is_file():
            parser.error(f"Required backup file is missing: {args.backup / rel}")
    return args
