#!/usr/bin/env python3
"""Read a safetensors header without loading tensors, Torch, or model code."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys


MAX_HEADER_BYTES = 16 * 1024 * 1024


def read_header(path: Path) -> dict:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("file is too short to be a safetensors file")
        (length,) = struct.unpack("<Q", prefix)
        if not 2 <= length <= MAX_HEADER_BYTES:
            raise ValueError(f"unsafe or invalid safetensors header length: {length}")
        raw = stream.read(length)
        if len(raw) != length:
            raise ValueError("truncated safetensors header")
    header = json.loads(raw)
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be a JSON object")
    return header


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--show", type=int, default=20, metavar="N", help="print at most N tensor keys")
    args = parser.parse_args()
    header = read_header(args.file)
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    metadata = header.get("__metadata__", {})
    print(f"Tensors: {len(tensors)}")
    if metadata:
        print("Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    print("Keys:")
    for key in list(tensors)[:max(args.show, 0)]:
        entry = tensors[key]
        shape = entry.get("shape") if isinstance(entry, dict) else None
        print(f"  {key}  shape={shape}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        sys.exit(f"ERROR: {error}")
