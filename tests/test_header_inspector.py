from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest


TOOL = Path(__file__).parents[1] / "tools" / "inspect_lora_header.py"
SPEC = importlib.util.spec_from_file_location("inspect_lora_header", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class HeaderTests(unittest.TestCase):
    def make_file(self, header: object) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "sample.safetensors"
        raw = json.dumps(header).encode("utf-8")
        path.write_bytes(struct.pack("<Q", len(raw)) + raw)
        return path

    def test_reads_metadata_without_torch(self):
        path = self.make_file({"__metadata__": {"ss_network_module": "networks.lora"}, "x.lora_up.weight": {"shape": [2, 3]}})
        self.assertEqual(MODULE.read_header(path)["x.lora_up.weight"]["shape"], [2, 3])

    def test_rejects_unsafe_length(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.safetensors"
            path.write_bytes(struct.pack("<Q", MODULE.MAX_HEADER_BYTES + 1))
            with self.assertRaises(ValueError):
                MODULE.read_header(path)


if __name__ == "__main__":
    unittest.main()
