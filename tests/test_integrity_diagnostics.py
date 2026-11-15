"""Aggregated integrity reporting must not weaken checks or rewrite payloads."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import install  # noqa: E402


class IntegrityDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="lora-phase3-diagnostics-")
        self.addCleanup(temp.cleanup)
        self.package = Path(temp.name)
        self.assets = [
            ("files/style.css", b"first\r\nsecond\r\n"),
            ("files/extensions-builtin/Lora/lora_schema.py", b"alpha\nbeta\n"),
            ("files/javascript/loraStudioView.js", b"one\ntwo\n"),
        ]
        self.manifest = {
            "format": 1, "complete": True, "name": "Test integrity",
            "assets": [], "operations": [], "required_runtime_paths": [],
        }
        for rel, data in self.assets:
            self.write(rel, data)
            self.manifest["assets"].append({"path": rel, "sha256": install.digest(data), "role": "runtime"})
            self.manifest["operations"].append({
                "path": rel[len("files/"):], "asset": rel, "kind": "replace", "known_before": [],
            })
        self.save_manifest()

    def write(self, name, data):
        path = self.package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def save_manifest(self):
        self.write("manifest.json", json.dumps(self.manifest).encode("utf-8"))

    def test_pristine_package_passes(self):
        self.assertEqual(install.verify_package(self.package)["name"], "Test integrity")

    def test_aggregate_all_mismatches_including_missing_file(self):
        self.write(self.assets[0][0], b"first\nsecond\n")
        self.write(self.assets[1][0], b"alpha\r\nbeta\r\n")
        (self.package / self.assets[2][0]).unlink()
        with self.assertRaises(install.PackageIntegrityError) as caught:
            install.verify_package(self.package)
        issues = caught.exception.issues
        self.assertEqual([item["path"] for item in issues], [x[0] for x in self.assets])
        self.assertIn("3 file(s)", str(caught.exception))
        self.assertIn("found LF/mixed; manifest expects CRLF", issues[0]["diagnosis"])
        self.assertIn("found CRLF; manifest expects LF", issues[1]["diagnosis"])
        self.assertEqual(issues[2]["diagnosis"], "missing")
        self.assertIsNone(issues[2]["actual"])
        self.assertEqual((self.package / self.assets[0][0]).read_bytes(), b"first\nsecond\n")

    def test_real_content_change_does_not_get_false_newline_diagnosis(self):
        self.write(self.assets[0][0], b"first\nDIFFERENT\n")
        with self.assertRaises(install.PackageIntegrityError) as caught:
            install.verify_package(self.package)
        self.assertEqual(len(caught.exception.issues), 1)
        self.assertIn("content/bytes differ", caught.exception.issues[0]["diagnosis"])

    def test_empty_or_binary_payload_change_is_not_called_newline_conversion(self):
        self.write(self.assets[0][0], b"\x00\r\x10\x99")
        issues = install.asset_integrity_issues(self.package, self.manifest["assets"])
        self.assertEqual(len(issues), 1)
        self.assertIn("content/bytes differ", issues[0]["diagnosis"])

    def test_manifest_validation_is_not_skipped(self):
        self.manifest["assets"][0]["sha256"] = "not-a-hash"
        self.save_manifest()
        with self.assertRaisesRegex(install.InstallError, "Invalid/duplicate package entry"):
            install.verify_package(self.package)

    def test_optional_docs_separate_from_code_integrity(self):
        self.manifest["documentation_operations"] = [{
            "path": "extensions-builtin/Lora/GUIDE.md",
            "assets": {"EN": "docs/EN/GUIDE_EN.md", "RUS": "docs/RUS/GUIDE_RUS.md"},
            "sha256": {"EN": install.digest(b"guide\n"), "RUS": install.digest(b"ru\n")},
            "known_before": [],
        }]
        self.write("docs/EN/GUIDE_EN.md", b"guide\r\n")
        self.save_manifest()
        install.verify_package(self.package)  # Optional docs are not runtime prerequisites.
        issues = install.optional_document_integrity_issues(self.package, self.manifest)
        self.assertEqual(len(issues), 2)
        self.assertIn("found CRLF; manifest expects LF", issues[0]["diagnosis"])
        self.assertEqual(issues[1]["diagnosis"], "missing")

    def test_traversal_and_symlink_protection_still_applies(self):
        unsafe = [{"path": "files/../outside.py", "sha256": "a" * 64}]
        with self.assertRaises(install.InstallError):
            install.asset_integrity_issues(self.package, unsafe)


if __name__ == "__main__":
    unittest.main()
