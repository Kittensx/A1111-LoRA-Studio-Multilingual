"""Optional documentation installation must never become a runtime dependency."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import install  # noqa: E402


class OptionalDocumentationManifestTests(unittest.TestCase):
    def test_package_has_six_optional_docs_and_no_runtime_markdown(self):
        manifest = install.verify_package(ROOT)
        self.assertEqual(len(manifest["documentation_operations"]), 6)
        self.assertFalse(any(op["path"].endswith(".md") and not install.is_runtime_help_document(op["path"]) for op in manifest["operations"]))
        self.assertEqual(len(install.selected_operations(manifest)), len(manifest["operations"]))
        self.assertEqual(len(install.selected_operations(manifest, "EN")), len(manifest["operations"]) + 6)
        self.assertEqual(len(install.selected_operations(manifest, "RUS")), len(manifest["operations"]) + 6)
        for language in ("EN", "RUS"):
            for entry in manifest["documentation_operations"]:
                self.assertEqual(Path(entry["assets"][language]).parent.as_posix(), f"docs/{language}")
                self.assertTrue(entry["assets"][language].endswith(f"_{language}.md"))
                self.assertTrue((ROOT / entry["assets"][language]).is_file())

    def test_unsupported_language_refused(self):
        with self.assertRaises(install.InstallError):
            install.selected_operations({"operations": []}, "FR")


class OptionalDocumentationFlowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="lora-optional-docs-")
        self.addCleanup(temp.cleanup)
        base = Path(temp.name)
        self.package = base / "package"
        self.host = base / "host"
        self.package.mkdir()
        self.host.mkdir()
        required = ("webui.py", "launch.py", "modules/processing.py",
                    "modules/sd_models.py", "modules/extra_networks.py")
        for rel in required:
            dest = self.host / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("# baseline\n", encoding="utf-8")
        (self.host / "extensions-builtin/Lora").mkdir(parents=True)
        assets = []
        doc_hashes = {}
        for rel, data in (
            ("files/style.css", b"NEW STYLE\n"),
            ("docs/EN/EXAMPLE_EN.md", b"English instructions\n"),
            ("docs/RUS/EXAMPLE_RUS.md", "Русские инструкции\n".encode("utf-8")),
        ):
            path = self.package / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            if rel.startswith("docs/"):
                doc_hashes["EN" if "/EN/" in rel else "RUS"] = hashlib.sha256(data).hexdigest()
            else:
                assets.append({"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "role": "runtime"})
        self.manifest = {
            "format": 1, "complete": True, "name": "Optional documentation test",
            "assets": assets,
            "operations": [{"path": "style.css", "asset": "files/style.css", "kind": "replace", "known_before": []}],
            "documentation_operations": [{
                "path": "extensions-builtin/Lora/EXAMPLE.md",
                "assets": {"EN": "docs/EN/EXAMPLE_EN.md", "RUS": "docs/RUS/EXAMPLE_RUS.md"},
                "sha256": doc_hashes,
                "known_before": [],
            }],
            "required_host_files": [],
            "required_runtime_paths": ["style.css"],
        }
        self.save()

    def save(self):
        (self.package / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_default_install_keeps_markdown_out_of_a1111(self):
        ident = install.install(self.host, self.package)
        self.assertEqual((self.host / "style.css").read_bytes(), b"NEW STYLE\n")
        self.assertFalse((self.host / "extensions-builtin/Lora/EXAMPLE.md").exists())
        self.assertEqual(install.rollback(self.host, ident), 1)
        self.assertFalse((self.host / "style.css").exists())

    def test_requested_english_docs_and_rollback(self):
        _, _, planned = install.plan_install(self.host, self.package, docs_language="EN")
        self.assertEqual(len(planned), 2)
        ident = install.install(self.host, self.package, docs_language="EN")
        self.assertEqual((self.host / "extensions-builtin/Lora/EXAMPLE.md").read_bytes(), b"English instructions\n")
        self.assertEqual(install.rollback(self.host, ident), 2)
        self.assertFalse((self.host / "extensions-builtin/Lora/EXAMPLE.md").exists())

    def test_requested_russian_docs(self):
        ident = install.install(self.host, self.package, docs_language="RUS")
        self.assertEqual((self.host / "extensions-builtin/Lora/EXAMPLE.md").read_text(encoding="utf-8"), "Русские инструкции\n")
        self.assertEqual(install.rollback(self.host, ident), 2)

    def test_existing_unrecognized_document_blocks_only_optional_mode(self):
        destination = self.host / "extensions-builtin/Lora/EXAMPLE.md"
        destination.write_text("user created document\n", encoding="utf-8")
        ident = install.install(self.host, self.package)
        self.assertEqual(destination.read_text(encoding="utf-8"), "user created document\n")
        install.rollback(self.host, ident)
        with self.assertRaises(install.InstallError):
            install.install(self.host, self.package, docs_language="EN")
        self.assertEqual(destination.read_text(encoding="utf-8"), "user created document\n")

    def test_corrupt_optional_doc_blocks_optional_install_but_not_code_only(self):
        (self.package / "docs/EN/EXAMPLE_EN.md").write_text("tampered", encoding="utf-8")
        install.verify_package(self.package)  # Documentation is not a runtime dependency.
        with self.assertRaises(install.InstallError):
            install.install(self.host, self.package, docs_language="EN")
        self.assertFalse((self.host / "style.css").exists())
        ident = install.install(self.host, self.package)
        self.assertIsNotNone(ident)
        install.rollback(self.host, ident)

    def test_missing_optional_docs_does_not_break_code_only_install(self):
        (self.package / "docs/RUS/EXAMPLE_RUS.md").unlink()
        install.verify_package(self.package)
        ident = install.install(self.host, self.package)
        self.assertIsNotNone(ident)
        with self.assertRaises(install.InstallError):
            install.plan_install(self.host, self.package, docs_language="RUS")
        install.rollback(self.host, ident)

    def test_invalid_optional_doc_destination_is_rejected(self):
        self.manifest["documentation_operations"][0]["path"] = "config.json"
        self.save()
        with self.assertRaises(install.InstallError):
            install.verify_package(self.package)


if __name__ == "__main__":
    unittest.main()
