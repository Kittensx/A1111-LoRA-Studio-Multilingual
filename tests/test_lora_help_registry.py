"""Offline tests for help article discovery, localization, and URL/path safety."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "files/extensions-builtin/Lora/lora_help_registry.py"
spec = importlib.util.spec_from_file_location("lora_help_registry", MODULE)
help_registry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(help_registry)
sys.path.insert(0, str(ROOT))
import install  # noqa: E402


class RealHelpFiles(unittest.TestCase):
    def test_index_discovers_nine_topics_with_related_links(self):
        index = help_registry.index("en")
        self.assertEqual(index["schema_version"], 1)
        self.assertEqual(len(index["topics"]), 9)
        topics = {topic["id"] for topic in index["topics"]}
        self.assertIn("getting-started", topics)
        for item in index["topics"]:
            self.assertFalse(item["fallback"])
            self.assertTrue(item["title"])
            self.assertTrue(set(item["related"]).issubset(topics))

    def test_ru_article_is_translated_and_other_languages_fallback(self):
        ru = help_registry.article("getting-started", "ru")
        self.assertEqual(ru["locale"], "ru")
        self.assertIn("Начало работы", ru["title"])
        self.assertFalse(ru["fallback"])
        fr = help_registry.article("getting-started", "fr")
        self.assertEqual(fr["locale"], "en")
        self.assertTrue(fr["fallback"])
        self.assertEqual(help_registry.locale_code("fr-CA"), "fr")
        self.assertEqual(help_registry.locale_code("pt-PT"), "pt-BR")

    def test_runtime_help_docs_are_installed_and_historical_docs_are_not(self):
        m = install.verify_package(ROOT)
        help_ops = [o for o in m["operations"] if install.is_runtime_help_document(o["path"])]
        self.assertEqual(len(help_ops), 27)
        self.assertTrue(all(o["asset"] == "files/" + o["path"] for o in help_ops))
        self.assertTrue(all("/help_documentation/" in o["path"] for o in help_ops))
        self.assertEqual(len(m["documentation_operations"]), 6)

    def test_reject_help_doc_outside_explicit_folder(self):
        self.assertFalse(install.allowed_destination("modules/fake.md"))
        with self.assertRaises(install.InstallError):
            install.allowed_destination("extensions-builtin/Lora/help_documentation/en/../password.md")
        self.assertFalse(install.allowed_destination("extensions-builtin/Lora/help_documentation/en/bad name.md"))
        self.assertFalse(install.allowed_destination("extensions-builtin/Lora/help_documentation/en/bad.html"))
        self.assertTrue(install.is_runtime_help_document("extensions-builtin/Lora/help_documentation/ja/help-me.md"))

    def test_validation_rejects_traversal(self):
        for unsafe in ("../etc/passwd", "start/../../", "", "some_topic", "bad.md", "écrire"):
            with self.subTest(unsafe=unsafe), self.assertRaises(help_registry.HelpError):
                help_registry.article(unsafe)


class DynamicRegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="lora-help-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "en").mkdir()
        (self.root / "ru").mkdir()
        (self.root / "registry").mkdir()
        (self.root / "en/topic-one.md").write_text("# Topic One\n\nFirst guide description.\n", encoding="utf-8")

    def test_scan_discovers_new_article_without_manifest_change(self):
        self.assertEqual([a["id"] for a in help_registry.index(root=self.root)["topics"]], ["topic-one"])
        (self.root / "en/topic-two.md").write_text("# Topic Two\n\nAnother guide.\n", encoding="utf-8")
        (self.root / "registry/topic-one.json").write_text(json.dumps({"related": ["topic-two"], "contexts": ["studio.root"], "order": 5}),encoding="utf-8")
        result = help_registry.index(root=self.root)
        self.assertEqual(len(result["topics"]), 2)
        self.assertEqual(result["topics"][0]["related"], ["topic-two"])
        (self.root / "ru/topic-one.md").write_text("# Тема один\n\nОписание на русском.\n",encoding="utf-8")
        self.assertEqual(help_registry.article("topic-one", "ru", root=self.root)["locale"], "ru")

    def test_missing_metadata_is_optional_and_bad_json_is_safe_error(self):
        self.assertEqual(help_registry.metadata("topic-one", self.root), {})
        (self.root / "registry/topic-one.json").write_text("{broken",encoding="utf-8")
        with self.assertRaises(help_registry.HelpError):
            help_registry.article("topic-one", root=self.root)
        self.assertEqual(help_registry.index(root=self.root)["topics"], [])

    def test_oversized_and_symlink_articles_are_skipped(self):
        (self.root / "en/topic-one.md").write_text("# Huge\n" + "x"*66000, encoding="utf-8")
        self.assertEqual(help_registry.index(root=self.root)["topics"], [])
        (self.root / "en/topic-one.md").unlink()
        try:
            (self.root / "en/topic-one.md").symlink_to(self.root / "registry")
        except (OSError, NotImplementedError):
            self.skipTest("Symlinks not available")
        self.assertEqual(help_registry.index(root=self.root)["topics"], [])


if __name__ == "__main__":
    unittest.main()
