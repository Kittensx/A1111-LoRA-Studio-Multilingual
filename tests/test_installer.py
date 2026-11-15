"""Run with: python -B -m unittest discover -s tests -v (no external libraries)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
import install  # noqa: E402


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lora-installer-test-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.target = self.base / "старый A1111"
        self.package = self.base / "пакет"
        self.target.mkdir()
        self.package.mkdir()
        for name in ("webui.py", "launch.py", "modules/processing.py", "modules/sd_models.py", "modules/extra_networks.py"):
            self.write(self.target / name, b"# original host\n")
        (self.target / "extensions-builtin/Lora").mkdir(parents=True)
        self.write(self.target / "style.css", b"old styles\n")
        self.write(self.target / "config.json", b'{"keep": true}')
        self.write(self.target / "models/Lora/keep.safetensors", b"not a real model; must remain untouched")
        self.recipe = {"before": "    old()\n    next()\n", "after": "    new()\n    next()\n"}
        self.write(self.target / "modules/processing.py", b"# user header\n" + self.recipe["before"].encode() + b"# user footer\n")
        self.manifest = {"format": 1, "complete": True, "name": "Test only", "assets": [], "operations": [], "required_host_files": []}
        self.asset("style.css", b"new styles\n", known=[install.digest(b"old styles\n")])
        self.asset("extensions-builtin/Lora/lora_schema.py", b"# new file\n")
        self.asset("modules/processing.py", json.dumps(self.recipe).encode(), kind="text-patch")
        self.save_manifest()

    @staticmethod
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def asset(self, destination, content, known=(), kind="replace"):
        name = "files/" + destination if kind == "replace" else "patches/hires.json"
        self.write(self.package / name, content)
        self.manifest["assets"].append({"path": name, "sha256": install.digest(content)})
        self.manifest["operations"].append({"path": destination, "asset": name, "kind": kind, "known_before": list(known)})

    def save_manifest(self):
        self.write(self.package / "manifest.json", json.dumps(self.manifest).encode())

    def run_install(self, **kwargs):
        return install.install(self.target, self.package, **kwargs)

    def test_check_is_read_only(self):
        _, _, rows = install.plan_install(self.target, self.package)
        self.assertEqual([r["status"] for r in rows], ["KNOWN", "ADD", "PATCH"])
        self.assertFalse((self.target / install.BACKUPS).exists())
        self.assertEqual((self.target / "style.css").read_bytes(), b"old styles\n")

    def test_install_backup_idempotence_and_recoverable_rollback(self):
        identifier = self.run_install()
        self.assertIsNotNone(identifier)
        backup = self.target / install.BACKUPS / identifier
        self.assertEqual((backup / "original/style.css").read_bytes(), b"old styles\n")
        self.assertEqual((self.target / "style.css").read_bytes(), b"new styles\n")
        self.assertIsNone(self.run_install())
        self.assertEqual(len(list((self.target / install.BACKUPS).iterdir())), 1)
        self.assertEqual(install.rollback(self.target, identifier), 3)
        self.assertEqual(install.rollback(self.target, identifier), 0)
        self.assertEqual((self.target / "style.css").read_bytes(), b"old styles\n")
        self.assertFalse((self.target / "extensions-builtin/Lora/lora_schema.py").exists())
        self.assertEqual((backup / "removed/extensions-builtin/Lora/lora_schema.py").read_bytes(), b"# new file\n")
        self.assertEqual((self.target / "config.json").read_bytes(), b'{"keep": true}')
        self.assertEqual((self.target / "models/Lora/keep.safetensors").read_bytes(), b"not a real model; must remain untouched")
        self.assertIsNotNone(self.run_install())

    def test_unknown_version_blocks_without_writes_then_explicit_opt_in(self):
        self.write(self.target / "style.css", b"custom styles")
        with self.assertRaises(install.InstallError):
            self.run_install()
        self.assertFalse((self.target / install.BACKUPS).exists())
        identifier = self.run_install(allow_unrecognized=True)
        install.rollback(self.target, identifier)
        self.assertEqual((self.target / "style.css").read_bytes(), b"custom styles")

    def test_tampered_package_blocks(self):
        self.write(self.package / "files/style.css", b"corrupt")
        with self.assertRaises(install.InstallError):
            self.run_install()
        self.assertFalse((self.target / install.BACKUPS).exists())

    def test_incomplete_package_blocks(self):
        self.manifest["complete"] = False
        self.save_manifest()
        with self.assertRaises(install.InstallError):
            self.run_install()

    def test_missing_host_dependency_blocks_even_override(self):
        self.manifest["required_host_files"] = ["modules/models/sd3/mmdit.py"]
        self.save_manifest()
        with self.assertRaises(install.InstallError):
            self.run_install(allow_unrecognized=True)

    def test_unsupported_fork_blocks(self):
        (self.target / "modules_forge").mkdir()
        with self.assertRaises(install.InstallError):
            self.run_install(allow_unrecognized=True)

    def test_wrong_target_blocks(self):
        with self.assertRaises(install.InstallError):
            install.install(self.base, self.package, allow_unrecognized=True)

    def test_directory_in_place_of_file_blocks(self):
        (self.target / "extensions-builtin/Lora/lora_schema.py").mkdir()
        with self.assertRaises(install.InstallError):
            self.run_install()

    def test_nested_package_is_rejected(self):
        nested = self.target / "bundle"
        shutil.copytree(self.package, nested)
        with self.assertRaises(install.InstallError):
            install.plan_install(self.target, nested)

    def test_path_traversal_ads_and_windows_aliases_rejected(self):
        for name in ("../evil", "/evil", "C:/evil", "C:evil", "a\\b", "a//b", "a/./b", "a/../b", "a/file ", "a/file.", "a/CON.py", "a/x:stream", "a/\x00bad"):
            with self.subTest(name=name), self.assertRaises(install.InstallError):
                install.relative(name)

    def test_forbidden_destination_and_case_collision_rejected(self):
        self.manifest["operations"][0]["path"] = "config.json"
        self.save_manifest()
        with self.assertRaises(install.InstallError):
            self.run_install()
        self.manifest["operations"][0]["path"] = "style.css"
        self.manifest["operations"].append(dict(self.manifest["operations"][0]))
        self.save_manifest()
        with self.assertRaises(install.InstallError):
            self.run_install()

    def test_target_symlink_rejected(self):
        alias = self.base / "linked-target"
        try:
            os.symlink(self.target, alias, target_is_directory=True)
        except OSError:
            self.skipTest("OS disallows creating symlinks for this user")
        with self.assertRaises(install.InstallError):
            install.install(alias, self.package)

    def test_windows_reparse_flag_rejected(self):
        real = Path.lstat
        def altered(path, *args, **kwargs):
            value = real(path, *args, **kwargs)
            if path == self.target:
                from types import SimpleNamespace
                return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
            return value
        with patch.object(Path, "lstat", altered), self.assertRaises(install.InstallError):
            install.reject_links(self.target)

    def test_patch_preserves_unrelated_code_and_newlines(self):
        for newline in (b"\n", b"\r\n"):
            with self.subTest(newline=newline):
                original = b"# prefix\r\n" + self.recipe["before"].encode().replace(b"\n", newline) + b"# suffix\n"
                result = install.patch_text(original, self.recipe)
                expected = b"# prefix\r\n" + self.recipe["after"].encode().replace(b"\n", newline) + b"# suffix\n"
                self.assertEqual(result, expected)
                self.assertEqual(install.patch_text(result, self.recipe), result)
        mixed = b"    new()\r\n    next()\n"
        self.assertEqual(install.patch_text(mixed, self.recipe), mixed)

    def test_patch_missing_and_ambiguous_context_block(self):
        for data in (None, b"different", self.recipe["before"].encode() * 2, (self.recipe["before"] + self.recipe["after"]).encode()):
            with self.subTest(data=data), self.assertRaises(install.InstallError):
                install.patch_text(data, self.recipe)

    def test_second_hunk_failure_prevents_all_writes(self):
        content = json.dumps({'hunks': [self.recipe, {'before': 'missing anchor', 'after': 'new anchor'}]}).encode()
        self.write(self.package / 'patches/hires.json', content)
        for asset in self.manifest['assets']:
            if asset['path'] == 'patches/hires.json':
                asset['sha256'] = install.digest(content)
        self.save_manifest()
        before = (self.target / 'modules/processing.py').read_bytes()
        with self.assertRaises(install.InstallError):
            self.run_install()
        self.assertEqual((self.target / 'modules/processing.py').read_bytes(), before)
        self.assertEqual((self.target / 'style.css').read_bytes(), b'old styles\n')
        self.assertFalse((self.target / install.BACKUPS).exists())

    def test_rollback_preserves_post_install_user_edits(self):
        identifier = self.run_install()
        self.write(self.target / "extensions-builtin/Lora/lora_schema.py", b"user changed new file")
        with self.assertRaises(install.InstallError):
            install.rollback(self.target, identifier)
        self.assertEqual((self.target / "style.css").read_bytes(), b"new styles\n")

    def test_old_completed_rollback_cannot_undo_new_install(self):
        old_id = self.run_install()
        install.rollback(self.target, old_id)
        new_id = self.run_install()
        with self.assertRaises(install.InstallError):
            install.rollback(self.target, old_id)
        self.assertEqual((self.target / "style.css").read_bytes(), b"new styles\n")
        self.assertEqual(install.rollback(self.target, new_id), 3)

    def test_rollback_corrupted_backup_blocks_before_changes(self):
        identifier = self.run_install()
        self.write(self.target / install.BACKUPS / identifier / "original/style.css", b"corrupt")
        with self.assertRaises(install.InstallError):
            install.rollback(self.target, identifier)
        self.assertEqual((self.target / "style.css").read_bytes(), b"new styles\n")

    def test_rollback_foreign_backup_and_bad_id_rejected(self):
        identifier = self.run_install()
        journal_path = self.target / install.BACKUPS / identifier / "journal.json"
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        journal["target"] = str(self.base)
        self.write(journal_path, json.dumps(journal).encode())
        for backup_id in (identifier, "../escape", str(journal_path)):
            with self.subTest(backup_id=backup_id), self.assertRaises(install.InstallError):
                install.rollback(self.target, backup_id)

    def test_tampered_journal_cannot_target_user_config(self):
        identifier = self.run_install()
        journal_path = self.target / install.BACKUPS / identifier / "journal.json"
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        journal["entries"][0]["path"] = "config.json"
        self.write(journal_path, json.dumps(journal).encode())
        with self.assertRaises(install.InstallError):
            install.rollback(self.target, identifier)
        self.assertEqual((self.target / "config.json").read_bytes(), b'{"keep": true}')

    def test_failure_mid_install_retains_complete_backup_and_can_rollback(self):
        actual = install.atomic_write
        def fail(path, data, mode=None):
            if path == self.target / "extensions-builtin/Lora/lora_schema.py":
                raise OSError("simulated disk failure")
            return actual(path, data, mode)
        with patch.object(install, "atomic_write", fail), self.assertRaises(OSError):
            self.run_install()
        identifier = next((self.target / install.BACKUPS).iterdir()).name
        journal = json.loads((self.target / install.BACKUPS / identifier / "journal.json").read_text(encoding="utf-8"))
        self.assertEqual(journal["status"], "interrupted")
        self.assertEqual(len(journal["entries"]), 3)
        self.assertEqual(install.rollback(self.target, identifier), 1)
        self.assertEqual((self.target / "style.css").read_bytes(), b"old styles\n")


class FullPackageTests(unittest.TestCase):
    def test_missing_region_operation_is_rejected_even_when_asset_is_present(self):
        package = Path(__file__).absolute().parents[1]
        manifest = install.read_json(package / 'manifest.json')
        self.assertIn('modules/region_utils.py', manifest['required_runtime_paths'])
        manifest['operations'] = [op for op in manifest['operations'] if op['path'] != 'modules/region_utils.py']
        with patch.object(install, 'read_json', return_value=manifest), self.assertRaisesRegex(install.InstallError, 'Required runtime'):
            install.verify_package(package)

    def test_actual_payload_install_and_rollback_in_synthetic_host(self):
        package = Path(__file__).absolute().parents[1]
        if not (package / "manifest.json").is_file():
            self.skipTest("Package has not been built yet")
        manifest = install.verify_package(package)
        recipe = install.read_json(package / "patches/processing-hires.json")
        current_processing = (package / "reference-only/modules/processing.py").read_bytes()
        hunks = recipe.get("hunks", [recipe])
        reverse = {"hunks": [{"before": hunk["after"], "after": hunk["before"]} for hunk in reversed(hunks)]}
        before_processing = install.patch_text(current_processing, reverse)
        with tempfile.TemporaryDirectory(prefix="lora-full-package-test-") as directory:
            target = Path(directory) / "A1111"
            target.mkdir()
            for name in {"webui.py", "launch.py", "modules/sd_models.py", "modules/extra_networks.py", *manifest["required_host_files"]}:
                InstallerTests.write(target / name, b"# synthetic host dependency; not executed\n")
            (target / "extensions-builtin/Lora").mkdir(parents=True)
            InstallerTests.write(target / "modules/processing.py", before_processing)
            InstallerTests.write(target / "config.json", b'{"personal": true}')
            InstallerTests.write(target / "extensions/keep/scripts/keep.py", b"# user extension\n")
            snapshot = {p.relative_to(target).as_posix(): p.read_bytes() for p in target.rglob("*") if p.is_file()}
            identifier = install.install(target, package, allow_unrecognized=True)
            _, _, rows = install.plan_install(target, package)
            self.assertTrue(all(row["status"] == "SAME" for row in rows))
            after_processing = (target / "modules/processing.py").read_bytes()
            self.assertEqual(after_processing.replace(b"\r\n", b"\n"), current_processing.replace(b"\r\n", b"\n"))
            self.assertFalse((target / "test/lora_studio_fixture.html").exists())
            self.assertFalse((target / "script.js").exists())
            install.rollback(target, identifier)
            restored = {p.relative_to(target).as_posix(): p.read_bytes() for p in target.rglob("*")
                        if p.is_file() and install.BACKUPS not in p.relative_to(target).parts}
            self.assertEqual(snapshot, restored)


if __name__ == "__main__":
    unittest.main()
