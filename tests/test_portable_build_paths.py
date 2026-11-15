"""Phase 2: historical builders must require portable, isolated inputs."""
from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools._historical_build_paths import resolve_build_paths  # noqa: E402


class PortableHistoricalBuildTests(unittest.TestCase):
    def test_all_six_historical_builders_have_help(self):
        for version in range(10, 16):
            with self.subTest(version=version):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "tools" / f"build_v{version}.py"), "--help"],
                    capture_output=True, text=True, cwd=ROOT, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--source", result.stdout)
                self.assertIn("--output", result.stdout)
                self.assertNotIn("D:/AI-tools/", result.stdout)

    def test_missing_input_is_rejected_without_writing(self):
        with patch.object(sys, "argv", ["builder"]), patch.dict(os.environ, {}, clear=True):
            with redirect_stderr(StringIO()) as error:
                with self.assertRaises(SystemExit) as raised:
                    resolve_build_paths(13)
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--source is required", error.getvalue())

    def test_repository_is_not_valid_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            with patch.object(sys, "argv", ["builder", "--source", str(source), "--output", str(ROOT)]):
                with redirect_stderr(StringIO()) as error:
                    with self.assertRaises(SystemExit) as raised:
                        resolve_build_paths(14)
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("isolated", error.getvalue())

    def test_overlapping_input_and_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            output = source / "out"
            output.mkdir(parents=True)
            (output / "manifest.json").write_text("{}", encoding="utf-8")
            with patch.object(sys, "argv", ["builder", "--source", str(source), "--output", str(output)]):
                with redirect_stderr(StringIO()) as error:
                    with self.assertRaises(SystemExit):
                        resolve_build_paths(14)
            self.assertIn("overlaps", error.getvalue())

    def test_output_must_be_preseeded(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            output = Path(temporary) / "out"
            source.mkdir()
            output.mkdir()
            with patch.object(sys, "argv", ["builder", "--source", str(source), "--output", str(output)]):
                with redirect_stderr(StringIO()) as error:
                    with self.assertRaises(SystemExit):
                        resolve_build_paths(14)
            self.assertIn("copy of a prior package", error.getvalue())

    def test_explicit_source_and_output_paths_are_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            output = Path(temporary) / "out"
            source.mkdir()
            output.mkdir()
            (source / "modules").mkdir()
            (source / "modules" / "options.py").write_text("# source", encoding="utf-8")
            (output / "manifest.json").write_text("{}", encoding="utf-8")
            with patch.object(sys, "argv", ["builder", "--source", str(source), "--output", str(output)]):
                args = resolve_build_paths(13, source_paths=("modules/options.py",))
            self.assertEqual(args.source, source)
            self.assertEqual(args.output, output)


if __name__ == "__main__":
    unittest.main()
