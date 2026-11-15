#!/usr/bin/env python3
"""Single entrypoint for the A1111 LoRA Studio package and its offline tools.

No bundled A1111 module is imported. The original installer remains the only
writer of installed A1111 files and the only owner of rollback journals.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
HELP = """A1111 LoRA Studio Multilingual — package launcher

  python run.py verify
  python run.py docs-audit
  python run.py help-audit
  python run.py check PATH_TO_A1111 [--docs-language EN|RUS]
  python run.py install PATH_TO_A1111 [--docs-language EN|RUS] [--yes]
  python run.py backups PATH_TO_A1111
  python run.py rollback PATH_TO_A1111 BACKUP_ID [--yes]
  python run.py test
  python run.py smoke --base-zip ZIP --modules-zip ZIP --temp-root DIR --torch-python PYTHON
                     [--docs-language EN|RUS]

Without --docs-language, Markdown remains in docs/ and is NOT copied to A1111.
The installer verifies required code assets independently of documentation.
Run docs-audit to verify both language folders and optional document hashes.
Run 'check' and inspect UNKNOWN entries before any installation.
"""


def execute(arguments):
    return subprocess.run(arguments, cwd=ROOT, check=False).returncode


def docs_audit():
    """Check repository documentation pairs without modifying any files."""
    english = ROOT / "docs" / "EN"
    russian = ROOT / "docs" / "RUS"
    if not english.is_dir() or not russian.is_dir():
        print("ERROR: docs/EN and docs/RUS must both exist.", file=sys.stderr)
        return 1
    en = {p.name[:-len("_EN.md")] for p in english.glob("*_EN.md")}
    ru = {p.name[:-len("_RUS.md")] for p in russian.glob("*_RUS.md")}
    if not en or en != ru:
        print("ERROR: English/Russian Markdown pairs do not match.", file=sys.stderr)
        print("Missing English:", ", ".join(sorted(ru - en)) or "none", file=sys.stderr)
        print("Missing Russian:", ", ".join(sorted(en - ru)) or "none", file=sys.stderr)
        return 1
    # Also verify every referenced optional document exists at its mapped path.
    sys.path.insert(0, str(ROOT))
    import install  # noqa: E402

    try:
        manifest = install.verify_package(ROOT)
        for op in manifest.get("documentation_operations", []):
            for language in ("EN", "RUS"):
                doc_path = install.child(ROOT, op["assets"][language])
                if install.file_hash(doc_path) != op["sha256"][language]:
                    raise install.InstallError(f"Missing or changed optional {language} document: {doc_path}")
    except (install.InstallError, OSError, KeyError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Documentation verified: {len(en)} EN/RUS pairs; "
          f"{len(manifest.get('documentation_operations', []))} optional installable documents.")
    return 0


def test_all():
    code = execute([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"])
    if code:
        return code
    node = shutil.which("node")
    if not node:
        print("ERROR: Node.js is required to run JavaScript tests.", file=sys.stderr)
        return 1
    tests = [str(path.relative_to(ROOT)) for path in sorted((ROOT / "files" / "test").glob("*.test.cjs"))]
    if not tests:
        print("ERROR: No JavaScript tests found.", file=sys.stderr)
        return 1
    return execute([node, "--test", *tests])


def main(argv):
    if len(argv) < 2 or argv[1] in ("help", "-h", "--help"):
        print(HELP)
        return 0 if len(argv) > 1 else 2
    command, args = argv[1], argv[2:]
    if command == "help-audit":
        if args:
            print("ERROR: help-audit accepts no additional arguments.", file=sys.stderr)
            return 2
        return execute([sys.executable, "-B", str(ROOT / "tools" / "help_audit.py")])
    if command == "docs-audit":
        if args:
            print("ERROR: docs-audit accepts no additional arguments.", file=sys.stderr)
            return 2
        return docs_audit()
    if command == "test":
        if args:
            print("ERROR: test accepts no additional arguments.", file=sys.stderr)
            return 2
        return test_all()
    if command == "verify":
        if args:
            print("ERROR: verify accepts no additional arguments.", file=sys.stderr)
            return 2
        return execute([sys.executable, str(ROOT / "install.py"), "verify"])
    if command in ("check", "install", "backups", "rollback"):
        return execute([sys.executable, str(ROOT / "install.py"), command, *args])
    if command == "smoke":
        return execute([sys.executable, str(ROOT / "tools" / "post_install_smoke.py"), *args])
    print(f"ERROR: Unknown command: {command}\n\n{HELP}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
