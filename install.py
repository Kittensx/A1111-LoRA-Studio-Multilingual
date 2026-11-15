#!/usr/bin/env python3
"""Offline, stdlib-only installer. Never imports the bundled WebUI code."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
from datetime import datetime, timezone
import uuid

PACKAGE = Path(__file__).absolute().parent
BACKUPS = ".lora-studio-backups"
HEX = re.compile(r"[0-9a-f]{64}\Z")
RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


class InstallError(Exception):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise InstallError(f"Unsafe relative path: {value!r}")
    parts = value.split("/")
    if PurePosixPath(value).is_absolute() or any(
        not p or p in (".", "..") or p.endswith((".", " "))
        or any(ord(c) < 32 or c in ':<>"|?*' for c in p)
        or p.split(".")[0].upper() in RESERVED for p in parts
    ):
        raise InstallError(f"Unsafe relative path: {value!r}")
    return parts


def reject_links(path):
    """Reject symlinks and Windows junction/reparse points, including ancestors."""
    path = Path(os.path.abspath(path))
    for item in (*reversed(path.parents), path):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise InstallError(f"Symlink/junction/reparse point is not supported: {item}")
    return path


def child(root, name):
    return reject_links(Path(root).joinpath(*relative(name)))


def read_json(path):
    reject_links(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as error:
        raise InstallError(f"Invalid JSON: {path}") from error


def atomic_write(path, data, mode=None):
    reject_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    reject_links(path)
    fd, name = tempfile.mkstemp(prefix=".lora-install-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()  # Only our exact temporary file, never a directory.


def save_json(path, value):
    atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def file_hash(path):
    reject_links(path)
    if not path.exists():
        return None
    if not path.is_file():
        raise InstallError(f"Expected a regular file: {path}")
    return digest(path.read_bytes())


HELP_LOCALES = {"en", "ru", "fr", "es", "it", "ja", "zh-CN", "pt-BR", "de", "ko"}
HELP_TOPIC = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")


def is_runtime_help_document(name):
    parts = name.split("/")
    return (len(parts) == 5 and parts[:3] == ["extensions-builtin", "Lora", "help_documentation"]
            and ((parts[3] in HELP_LOCALES and parts[4].endswith(".md") and HELP_TOPIC.fullmatch(parts[4][:-3]))
                 or (parts[3] == "registry" and parts[4].endswith(".json") and HELP_TOPIC.fullmatch(parts[4][:-5]))))


def allowed_destination(name):
    parts = relative(name)
    if name in {"style.css", "modules/extra_networks.py", "modules/ui_extra_networks.py", "modules/sd_models.py", "modules/processing.py",
                "modules/sd_disable_initialization.py", "modules/devices.py", "modules/patches.py", "modules/safe.py", "modules/sd_vae.py", "modules/textual_inversion/textual_inversion.py",
                "modules/options.py", "modules/config_store.py", "modules/lowvram.py", "modules/sd_emphasis.py",
                "modules/cache.py", "modules/hashes.py", "modules/api/api.py",
                "modules/region_utils.py", "modules/asset_files.py", "modules/sd_models_config.py",
                "modules/sd_hijack_clip_old.py", "modules/ui_settings.py", "modules/shared_options.py"}:
        return True
    if len(parts) == 2 and parts[0] == "javascript":
        return parts[1] == "extraNetworks.js" or bool(re.fullmatch(r"loraStudio[A-Za-z0-9]+\.js", parts[1]))
    if len(parts) == 2 and parts[0] == "html":
        return bool(re.fullmatch(r"extra-networks-[a-z-]+\.html", parts[1]))
    if is_runtime_help_document(name):
        return True
    if parts[:2] == ["extensions-builtin", "Lora"]:
        return ((len(parts) == 3 and Path(parts[-1]).suffix in (".py", ".md"))
                or (len(parts) == 4 and parts[2] == "scripts" and parts[3].endswith(".py")))
    return False


def target_root(target):
    root = reject_links(Path(target).absolute())
    if root == Path(root.anchor) or root == Path.home().absolute() or not root.is_dir():
        raise InstallError("Target must be a specific, existing A1111 installation directory.")
    for name in ("webui.py", "launch.py", "modules/processing.py", "modules/sd_models.py", "modules/extra_networks.py"):
        if file_hash(child(root, name)) is None:
            raise InstallError(f"Not a supported A1111 directory; missing: {name}")
    if not child(root, "extensions-builtin/Lora").is_dir():
        raise InstallError("Native extensions-builtin/Lora directory is missing.")
    for name in ("modules_forge", "backend", "modules/sd_models_compile.py"):
        if child(root, name).exists():
            raise InstallError("Forge/SD.Next variants are not supported by this snapshot.")
    return root


def verify_package(package=PACKAGE):
    package = reject_links(package)
    manifest = read_json(child(package, "manifest.json"))
    if manifest.get("format") != 1 or manifest.get("complete") is not True:
        raise InstallError("Package is incomplete or its format is unsupported.")
    seen = set()
    assets = manifest.get("assets", [])
    if not assets:
        raise InstallError("Empty package.")
    for asset in assets:
        name = asset["path"]
        relative(name)
        if name.casefold() in seen or not HEX.fullmatch(asset["sha256"]):
            raise InstallError(f"Invalid/duplicate package entry: {name}")
        seen.add(name.casefold())
        if file_hash(child(package, name)) != asset["sha256"]:
            raise InstallError(f"Package integrity check failed: {name}")
    asset_names = {a["path"] for a in assets}
    destinations = set()
    for entry in manifest.get("operations", []):
        name = entry["path"]
        if name.endswith(".md") and not is_runtime_help_document(name):
            raise InstallError(f"Only contextual help Markdown may be a required runtime asset: {name}")
        if is_runtime_help_document(name) and entry.get("asset") != "files/" + name:
            raise InstallError(f"Contextual help source must match its destination: {name}")
        if not allowed_destination(name) or name.casefold() in destinations:
            raise InstallError(f"Forbidden/duplicate destination: {name}")
        destinations.add(name.casefold())
        if entry.get("kind") not in ("replace", "text-patch") or entry["asset"] not in asset_names:
            raise InstallError(f"Invalid operation: {name}")
        for known in entry.get("known_before", []):
            if not HEX.fullmatch(known):
                raise InstallError(f"Invalid baseline hash: {name}")
    # Documentation is independently verified, but is not part of the default
    # A1111 runtime installation. The two language variants are kept in docs/.
    doc_destinations = set(destinations)
    for entry in manifest.get("documentation_operations", []):
        name = entry["path"]
        parts = relative(name)
        if (parts[:2] != ["extensions-builtin", "Lora"] or len(parts) != 3
                or not name.endswith(".md") or not allowed_destination(name)
                or name.casefold() in doc_destinations):
            raise InstallError(f"Invalid/duplicate optional documentation destination: {name}")
        doc_destinations.add(name.casefold())
        variants = entry.get("assets")
        if not isinstance(variants, dict) or set(variants) != {"EN", "RUS"}:
            raise InstallError(f"Expected English and Russian document variants: {name}")
        hashes = entry.get("sha256")
        if not isinstance(hashes, dict) or set(hashes) != {"EN", "RUS"}:
            raise InstallError(f"Missing optional document checksum mapping: {name}")
        for language, asset_name in variants.items():
            expected_asset = f"docs/{language}/{Path(name).stem}_{language}.md"
            if asset_name != expected_asset or not HEX.fullmatch(hashes[language]):
                raise InstallError(f"Invalid optional document source/checksum for {name}: {asset_name}")
        for known in entry.get("known_before", []):
            if not HEX.fullmatch(known):
                raise InstallError(f"Invalid optional document baseline hash: {name}")
    if not destinations:
        raise InstallError("No install operations.")
    for name in manifest.get('required_runtime_paths', []):
        relative(name)
        if name.casefold() not in destinations:
            raise InstallError(f'Required runtime has no install operation: {name}')
    installed_assets = {entry['asset'] for entry in manifest['operations']}
    for asset in assets:
        if asset.get('role') == 'runtime' and asset['path'] not in installed_assets:
            raise InstallError(f'Runtime asset is not installed: {asset["path"]}')
    return manifest


def patch_text(data, recipe):
    """Exact, unique anchor only. Preserve unrelated bytes and original newlines."""
    if "hunks" in recipe:
        hunks = recipe["hunks"]
        if not isinstance(hunks, list) or not hunks or any(not isinstance(hunk, dict) or "hunks" in hunk for hunk in hunks):
            raise InstallError("Invalid multi-hunk patch recipe.")
        for hunk in hunks:
            data = patch_text(data, hunk)
        return data
    if data is None:
        raise InstallError("Cannot patch a missing host file.")
    old = recipe["before"].encode("utf-8")
    new = recipe["after"].encode("utf-8")
    def matches(fragment):
        pattern = b"\r?\n".join(re.escape(line) for line in fragment.split(b"\n"))
        return list(re.finditer(pattern, data))
    old_matches, new_matches = matches(old), matches(new)
    # A source snapshot can have mixed line endings around a previously patched block.
    if len(new_matches) == 1 and not old_matches:
        return data
    if len(old_matches) != 1 or new_matches:
        raise InstallError("WebUI patch context differs or is ambiguous; manual port required.")
    match = old_matches[0]
    fragment = match.group()
    if fragment.count(b"\r\n") > fragment.count(b"\n") / 2:
        new = new.replace(b"\n", b"\r\n")
    return data[:match.start()] + new + data[match.end():]


def selected_operations(manifest, docs_language=None):
    """Use documented source paths only when explicitly requested by the caller."""
    if docs_language not in (None, "EN", "RUS"):
        raise InstallError("Documentation language must be EN or RUS.")
    operations = list(manifest["operations"])
    if docs_language is not None:
        for entry in manifest.get("documentation_operations", []):
            operations.append({
                "path": entry["path"],
                "asset": entry["assets"][docs_language],
                "sha256": entry["sha256"][docs_language],
                "kind": "replace",
                "known_before": list(entry.get("known_before", [])),
            })
    return operations


def plan_install(target, package=PACKAGE, docs_language=None):
    manifest = verify_package(package)
    root = target_root(target)
    package = Path(package).absolute()
    if root == package or root in package.parents or package in root.parents:
        raise InstallError("Keep the patch package outside the target A1111 directory.")
    for name in manifest.get("required_host_files", []):
        if file_hash(child(root, name)) is None:
            raise InstallError(f"This A1111 version is too old/incompatible; missing host dependency: {name}")
    asset_hashes = {asset["path"]: asset["sha256"] for asset in manifest["assets"]}
    plan = []
    for entry in selected_operations(manifest, docs_language):
        dest = child(root, entry["path"])
        before_hash = file_hash(dest)
        before = dest.read_bytes() if before_hash is not None else None
        asset = child(package, entry["asset"])
        expected_hash = asset_hashes.get(entry["asset"], entry.get("sha256"))
        if file_hash(asset) != expected_hash:
            raise InstallError(f'Package document or runtime asset missing/changed: {entry["asset"]}')
        content = asset.read_bytes()
        if digest(content) != expected_hash:
            raise InstallError(f'Package changed during preflight: {entry["asset"]}')
        after = content if entry["kind"] == "replace" else patch_text(before, json.loads(content))
        after_hash = digest(after)
        if before_hash == after_hash:
            status = "SAME"
        elif before_hash is None:
            status = "ADD"
        elif entry["kind"] == "text-patch":
            status = "PATCH"
        elif before_hash in entry.get("known_before", []):
            status = "KNOWN"
        else:
            status = "UNKNOWN"
        plan.append({"path": entry["path"], "status": status, "before": before_hash, "after": after_hash, "data": after})
    return root, manifest, plan


def show_plan(plan):
    for row in plan:
        print(f'{row["status"]:7} {row["path"]}')
    unknown = sum(row["status"] == "UNKNOWN" for row in plan)
    changed = sum(row["status"] != "SAME" for row in plan)
    print(f"\nChanges: {changed}; unrecognized existing files: {unknown}.")
    print("Hashes/context checks are NOT proof of runtime or image-quality compatibility.")
    return unknown


def install(target, package=PACKAGE, allow_unrecognized=False, docs_language=None):
    root, manifest, plan = plan_install(target, package, docs_language=docs_language)
    if any(row["status"] == "UNKNOWN" for row in plan) and not allow_unrecognized:
        raise InstallError("Unknown existing files: inspect 'check' output, then explicitly use --allow-unrecognized if appropriate.")
    changed = [row for row in plan if row["status"] != "SAME"]
    if not changed:
        return None
    backup_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:10]
    backup = child(root, f"{BACKUPS}/{backup_id}")
    backup.mkdir(parents=True, exist_ok=False)
    journal = {"format": 1, "target": str(root), "package": manifest["name"], "status": "preparing", "entries": []}
    journal_path = backup / "journal.json"
    try:
        for row in changed:
            dest = child(root, row["path"])
            if file_hash(dest) != row["before"]:
                raise InstallError(f'File changed during preflight: {row["path"]}')
            mode = stat.S_IMODE(dest.stat().st_mode) if row["before"] is not None else None
            if row["before"] is not None:
                original = dest.read_bytes()
                if digest(original) != row["before"]:
                    raise InstallError(f'File changed while backing up: {row["path"]}')
                saved = child(backup, "original/" + row["path"])
                atomic_write(saved, original)
                if file_hash(saved) != row["before"]:
                    raise InstallError("Backup verification failed.")
            journal["entries"].append({k: row[k] for k in ("path", "before", "after") } | {"mode": mode})
        journal["status"] = "installing"
        save_json(journal_path, journal)  # Full verified backup before the first target change.
        for row in changed:
            dest = child(root, row["path"])
            if file_hash(dest) != row["before"]:
                raise InstallError(f'Concurrent target edit: {row["path"]}')
            mode = next(e["mode"] for e in journal["entries"] if e["path"] == row["path"])
            atomic_write(dest, row["data"], mode)
            if file_hash(dest) != row["after"]:
                raise InstallError(f'Post-install hash mismatch: {row["path"]}')
        journal["status"] = "installed"
        save_json(journal_path, journal)
    except BaseException:
        journal["status"] = "interrupted"
        try:
            save_json(journal_path, journal)
        finally:
            print(f"Installation interrupted. Backup retained: {backup}", file=sys.stderr)
            print(f'After resolving the error, run: python install.py rollback "{root}" {backup_id}', file=sys.stderr)
        raise
    return backup_id


def rollback(target, backup_id):
    # Rollback must work even if a failed install prevents WebUI from starting.
    root = reject_links(Path(target).absolute())
    if not re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{10}", backup_id):
        raise InstallError("Supply the exact backup ID printed by install/backups, not a path.")
    backup = child(root, f"{BACKUPS}/{backup_id}")
    journal_path = child(backup, "journal.json")
    journal = read_json(journal_path)
    if journal.get("format") != 1 or Path(journal.get("target", "")).absolute() != root:
        raise InstallError("Backup belongs to a different target or has an unsupported format.")
    actions = []
    seen = set()
    for row in journal["entries"]:
        name = row["path"]
        if not allowed_destination(name) or name.casefold() in seen:
            raise InstallError(f"Invalid backup destination: {name}")
        seen.add(name.casefold())
        if not HEX.fullmatch(row["after"]) or (row["before"] is not None and not HEX.fullmatch(row["before"])):
            raise InstallError("Invalid backup hashes.")
        data = None
        if row["before"] is not None:
            original = child(backup, "original/" + name)
            if file_hash(original) != row["before"]:
                raise InstallError(f"Backup damaged: {name}")
            data = original.read_bytes()
        current = file_hash(child(root, name))
        if journal.get("status") == "rolled_back" and current != row["before"]:
            raise InstallError("This backup was already rolled back; the target changed again. Use the appropriate newer backup.")
        if current == row["before"]:
            continue  # Not installed yet, or already rolled back.
        if current != row["after"]:
            raise InstallError(f"Changed since installation; rollback refuses to overwrite: {name}")
        actions.append((row, data))
    journal["status"] = "rolling_back"
    save_json(journal_path, journal)
    for row, data in reversed(actions):
        dest = child(root, row["path"])
        if file_hash(dest) != row["after"]:
            raise InstallError(f'Concurrent edit during rollback: {row["path"]}')
        if data is None:
            removed = child(backup, "removed/" + row["path"])
            if removed.exists():
                raise InstallError(f"Recovery destination already exists: {removed}")
            removed.parent.mkdir(parents=True, exist_ok=True)
            reject_links(removed)
            os.replace(dest, removed)  # Recoverable removal of only a newly added file.
        else:
            atomic_write(dest, data, row["mode"])
        if file_hash(dest) != row["before"]:
            raise InstallError(f'Rollback verification failed: {row["path"]}')
    journal["status"] = "rolled_back"
    save_json(journal_path, journal)
    return len(actions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="Verify package checksums without a target.")
    for name in ("check", "install", "backups", "rollback"):
        command = sub.add_parser(name)
        command.add_argument("target", type=Path, help="Path to the old/target A1111, NOT the package.")
        if name in ("install", "rollback"):
            command.add_argument("--yes", action="store_true", help="Confirm WebUI is stopped and approve listed changes.")
        if name in ("check", "install"):
            command.add_argument("--docs-language", choices=("EN", "RUS"), default=None,
                                 help="Optionally install 6 documentation files from docs/EN or docs/RUS; default is code only.")
        if name == "install":
            command.add_argument("--allow-unrecognized", action="store_true", help="Allow full replacement of unknown file versions (backup is mandatory).")
        if name == "rollback":
            command.add_argument("backup_id")
    args = parser.parse_args()
    try:
        if args.command == "verify":
            manifest = verify_package()
            print(f'Package verified: {len(manifest["assets"])} assets. No A1111 installation changed.')
        elif args.command == "backups":
            root = reject_links(args.target.absolute())
            parent = child(root, BACKUPS)
            if parent.is_dir():
                for item in sorted(parent.iterdir()):
                    if re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{10}", item.name):
                        print(item.name, read_json(child(item, "journal.json")).get("status"))
        else:
            if args.command in ("check", "install"):
                _, _, plan = plan_install(args.target, docs_language=args.docs_language)
                unknown = show_plan(plan)
                if args.command == "check":
                    return 2 if unknown else 0
                if unknown and not args.allow_unrecognized:
                    raise InstallError("Review UNKNOWN files. Installation blocked without --allow-unrecognized.")
            if not args.yes:
                print("Stop WebUI first. This changes code files; retain a separate full backup of your installation.")
                if input("Type STOPPED to confirm and continue: ").strip() != "STOPPED":
                    raise InstallError("Cancelled; target not changed.")
            if args.command == "install":
                backup_id = install(args.target, allow_unrecognized=args.allow_unrecognized, docs_language=args.docs_language)
                if backup_id:
                    print(f"Installed. Backup ID: {backup_id}")
                    print(f'Rollback: python install.py rollback "{args.target.absolute()}" {backup_id}')
                    print("Restart WebUI, then hard-refresh the browser (Ctrl+F5).")
                else:
                    print("Already installed; no files changed and no new backup created.")
            else:
                print(f"Rolled back {rollback(args.target, args.backup_id)} files. Backups/removed files retained.")
        return 0
    except (InstallError, OSError, ValueError, KeyError, TypeError, EOFError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
