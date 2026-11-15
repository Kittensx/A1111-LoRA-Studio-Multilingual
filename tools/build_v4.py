"""Collect the v4 recovery candidate over a verified v3 snapshot."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import install  # noqa: E402

CHANGED = (
    'modules/patches.py', 'modules/sd_models.py',
    'extensions-builtin/Lora/lora_patches.py', 'extensions-builtin/Lora/networks.py',
    'extensions-builtin/Lora/RUNTIME_STABILITY_FIXES.md',
)


def close_method(text, name):
    cls = next(node for node in ast.parse(text).body if isinstance(node, ast.ClassDef) and node.name == name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'close')
    return ''.join(text.splitlines(keepends=True)[method.lineno - 1:method.end_lineno])


def build(source, baseline, base):
    manifest = json.loads(json.dumps(install.verify_package(base)))
    assets = {asset['path']: asset for asset in manifest['assets']}
    operations = {op['path']: op for op in manifest['operations']}

    def asset(path, data, role):
        install.atomic_write(install.child(ROOT, path), data)
        assets[path] = dict(path=path, sha256=install.digest(data), bytes=len(data), role=role)

    for relative in CHANGED:
        data = install.child(source, relative).read_bytes()
        name = 'files/' + relative
        op = operations[relative]
        known = set(op['known_before'])
        known.add(assets[name]['sha256'])
        old = install.child(baseline, relative)
        if old.is_file():
            known.add(install.file_hash(old))
        op['known_before'] = sorted(known)
        asset(name, data, 'runtime')
        manifest['source_fingerprints'][relative] = install.digest(data)

    old = install.child(base, 'reference-only/modules/processing.py').read_text(encoding='utf-8')
    new = install.child(source, 'modules/processing.py').read_text(encoding='utf-8')
    recipe = install.read_json(base / 'patches/processing-hires.json')
    hunks = list(recipe.get('hunks', [recipe]))
    for name in ('StableDiffusionProcessing', 'StableDiffusionProcessingTxt2Img'):
        hunks.append(dict(before=close_method(old, name), after=close_method(new, name)))
    recipe = dict(description='Scoped Hires, generation cleanup, and independent base/Hires close.', hunks=hunks)
    if install.patch_text(new.encode(), recipe) != new.encode():
        raise install.InstallError('Processing recipe is not idempotent')
    if install.patch_text(old.encode(), recipe) != new.encode():
        raise install.InstallError('Processing recipe does not reproduce the tested source')
    asset('patches/processing-hires.json', (json.dumps(recipe, ensure_ascii=False, indent=2) + '\n').encode(), 'targeted-core-patch')
    asset('reference-only/modules/processing.py', new.encode(), 'reference-only-not-installed')
    asset('files/modules/processing.py', new.encode(), 'verification-only-not-installed')
    manifest['source_fingerprints']['modules/processing.py'] = install.file_hash(source / 'modules/processing.py')
    for name in ('test_runtime_stability.py', 'test_runtime_recovery.py'):
        asset('files/test/' + name, install.child(source, 'test/' + name).read_bytes(), 'verification-only-not-installed')
    name = 'files/extensions-builtin/Lora/tests/test_lora_loading.py'
    data = install.child(ROOT, name).read_bytes()
    assets[name].update(sha256=install.digest(data), bytes=len(data))
    manifest.update(name='A1111 LoRA Studio v4 - recovery candidate',
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    source='Verified v3 snapshot plus review R1-R5 fixes; GPU integration pending',
                    assets=sorted(assets.values(), key=lambda item: item['path']), operations=list(operations.values()))
    install.save_json(ROOT / 'manifest.json', manifest)
    install.verify_package(ROOT)
    print(f'Built {len(assets)} assets; {len(operations)} operations.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('--base', type=Path, default=ROOT.parent / 'A1111-LoRA-Studio-Pack-v3')
    args = parser.parse_args()
    build(args.source, args.baseline, args.base)
