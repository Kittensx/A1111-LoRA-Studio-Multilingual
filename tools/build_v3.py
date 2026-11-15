"""Collect the tested v3 fixes over a verified v2 portable snapshot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import install  # noqa: E402

CHANGED = (
    'modules/sd_disable_initialization.py', 'modules/devices.py', 'modules/patches.py',
    'modules/sd_models.py', 'modules/textual_inversion/textual_inversion.py',
    'extensions-builtin/Lora/network.py', 'extensions-builtin/Lora/networks.py',
    'extensions-builtin/Lora/RUNTIME_STABILITY_FIXES.md',
)


def build(source, baseline, base):
    previous = install.verify_package(base)
    manifest = json.loads(json.dumps(previous))
    assets = {asset['path']: asset for asset in manifest['assets']}
    operations = {op['path']: op for op in manifest['operations']}

    def asset(path, data, role):
        install.atomic_write(install.child(ROOT, path), data)
        assets[path] = dict(path=path, sha256=install.digest(data), bytes=len(data), role=role)

    for relative in CHANGED:
        data = install.child(source, relative).read_bytes()
        name = 'files/' + relative
        op = operations.setdefault(relative, dict(path=relative, asset=name, kind='replace', known_before=[]))
        known = set(op['known_before'])
        original = install.child(baseline, relative)
        if original.is_file():
            known.add(install.digest(original.read_bytes()))
        if name in assets:
            known.add(assets[name]['sha256'])
        op['known_before'] = sorted(known)
        asset(name, data, 'runtime')
        manifest['source_fingerprints'][relative] = install.digest(data)

    old = install.child(baseline, 'modules/processing.py').read_text(encoding='utf-8')
    new = install.child(source, 'modules/processing.py').read_text(encoding='utf-8')
    manifest['source_fingerprints']['modules/processing.py'] = install.file_hash(install.child(source, 'modules/processing.py'))
    def cleanup_block(text):
        start = text.index('    finally:\n        sd_models.apply_token_merging(p.sd_model, 0)', text.index('def process_images(')) if '    finally:\n        sd_models.apply_token_merging(p.sd_model, 0)' in text else text.index('    finally:\n        cleanup_processing(', text.index('def process_images('))
        end = text.index('def process_images_inner(', start)
        return text[start:end]
    recipe = install.read_json(base / 'patches/processing-hires.json')
    combined = {'description': 'Scoped Hires marker and exception-safe generation cleanup only.',
                'hunks': [recipe, {'before': cleanup_block(old), 'after': cleanup_block(new)}]}
    if install.patch_text(new.encode(), combined) != new.encode():
        raise install.InstallError('Processing recipe is not idempotent')
    asset('patches/processing-hires.json', (json.dumps(combined, ensure_ascii=False, indent=2) + '\n').encode(), 'targeted-core-patch')
    asset('reference-only/modules/processing.py', new.encode(), 'reference-only-not-installed')
    asset('files/modules/processing.py', new.encode(), 'verification-only-not-installed')
    for relative in ('test/test_runtime_stability.py',):
        asset('files/' + relative, install.child(source, relative).read_bytes(), 'verification-only-not-installed')
    # The old NumPy-only harness needs to select the new internal merge helper.
    name = 'files/extensions-builtin/Lora/tests/test_lora_loading.py'
    data = (ROOT / name).read_bytes()
    assets[name].update(sha256=install.digest(data), bytes=len(data))
    manifest.update(name='A1111 LoRA Studio v3 - runtime stability',
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    source='Verified v2 snapshot plus tested runtime fixes from the user checkout',
                    assets=sorted(assets.values(), key=lambda a: a['path']), operations=list(operations.values()))
    install.save_json(ROOT / 'manifest.json', manifest)
    install.verify_package(ROOT)
    print(f'Built {len(assets)} assets; {len(operations)} operations.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('--base', type=Path, default=ROOT.parent / 'A1111-LoRA-Studio-Pack-v2')
    args = parser.parse_args()
    build(args.source, args.baseline, args.base)
