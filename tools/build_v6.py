"""Collect the v6 failure-path candidate over a verified v5 snapshot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import install  # noqa: E402

CHANGED = ('modules/sd_models.py', 'modules/sd_disable_initialization.py', 'extensions-builtin/Lora/networks.py',
           'extensions-builtin/Lora/RUNTIME_STABILITY_FIXES.md', 'extensions-builtin/Lora/LORA_EVOLUTION_PLAN.md')


def build(source, baseline, base):
    manifest = json.loads(json.dumps(install.verify_package(base)))
    assets = {item['path']: item for item in manifest['assets']}
    operations = {item['path']: item for item in manifest['operations']}

    def asset(path, data, role):
        install.atomic_write(install.child(ROOT, path), data)
        assets[path] = dict(path=path, sha256=install.digest(data), bytes=len(data), role=role)

    for relative in CHANGED:
        data = install.child(source, relative).read_bytes()
        name = 'files/' + relative
        op = operations.setdefault(relative, dict(path=relative, asset=name, kind='replace', known_before=[]))
        known = set(op['known_before'])
        if name in assets:
            known.add(assets[name]['sha256'])
        original = install.child(baseline, relative)
        if original.is_file():
            known.add(install.file_hash(original))
        op['known_before'] = sorted(known)
        asset(name, data, 'runtime')
        manifest['source_fingerprints'][relative] = install.digest(data)
    relative = 'test/test_runtime_edgecases.py'
    asset('files/' + relative, install.child(source, relative).read_bytes(), 'verification-only-not-installed')
    manifest.update(name='A1111 LoRA Studio v6 - failure-path candidate',
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    source='Verified v5 plus tested cached-model, packed-FP8 and temporary-loader-hook fixes; GPU integration pending',
                    assets=sorted(assets.values(), key=lambda item: item['path']), operations=list(operations.values()))
    install.save_json(ROOT / 'manifest.json', manifest)
    install.verify_package(ROOT)
    print(f'Built {len(assets)} assets; {len(operations)} operations.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('--base', type=Path, default=ROOT.parent / 'A1111-LoRA-Studio-Pack-v5')
    args = parser.parse_args()
    build(args.source, args.baseline, args.base)
