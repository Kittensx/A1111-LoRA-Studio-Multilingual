"""Collect the v11 interaction fixes without changing the v10 release."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import install  # noqa: E402
from _historical_build_paths import resolve_build_paths

CHANGED = (
    'extensions-builtin/Lora/network.py',
    'extensions-builtin/Lora/network_lora.py',
    'extensions-builtin/Lora/networks.py',
    'extensions-builtin/Lora/lora_advanced.py',
    'extensions-builtin/Lora/LORA_EVOLUTION_PLAN.md',
    'extensions-builtin/Lora/RUNTIME_STABILITY_FIXES.md',
    'javascript/loraStudioModel.js', 'javascript/loraStudioCurves.js',
    'javascript/loraStudioBlocks.js', 'javascript/loraStudioView.js',
    'style.css',
)
TESTS = (
    'test/test_runtime_combinations.py',
    'test/test_lora_interaction_contract.py',
    'test/lora_interactions.test.cjs',
    'test/lora_studio_fixture.js',
)


def build():

    global ROOT, SOURCE, BASE, BACKUP
    args = resolve_build_paths(11, needs_backup=True,
                              previous='A1111-LoRA-Studio-Pack-v10',
                              source_paths=(*CHANGED, *TESTS),
                              backup_paths=('modules/processing.py',) if 11 == 10 else ())
    SOURCE, BASE, BACKUP, ROOT = args.source, args.base, args.backup, args.output
    manifest = json.loads(json.dumps(install.verify_package(BASE)))
    assets = {item['path']: item for item in manifest['assets']}
    operations = {item['path']: item for item in manifest['operations']}

    def asset(name, data, role):
        install.atomic_write(install.child(ROOT, name), data)
        assets[name] = dict(path=name, bytes=len(data), sha256=install.digest(data), role=role)

    for relative in CHANGED:
        data = install.child(SOURCE, relative).read_bytes()
        name = 'files/' + relative
        # Do not silently expand the install scope while collecting corrections.
        op = operations[relative]
        assert op['kind'] == 'replace' and op['asset'] == name, relative
        known = set(op['known_before'])
        known.add(assets[name]['sha256'])
        previous = install.child(BACKUP, relative)
        if previous.is_file():
            known.add(install.file_hash(previous))
        op['known_before'] = sorted(known)
        asset(name, data, 'runtime')
        manifest['source_fingerprints'][relative] = install.digest(data)

    for relative in TESTS:
        asset('files/' + relative, install.child(SOURCE, relative).read_bytes(),
              'verification-only-not-installed')

    manifest.update(
        name='A1111 LoRA Studio v11 - interaction and block UI corrections candidate',
        created_utc=datetime.now(timezone.utc).isoformat(),
        source='v10 + cross-feature CPU/AST and browser-fixture corrections; actual GPU/Gradio/model acceptance pending',
        assets=sorted(assets.values(), key=lambda item: item['path']),
        operations=list(operations.values()),
    )
    install.save_json(ROOT / 'manifest.json', manifest)
    install.verify_package(ROOT)
    print(f'Built {len(assets)} assets; {len(operations)} operations.')


if __name__ == '__main__':
    build()
