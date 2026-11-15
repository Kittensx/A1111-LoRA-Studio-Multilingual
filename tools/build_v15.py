"""Build v15 from the preserved v14 manifest; never modify earlier releases."""
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import install  # noqa: E402
from _historical_build_paths import resolve_build_paths

RUNTIME = ('modules/cache.py', 'modules/hashes.py', 'modules/options.py', 'modules/api/api.py',
           'extensions-builtin/Lora/network.py', 'extensions-builtin/Lora/networks.py',
           'javascript/loraStudioView.js')
TESTS = ('test/test_config_store.py', 'test/test_runtime_asset_identity.py',
         'test/test_runtime_external_audit.py', 'test/test_runtime_v10_lifecycle.py',
         'test/test_runtime_known_routes.py', 'test/lora_studio_fixture.js')


def main():

    global ROOT, SOURCE, BEFORE
    args = resolve_build_paths(15, needs_backup=True,
                              previous='A1111-LoRA-Studio-Pack-v14',
                              source_paths=(*RUNTIME, *TESTS), backup_paths=RUNTIME)
    SOURCE, BEFORE, ROOT = args.source, args.backup, args.output
    previous_root = args.base
    manifest = install.verify_package(previous_root)
    assets = {item['path']: item for item in manifest['assets']}
    operations = {item['path']: item for item in manifest['operations']}
    for relative in (*RUNTIME, *TESTS):
        data = (SOURCE / relative).read_bytes()
        asset = 'files/' + relative
        previous = assets.get(asset)
        if relative in RUNTIME:
            before = install.digest((BEFORE / relative).read_bytes())
            if previous:
                assert previous['sha256'] == before, relative
            operation = operations.setdefault(relative, dict(kind='replace', path=relative, asset=asset, known_before=[]))
            operation['known_before'] = sorted(set(operation['known_before']) | {before})
        install.atomic_write(ROOT / asset, data)
        assets[asset] = dict(path=asset, bytes=len(data), sha256=install.digest(data),
                             role='runtime' if relative in RUNTIME else 'verification')
        manifest['source_fingerprints'][relative] = install.digest(data)
    manifest.update(
        name='A1111 LoRA Studio v15 - coordinated asset caches and settings conflicts',
        created_utc=datetime.now(timezone.utc).isoformat(),
        source='v14 plus verified independent audit fixes; full v14 audit findings tracked separately; GPU acceptance pending',
        assets=sorted(assets.values(), key=lambda item: item['path']),
        operations=list(operations.values()),
        required_runtime_paths=sorted(set(manifest['required_runtime_paths']) | set(RUNTIME)),
    )
    install.save_json(ROOT / 'manifest.json', manifest)
    install.verify_package(ROOT)
    print(f'Built v15: {len(assets)} assets; {len(operations)} operations.')


if __name__ == '__main__':
    main()
