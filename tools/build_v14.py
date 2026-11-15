"""Prepare v14 from v13 without mutating any earlier release."""
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import install  # noqa: E402

SOURCE = Path('D:/Users/Asus/Music/Musify/test/stable-diffusion-webui-dev')
CHANGED = ('extensions-builtin/Lora/networks.py', 'test/test_runtime_external_audit.py')


def main():
    # The prior interrupted build may already have copied one changed asset.
    # Read the last valid manifest and verify only after all coupled hashes are
    # regenerated below; never install from this intermediate state.
    manifest = install.read_json(ROOT / 'manifest.json')
    assets = {item['path']: item for item in manifest['assets']}
    operations = {item['path']: item for item in manifest['operations']}
    for relative in CHANGED:
        data = (SOURCE / relative).read_bytes()
        asset = 'files/' + relative
        previous = assets[asset]['sha256']
        install.atomic_write(ROOT / asset, data)
        assets[asset] = dict(path=asset, bytes=len(data), sha256=install.digest(data), role=assets[asset]['role'])
        operation = operations.get(relative)
        if operation is not None:
            operation['known_before'] = sorted(set(operation['known_before']) | {previous})
        manifest['source_fingerprints'][relative] = install.digest(data)
    manifest.update(
        name='A1111 LoRA Studio v14 - deep LoRA cache audit correction candidate',
        created_utc=datetime.now(timezone.utc).isoformat(),
        source='v13 plus independently reproduced LoRA cache replacement correction; CPU and installed-host smoke, GPU acceptance pending',
        assets=sorted(assets.values(), key=lambda item: item['path']),
        operations=list(operations.values()),
    )
    install.save_json(ROOT / 'manifest.json', manifest)
    install.verify_package(ROOT)
    print(f'Built v14: {len(assets)} assets; {len(operations)} operations.')


if __name__ == '__main__':
    main()
