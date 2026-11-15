"""Collect reviewed external-audit corrections; never rewrite the v11 release."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import install  # noqa: E402
from _historical_build_paths import resolve_build_paths

CHANGED = (
    'modules/asset_files.py', 'modules/sd_vae.py', 'modules/sd_models.py',
    'modules/sd_models_config.py', 'modules/sd_hijack_clip_old.py',
    'modules/textual_inversion/textual_inversion.py', 'modules/options.py',
    'modules/region_utils.py', 'modules/ui_settings.py', 'modules/shared_options.py',
    'extensions-builtin/Lora/scripts/lora_script.py', 'extensions-builtin/Lora/network.py',
    'extensions-builtin/Lora/ui_extra_networks_lora.py',
    'extensions-builtin/Lora/LORA_EVOLUTION_PLAN.md',
    'extensions-builtin/Lora/RUNTIME_STABILITY_FIXES.md',
)
TESTS = ('test/test_runtime_external_audit.py', 'test/test_lora_studio_access.py',
         'test/test_lora_presets_transaction.py', 'test/test_config_store.py')


def build():

    global ROOT, SOURCE, BASE, BACKUP
    args = resolve_build_paths(12, needs_backup=True,
                              previous='A1111-LoRA-Studio-Pack-v11',
                              source_paths=(*CHANGED, *TESTS, "modules.zip"),
                              backup_paths=('modules/processing.py',) if 12 == 10 else ())
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
        op = operations.setdefault(relative, dict(path=relative, asset=name, kind='replace', known_before=[]))
        assert op['kind'] == 'replace' and op['asset'] == name
        known = set(op['known_before'])
        if name in assets:
            known.add(assets[name]['sha256'])
        previous = install.child(BACKUP, relative)
        if previous.is_file():
            known.add(install.file_hash(previous))
        # Reviewed old REGION baseline from the user's local modules archive.
        # Unknown edits still block; this does not grant arbitrary replacement.
        if relative == 'modules/region_utils.py':
            with zipfile.ZipFile(SOURCE / 'modules.zip') as archive:
                known.add(install.digest(archive.read(relative)))
        op['known_before'] = sorted(known)
        asset(name, data, 'runtime')
        manifest['source_fingerprints'][relative] = install.digest(data)
    for relative in TESTS:
        asset('files/' + relative, (SOURCE / relative).read_bytes(), 'verification-only-not-installed')

    manifest.update(name='A1111 LoRA Studio v12 - external audit corrections candidate',
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    source='v11 + supplied Nov 8 audit corrections; CPU and installed-host smoke, GPU acceptance pending',
                    assets=sorted(assets.values(), key=lambda item: item['path']), operations=list(operations.values()))
    manifest['required_runtime_paths'] = ['modules/region_utils.py', 'modules/asset_files.py',
                                          'extensions-builtin/Lora/lora_studio_access.py']
    # Every declared runtime asset must actually be installed (or have an explicit recipe).
    installed_assets = {op['asset'] for op in manifest['operations']}
    assert all(item['path'] in installed_assets for item in manifest['assets'] if item['role'] == 'runtime')
    install.save_json(ROOT / 'manifest.json', manifest)
    install.verify_package(ROOT)
    print(f'Built {len(assets)} assets; {len(operations)} operations.')


if __name__ == '__main__':
    build()
