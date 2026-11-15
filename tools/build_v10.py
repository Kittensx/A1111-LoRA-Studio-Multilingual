"""Mechanical collection of the v10 candidate; keeps v9 immutable."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import install  # noqa: E402
from _historical_build_paths import resolve_build_paths

CHANGED = (
    'extensions-builtin/Lora/network.py', 'extensions-builtin/Lora/network_lora.py',
    'extensions-builtin/Lora/networks.py', 'extensions-builtin/Lora/lora_keys.py',
    'extensions-builtin/Lora/lora_studio_access.py', 'extensions-builtin/Lora/scripts/lora_script.py',
    'modules/textual_inversion/textual_inversion.py', 'modules/options.py',
    'modules/config_store.py', 'modules/lowvram.py', 'modules/sd_emphasis.py',
    'extensions-builtin/Lora/RUNTIME_STABILITY_FIXES.md', 'extensions-builtin/Lora/LORA_EVOLUTION_PLAN.md',
)


def build():

    global ROOT, SOURCE, BASE, BACKUP
    args = resolve_build_paths(10, needs_backup=True,
                              previous='A1111-LoRA-Studio-Pack-v9',
                              source_paths=(*CHANGED, "modules/processing.py", "modules/sd_hijack.py",
                                            "modules/api/api.py", "modules/script_callbacks.py",
                                            "modules/region_utils.py", "test/test_lora_studio_access.py",
                                            "test/test_lora_presets_transaction.py", "test/test_config_store.py"),
                              backup_paths=('modules/processing.py',) if 10 == 10 else ())
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
        known = set(op['known_before'])
        if name in assets:
            known.add(assets[name]['sha256'])
        previous = install.child(BACKUP, relative)
        if previous.is_file():
            known.add(install.file_hash(previous))
        op['known_before'] = sorted(known)
        asset(name, data, 'runtime')
        manifest['source_fingerprints'][relative] = install.digest(data)

    # Extend the existing contextual patch; never replace this large custom file.
    name = 'patches/processing-hires.json'
    recipe = json.loads((BASE / name).read_text(encoding='utf-8'))
    recipe['hunks'].extend([
        dict(before='            opts.emphasis,\n        )',
             after='            opts.emphasis,\n            model_hijack.embedding_db.revision,\n            getattr(shared.sd_model, \'lora_model_token\', None),\n            getattr(shared.sd_model, \'lora_conditioning_signature\', ()),\n        )'),
        dict(before='        caches is a list with items described above.\n        """\n\n        if shared.opts.use_old_scheduling:',
             after='        caches is a list with items described above.\n        """\n\n        model_hijack.embedding_db.ensure_model_compatibility()\n        if shared.opts.use_old_scheduling:'),
    ])
    previous = (BACKUP / 'modules/processing.py').read_bytes()
    current = (SOURCE / 'modules/processing.py').read_bytes()
    assert install.patch_text(previous, recipe).replace(b'\r\n', b'\n') == current.replace(b'\r\n', b'\n')
    assert install.patch_text(current, recipe) == current
    asset(name, (json.dumps(recipe, indent=2, ensure_ascii=False) + '\n').encode(), 'patch')
    asset('reference-only/modules/processing.py', current, 'reference-only-not-installed')
    asset('files/modules/processing.py', current, 'verification-only-not-installed')
    manifest['source_fingerprints']['modules/processing.py'] = install.digest(current)

    tests = list((SOURCE / 'test').glob('test_runtime_*.py'))
    tests += [SOURCE / 'test' / name for name in ('test_lora_studio_access.py', 'test_lora_presets_transaction.py', 'test_config_store.py')]
    for path in tests:
        asset('files/' + path.relative_to(SOURCE).as_posix(), path.read_bytes(), 'verification-only-not-installed')
    # Host files needed by new isolated production-AST tests, not install targets.
    for relative in ('modules/sd_hijack.py', 'modules/api/api.py', 'modules/script_callbacks.py', 'modules/region_utils.py'):
        asset('files/' + relative, (SOURCE / relative).read_bytes(), 'verification-only-not-installed')
    manifest.update(name='A1111 LoRA Studio v10 - cross-module and loader corrections candidate',
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    source='v9 + V9-1..5, C1..4, L1..2 corrections; CPU/in-process tests, actual user checkpoints/GPU/Gradio pending',
                    assets=sorted(assets.values(), key=lambda item: item['path']), operations=list(operations.values()))
    install.save_json(ROOT / 'manifest.json', manifest)
    install.verify_package(ROOT)
    print(f'Built {len(assets)} assets; {len(operations)} operations.')


if __name__ == '__main__':
    build()
