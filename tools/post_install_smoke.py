"""Install in a NEW disposable host from local archives, run installed CPU tests.

Never overlays package reference/verification modules onto the installed host.
Only test/ files are copied after installation. No dependencies are installed.
The host and rollback journal are retained for inspection (no recursive delete).
"""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import zipfile

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
import install  # noqa: E402


def extract(archive_path, target, strip_root=False):
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            parts = PurePosixPath(entry.filename).parts
            name = '/'.join(parts[1:] if strip_root else parts)
            # Same explicit path-safety rules as installer, even for local ZIPs.
            install.atomic_write(install.child(target, name), archive.read(entry))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-zip', type=Path, required=True)
    parser.add_argument('--modules-zip', type=Path, required=True)
    parser.add_argument('--temp-root', type=Path, required=True)
    parser.add_argument('--torch-python', type=Path, required=True)
    parser.add_argument('--docs-language', choices=('EN', 'RUS'), default=None, help='Optionally install documentation in the disposable test host.')
    args = parser.parse_args()
    host = Path(tempfile.mkdtemp(prefix='lora-v15-installed-', dir=args.temp_root))
    print(f'Retained test host: {host}', flush=True)
    extract(args.base_zip, host, strip_root=True)
    extract(args.modules_zip, host)
    before = {p.relative_to(host).as_posix(): p.read_bytes() for p in host.rglob('*') if p.is_file()}
    _, _, plan = install.plan_install(host, PACKAGE, docs_language=args.docs_language)
    install.show_plan(plan)
    # Unknown baseline versions are allowed ONLY in this newly-created test host.
    # Normal install.py still refuses them. Never use this flag on a live WebUI.
    backup_id = install.install(host, PACKAGE, allow_unrecognized=True, docs_language=args.docs_language)
    manifest = install.verify_package(PACKAGE)
    for op in install.selected_operations(manifest, args.docs_language):
        if op['kind'] == 'replace':
            assert (host / op['path']).read_bytes() == (PACKAGE / op['asset']).read_bytes()
    for path in (PACKAGE / 'files/test').iterdir():
        if path.is_file() and path.suffix in ('.py', '.json', '.js', '.cjs', '.html'):
            install.atomic_write(host / 'test' / path.name, path.read_bytes())
    env = dict(os.environ, TEMP=str(args.temp_root), TMP=str(args.temp_root))
    env.pop('WEBUI_AUDIT_BASELINE', None)
    env.pop('PYTHONPATH', None)
    result = subprocess.run([str(args.torch_python), '-B', '-m', 'unittest', 'discover', '-s', 'test', '-p', 'test_runtime_*.py'], cwd=host, env=env)
    if result.returncode:
        raise RuntimeError(f'Installed-host tests failed; inspect {host}')
    env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    parser_tests = ['test/test_lora_interaction_contract.py', 'test/test_config_store.py',
                    'test/test_lora_studio_access.py', 'test/test_lora_presets_transaction.py',
                    'test/test_lora_advanced.py', 'test/test_lora_schema.py',
                    'test/test_extra_networks_parser.py', 'test/test_lora_quick_presets.py']
    subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', '--noconftest', '-o', 'addopts=', *parser_tests],
                   cwd=host, env=env, check=True)
    node_tests = [str(path.relative_to(host)) for path in sorted((host / 'test').glob('lora*.test.cjs'))]
    subprocess.run(['node', '--test', *node_tests], cwd=host, env=env, check=True)
    _, _, after = install.plan_install(host, PACKAGE, docs_language=args.docs_language)
    assert all(row['status'] == 'SAME' for row in after)
    count = install.rollback(host, backup_id)
    for op in install.selected_operations(manifest, args.docs_language):
        filename = host / op['path']
        if op['path'] in before:
            assert filename.read_bytes() == before[op['path']], op['path']
        else:
            assert not filename.exists(), op['path']
    print(json.dumps(dict(installed_tests='PASS', rollback_files=count, host=str(host)), indent=2))


if __name__ == '__main__':
    main()
