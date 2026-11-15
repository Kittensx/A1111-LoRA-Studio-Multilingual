"""In-process authentication tests; no model, real config or HTTP listener."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import builtins

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'extensions-builtin/Lora'))
import lora_presets  # noqa: E402
from lora_studio_access import create_access_dependency  # noqa: E402


class AccessTests(unittest.TestCase):
    def test_callback_does_not_import_extension_after_script_path_restored(self):
        tree = ast.parse((ROOT / 'extensions-builtin/Lora/scripts/lora_script.py').read_bytes())
        imports = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == 'lora_studio_access']
        self.assertEqual(len(imports), 1, 'extension dependency must be bound during script loading')
        real_import = builtins.__import__
        def no_late_import(name, *args, **kwargs):
            if name == 'lora_studio_access':
                raise ModuleNotFoundError('extension sys.path has been restored')
            return real_import(name, *args, **kwargs)
        with patch.object(builtins, '__import__', no_late_import):
            client, _ = self.fixture()
            with client:
                self.assertEqual(client.get('/sdapi/v1/lora-studio/presets').status_code, 200)

    def fixture(self, api_auth=None, ui_auth=None, cli_ui_auth=None):
        writes = []
        opts = NS(lora_studio_presets='{"schema_version":3,"presets":[]}', lora_block_weight_presets='', lora_elemental_weight_presets='')
        opts.set = lambda key, value, **kw: setattr(opts, key, value)
        opts.save = lambda filename: writes.append(filename)
        opts.set_and_save = lambda key, value, filename: (opts.set(key, value), opts.save(filename))
        env = dict(FastAPI=FastAPI, HTTPException=HTTPException, lora_presets=lora_presets, create_access_dependency=create_access_dependency,
                   shared=NS(opts=opts, config_filename='memory', cmd_opts=NS(api_auth=api_auth, gradio_auth=cli_ui_auth)),
                   networks=NS(available_networks={}, list_available_networks=lambda: [], get_network_on_disk=lambda name: None))
        tree = ast.parse((ROOT / 'extensions-builtin/Lora/scripts/lora_script.py').read_bytes())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'api_networks')
        module = ast.Module(body=[ast.ImportFrom('__future__', [ast.alias('annotations')], 0), function], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), 'production-api_networks', 'exec'), env)
        app = FastAPI()
        app.auth, app.tokens = ui_auth, {'valid-session': 'ui-user'}
        env['api_networks'](None, app)
        return TestClient(app), writes

    def put(self, client, **kwargs):
        return client.put('/sdapi/v1/lora-studio/presets', json={'schema_version': 3, 'presets': []}, **kwargs)

    def test_every_lora_route_requires_auth(self):
        client, writes = self.fixture('api:secret')
        routes = [('GET', '/sdapi/v1/loras'), ('POST', '/sdapi/v1/refresh-loras'),
                  ('GET', '/sdapi/v1/lora-studio/coverage?name=x'), ('GET', '/sdapi/v1/lora-studio/sidecar?name=x'),
                  ('GET', '/sdapi/v1/lora-studio/presets'), ('PUT', '/sdapi/v1/lora-studio/presets')]
        with client:
            for method, path in routes:
                with self.subTest(path=path, method=method):
                    self.assertEqual(client.request(method, path, json={}).status_code, 401)
        self.assertEqual(writes, [])

    def test_basic_auth_correct_wrong_and_malformed(self):
        client, writes = self.fixture('api:secret,second:other:password')
        with client:
            self.assertEqual(self.put(client, auth=('api', 'wrong')).status_code, 401)
            self.assertEqual(self.put(client, auth=('unknown', 'secret')).status_code, 401)
            self.assertEqual(self.put(client, headers={'Authorization': 'Basic invalid'}).status_code, 401)
            self.assertEqual(writes, [])
            self.assertEqual(self.put(client, auth=('api', 'secret')).status_code, 200)
            self.assertEqual(self.put(client, auth=('second', 'other:password')).status_code, 200)
        self.assertEqual(len(writes), 2)

    def test_gradio_sessions_and_mixed_mode(self):
        for api in (None, 'api:secret'):
            for cookie in ('access-token', 'access-token-unsecure'):
                with self.subTest(api=api, cookie=cookie):
                    client, writes = self.fixture(api, ui_auth={'ui-user': 'ui-password'})
                    with client:
                        self.assertEqual(self.put(client).status_code, 401)
                        client.cookies.set(cookie, 'fake-session')
                        self.assertEqual(self.put(client).status_code, 401)
                        client.cookies.set(cookie, 'valid-session')
                        self.assertEqual(self.put(client).status_code, 200)
                    self.assertEqual(len(writes), 1)

    def test_session_is_not_an_api_only_bypass(self):
        client, writes = self.fixture('api:secret')
        with client:
            client.cookies.set('access-token', 'valid-session')
            self.assertEqual(self.put(client).status_code, 401)
        self.assertEqual(writes, [])

    def test_no_auth_mode_and_configured_ui_without_session_backend(self):
        client, writes = self.fixture()
        with client:
            self.assertEqual(self.put(client).status_code, 200)
        self.assertEqual(len(writes), 1)
        client, writes = self.fixture(cli_ui_auth='user:password')
        with client:
            self.assertEqual(self.put(client).status_code, 401)
        self.assertEqual(writes, [])

    def test_cookie_writes_reject_foreign_origins(self):
        client, writes = self.fixture(ui_auth={'ui-user': 'password'})
        with client:
            client.cookies.set('access-token', 'valid-session')
            for origin in ('https://foreign.example', 'null', 'http://testserver.evil'):
                self.assertEqual(self.put(client, headers={'Origin': origin}).status_code, 403)
            self.assertEqual(self.put(client, headers={'Sec-Fetch-Site': 'cross-site'}).status_code, 403)
            self.assertEqual(writes, [])
            self.assertEqual(self.put(client, headers={'Origin': 'http://testserver'}).status_code, 200)
        self.assertEqual(len(writes), 1)


if __name__ == '__main__':
    unittest.main()
