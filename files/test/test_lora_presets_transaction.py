"""Desired-contract audit of production Options + Studio HTTP routes.

Run with system Python (FastAPI installed). All disk writes are mocked.
Assertions cover publication, rejection and write-failure boundaries.
"""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, mock_open, patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'extensions-builtin/Lora'))
import lora_presets  # noqa: E402
from lora_studio_access import create_access_dependency  # noqa: E402


def selected(path, names, **env):
    tree = ast.parse(path.read_bytes(), str(path))
    body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert {node.name for node in body} == set(names)
    body.insert(0, ast.ImportFrom('__future__', [ast.alias('annotations')], 0))
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), 'exec'), env)
    return env


class PresetContracts(unittest.TestCase):
    def fixture(self, hidden=False):
        flags = NS(freeze_settings=False, freeze_settings_in_sections=None,
                   freeze_specific_settings=None, hide_ui_dir_config=False,
                   api_auth=None, gradio_auth=None)
        types = selected(ROOT / 'modules/options.py', {'OptionInfo', 'Options'},
                         cmd_opts=flags, json=json,
                         options_builtin_fields={'data_labels', 'data', 'restricted_opts', 'typemap'})
        info = types['OptionInfo']
        old = json.dumps({'schema_version': 3, 'presets': [{
            'schema_version': 3, 'preset_id': 'keep', 'title': 'Keep',
            'weights': {'te': None, 'unet': None}, 'hires': None, 'elemental_weights': None}]})
        options = types['Options']({
            'lora_studio_presets': info(old, component_args={'visible': not hidden}),
            'lora_block_weight_presets': info(''), 'lora_elemental_weight_presets': info(''),
        }, set())
        # Assert this fixture is valid independently of the request we test.
        self.assertEqual(len(lora_presets.parse_preset_library(old)), 1)
        env = selected(ROOT / 'extensions-builtin/Lora/scripts/lora_script.py', {'api_networks'},
                       FastAPI=FastAPI, HTTPException=HTTPException, lora_presets=lora_presets, create_access_dependency=create_access_dependency,
                       shared=NS(opts=options, cmd_opts=flags, config_filename='NEVER_WRITE_REAL_CONFIG.json'),
                       networks=NS(available_networks={}))
        app = FastAPI()
        env['api_networks'](None, app)
        return TestClient(app, raise_server_exceptions=False), options, old

    def put(self, client, **kwargs):
        return client.put('/sdapi/v1/lora-studio/presets', json={'schema_version': 3, 'presets': []}, **kwargs)

    def test_success_path_with_real_options(self):
        client, opts, old = self.fixture()
        with client, patch('modules.config_store.atomic_save') as opened:
            response = self.put(client)
        self.assertEqual(response.status_code, 200)
        self.assertNotEqual(opts.lora_studio_presets, old)
        self.assertTrue(opened.called)

    def test_failed_save_does_not_publish_unsaved_library(self):
        client, opts, old = self.fixture()
        with client, patch('modules.config_store.atomic_save', side_effect=OSError('injected read-only config')):
            response = self.put(client)
            after = client.get('/sdapi/v1/lora-studio/presets')
        self.assertEqual(response.status_code, 409)
        self.assertEqual(opts.lora_studio_presets, old, 'failed save already changed live options')
        self.assertEqual(len(after.json()['presets']), 1)

    def test_refused_setting_is_not_reported_success(self):
        client, opts, old = self.fixture(hidden=True)
        with client, patch('modules.config_store.atomic_save'):
            response = self.put(client)
        self.assertEqual(opts.lora_studio_presets, old)
        self.assertEqual(response.status_code, 409, 'Options.set returned False, endpoint ignored refusal')

    def test_failed_write_never_truncates_live_config(self):
        client, opts, old = self.fixture()
        opened = mock_open()
        temporary = mock_open()
        temporary.return_value.name = 'injected-temporary-file'
        temporary.return_value.write = Mock(side_effect=OSError('injected disk full'))
        with client, patch('builtins.open', opened), patch('modules.config_store.tempfile.NamedTemporaryFile', temporary), patch('modules.config_store.os.unlink'), patch('modules.config_store.os.replace') as replace:
            response = self.put(client)
        self.assertEqual(response.status_code, 409)
        replace.assert_not_called()
        self.assertEqual(opts.lora_studio_presets, old)
        self.assertFalse(any(call.args[:2] == ('NEVER_WRITE_REAL_CONFIG.json', 'w')
                             for call in opened.call_args_list), 'live config opened in truncating mode before failed write')

    def test_malformed_origin_is_rejected_not_server_error(self):
        client, opts, old = self.fixture()
        with client, patch('builtins.open', mock_open()):
            response = self.put(client, headers={'Origin': 'http://['})
        self.assertEqual(opts.lora_studio_presets, old)
        self.assertEqual(response.status_code, 403)


if __name__ == '__main__':
    unittest.main(verbosity=2)
