"""Real temporary-file config transactions; no WebUI/Torch/user config."""
from concurrent.futures import ThreadPoolExecutor
import ast
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'test'))
from modules import config_store  # noqa: E402
from test_lora_presets_transaction import selected  # noqa: E402


class ConfigTests(unittest.TestCase):
    def test_api_reports_conflict_without_saving_pending_or_rejected_value(self):
        from fastapi import HTTPException
        tree = ast.parse((ROOT / 'modules/api/api.py').read_text(encoding='utf-8'))
        api = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Api')
        method = next(node for node in api.body if isinstance(node, ast.FunctionDef) and node.name == 'set_config')
        # Execute the actual API method without importing WebUI/model dependencies.
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method], type_ignores=[])
        ast.fix_missing_locations(module)
        opts, _ = self.options()
        entered, release = threading.Event(), threading.Event()
        def callback():
            entered.set()
            if not release.wait(5):
                raise RuntimeError('timeout')
            raise ValueError('injected failure')
        opts.data_labels['a'].onchange = callback
        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(max_workers=1) as pool:
            path = Path(directory) / 'config.json'
            opts.save(path)
            before = path.read_bytes()
            env = dict(shared=NS(opts=opts, config_filename=path), sd_models=NS(checkpoint_aliases={}), HTTPException=HTTPException)
            exec(compile(module, 'modules/api/api.py', 'exec'), env)
            request = env['set_config']
            pending = pool.submit(opts.set, 'a', 'first')
            try:
                self.assertTrue(entered.wait(5))
                for body in ({'a': 'next'}, {}):
                    with self.assertRaises(HTTPException) as error:
                        request(None, body)
                    self.assertEqual(error.exception.status_code, 409)
                    self.assertEqual(path.read_bytes(), before)
            finally:
                release.set()
            self.assertFalse(pending.result(timeout=5))
            with self.assertRaises(HTTPException) as error:
                request(None, {'a': 'fails'})
            self.assertEqual(error.exception.status_code, 409)
            self.assertEqual(path.read_bytes(), before)
            opts.data_labels['a'].onchange = None
            request(None, {'a': 'retry'})
            self.assertEqual(json.loads(path.read_text())['a'], 'retry')

    def test_slow_callback_cannot_finish_after_newer_value_unreconciled(self):
        opts, _ = self.options()
        entered, release = threading.Event(), threading.Event()
        applied = []
        def callback():
            value = opts.a
            if value == 'first':
                entered.set()
                if not release.wait(5):
                    raise RuntimeError('test synchronization timed out')
            applied.append(value)
        opts.data_labels['a'].onchange = callback
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(opts.set, 'a', 'first')
            try:
                self.assertTrue(entered.wait(5))
                second = pool.submit(opts.set, 'a', 'latest')
                with self.assertRaisesRegex(RuntimeError, 'still being applied'):
                    second.result(timeout=5)
            finally:
                release.set()
            self.assertTrue(first.result(timeout=5))
        self.assertEqual(opts.a, 'first')
        self.assertTrue(opts.set('a', 'latest'))
        self.assertEqual(applied[-1], 'latest')
        self.assertEqual(applied, ['first', 'latest'])

    def test_data_only_update_does_not_become_an_implicit_callback(self):
        opts, _ = self.options()
        entered, release = threading.Event(), threading.Event()
        applied = []

        def callback():
            entered.set()
            if not release.wait(5):
                raise RuntimeError('test synchronization timed out')
            applied.append(opts.a)  # Read AFTER the concurrent data-only attempt.

        opts.data_labels['a'].onchange = callback
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(opts.set, 'a', 'first')
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaisesRegex(RuntimeError, 'still being applied'):
                    opts.set('a', 'data-only', run_callbacks=False)
            finally:
                release.set()
            self.assertTrue(first.result(timeout=5))
        self.assertEqual(opts.a, 'first')
        self.assertEqual(applied, ['first'])
        self.assertTrue(opts.set('a', 'data-only', run_callbacks=False))
        self.assertEqual(opts.a, 'data-only')
        self.assertEqual(applied, ['first'])

    def test_pending_failed_callback_cannot_be_acknowledged_or_saved(self):
        opts, _ = self.options()
        entered, release = threading.Event(), threading.Event()
        def callback():
            entered.set()
            if not release.wait(5):
                raise RuntimeError('timeout')
            raise ValueError('injected failure')
        opts.data_labels['a'].onchange = callback
        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(max_workers=1) as pool:
            path = Path(directory) / 'config.json'
            opts.save(path)
            before = path.read_bytes()
            first = pool.submit(opts.set, 'a', 'first')
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaises(RuntimeError):
                    opts.set('a', 'first')  # Even same-value requests are pending.
                with self.assertRaises(RuntimeError):
                    opts.save(path)
                with self.assertRaises(RuntimeError):
                    opts.set_and_save('b', 'preset', path)
                self.assertEqual(path.read_bytes(), before)
            finally:
                release.set()
            self.assertFalse(first.result(timeout=5))
            self.assertEqual(opts.a, 'old')
            self.assertEqual(path.read_bytes(), before)
            opts.save(path)

    def options(self):
        flags = NS(freeze_settings=False, freeze_settings_in_sections=None,
                   freeze_specific_settings=None, hide_ui_dir_config=False)
        env = selected(ROOT / 'modules/options.py', {'OptionInfo', 'Options'},
                       cmd_opts=flags, json=json, errors=NS(display=lambda *args: None),
                       options_builtin_fields={'data_labels', 'data', 'restricted_opts', 'typemap'})
        opts = env['Options']({key: env['OptionInfo']('old') for key in ('a', 'b')}, set())
        return opts, flags

    def test_real_atomic_success_and_noop(self):
        opts, _ = self.options()
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'config.json'
            opts.save(filename)
            self.assertTrue(opts.set_and_save('a', 'new', filename))
            self.assertFalse(opts.set_and_save('a', 'new', filename))
            self.assertEqual(json.loads(filename.read_text()), opts.data)
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])

    def test_commit_failures_preserve_real_file_and_memory(self):
        for target in ('os.fsync', 'os.replace', 'tempfile.NamedTemporaryFile'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                opts, _ = self.options()
                filename = Path(directory) / 'config.json'
                opts.save(filename)
                before = filename.read_bytes()
                with patch('modules.config_store.' + target, side_effect=OSError('injected I/O failure')):
                    with self.assertRaises(OSError):
                        opts.set_and_save('a', 'new', filename)
                self.assertEqual(filename.read_bytes(), before)
                self.assertEqual(opts.a, 'old')
                self.assertEqual(list(Path(directory).glob('*.tmp')), [])
                opts.set_and_save('a', 'retry', filename)
                self.assertEqual(json.loads(filename.read_text())['a'], 'retry')

    def test_serialization_failure_does_not_truncate_file(self):
        opts, _ = self.options()
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'config.json'
            opts.save(filename)
            before = filename.read_bytes()
            with self.assertRaises(TypeError):
                opts.set_and_save('a', object(), filename)
            self.assertEqual(filename.read_bytes(), before)
            self.assertEqual(opts.a, 'old')

    def test_concurrent_transactions_do_not_lose_other_keys(self):
        opts, _ = self.options()
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'config.json'
            opts.save(filename)
            with ThreadPoolExecutor(max_workers=4) as pool:
                jobs = [pool.submit(opts.set_and_save, key, str(index), filename)
                        for index in range(12) for key in ('a', 'b')]
                for job in jobs:
                    job.result(timeout=10)
            result = json.loads(filename.read_text())
            self.assertEqual(result, opts.data)
            self.assertNotEqual(result['a'], 'old')
            self.assertNotEqual(result['b'], 'old')
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])

    def test_frozen_and_restricted_settings_never_write(self):
        for restriction in ('freeze_settings', 'hidden', 'restrict_api'):
            opts, flags = self.options()
            if restriction == 'freeze_settings':
                flags.freeze_settings = True
            elif restriction == 'hidden':
                opts.data_labels['a'].component_args = {'visible': False}
            else:
                opts.data_labels['a'].restrict_api = True
            with patch.object(config_store, 'atomic_save') as save:
                with self.assertRaises((AssertionError, RuntimeError)):
                    opts.set_and_save('a', 'new', 'not-used.json')
                save.assert_not_called()
            self.assertEqual(opts.a, 'old')

    def test_callback_does_not_hold_config_lock(self):
        opts, _ = self.options()
        acquired_during_callback = []
        workers = []

        def callback():
            # Simulate a queued generation changing another option while the
            # settings callback waits for it. Timed join also makes failure safe.
            worker = threading.Thread(target=lambda: opts.set('b', 'generation'), daemon=True)
            workers.append(worker)
            worker.start()
            worker.join(timeout=2)
            acquired_during_callback.append(not worker.is_alive())

        opts.data_labels['a'].onchange = callback
        self.assertTrue(opts.set('a', 'new'))
        for worker in workers:
            worker.join(timeout=2)
        self.assertEqual(acquired_during_callback, [True])
        self.assertEqual(opts.b, 'generation')


if __name__ == '__main__':
    unittest.main()
