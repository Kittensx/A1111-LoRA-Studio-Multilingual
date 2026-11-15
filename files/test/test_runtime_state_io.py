"""Real CPU state-dict and file-write regressions, without a WebUI server."""
from __future__ import annotations

import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import test_runtime_stability as harness
import test_runtime_compatibility as compatibility
import test_runtime_recovery as recovery

torch = harness.torch


class StateIOTests(unittest.TestCase):
    def meta(self, state, conversion=None):
        module = harness.selected('modules/sd_disable_initialization.py', {'ReplaceHelper', 'LoadStateDictOnMeta'},
            torch=torch, logging=logging, shared=NS(cmd_opts=NS(disable_model_loading_ram_optimization=False)))
        return module.LoadStateDictOnMeta(state, torch.device('cpu'), conversion)

    def test_meta_preserves_version_metadata(self):
        class Versioned(torch.nn.Module):
            _version = 7
            def __init__(self, device):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(2, device=device))
                self.seen_version = None
            def _load_from_state_dict(self, state, prefix, metadata, *args, **kwargs):
                self.seen_version = metadata.get('version')
                return super()._load_from_state_dict(state, prefix, metadata, *args, **kwargs)
        state = torch.nn.Sequential(Versioned('cpu')).state_dict()
        model = torch.nn.Sequential(Versioned('meta'))
        with self.meta(state):
            model.load_state_dict(state)
        self.assertEqual(model[0].seen_version, 7)
        torch.testing.assert_close(model[0].weight, torch.ones(2))

    def test_meta_preserves_dictionary_and_tensor_extra_state(self):
        class Extra(torch.nn.Module):
            def __init__(self, device, extra):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(2, device=device))
                self.extra = extra
            def get_extra_state(self):
                return self.extra
            def set_extra_state(self, value):
                self.extra = value
        for extra in ({'version': 1, 'labels': ['a', 'b']}, torch.arange(3)):
            with self.subTest(kind=type(extra).__name__):
                state = Extra('cpu', extra).state_dict()
                model = Extra('meta', None)
                with self.meta(state):
                    model.load_state_dict(state)
                if isinstance(extra, torch.Tensor):
                    torch.testing.assert_close(model.extra, extra)
                else:
                    self.assertEqual(model.extra, extra)
                torch.testing.assert_close(model.weight, torch.ones(2))

    def test_meta_materializes_converted_parameter_dtype(self):
        model = torch.nn.Linear(2, 2, device='meta')
        state = {'weight': torch.full((2, 2), 0.25), 'bias': torch.ones(2)}
        with self.meta(state, {'': torch.float16}):
            model.load_state_dict(state)
        self.assertEqual(model.weight.dtype, torch.float16)
        self.assertEqual(model.bias.dtype, torch.float16)
        torch.testing.assert_close(model.weight, torch.full((2, 2), 0.25, dtype=torch.float16))

    def test_embedding_serialization_failure_preserves_previous_files(self):
        ti = compatibility.CompatibilityTests().ti(optimizer=True)
        embedding = ti.Embedding(torch.ones(2, 3), 'original', step=1)
        embedding.optimizer_state_dict = {'state': {}, 'param_groups': []}
        for fail_at in (1, 2):
            with self.subTest(fail_at=fail_at), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'test.pt'
                embedding.save(path)
                before = {p.name: p.read_bytes() for p in Path(directory).iterdir()}
                real_save = torch.save
                calls = []
                def fail(data, destination, *args, calls=calls, fail_at=fail_at, real_save=real_save, **kwargs):
                    calls.append(destination)
                    if len(calls) == fail_at:
                        # Simulate a serializer which writes a prefix then fails.
                        if hasattr(destination, 'write'):
                            destination.write(b'incomplete')
                        else:
                            Path(destination).write_bytes(b'incomplete')
                        raise OSError('serialization interrupted')
                    return real_save(data, destination, *args, **kwargs)
                with patch.object(torch, 'save', side_effect=fail), self.assertRaisesRegex(OSError, 'serialization interrupted'):
                    embedding.save(path)
                self.assertEqual({p.name: p.read_bytes() for p in Path(directory).iterdir()}, before)

    def test_embedding_replace_failures_leave_readable_files_and_reject_stale_optimizer(self):
        ti = compatibility.CompatibilityTests().ti(optimizer=True)
        for fail_at in (1, 2):
            with self.subTest(fail_at=fail_at), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'saved.pt'
                old = ti.Embedding(torch.ones(2, 3), 'test', step=1)
                old.optimizer_state_dict = {'state': {}, 'param_groups': []}
                old.save(path)
                before = path.read_bytes()
                # Same vectors/checksum, different training step: old sidecar
                # must still be rejected after only the main file was committed.
                new = ti.Embedding(torch.ones(2, 3), 'test', step=2)
                new.optimizer_state_dict = {'state': {0: {'step': 2}}, 'param_groups': []}
                calls, replace = [], os.replace
                def fail(source, destination, calls=calls, replace=replace, fail_at=fail_at):
                    calls.append(destination)
                    if len(calls) == fail_at:
                        raise OSError('replace interrupted')
                    return replace(source, destination)
                with patch.object(os, 'replace', side_effect=fail), self.assertRaisesRegex(OSError, 'replace interrupted'):
                    new.save(path)
                self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ['saved.pt', 'saved.pt.optim'])
                restored = ti.create_embedding_from_data(torch.load(path, weights_only=True), 'test')
                if fail_at == 1:
                    self.assertEqual(path.read_bytes(), before)
                    self.assertEqual(ti.load_embedding_optimizer(path, restored), old.optimizer_state_dict)
                else:
                    self.assertEqual(restored.step, 2)
                    self.assertIsNone(ti.load_embedding_optimizer(path, restored))
                new.save(path)
                self.assertEqual(ti.load_embedding_optimizer(path, new), new.optimizer_state_dict)

    def test_legacy_and_rejected_optimizer_sidecars(self):
        ti = compatibility.CompatibilityTests().ti()
        embedding = ti.Embedding(torch.ones(2, 3), 'test')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'saved.pt'
            torch.save({'string_to_param': {'*': embedding.vec}}, path)  # genuinely legacy .pt
            payload = {'hash': embedding.checksum(), 'optimizer_state_dict': {'state': {}}}
            torch.save(payload, str(path) + '.optim')
            self.assertEqual(ti.load_embedding_optimizer(path, embedding), {'state': {}})
            torch.save(None, str(path) + '.optim')
            self.assertIsNone(ti.load_embedding_optimizer(path, embedding))
            torch.save(dict(payload, embedding_sha256='wrong'), str(path) + '.optim')
            self.assertIsNone(ti.load_embedding_optimizer(path, embedding))

    def test_new_embedding_cannot_fall_back_to_legacy_optimizer_after_partial_save(self):
        ti = compatibility.CompatibilityTests().ti(optimizer=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'saved.pt'
            embedding = ti.Embedding(torch.ones(2, 3), 'test', step=2)
            embedding.optimizer_state_dict = {'state': {0: {'step': 2}}}
            # Old vectors have the same legacy checksum, but training state is old.
            torch.save({'string_to_param': {'*': embedding.vec}, 'step': 1}, path)
            torch.save({'hash': embedding.checksum(), 'optimizer_state_dict': {'state': {0: {'step': 1}}}}, str(path) + '.optim')
            replace = os.replace
            def fail_sidecar(source, destination):
                if str(destination).endswith('.optim'):
                    raise OSError('sidecar locked')
                return replace(source, destination)
            with patch.object(os, 'replace', side_effect=fail_sidecar), self.assertRaisesRegex(OSError, 'sidecar locked'):
                embedding.save(path)
            restored = ti.create_embedding_from_data(torch.load(path, weights_only=True), 'test')
            self.assertEqual(restored.step, 2)
            self.assertIsNone(ti.load_embedding_optimizer(path, restored))

    def test_cached_fp8_selection_checks_target_storage_not_previous_model(self):
        for desired, stored in ((False, True), (True, False), (True, True), (False, False)):
            with self.subTest(desired=desired, stored=stored):
                mod, data, previous, candidate, _, _ = recovery.RecoveryTests().loader(extra_cached=True)
                previous.fp8 = False
                candidate.fp8 = stored
                mod.model_has_fp8_weights = lambda model: model.fp8
                mod.check_fp8 = lambda model, desired=desired, candidate=candidate: desired if model is candidate else False
                reads, loads = [], []
                mod.get_checkpoint_state_dict = lambda *args, reads=reads: reads.append('read') or {'weight': 'new'}
                def weights(model, *args, loads=loads, desired=desired, mod=mod):
                    loads.append(model)
                    model.fp8 = desired
                    mod.devices.fp8 = desired
                mod.load_model_weights = weights
                self.assertIs(mod.reload_model_weights(info=candidate.sd_checkpoint_info), candidate)
                self.assertEqual(len(loads), int(desired != stored))
                self.assertEqual(len(reads), int(desired != stored))
                self.assertEqual(mod.devices.fp8, desired)

    def test_fp8_detection_uses_real_tensor_storage(self):
        mod = harness.selected('modules/sd_models.py', {'model_has_fp8_weights'}, torch=torch)
        self.assertFalse(mod.model_has_fp8_weights(None))
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
        self.assertFalse(mod.model_has_fp8_weights(model))
        model[1].to(torch.float8_e4m3fn)
        self.assertTrue(mod.model_has_fp8_weights(model))
        model.half()
        self.assertFalse(mod.model_has_fp8_weights(model))

    def test_fp8_target_preflight_failure_preserves_active_model(self):
        mod, data, previous, candidate, _, events = recovery.RecoveryTests().loader(extra_cached=True)
        mod.model_has_fp8_weights = lambda model: model is candidate
        mod.check_fp8 = lambda model: False
        def fail(*args):
            raise OSError('checkpoint read failed')
        mod.get_checkpoint_state_dict = fail
        with self.assertRaisesRegex(OSError, 'checkpoint read failed'):
            mod.reload_model_weights(info=candidate.sd_checkpoint_info)
        self.assertIs(data.sd_model, previous)
        self.assertEqual(events, [])
        self.assertIn(candidate, data.loaded_sd_models)

    def test_loading_fp8_object_upcasts_even_if_global_mode_is_false(self):
        model = torch.nn.Linear(2, 2).to(torch.float8_e4m3fn)
        stop = RuntimeError('stop after storage preparation')
        def check(*args):
            self.assertEqual(model.weight.dtype, torch.float16)
            raise stop
        mod = harness.selected('modules/sd_models.py', {'model_has_fp8_weights', 'load_model_weights'},
            torch=torch, devices=NS(fp8=False), SkipWritingToConfig=NS(skip=True), set_model_type=check)
        with self.assertRaises(RuntimeError) as caught:
            mod.load_model_weights(model, NS(calculate_shorthash=lambda: 'hash'), {}, NS(record=lambda *a: None))
        self.assertIs(caught.exception, stop)

    def test_failed_temp_cleanup_does_not_replace_serialization_error(self):
        ti = compatibility.CompatibilityTests().ti()
        error = OSError('primary save error')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'saved.pt'
            with patch.object(torch, 'save', side_effect=error), patch.object(os, 'unlink', side_effect=PermissionError('temp locked')):
                with self.assertLogs(level='ERROR'), self.assertRaises(OSError) as caught:
                    ti.Embedding(torch.ones(1, 2), 'test').save(path)
            self.assertIs(caught.exception, error)
            self.assertFalse(path.exists())

    def test_meta_assign_and_dtype_exclusion_keep_values(self):
        model = torch.nn.Module()
        model.first_stage_model = torch.nn.Linear(2, 2, device='meta')
        model.other = torch.nn.Linear(2, 2, device='meta')
        state = {key: torch.full(value.shape, 0.25) for key, value in model.state_dict().items()}
        with self.meta(state, {'': torch.float16, 'first_stage_model': None}):
            model.load_state_dict(state, assign=True)
        self.assertEqual(model.first_stage_model.weight.dtype, torch.float32)
        self.assertEqual(model.other.weight.dtype, torch.float16)
        self.assertTrue(model.other.weight.requires_grad)
        torch.testing.assert_close(model.other.weight, torch.full((2, 2), 0.25, dtype=torch.float16))


if __name__ == '__main__':
    unittest.main(verbosity=2)
