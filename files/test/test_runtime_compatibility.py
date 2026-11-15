"""Real CPU tensor/file compatibility checks; no WebUI boot or downloads.

Run: python -B test/test_runtime_compatibility.py
Temporary .pt files are self-generated test data, not downloaded checkpoints.
"""
from __future__ import annotations

import ast
import collections
import contextlib
import copy
import hashlib
import inspect
import logging
import os
import pickle
from pathlib import Path
import re
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch
import zipfile

import test_runtime_stability as harness

torch = harness.torch
selected = harness.selected


class CompatibilityTests(unittest.TestCase):
    def ti(self, optimizer=False):
        return selected('modules/textual_inversion/textual_inversion.py',
                        {'Embedding', 'create_embedding_from_data', 'save_embedding',
                         'save_embedding_files', 'embedding_file_hash', 'load_embedding_optimizer'},
                        torch=torch, copy=copy, hashlib=hashlib, tempfile=tempfile, os=os, logging=logging,
                        devices=NS(device=torch.device('cpu')),
                        shared=NS(opts=NS(save_optimizer_state=optimizer)))

    def safe_loader(self):
        # Production restricted unpickler and verification functions. Only
        # WebUI reporting/config is stubbed; no numpy paths are used here.
        reports = []
        safe = selected('modules/safe.py', {'RestrictedUnpickler', 'check_zip_filenames', 'check_pt', 'load_with_extra'},
                        torch=torch, pickle=pickle, collections=collections,
                        zipfile=zipfile, TypedStorage=torch.storage.TypedStorage,
                        unsafe_torch_load=torch.load, errors=NS(report=lambda *a, **k: reports.append(a)))
        tree = ast.parse((harness.ROOT / 'modules/safe.py').read_bytes())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('allowed_zip_names_re', 'data_pkl_re') for t in node.targets):
                exec(compile(ast.Module(body=[node], type_ignores=[]), 'production-safe-patterns', 'exec'), {'re': re}, safe.__dict__)
        dummy = ModuleType('modules')
        dummy.shared = NS(cmd_opts=NS(disable_safe_unpickle=False))
        return safe, reports, dummy

    def test_safe_loader_reads_self_saved_ti_and_optimizer(self):
        ti = self.ti(optimizer=True)
        embedding = ti.Embedding(torch.arange(12, dtype=torch.float32).reshape(3, 4), 'test', step=7)
        embedding.optimizer_state_dict = {'state': {0: {'step': torch.tensor(2.), 'exp_avg': torch.ones(3, 4)}}, 'param_groups': []}
        safe, reports, dummy = self.safe_loader()
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'modules': dummy}):
            path = Path(directory) / 'test.pt'
            embedding.save(path)
            data = safe.load_with_extra(path, map_location='cpu')
            optim = safe.load_with_extra(str(path) + '.optim', map_location='cpu')
            self.assertEqual(reports, [], 'safe loader rejected its own newly saved Torch files')
            restored = ti.create_embedding_from_data(data, 'test')
            torch.testing.assert_close(restored.vec, embedding.vec)
            self.assertEqual(optim['hash'], embedding.checksum())
            torch.testing.assert_close(optim['optimizer_state_dict']['state'][0]['exp_avg'], torch.ones(3, 4))

    def test_safe_loader_still_rejects_unknown_zip_members_and_pickle_globals(self):
        safe, reports, dummy = self.safe_loader()
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'modules': dummy}):
            path = Path(directory) / 'test.pt'
            torch.save({'tensor': torch.ones(2)}, path)
            with zipfile.ZipFile(path, 'a') as archive:
                archive.writestr('test/.format_version.evil', b'x')
            self.assertIsNone(safe.load_with_extra(path))
            self.assertTrue(reports)
            reports.clear()
            # A harmless but forbidden global. No executable malicious payload.
            torch.save({'unsupported': Path('example')}, path)
            self.assertIsNone(safe.load_with_extra(path))
            self.assertTrue(reports)

    def test_classic_ti_snapshot_roundtrip_does_not_mutate_live_state(self):
        ti = self.ti(optimizer=True)
        live = ti.Embedding(torch.arange(12, dtype=torch.float32).reshape(3, 4), 'live', step=9)
        checksum = live.checksum()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'saved.pt'
            ti.save_embedding(live, NS(state_dict=lambda: {'state': {}, 'param_groups': []}),
                              NS(shorthash='hash', model_name='model'), 'export', path)
            data = torch.load(path, map_location='cpu', weights_only=True)
            restored = ti.create_embedding_from_data(data, 'export')
            torch.testing.assert_close(restored.vec, live.vec)
            self.assertEqual((restored.step, restored.sd_checkpoint, restored.sd_checkpoint_name), (9, 'hash', 'model'))
            self.assertEqual((live.name, live.sd_checkpoint, live.cached_checksum), ('live', None, checksum))

    def test_sdxl_ti_roundtrip_with_metadata_and_optimizer(self):
        ti = self.ti(optimizer=True)
        data = {'clip_l': torch.arange(12, dtype=torch.float32).reshape(3, 4),
                'clip_g': torch.arange(18, dtype=torch.float32).reshape(3, 6), 'step': 7, 'sd_checkpoint': 'hash'}
        live = ti.create_embedding_from_data(data, 'xl')
        self.assertEqual((live.vectors, live.shape, live.step), (3, 10, 7))
        before = {key: value.clone() for key, value in live.vec.items()}
        safe, reports, dummy = self.safe_loader()
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'modules': dummy}):
            path = Path(directory) / 'xl.pt'
            ti.save_embedding(live, NS(state_dict=lambda: {'state': {}, 'param_groups': []}),
                              NS(shorthash='hash2', model_name='model'), 'export', path)
            saved = safe.load_with_extra(path, map_location='cpu', weights_only=True)
            self.assertEqual(reports, [])
            restored = ti.create_embedding_from_data(saved, 'export')
            for key, value in before.items():
                torch.testing.assert_close(restored.vec[key], value)
                torch.testing.assert_close(live.vec[key], value)
            self.assertEqual(restored.step, 7)
            optimizer = torch.load(str(path) + '.optim', weights_only=True)
            self.assertEqual(optimizer['hash'], restored.checksum())

    def test_sdxl_ti_requires_matching_vector_counts(self):
        ti = self.ti()
        with self.assertRaisesRegex(ValueError, 'vector|shape|row'):
            ti.create_embedding_from_data({'clip_l': torch.ones(2, 4), 'clip_g': torch.ones(3, 6)}, 'bad')

    def test_sdxl_checksum_tracks_both_encoders(self):
        ti = self.ti()
        live = ti.Embedding({'clip_l': torch.zeros(2, 4), 'clip_g': torch.zeros(2, 6)}, 'xl')
        first = live.checksum()
        live.cached_checksum = None
        live.vec['clip_g'][0, 0] = 1
        second = live.checksum()
        live.cached_checksum = None
        live.vec['clip_l'][0, 0] = 1
        self.assertEqual(len({first, second, live.checksum()}), 3)

    def test_sdxl_training_refused_before_state_or_dataset_changes(self):
        state = NS(job='idle', textinfo='untouched')
        shared = NS(state=state)
        mod = selected('modules/textual_inversion/textual_inversion.py', {'train_embedding'},
                       shared=shared, textual_inversion_templates={'test': NS(path='unused')},
                       validate_train_inputs=lambda *a, **k: None,
                       sd_hijack=NS(model_hijack=NS(embedding_db=NS(word_embeddings={'xl': NS(vec={'clip_l': torch.ones(1, 4), 'clip_g': torch.ones(1, 6)})}))))
        args = dict.fromkeys(inspect.signature(mod.train_embedding).parameters)
        args.update(embedding_name='xl', template_filename='test')
        dummy = ModuleType('modules')
        dummy.processing = NS()
        with patch.dict(sys.modules, {'modules': dummy}), self.assertRaisesRegex(ValueError, 'single-encoder trainer'):
            mod.train_embedding(**args)
        self.assertEqual((state.job, state.textinfo), ('idle', 'untouched'))

    def test_safe_loader_reads_legacy_serialization(self):
        safe, reports, dummy = self.safe_loader()
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'modules': dummy}):
            path = Path(directory) / 'legacy.pt'
            torch.save({'vec': torch.ones(2, 4)}, path, _use_new_zipfile_serialization=False)
            data = safe.load_with_extra(path, weights_only=True)
            self.assertEqual(reports, [])
            torch.testing.assert_close(data['vec'], torch.ones(2, 4))

    def test_fp8_cache_preserves_later_layers_and_disable_restores_fp8(self):
        h = harness.RuntimeTests()
        h.setUp()
        a, b = torch.nn.Linear(2, 2, bias=False), torch.nn.Linear(2, 2, bias=False)
        for layer in (a, b):
            with torch.no_grad():
                layer.weight.copy_(torch.tensor([[0.123, 0.251], [0.376, 0.512]]))
        precise = {id(layer): (layer.weight.detach().clone().half(), None) for layer in (a, b)}
        model = NS(_fp16_cache=dict(precise))
        for layer in (a, b):
            layer.to(torch.float8_e4m3fn)
        native = {id(layer): layer.weight.float().clone() for layer in (a, b)}
        finalizer = selected('extensions-builtin/Lora/networks.py', {'finalize_fp16_cache'}, loaded_networks=h.runtime.loaded_networks)
        finalizer.finalize_fp16_cache(model)
        self.assertEqual(len(model._fp16_cache), 2)
        for layer in (a, b):
            activation = h.activation(layer, value=0.125)
            activation.modules[layer.network_layer_name].sd_module = layer
            h.runtime.loaded_networks[:] = [activation]
            finalizer.finalize_fp16_cache(model)
            torch.testing.assert_close(layer.fp16_weight, precise[id(layer)][0])
            h.runtime.network_apply_weights(layer)
            expected = (precise[id(layer)][0] + 0.125).to(torch.float8_e4m3fn).float()
            torch.testing.assert_close(layer.weight.float(), expected)
            h.runtime.loaded_networks.clear()
            h.runtime.network_apply_weights(layer)
            torch.testing.assert_close(layer.weight.float(), native[id(layer)])
        self.assertFalse(hasattr(model, '_fp16_cache'))

    def test_manual_cast_setup_failure_restores_earlier_classes(self):
        class First:
            def forward(self):
                return 'original'
        class Refuse(type):
            def __setattr__(cls, key, value):
                if key == 'org_forward':
                    raise RuntimeError('setup denied')
                super().__setattr__(key, value)
        class Second(metaclass=Refuse):
            def forward(self):
                return 'original'
        originals = (First.forward, Second.forward)
        dev = selected('modules/devices.py', {'manual_cast', 'manual_cast_forward'}, torch=torch,
                       contextlib=contextlib, patch_module_list=[First, Second], dtype_inference=torch.float32)
        try:
            with self.assertRaisesRegex(RuntimeError, 'setup denied'):
                with dev.manual_cast(torch.float16):
                    self.fail('failed setup must not enter body')
            self.assertEqual((First.forward, Second.forward), originals)
            self.assertFalse(hasattr(First, 'org_forward'))
            self.assertFalse(hasattr(Second, 'org_forward'))
        finally:
            First.forward, Second.forward = originals
            if hasattr(First, 'org_forward'):
                del First.org_forward

    def test_manual_cast_preserves_inherited_forward_and_nested_context(self):
        class Child(torch.nn.Linear):
            pass
        dev = selected('modules/devices.py', {'manual_cast', 'manual_cast_forward'}, torch=torch,
                       contextlib=contextlib, patch_module_list=[Child], dtype_inference=torch.float32)
        self.assertNotIn('forward', vars(Child))
        layer = Child(2, 2)
        with dev.manual_cast(torch.float16):
            outer = Child.forward
            with dev.manual_cast(torch.float16):
                self.assertIs(Child.forward, outer)
                self.assertEqual(layer(torch.ones(1, 2)).dtype, torch.float32)
            self.assertIs(Child.forward, outer)
        self.assertNotIn('forward', vars(Child))
        self.assertNotIn('org_forward', vars(Child))

    def test_lora_and_manual_cast_in_both_stack_orders(self):
        original = torch.nn.Linear.forward
        for lora_first in (True, False):
            with self.subTest(lora_first=lora_first):
                h = harness.RuntimeTests()
                h.setUp()
                registry = harness.import_file('modules/patches.py')
                hooks = selected('extensions-builtin/Lora/lora_patches.py', {'LoraPatches'}, torch=torch,
                                 networks=h.runtime, patches=registry)
                dev = selected('modules/devices.py', {'manual_cast', 'manual_cast_forward'}, torch=torch,
                               contextlib=contextlib, patch_module_list=[torch.nn.Linear], dtype_inference=torch.float32)
                layer = torch.nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    layer.weight.fill_(0.5)
                h.runtime.loaded_networks[:] = [h.activation(layer, value=0.25)]
                @contextlib.contextmanager
                def lora_hooks(hooks=hooks, h=h):
                    holder = hooks.LoraPatches()
                    h.runtime.originals = holder
                    try:
                        yield
                    finally:
                        holder.undo()
                contexts = (lora_hooks(), dev.manual_cast(torch.float16))
                if not lora_first:
                    contexts = contexts[::-1]
                with contexts[0], contexts[1]:
                    torch.testing.assert_close(layer(torch.ones(1, 2)), torch.full((1, 2), 1.5))
                    self.assertEqual(layer.weight.dtype, torch.float32)
                    h.runtime.loaded_networks.clear()
                    torch.testing.assert_close(layer(torch.ones(1, 2)), torch.ones(1, 2))
                self.assertIs(torch.nn.Linear.forward, original)
                self.assertNotIn('org_forward', vars(torch.nn.Linear))


if __name__ == '__main__':
    unittest.main(verbosity=2)
