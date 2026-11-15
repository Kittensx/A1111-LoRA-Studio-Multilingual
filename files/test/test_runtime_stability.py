"""Standalone CPU regressions: real torch, production AST, no WebUI boot.

Run: python -B test/test_runtime_stability.py
AST selection isolates unavailable UI/model dependencies, not tensor semantics.
"""
from __future__ import annotations

import ast
import contextlib
import copy
from collections import namedtuple
from dataclasses import dataclass
import importlib.util
import logging
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
LORA = ROOT / 'extensions-builtin/Lora'
sys.path.insert(0, str(LORA))
import lora_keys  # noqa: E402
import lora_runtime  # noqa: E402


def selected(relative, names, **env):
    path = ROOT / relative
    tree = ast.parse(path.read_bytes(), str(path))
    body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in body} == set(names)
    module = ModuleType('_runtime_' + relative.replace('/', '_'))
    sys.modules[module.__name__] = module
    module.__dict__.update(env)
    exec(compile(ast.fix_missing_locations(ast.Module([ast.ImportFrom('__future__', [ast.alias('annotations')], 0), *body], [])), str(path), 'exec'), module.__dict__)
    return module


def import_file(relative):
    spec = importlib.util.spec_from_file_location('_test_' + relative.replace('/', '_'), ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.shared = NS(opts=NS(lora_functional=False, lora_gpu_backup=False),
                         state=NS(sampling_step=0, sampling_steps=10),
                         cmd_opts=NS(disable_model_loading_ram_optimization=False))
        self.devices = NS(cpu=torch.device('cpu'), device=torch.device('cpu'), dtype=torch.float32)
        class QkvLinear(torch.nn.Linear):
            pass
        self.modules = NS(models=NS(sd3=NS(mmdit=NS(QkvLinear=QkvLinear))))
        self.base = selected('extensions-builtin/Lora/network.py',
                             {'Network', 'NetworkActivation', 'NetworkModule', 'ModuleType'},
                             nn=torch.nn, F=torch.nn.functional, modules=self.modules,
                             shared=self.shared, dataclass=dataclass, lora_runtime=lora_runtime)
        self.base.NetworkWeights = namedtuple('NetworkWeights', 'network_key sd_key w sd_module')
        self.runtime = selected('extensions-builtin/Lora/networks.py',
            {'network_apply_weights', '_network_apply_weights', 'network_restore_weights_from_backup',
             'network_merge_projection_bias', 'network_set_bias',
             'store_weights_backup', 'restore_weights_backup', 'network_reset_cached_weight',
             'network_Embedding_forward', 'network_Embedding_load_state_dict', 'allowed_layer_without_weight',
             'assign_network_names_to_compvis_modules', 'network_forward'} |
            {'network_' + kind + '_' + method for kind in ('Linear', 'Conv2d', 'GroupNorm', 'LayerNorm', 'MultiheadAttention')
             for method in ('forward', 'load_state_dict')},
            torch=torch, modules=self.modules, shared=self.shared, devices=self.devices,
            loaded_networks=[], networks_in_memory={}, logging=logging, extra_network_lora=NS(errors={}))
        self.runtime.originals = NS(Embedding_forward=torch.nn.Embedding.forward,
                                   Embedding_load_state_dict=torch.nn.Embedding._load_from_state_dict)

    def activation(self, layer, value=1, fail=False):
        layer.network_layer_name = 'transformer_test'
        net = self.base.Network('test', NS(filename='same.safetensors'))
        def delta(weight, activation):
            if fail:
                raise RuntimeError('injected adapter failure')
            return torch.full_like(weight, value), None
        net.modules[layer.network_layer_name] = NS(calc_updown=delta)
        return self.base.NetworkActivation(net, 'test')

    def test_hot_reload_same_path(self):
        layer = torch.nn.Linear(3, 2, bias=False)
        base = layer.weight.detach().clone()
        self.runtime.loaded_networks[:] = [self.activation(layer, 1)]
        self.runtime.network_apply_weights(layer)
        torch.testing.assert_close(layer.weight, base + 1)
        self.runtime.loaded_networks[:] = [self.activation(layer, 3)]
        self.runtime.network_apply_weights(layer)
        torch.testing.assert_close(layer.weight, base + 3)

    def test_failure_rolls_back_then_retry(self):
        layer = torch.nn.Linear(3, 2, bias=False)
        base = layer.weight.detach().clone()
        good, bad = self.activation(layer), self.activation(layer, fail=True)
        self.runtime.loaded_networks[:] = [good, bad]
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                self.runtime.network_apply_weights(layer)
            torch.testing.assert_close(layer.weight, base)
            self.assertEqual(layer.network_current_names, ())
        self.runtime.loaded_networks[:] = [good]
        self.runtime.network_apply_weights(layer)
        torch.testing.assert_close(layer.weight, base + 1)

    def test_fp16_cache_combines_two_loras(self):
        layer = torch.nn.Linear(3, 2, bias=False)
        layer.fp16_weight = layer.weight.detach().clone().half()
        base = layer.fp16_weight.float()
        self.runtime.loaded_networks[:] = [self.activation(layer, 1), self.activation(layer, 2)]
        self.runtime.network_apply_weights(layer)
        torch.testing.assert_close(layer.weight, (base.half() + 1 + 2).float())

    def test_no_target_no_backup_and_disable_restores(self):
        layer = torch.nn.Linear(2, 2, bias=False)
        other = torch.nn.Linear(2, 2, bias=False)
        net = self.activation(layer)
        other.network_layer_name = 'unrelated'
        self.runtime.loaded_networks[:] = [net]
        self.runtime.network_apply_weights(other)
        self.assertIsNone(getattr(other, 'network_weights_backup', None))
        base = layer.weight.detach().clone()
        self.runtime.network_apply_weights(layer)
        self.runtime.loaded_networks.clear()
        self.runtime.network_apply_weights(layer)
        torch.testing.assert_close(layer.weight, base)
        self.assertIsNone(layer.network_weights_backup)

    def test_model_callback_invalidates_binding(self):
        model = NS(is_sdxl=False, cond_stage_model=torch.nn.Linear(2, 2), model=torch.nn.Linear(2, 2))
        self.shared.sd_model = None  # callback must use its argument
        for _ in range(2):
            previous = getattr(model, 'lora_model_token', None)
            self.runtime.loaded_networks.append('stale')
            self.runtime.networks_in_memory['stale'] = object()
            self.runtime.assign_network_names_to_compvis_modules(model)
            self.assertIsNot(model.lora_model_token, previous)
            self.assertEqual(self.runtime.loaded_networks, [])
            self.assertEqual(self.runtime.networks_in_memory, {})

    def test_cached_model_switch_without_callback_rebinds_adapter(self):
        old_layer = torch.nn.Linear(2, 2)
        previous = self.activation(old_layer)
        previous.network.file_signature = (10, 20)
        previous.network.model_token = object()
        self.runtime.loaded_networks[:] = [previous]
        new_layer = torch.nn.Linear(2, 2)
        model = NS(network_layer_mapping={'transformer_test': new_layer}, lora_model_token=object())
        self.shared.sd_model = model
        entry = NS(filename='same.safetensors', name='test', read_hash=lambda: None)
        calls = []
        def load(entry):
            calls.append(entry)
            net = self.base.Network(entry.name, entry)
            net.file_signature = (10, 20)
            net.model_token = model.lora_model_token
            net.modules['transformer_test'] = NS(sd_module=new_layer)
            return net
        mod = selected('extensions-builtin/Lora/networks.py', {'load_networks'},
            shared=self.shared, network=self.base, loaded_networks=self.runtime.loaded_networks,
            networks_in_memory={}, network_cache_key=lambda e: e.filename,
            network_file_signature=lambda p: (10, 20), get_network_on_disk=lambda n: entry,
            load_network=load, purge_networks_from_memory=lambda: None, finalize_fp16_cache=lambda m: None,
            sd_hijack=NS(model_hijack=NS(embedding_db=NS(ensure_model_compatibility=lambda: None), comments=[])))
        mod.load_networks(['test'])
        self.assertEqual(calls, [entry])
        self.assertIs(mod.loaded_networks[0].modules['transformer_test'].sd_module, new_layer)
        mod.load_networks(['test'])
        self.assertEqual(calls, [entry])

    def test_mha_and_packed_qkv(self):
        for layer in (torch.nn.MultiheadAttention(4, 1), self.modules.models.sd3.mmdit.QkvLinear(4, 12, bias=False)):
            activation = self.activation(layer)
            adapter = activation.modules.pop(layer.network_layer_name)
            for suffix in ('_q_proj', '_k_proj', '_v_proj', '_out_proj'):
                activation.modules[layer.network_layer_name + suffix] = adapter
            self.runtime.loaded_networks[:] = [activation]
            weight = layer.in_proj_weight if isinstance(layer, torch.nn.MultiheadAttention) else layer.weight
            base = weight.detach().clone()
            self.runtime.network_apply_weights(layer)
            torch.testing.assert_close(weight, base + 1)
            self.runtime.network_apply_weights(layer)
            torch.testing.assert_close(weight, base + 1)
            self.runtime.loaded_networks.clear()
            self.runtime.network_apply_weights(layer)
            torch.testing.assert_close(weight, base)

    def test_embedding_real_factors_merge_functional_disable(self):
        lora = selected('extensions-builtin/Lora/network_lora.py', {'NetworkModuleLoraEmbedding'},
                        network=self.base, lora_keys=lora_keys, devices=self.devices)
        layer = torch.nn.Embedding(7, 3)
        act = self.activation(layer)
        up, down = torch.ones(7, 2), torch.ones(2, 3)
        weights = self.base.NetworkWeights('test', layer.network_layer_name,
                        {'lora_up.weight': up, 'lora_down.weight': down, 'alpha': torch.tensor(2.)}, layer)
        act.modules[layer.network_layer_name] = lora.NetworkModuleLoraEmbedding(act.network, weights, 'lora_up.weight', 'lora_down.weight')
        self.runtime.loaded_networks[:] = [act]
        indices = torch.tensor([0, 4], dtype=torch.long)
        base = layer.weight.detach().clone()
        for functional in (False, True, False):
            self.shared.opts.lora_functional = functional
            result = self.runtime.network_Embedding_forward(layer, indices)
            torch.testing.assert_close(result, base[indices] + 2)
        self.runtime.loaded_networks.clear()
        self.shared.opts.lora_functional = True
        torch.testing.assert_close(self.runtime.network_Embedding_forward(layer, indices), base[indices])

    def test_meta_loader_with_actual_embedding_hook(self):
        meta = selected('modules/sd_disable_initialization.py', {'ReplaceHelper', 'LoadStateDictOnMeta'}, torch=torch, shared=self.shared)
        patches = import_file('modules/patches.py')
        patch_class = selected('extensions-builtin/Lora/lora_patches.py', {'LoraPatches'},
                               torch=torch, networks=self.runtime, patches=patches)
        for hook in (False, True):
            for device in ('cpu', 'meta'):
                model = torch.nn.Sequential(torch.nn.Embedding(7, 3, device=device), torch.nn.Linear(3, 2, device=device))
                expected = {key: torch.randn(value.shape) for key, value in model.state_dict().items()}
                # Same actual original-reference chain as the production hook.
                if hook:
                    self.runtime.originals = patch_class.LoraPatches()
                try:
                    state = dict(expected)
                    with meta.LoadStateDictOnMeta(state, torch.device('cpu')):
                        result = model.load_state_dict(state)
                    self.assertFalse(result.missing_keys)
                    for key, value in model.state_dict().items():
                        torch.testing.assert_close(value, expected[key])
                finally:
                    if hook:
                        self.runtime.originals.undo()
        self.assertNotIn('_load_from_state_dict', torch.nn.Embedding.__dict__)
        model = torch.nn.Linear(3, 2, device='meta')
        state = {'weight': torch.ones(2, 3), 'bias': torch.ones(2)}
        with meta.LoadStateDictOnMeta(state, torch.device('cpu')):
            model.load_state_dict(state)
        torch.testing.assert_close(model.weight, torch.ones(2, 3))

    def test_hook_order_rejected_without_corrupting_stack(self):
        patches = import_file('modules/patches.py')
        class Parent:
            method = object()
        class Child(Parent):
            pass
        original = Child.method
        a, b = object(), object()
        patches.patch('a', Child, 'method', a)
        patches.patch('b', Child, 'method', b)
        with self.assertRaisesRegex(RuntimeError, 'reverse'):
            patches.undo('a', Child, 'method')
        self.assertIs(Child.method, b)
        patches.undo('b', Child, 'method')
        patches.undo('a', Child, 'method')
        self.assertIs(Child.method, original)
        self.assertNotIn('method', Child.__dict__)
        patches.patch('a', Child, 'method', a)
        Child.method = b
        with self.assertRaisesRegex(RuntimeError, 'external'):
            patches.undo('a', Child, 'method')
        self.assertIs(Child.method, b)

    def test_meta_buffers_and_missing_source_diagnostic(self):
        meta = selected('modules/sd_disable_initialization.py', {'ReplaceHelper', 'LoadStateDictOnMeta'}, torch=torch, shared=self.shared)
        class WithBuffer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('values', torch.empty(3, device='meta'))
        model = WithBuffer()
        state = {'values': torch.arange(3.)}
        with meta.LoadStateDictOnMeta(state, torch.device('cpu')):
            model.load_state_dict(state)
        torch.testing.assert_close(model.values, torch.arange(3.))
        state = {'weight': torch.empty(2, 2, device='meta')}
        with self.assertRaisesRegex(RuntimeError, 'weight.*has no data'):
            with meta.LoadStateDictOnMeta(state, torch.device('cpu')):
                torch.nn.Linear(2, 2, bias=False).load_state_dict(state)

    def test_manual_cast_mixed_dtypes_and_keyword_mask(self):
        dev = selected('modules/devices.py', {'manual_cast_forward'}, torch=torch, dtype_inference=torch.float32)
        class Mixed(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
                self.b = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
                self.register_buffer('stat', torch.ones(2, dtype=torch.float64))
            def org_forward(self, *, x, mask):
                assert x.dtype == torch.float16
                assert mask.dtype == torch.bool
                assert self.a.dtype == self.b.dtype == self.stat.dtype == torch.float16
                raise RuntimeError('mixed failure')
        model = Mixed()
        with self.assertRaisesRegex(RuntimeError, 'mixed failure'):
            dev.manual_cast_forward(torch.float16)(model, x=torch.ones(2), mask=torch.ones(2, dtype=torch.bool))
        self.assertEqual((model.a.dtype, model.b.dtype, model.stat.dtype), (torch.float32, torch.float64, torch.float64))

    def test_failed_published_model_is_discarded_without_lazy_retry(self):
        previous, candidate = object(), object()
        data = NS(sd_model=previous, loaded_sd_models=[previous], was_loaded_at_least_once=False)
        def failing_load(*args):
            # Real _load_model trashes/unregisters the previous object before
            # constructing and publishing a replacement. Model that phase too.
            mod.discard_failed_model(previous)
            data.sd_model = candidate
            data.loaded_sd_models.insert(0, candidate)
            raise RuntimeError('empty conditioning failed')
        mod = selected('modules/sd_models.py', {'load_model', 'discard_failed_model'}, model_data=data, _load_model=failing_load)
        with self.assertRaisesRegex(RuntimeError, 'empty conditioning failed'):
            mod.load_model()
        self.assertIsNone(data.sd_model)
        self.assertEqual(data.loaded_sd_models, [])
        self.assertTrue(data.was_loaded_at_least_once)

    def test_trashed_model_is_not_available_for_cached_reuse(self):
        model = torch.nn.Linear(2, 2)
        retained = torch.nn.Linear(2, 2)
        data = NS(sd_model=model, loaded_sd_models=[model, retained], was_loaded_at_least_once=True)
        mod = selected('modules/sd_models.py', {'send_model_to_trash', 'discard_failed_model'},
                       model_data=data, devices=NS(torch_gc=lambda: None))
        mod.send_model_to_trash(model)
        self.assertTrue(model.weight.is_meta)
        self.assertIsNone(data.sd_model)
        self.assertEqual(data.loaded_sd_models, [retained])

    def test_checkpoint_rollback_failure_preserves_original_exception(self):
        old, new = NS(filename='old'), NS(filename='new')
        model = NS(sd_checkpoint_info=old, sd_model_checkpoint='old', used_config='same', lowvram=False)
        data = NS(sd_model=model, loaded_sd_models=[model], was_loaded_at_least_once=True)
        calls = []
        def load_weights(*args):
            calls.append(args[1])
            raise RuntimeError('original failure' if len(calls) == 1 else 'rollback failure')
        env = dict(model_data=data, logging=logging, select_checkpoint=lambda: new,
                   Timer=lambda: NS(record=lambda *a: None), check_fp8=lambda m: False,
                   devices=NS(fp8=False), model_has_fp8_weights=lambda model: False,
                   reuse_model_from_already_loaded=lambda *a: model,
                   sd_unet=NS(apply_unet=lambda *a: None), send_model_to_cpu=lambda *a: None,
                   sd_hijack=NS(model_hijack=NS(undo_hijack=lambda *a: None)),
                   get_checkpoint_state_dict=lambda *a: {},
                   sd_models_config=NS(find_checkpoint_config=lambda *a: 'same'),
                   load_model_weights=load_weights,
                   finish_model_reload=lambda *a: self.fail('must not finalize broken model'))
        mod = selected('modules/sd_models.py', {'reload_model_weights', 'discard_failed_model'}, **env)
        with self.assertLogs(level='ERROR'), self.assertRaisesRegex(RuntimeError, 'original failure'):
            mod.reload_model_weights()
        self.assertEqual(calls, [new, old])
        self.assertIsNone(data.sd_model)
        self.assertEqual(data.loaded_sd_models, [])

    def test_manual_cast_exception_and_foreign_owner(self):
        dev = selected('modules/devices.py', {'manual_cast_forward', 'manual_cast'},
                       torch=torch, contextlib=contextlib, dtype_inference=torch.float32,
                       patch_module_list=[torch.nn.Linear, torch.nn.Conv2d])
        layer = torch.nn.Linear(3, 2)
        def fail(*args):
            raise RuntimeError('forward failure')
        layer.org_forward = fail
        with self.assertRaisesRegex(RuntimeError, 'forward failure'):
            dev.manual_cast_forward(torch.float16)(layer, torch.ones(1, 3))
        self.assertEqual(layer.weight.dtype, torch.float32)
        foreign = torch.nn.Conv2d.forward
        torch.nn.Conv2d.org_forward = foreign
        try:
            with dev.manual_cast(torch.float16):
                pass
            self.assertIs(torch.nn.Conv2d.org_forward, foreign)
            self.assertNotIn('org_forward', torch.nn.Linear.__dict__)
        finally:
            del torch.nn.Conv2d.org_forward

    def test_cleanup_does_not_reload_and_preserves_primary_failure(self):
        calls = NS(load=0, close=0)
        def fail_load():
            calls.load += 1
            raise RuntimeError('primary load failure')
        class Processing:
            scripts = None
            override_settings = {}
            override_settings_restore_afterwards = True
            @property
            def sd_model(self):
                return fail_load()
            def close(self):
                calls.close += 1
        opts = NS(data={})
        sd_models = NS(checkpoint_aliases={}, reload_model_weights=fail_load, model_data=NS(sd_model=None))
        p = selected('modules/processing.py', {'process_images', 'cleanup_processing'},
                     sd_models=sd_models, opts=opts, sys=sys, logging=logging, sd_vae=NS(reload_vae_weights=lambda: None))
        with self.assertRaisesRegex(RuntimeError, 'primary load failure'):
            p.process_images(Processing())
        self.assertEqual((calls.load, calls.close), (1, 1))
        sd_models.model_data.sd_model = object()
        sd_models.apply_token_merging = lambda *args: (_ for _ in ()).throw(RuntimeError('cleanup failure'))
        with self.assertLogs(level='ERROR'):
            p.cleanup_processing(Processing(), {'restored': 4}, primary_error=RuntimeError('primary'))
        self.assertEqual(opts.restored, 4)
        self.assertEqual(calls.close, 2)

    def test_ti_nested_changes_and_root_removal(self):
        ti = selected('modules/textual_inversion/textual_inversion.py', {'DirWithTextualInversionEmbeddings'}, os=os)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'embeddings'
            (root / 'nested').mkdir(parents=True)
            file = root / 'nested/test.pt'
            file.write_bytes(b'old')
            watcher = ti.DirWithTextualInversionEmbeddings(root)
            watcher.update()
            self.assertFalse(watcher.has_changed())
            file.write_bytes(b'new and larger')
            self.assertTrue(watcher.has_changed())
            watcher.update()
            file.unlink()
            self.assertTrue(watcher.has_changed())

    def test_ti_save_snapshot_even_on_failure(self):
        ti = selected('modules/textual_inversion/textual_inversion.py', {'save_embedding'}, copy=copy)
        seen = []
        class Emb:
            name = 'live'
            cached_checksum = 'cached'
            vec = torch.ones(2, 3)
            def save(self, filename):
                seen.append(self)
                self.vec.zero_()
                raise RuntimeError('disk failure')
        live = Emb()
        with self.assertRaisesRegex(RuntimeError, 'disk failure'):
            ti.save_embedding(live, NS(state_dict=lambda: {}), NS(shorthash='hash', model_name='model'), 'export', 'unused')
        self.assertEqual(live.name, 'live')
        self.assertEqual(live.cached_checksum, 'cached')
        torch.testing.assert_close(live.vec, torch.ones(2, 3))
        self.assertEqual(seen[0].name, 'export')

    def test_ti_multiple_vectors_and_explicit_multi_term_error(self):
        ti = selected('modules/textual_inversion/textual_inversion.py', {'Embedding', 'create_embedding_from_data'},
                      torch=torch, devices=self.devices)
        embedding = ti.create_embedding_from_data({'string_to_param': {'*': torch.ones(4, 8)}}, 'four-vectors')
        self.assertEqual((embedding.vectors, embedding.shape), (4, 8))
        with self.assertRaisesRegex(ValueError, 'export separate files'):
            ti.create_embedding_from_data({'string_to_param': {'a': torch.ones(1, 8), 'b': torch.ones(1, 8)}}, 'two-terms')

    def test_projection_failure_restores_both_mha_weights(self):
        layer = torch.nn.MultiheadAttention(4, 1)
        act = self.activation(layer)
        adapter = act.modules.pop(layer.network_layer_name)
        for suffix in ('_q_proj', '_k_proj', '_v_proj', '_out_proj'):
            act.modules[layer.network_layer_name + suffix] = adapter
        base_qkv, base_out = layer.in_proj_weight.detach().clone(), layer.out_proj.weight.detach().clone()
        failing = self.activation(layer, fail=True)
        # Remove the direct entry and make the second complete projection group fail.
        bad = failing.modules.pop(layer.network_layer_name)
        for suffix in ('_q_proj', '_k_proj', '_v_proj', '_out_proj'):
            failing.modules[layer.network_layer_name + suffix] = bad
        self.runtime.loaded_networks[:] = [act, failing]
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            self.runtime.network_apply_weights(layer)
        torch.testing.assert_close(layer.in_proj_weight, base_qkv)
        torch.testing.assert_close(layer.out_proj.weight, base_out)
        self.assertEqual(layer.network_current_names, ())


if __name__ == '__main__':
    unittest.main(verbosity=2)
