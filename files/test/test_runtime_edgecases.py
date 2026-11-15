"""Adversarial CPU tests for cache failures, packed FP8 and temporary hooks.

Production functions are AST-selected; WebUI services are stubs, Torch is real.
Run directly with Python; no server or checkpoint downloads are required.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import test_runtime_recovery as recovery
import test_runtime_stability as harness

torch = harness.torch


class EdgeCaseTests(unittest.TestCase):
    def test_cached_model_failure_quarantines_candidate(self):
        for stage in ('device', 'vae'):
            with self.subTest(stage=stage):
                mod, data, previous, candidate, _, _ = recovery.RecoveryTests().loader(extra_cached=True)
                original_options = {'sd_model_checkpoint': 'old', 'sd_checkpoint_hash': 'old'}
                mod.shared.opts.data.update(original_options)
                error = RuntimeError('cached ' + stage)
                def fail(*args, candidate=candidate, error=error):
                    candidate.hijacked = False  # transfer/load can fail after mutation
                    raise error
                if stage == 'device':
                    mod.send_model_to_device = fail
                else:
                    mod.sd_vae.reload_vae_weights = fail
                with self.assertRaises(RuntimeError) as caught:
                    mod.reload_model_weights(info=candidate.sd_checkpoint_info)
                self.assertIs(caught.exception, error)
                self.assertNotIn(candidate, data.loaded_sd_models)
                self.assertIn(previous, data.loaded_sd_models)
                self.assertIsNot(data.sd_model, candidate)
                self.assertEqual(mod.shared.opts.data, original_options)

    def projection_fixture(self, *, cached):
        h = harness.RuntimeTests()
        h.setUp()
        layer = h.modules.models.sd3.mmdit.QkvLinear(4, 12, bias=False)
        layer.network_layer_name = 'transformer_test'
        precise = torch.linspace(0.11, 0.54, 48).reshape(12, 4).half()
        with torch.no_grad():
            layer.weight.copy_(precise)
        layer.to(torch.float8_e4m3fn)
        if cached:
            layer.fp16_weight = precise.clone()
        base = precise if cached else layer.weight.float().clone()
        lora = harness.selected('extensions-builtin/Lora/network_lora.py', {'NetworkModuleLora'},
            torch=torch, network=h.base, modules=h.modules,
            lyco_helpers=harness.import_file('extensions-builtin/Lora/lyco_helpers.py'))
        activations = []
        for index in range(2):
            net = h.base.Network(str(index), NS(filename=str(index)))
            for suffix in ('q', 'k', 'v'):
                key = layer.network_layer_name + '_' + suffix + '_proj'
                factors = {'lora_up.weight': torch.full((4, 2), 0.0625),
                           'lora_down.weight': torch.ones(2, 4)}
                net.modules[key] = lora.NetworkModuleLora(net, h.base.NetworkWeights(key, key, factors, layer))
            activations.append(h.base.NetworkActivation(net, str(index), te_multiplier=0.5, dyn_dim=1))
        return h, layer, base, activations

    def test_packed_fp8_accumulates_before_cast_and_restores(self):
        for cached in (True, False):
            with self.subTest(cached=cached):
                h, layer, base, activations = self.projection_fixture(cached=cached)
                native = layer.weight.float().clone()
                h.runtime.loaded_networks[:] = activations
                h.runtime.network_apply_weights(layer)
                expected = (base.float() + 0.0625).to(torch.float8_e4m3fn).float()
                torch.testing.assert_close(layer.weight.float(), expected)
                h.runtime.network_apply_weights(layer)
                torch.testing.assert_close(layer.weight.float(), expected)
                h.runtime.loaded_networks.clear()
                h.runtime.network_apply_weights(layer)
                torch.testing.assert_close(layer.weight.float(), native)

    def test_packed_failure_rolls_back_all_projections_then_retries(self):
        h, layer, base, activations = self.projection_fixture(cached=True)
        native = layer.weight.float().clone()
        h.runtime.loaded_networks[:] = activations
        last = activations[-1].modules['transformer_test_v_proj']
        with patch.object(last, 'calc_updown', side_effect=RuntimeError('last projection')):
            with self.assertRaisesRegex(RuntimeError, 'last projection'):
                h.runtime.network_apply_weights(layer)
        torch.testing.assert_close(layer.weight.float(), native)
        self.assertEqual(layer.network_current_names, ())
        h.runtime.network_apply_weights(layer)
        torch.testing.assert_close(layer.weight.float(), (base.float() + 0.0625).to(torch.float8_e4m3fn).float())

    def helper(self):
        return harness.selected('modules/sd_disable_initialization.py', {'ReplaceHelper'}, logging=logging).ReplaceHelper

    def test_replace_helper_restores_classmethod_descriptor(self):
        class Target:
            @classmethod
            def method(cls):
                return cls
        class Child(Target):
            pass
        original = vars(Target)['method']
        helper = self.helper()()
        helper.replace(Target, 'method', lambda: None)
        helper.restore()
        self.assertIs(vars(Target)['method'], original)
        self.assertIs(Child.method(), Child)

    def test_replace_helper_does_not_clobber_foreign_owner(self):
        original, ours, foreign = (object() for _ in range(3))
        target, independent = NS(method=original), NS(method=original)
        helper = self.helper()()
        helper.replace(independent, 'method', ours)
        helper.replace(target, 'method', ours)
        target.method = foreign
        with self.assertRaisesRegex(RuntimeError, 'owner|changed|replacement'):
            helper.restore()
        self.assertIs(target.method, foreign)
        self.assertIs(independent.method, original)
        target.method = ours  # external owner released its layer; retry is possible
        helper.restore()
        self.assertIs(target.method, original)
        self.assertEqual(helper.replaced, [])

    def test_replace_helper_rejected_assignment_leaves_no_phantom_record(self):
        class Reject(type):
            def __setattr__(cls, name, value):
                raise RuntimeError('assignment refused')
        class Target(metaclass=Reject):
            method = staticmethod(lambda: None)
        helper = self.helper()()
        with self.assertRaisesRegex(RuntimeError, 'assignment refused'):
            helper.replace(Target, 'method', lambda: None)
        self.assertEqual(helper.replaced, [])

    def test_meta_context_setup_failure_unwinds_earlier_hooks(self):
        shared = NS(cmd_opts=NS(disable_model_loading_ram_optimization=False))
        meta = harness.selected('modules/sd_disable_initialization.py',
            {'ReplaceHelper', 'InitializeOnMeta'}, torch=torch, shared=shared, logging=logging)
        context = meta.InitializeOnMeta()
        original = torch.nn.Linear.__init__
        replace = context.replace
        def fail_second(obj, field, value):
            if obj is torch.nn.Conv2d:
                raise RuntimeError('second hook')
            return replace(obj, field, value)
        context.replace = fail_second
        try:
            with self.assertRaisesRegex(RuntimeError, 'second hook'):
                with context:
                    self.fail('body must not run')
            self.assertIs(torch.nn.Linear.__init__, original)
            self.assertEqual(context.replaced, [])
        finally:
            torch.nn.Linear.__init__ = original

    def test_nested_meta_contexts_and_reentry_guard(self):
        shared = NS(cmd_opts=NS(disable_model_loading_ram_optimization=False))
        meta = harness.selected('modules/sd_disable_initialization.py',
            {'ReplaceHelper', 'InitializeOnMeta'}, torch=torch, shared=shared, logging=logging)
        original = torch.nn.Linear.__init__
        outer = meta.InitializeOnMeta()
        with outer:
            wrapped = torch.nn.Linear.__init__
            with self.assertRaisesRegex(RuntimeError, 'already active'):
                with outer:
                    self.fail('same instance cannot be reentered')
            self.assertIs(torch.nn.Linear.__init__, wrapped)
            with meta.InitializeOnMeta():
                self.assertTrue(torch.nn.Linear(2, 2).weight.is_meta)
            self.assertIs(torch.nn.Linear.__init__, wrapped)
        self.assertIs(torch.nn.Linear.__init__, original)

    def test_primary_error_survives_restore_conflict(self):
        target = NS(method=object())
        ours, foreign = object(), object()
        helper_type = self.helper()
        class Context(helper_type):
            def _enter(self):
                self.replace(target, 'method', ours)
        context = Context()
        error = ValueError('primary failure')
        with self.assertLogs(level='ERROR'), self.assertRaises(ValueError) as caught:
            with context:
                target.method = foreign
                raise error
        self.assertIs(caught.exception, error)
        self.assertIs(target.method, foreign)
        self.assertEqual(len(context.replaced), 1)
        target.method = ours
        context.restore()

    def test_restore_failure_does_not_skip_independent_hook(self):
        original, ours = object(), object()
        class RefuseRestore(type):
            def __setattr__(cls, field, value):
                if field == 'method' and value is original:
                    raise RuntimeError('restore refused')
                super().__setattr__(field, value)
        class Target(metaclass=RefuseRestore):
            method = original
        independent = NS(method=original)
        context = self.helper()()
        context.replace(independent, 'method', ours)
        context.replace(Target, 'method', ours)
        with self.assertRaisesRegex(RuntimeError, 'restore refused'):
            context.restore()
        self.assertIs(independent.method, original)
        self.assertIs(Target.method, ours)
        self.assertEqual(len(context.replaced), 1)
        type.__setattr__(Target, 'method', original)

    def test_loader_setup_failure_at_each_hook_preserves_lora_stack(self):
        h = harness.RuntimeTests()
        h.setUp()
        registry = harness.import_file('modules/patches.py')
        hooks = harness.selected('extensions-builtin/Lora/lora_patches.py', {'LoraPatches'},
            torch=torch, networks=h.runtime, patches=registry)
        meta = harness.selected('modules/sd_disable_initialization.py', {'ReplaceHelper', 'LoadStateDictOnMeta'},
            torch=torch, shared=h.shared, logging=logging)
        holder = hooks.LoraPatches()
        targets = [(torch.nn.Module, 'load_state_dict')] + [(cls, '_load_from_state_dict') for cls in (
            torch.nn.Module, torch.nn.Linear, torch.nn.Conv2d, torch.nn.MultiheadAttention,
            torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.Embedding)]
        before = [(field in vars(cls), vars(cls).get(field)) for cls, field in targets]
        try:
            for denied in range(len(targets)):
                with self.subTest(denied=denied):
                    context = meta.LoadStateDictOnMeta({}, torch.device('cpu'))
                    replace = context.replace
                    calls = []
                    def reject(obj, field, value, calls=calls, denied=denied, replace=replace):
                        calls.append((obj, field))
                        if len(calls) - 1 == denied:
                            raise RuntimeError('denied loader hook')
                        return replace(obj, field, value)
                    context.replace = reject
                    with self.assertRaisesRegex(RuntimeError, 'denied loader hook'):
                        with context:
                            self.fail('body must not run')
                    self.assertEqual(context.replaced, [])
                    for (cls, field), (owned, original) in zip(targets, before):
                        self.assertEqual(field in vars(cls), owned)
                        self.assertIs(vars(cls).get(field), original)
        finally:
            holder.undo()


if __name__ == '__main__':
    unittest.main(verbosity=2)
