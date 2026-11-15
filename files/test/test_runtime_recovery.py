"""CPU/AST regressions for the 2026-11-03 review findings.

Run directly: python -B test/test_runtime_recovery.py
No checkpoint, WebUI server, downloads or workspace audit files are required.
"""
from __future__ import annotations

import ast
import logging
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import test_runtime_stability as harness

torch = harness.torch
selected = harness.selected


class RecoveryTests(unittest.TestCase):
    def loader(self, *, extra_cached=False):
        def checkpoint(name):
            return NS(filename=name, title=name, sha256=name)
        old, other, new = [checkpoint(name) for name in ('old', 'other', 'new')]
        def model(info):
            return NS(sd_checkpoint_info=info, sd_model_checkpoint=info.filename, used_config='same', lowvram=False, hijacked=True)
        previous, spare = model(old), model(other)
        data = NS(sd_model=previous, loaded_sd_models=[previous] + ([spare] if extra_cached else []), was_loaded_at_least_once=True)
        def publish(value, already_loaded=False):
            data.sd_model = value
            data.loaded_sd_models[:] = [value] + [m for m in data.loaded_sd_models if m is not value]
        data.set_sd_model = publish
        events = []
        def undo(value):
            value.hijacked = False
            events.append('undo')
        def finish(value, timer):
            value.hijacked = True
            events.append('finish')
        def weights(value, info, state, timer):
            value.sd_checkpoint_info = info
            value.sd_model_checkpoint = info.filename
        env = dict(model_data=data, logging=logging, select_checkpoint=lambda: new,
                   Timer=lambda: NS(record=lambda *a: None, summary=lambda: 'test'), check_fp8=lambda m: False,
                   devices=NS(fp8=False), model_has_fp8_weights=lambda model: False,
                   shared=NS(opts=NS(sd_checkpoints_keep_in_cpu=False, sd_checkpoints_limit=2, data={})),
                   sd_vae=NS(reload_vae_weights=lambda *a: events.append('vae')),
                   SkipWritingToConfig=NS(skip=False),
                   sd_unet=NS(apply_unet=lambda *a: events.append(('unet', a))),
                   send_model_to_cpu=lambda *a: events.append('cpu'),
                   send_model_to_device=lambda *a: events.append('device'),
                   send_model_to_trash=lambda *a: self.fail('unexpected eviction'),
                   sd_hijack=NS(model_hijack=NS(undo_hijack=undo)),
                   get_checkpoint_state_dict=lambda *a: {'weight': 'preflight'},
                   sd_models_config=NS(find_checkpoint_config=lambda *a: 'same'),
                   load_model_weights=weights, finish_model_reload=finish)
        mod = selected('modules/sd_models.py', {'reload_model_weights', 'reuse_model_from_already_loaded', 'discard_failed_model'}, **env)
        return mod, data, previous, spare, new, events

    def test_direct_load_preflight_preserves_previous(self):
        for failure in ('select', 'read', 'config', 'yaml', 'repair'):
            with self.subTest(failure=failure):
                previous = torch.nn.Linear(2, 2)
                original = previous.weight.detach().clone()
                data = NS(sd_model=previous, loaded_sd_models=[previous], was_loaded_at_least_once=True)
                error = RuntimeError(failure)
                def fail(*args, error=error, **kwargs):
                    raise error
                mod = selected('modules/sd_models.py', {'load_model', '_load_model', 'discard_failed_model'},
                    model_data=data, select_checkpoint=fail if failure == 'select' else lambda: object(),
                    Timer=lambda: NS(record=lambda *a: None),
                    get_checkpoint_state_dict=fail if failure == 'read' else lambda *a: {},
                    sd_models_config=NS(find_checkpoint_config=fail if failure == 'config' else lambda *a: 'same'),
                    OmegaConf=NS(load=fail if failure == 'yaml' else lambda *a: NS()),
                    repair_config=fail if failure == 'repair' else lambda *a: None,
                    sd1_clip_weight='one', sd2_clip_weight='two', sdxl_clip_weight='xl', sdxl_refiner_clip_weight='refiner')
                dummy = ModuleType('modules')
                dummy.sd_hijack = NS()
                with patch.dict(sys.modules, {'modules': dummy}), self.assertRaises(RuntimeError) as caught:
                    mod.load_model()
                self.assertIs(caught.exception, error)
                self.assertIs(data.sd_model, previous)
                self.assertEqual(data.loaded_sd_models, [previous])
                torch.testing.assert_close(previous.weight, original)

    def test_reload_preflight_precedes_reuse_and_unhook(self):
        for failure in ('read', 'config'):
            with self.subTest(failure=failure):
                mod, data, previous, _, _, events = self.loader()
                def fail(*args, failure=failure):
                    raise RuntimeError(failure)
                if failure == 'read':
                    mod.get_checkpoint_state_dict = fail
                else:
                    mod.sd_models_config.find_checkpoint_config = fail
                with self.assertRaisesRegex(RuntimeError, failure):
                    mod.reload_model_weights()
                self.assertEqual(events, [])
                self.assertIs(data.sd_model, previous)
                self.assertTrue(previous.hijacked)

    def test_rollback_restores_reused_object_not_original_active_checkpoint(self):
        mod, data, previous, spare, target, events = self.loader(extra_cached=True)
        restored_info = spare.sd_checkpoint_info
        calls = []
        def weights(model, info, state, timer):
            calls.append(info)
            if len(calls) == 1:
                raise RuntimeError('weight failure')
            model.sd_checkpoint_info = info
        mod.load_model_weights = weights
        with self.assertRaisesRegex(RuntimeError, 'weight failure'):
            mod.reload_model_weights()
        self.assertEqual(calls, [target, restored_info])
        self.assertIs(data.sd_model, spare)
        self.assertIn(previous, data.loaded_sd_models)
        self.assertTrue(spare.hijacked)
        self.assertEqual(events[-1], ('unet', ()))

    def test_cached_checkpoint_hit_avoids_read_and_rebinds_active_model(self):
        mod, data, previous, spare, _, events = self.loader(extra_cached=True)
        mod.get_checkpoint_state_dict = lambda *a: self.fail('cache hit read disk')
        result = mod.reload_model_weights(info=spare.sd_checkpoint_info)
        self.assertIs(result, spare)
        self.assertIs(data.sd_model, spare)
        self.assertIn(previous, data.loaded_sd_models)
        self.assertEqual(events, ['device', 'vae'])

    def test_free_cache_slot_passes_preflight_data_to_loader(self):
        mod, data, previous, spare, target, _ = self.loader()
        reads = []
        payload = {'weight': 'preflight'}
        def read(*args):
            reads.append(args)
            return payload
        def load(info, already_loaded_state_dict=None, checkpoint_config=None):
            self.assertIs(info, target)
            self.assertIs(already_loaded_state_dict, payload)
            self.assertEqual(checkpoint_config, 'same')
            spare.sd_checkpoint_info = info
            data.set_sd_model(spare)
        mod.get_checkpoint_state_dict, mod.load_model = read, load
        self.assertIs(mod.reload_model_weights(), spare)
        self.assertEqual(len(reads), 1)
        self.assertIn(previous, data.loaded_sd_models)

    def test_forced_reload_free_slot_does_not_reuse_consumed_state_dict(self):
        mod, data, previous, spare, target, _ = self.loader()
        payload = {'weight': 'preflight'}
        mod.get_checkpoint_state_dict = lambda *a: payload
        def load(info, already_loaded_state_dict=None, checkpoint_config=None):
            self.assertIs(already_loaded_state_dict, payload)
            already_loaded_state_dict.clear()  # production meta-loader consumes it
            spare.sd_checkpoint_info = target
            data.set_sd_model(spare)
        mod.load_model = load
        mod.load_model_weights = lambda *a: self.fail('fresh model loaded a second time')
        self.assertIs(mod.reload_model_weights(forced_reload=True), spare)
        self.assertIn(previous, data.loaded_sd_models)
        self.assertEqual(payload, {})

    def test_prepare_or_finalize_failure_does_not_publish_incomplete_model(self):
        for where in ('prepare', 'finish', 'unet'):
            with self.subTest(where=where):
                mod, data, previous, spare, _, _ = self.loader(extra_cached=True)
                def fail(*a, where=where):
                    raise RuntimeError(where)
                if where == 'prepare':
                    mod.sd_hijack.model_hijack.undo_hijack = fail
                elif where == 'finish':
                    mod.finish_model_reload = fail
                else:
                    mod.sd_unet.apply_unet = lambda *a: None if a else fail()
                with self.assertRaisesRegex(RuntimeError, where):
                    mod.reload_model_weights()
                self.assertIsNone(data.sd_model)
                self.assertNotIn(spare, data.loaded_sd_models)
                self.assertIn(previous, data.loaded_sd_models)

    def test_failed_candidate_does_not_remove_other_valid_cached_model(self):
        previous, candidate = object(), object()
        data = NS(sd_model=previous, loaded_sd_models=[previous], was_loaded_at_least_once=True)
        def fail(*a):
            # Simulate publication without destroying the separately retained model.
            data.sd_model = candidate
            data.loaded_sd_models.append(candidate)
            raise RuntimeError('candidate failed')
        mod = selected('modules/sd_models.py', {'load_model', 'discard_failed_model'}, model_data=data, _load_model=fail)
        with self.assertRaisesRegex(RuntimeError, 'candidate failed'):
            mod.load_model()
        self.assertIsNone(data.sd_model)
        self.assertEqual(data.loaded_sd_models, [previous])

    def test_projection_settings_use_real_factors_and_cache(self):
        h = harness.RuntimeTests()
        h.setUp()
        lora = selected('extensions-builtin/Lora/network_lora.py', {'NetworkModuleLora'},
                        torch=torch, network=h.base, modules=h.modules,
                        lyco_helpers=harness.import_file('extensions-builtin/Lora/lyco_helpers.py'))
        for layer in (torch.nn.MultiheadAttention(4, 1), h.modules.models.sd3.mmdit.QkvLinear(4, 12, bias=False)):
            with self.subTest(kind=type(layer).__name__):
                layer.network_layer_name = 'transformer_test'
                net = h.base.Network('test', NS(filename='same'))
                suffixes = ('q', 'k', 'v', 'out') if isinstance(layer, torch.nn.MultiheadAttention) else ('q', 'k', 'v')
                for suffix in suffixes:
                    key = layer.network_layer_name + '_' + suffix + '_proj'
                    factors = {'lora_up.weight': torch.ones(4, 2), 'lora_down.weight': torch.ones(2, 4)}
                    net.modules[key] = lora.NetworkModuleLora(net, h.base.NetworkWeights(key, key, factors, layer))
                weight = layer.in_proj_weight if isinstance(layer, torch.nn.MultiheadAttention) else layer.weight
                baseline = weight.detach().clone()
                out_base = layer.out_proj.weight.detach().clone() if isinstance(layer, torch.nn.MultiheadAttention) else None
                for muted in (*suffixes, None):
                    rules = harness.lora_runtime.parse_elemental_weights('MUTE', f'MUTE::{muted}_proj:0') if muted else None
                    activation = h.base.NetworkActivation(net, 'test', te_multiplier=0.5, dyn_dim=1, elemental_weights=rules)
                    h.runtime.loaded_networks[:] = [activation]
                    h.runtime.network_apply_weights(layer)
                    for idx, suffix in enumerate(('q', 'k', 'v')):
                        torch.testing.assert_close(weight[idx * 4:(idx + 1) * 4], baseline[idx * 4:(idx + 1) * 4] + (0 if muted == suffix else 0.5))
                    if out_base is not None:
                        torch.testing.assert_close(layer.out_proj.weight, out_base + (0 if muted == 'out' else 0.5))
                    for adapter in net.modules.values():
                        with patch.object(adapter, 'calc_updown', side_effect=AssertionError('unnecessary repeat merge')):
                            h.runtime.network_apply_weights(layer)
                h.runtime.loaded_networks.clear()
                h.runtime.network_apply_weights(layer)
                torch.testing.assert_close(weight, baseline)

    def hook_environment(self):
        registry = harness.import_file('modules/patches.py')
        types = ('Linear', 'Embedding', 'Conv2d', 'GroupNorm', 'LayerNorm', 'MultiheadAttention')
        nn = NS(**{name: type(name, (), {'forward': lambda *a: None, '_load_from_state_dict': lambda *a: None}) for name in types})
        networks = NS(**{'network_' + name + '_' + method: (lambda *a: None)
                        for name in types for method in ('forward', 'load_state_dict')})
        hooks = selected('extensions-builtin/Lora/lora_patches.py', {'LoraPatches'}, torch=NS(nn=nn), patches=registry, networks=networks)
        return registry, nn, hooks

    def test_install_failure_at_each_target_rolls_back_earlier_hooks(self):
        for fail_at in range(12):
            with self.subTest(fail_at=fail_at):
                registry, nn, hooks = self.hook_environment()
                before = {cls: dict(vars(cls)) for cls in vars(nn).values()}
                original = registry.patch
                count = 0
                def install(*args, fail_at=fail_at, original=original):
                    nonlocal count
                    count += 1
                    if count == fail_at + 1:
                        raise RuntimeError('injected install conflict')
                    return original(*args)
                registry.patch = install
                with self.assertRaisesRegex(RuntimeError, 'injected install conflict'):
                    hooks.LoraPatches()
                for cls, values in before.items():
                    self.assertEqual(dict(vars(cls)), values)
                self.assertFalse(registry.stacks)

    def test_uninstall_foreign_owner_preserves_entire_hook_set(self):
        registry, nn, hooks = self.hook_environment()
        holder = hooks.LoraPatches()
        before = {cls: (cls.forward, cls._load_from_state_dict) for cls in vars(nn).values()}
        registry.patch('foreign', nn.Linear, 'forward', lambda *a: None)
        with self.assertRaisesRegex(RuntimeError, 'reverse installation order'):
            holder.undo()
        for cls, methods in before.items():
            if cls is not nn.Linear:
                self.assertEqual((cls.forward, cls._load_from_state_dict), methods)
        self.assertEqual(len(holder._installed), 12)
        registry.undo('foreign', nn.Linear, 'forward')
        holder.undo()
        self.assertFalse(registry.stacks)
        self.assertEqual(holder._installed, [])

    def test_base_and_hires_cleanup_run_after_unhook_failure(self):
        tree = ast.parse((harness.ROOT / 'modules/processing.py').read_bytes())
        names = ('StableDiffusionProcessing', 'StableDiffusionProcessingTxt2Img')
        classes = []
        for name in names:
            source = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
            close = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'close')
            classes.append(ast.ClassDef(name=name, bases=[] if not classes else [ast.Name(names[0], ast.Load())],
                                        keywords=[], body=[close], decorator_list=[]))
        opts = NS(persistent_cond_cache=False)
        ns = {'opts': opts}
        exec(compile(ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[])), 'production-close-methods', 'exec'), ns)
        Base, Hires = (ns[name] for name in names)
        for fail_at in (0, 1):
            for persistent in (False, True):
                with self.subTest(fail_at=fail_at, persistent=persistent):
                    opts.persistent_cond_cache = persistent
                    Base.cached_c = Base.cached_uc = Hires.cached_hr_c = Hires.cached_hr_uc = ['old', 'old']
                    p = Hires()
                    p.hr_c = p.hr_uc = 'old'
                    calls = []
                    def unhook(idx, calls=calls, fail_at=fail_at):
                        calls.append(idx)
                        if idx == fail_at:
                            raise RuntimeError('unhook failed')
                    p._monkey_patcher = NS(unhook=lambda: unhook(0))
                    p._forge_shim = NS(unhook=lambda: unhook(1))
                    with self.assertRaisesRegex(RuntimeError, 'unhook failed'):
                        p.close()
                    self.assertEqual(calls, [0, 1])
                    self.assertIsNone(p.hr_c)
                    self.assertIsNone(p.hr_uc)
                    expected = ['old', 'old'] if persistent else [None, None]
                    self.assertEqual((Base.cached_c, Base.cached_uc, Hires.cached_hr_c, Hires.cached_hr_uc), (expected,) * 4)
                    self.assertIsNotNone(getattr(p, ('_monkey_patcher', '_forge_shim')[fail_at]))
                    self.assertIsNone(getattr(p, ('_monkey_patcher', '_forge_shim')[1 - fail_at]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
