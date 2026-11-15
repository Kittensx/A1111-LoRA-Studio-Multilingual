"""Independent regressions for the supplied November 8 audits, against current code.

Set WEBUI_AUDIT_BASELINE to the pre-fix backup to reproduce the old defects.
Only this test's synthetic temporary assets are written; no downloads/models.
"""
import collections
import contextlib
import glob
import json
import os
import enum
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
from types import ModuleType
import sys
import unittest
from unittest.mock import patch

import test_runtime_stability as h
import test_runtime_compatibility as compatibility
import test_runtime_recovery as recovery
import test_runtime_vae_failures as vae_failures
from modules import asset_files

torch = h.torch


class ExternalAuditTests(unittest.TestCase):
    def setUp(self):
        self.source = h.ROOT
        if os.environ.get('WEBUI_AUDIT_BASELINE'):
            h.ROOT = Path(os.environ['WEBUI_AUDIT_BASELINE'])

    def tearDown(self):
        h.ROOT = self.source

    def vae(self, root):
        return h.selected('modules/sd_vae.py', {'refresh_vae_list', 'get_filename', 'find_vae_near_checkpoint', 'load_vae'},
            os=os, glob=glob, collections=collections, vae_dict={}, base_vae=None,
            loaded_vae_file=None, checkpoints_loaded=collections.OrderedDict(),
            vae_path=str(root / 'vae'), sd_models=NS(model_path=str(root / 'models')),
            shared=NS(cmd_opts=NS(ckpt_dir=None, vae_dir=None), natural_sort_key=lambda s: s.casefold(),
                      opts=NS(sd_vae_checkpoint_cache=2), weight_load_location='cpu'),
            store_base_vae=lambda m: None, _load_vae_dict=lambda m, data: setattr(m, 'data', data),
            load_vae_dict=lambda fn, **kw: {'value': Path(fn).read_bytes()})

    def test_vae_exact_sibling_not_prefix_or_remote(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mod = self.vae(root)
            wrong = root / 'foobar.vae.safetensors'
            wrong.write_bytes(b'wrong')
            mod.vae_dict['foobar.vae.safetensors'] = str(wrong)
            self.assertIsNone(mod.find_vae_near_checkpoint(str(root / 'foo.safetensors')))
            exact = root / 'foo.vae.safetensors'
            exact.write_bytes(b'exact')
            self.assertEqual(mod.find_vae_near_checkpoint(str(root / 'foo.safetensors')), str(exact))
            (root / 'foo.vae.pt').write_bytes(b'ambiguous')
            self.assertIsNone(mod.find_vae_near_checkpoint(str(root / 'foo.safetensors')))

    def test_vae_duplicate_basename_both_visible_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / 'vae' / folder / 'shared.safetensors' for folder in ('b', 'a')]
            for path in paths:
                path.parent.mkdir(parents=True)
                path.write_bytes(b'vae')
            mod = self.vae(root)
            mod.refresh_vae_list()
            self.assertEqual(set(mod.vae_dict.values()), {str(p) for p in paths})
            self.assertEqual(list(mod.vae_dict), sorted(mod.vae_dict, key=str.casefold))
            self.assertNotIn('shared.safetensors', mod.vae_dict, 'ambiguous basename must not choose one silently')

    def test_vae_cache_replacement_and_force_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'same.safetensors'
            path.write_bytes(b'old')
            mod, model = self.vae(root), NS()
            mod.load_vae(model, str(path))
            before = path.stat()
            path.write_bytes(b'replaced weights')
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            mod.load_vae(model, str(path))
            self.assertEqual(model.data, {'value': b'replaced weights'})
            with patch.object(mod, 'load_vae_dict', return_value={'value': b'forced'}) as read:
                mod.load_vae(model, str(path), force_reload=True)
            self.assertEqual(read.call_count, 1)
            self.assertEqual(model.data, {'value': b'forced'})

    def test_active_vae_changed_on_disk_not_noop(self):
        mod, model, _, _, events, _ = vae_failures.VaeFailureTests().fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'active.vae'
            path.write_bytes(b'old')
            mod.loaded_vae_file = str(path)
            model.loaded_vae_signature = asset_files.file_signature(path)
            path.write_bytes(b'new weights')
            mod.reload_vae_weights(model, str(path))
            self.assertIn('weights', events)

    def test_checkpoint_cpu_cache_validates_file_generation_and_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint'
            path.write_bytes(b'old')
            class Info:
                filename = str(path)
                calculate_shorthash = lambda self: 'test'
            info = Info()
            info.cached_file_signature = asset_files.file_signature(path)
            mod = h.selected('modules/sd_models.py', {'get_checkpoint_state_dict'},
                checkpoints_loaded=collections.OrderedDict({info: {'weight': b'old'}}),
                read_state_dict=lambda fn: {'weight': Path(fn).read_bytes()})
            timer = NS(record=lambda *a: None)
            path.write_bytes(b'new longer weights')
            self.assertEqual(mod.get_checkpoint_state_dict(info, timer)['weight'], b'new longer weights')
            mod.checkpoints_loaded[info] = {'weight': b'wrong RAM'}
            self.assertEqual(mod.get_checkpoint_state_dict(info, timer, True)['weight'], b'new longer weights')

    def test_forced_reload_reaches_disk_read_flag(self):
        mod, _, previous, _, _, _ = recovery.RecoveryTests().loader()
        calls = []
        mod.get_checkpoint_state_dict = lambda *args: calls.append(args) or {'weight': 'disk'}
        mod.reload_model_weights(previous, previous.sd_checkpoint_info, forced_reload=True)
        self.assertTrue(calls and len(calls[0]) == 3 and calls[0][2] is True)

    def test_active_checkpoint_replaced_forces_reload(self):
        mod, _, previous, _, _, _ = recovery.RecoveryTests().loader()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'same.ckpt'
            path.write_bytes(b'old')
            previous.sd_checkpoint_info.filename = str(path)
            previous.loaded_checkpoint_signature = asset_files.file_signature(path)
            path.write_bytes(b'replacement')
            calls = []
            mod.get_checkpoint_state_dict = lambda *args: calls.append(args) or {'weight': 'disk'}
            mod.reload_model_weights(previous, previous.sd_checkpoint_info)
            self.assertTrue(calls and calls[0][2] is True)

    def test_lora_cache_signature_detects_same_size_replacement(self):
        signature = h.selected('extensions-builtin/Lora/networks.py', {'network_file_signature'}, os=os)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'adapter.safetensors'
            replacement = root / 'replacement.safetensors'
            path.write_bytes(b'old!')
            before = path.stat()
            replacement.write_bytes(b'new!')  # deliberately the same size
            os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
            old_signature = signature.network_file_signature(path)
            os.replace(replacement, path)
            new_signature = signature.network_file_signature(path)
            self.assertEqual(old_signature[3:5], new_signature[3:5])
            self.assertNotEqual(old_signature, new_signature)

    def test_lora_loader_rejects_file_changed_during_read(self):
        class Net:
            def __init__(self, name, disk):
                self.name, self.network_on_disk = name, disk
        signatures = iter(((1, 4, 1, 1, 1, 'a'), (2, 4, 2, 1, 2, 'a')))
        loader = h.selected('extensions-builtin/Lora/networks.py', {'load_network'},
            network=NS(Network=Net),
            network_file_signature=lambda _filename: next(signatures),
            sd_models=NS(read_state_dict=lambda _filename: {}))
        with self.assertRaisesRegex(RuntimeError, 'changed while reading'):
            loader.load_network(NS(name='changing', filename='changing.safetensors'))

    def test_metadata_bounds_before_allocating_and_accepts_json_whitespace(self):
        mod = h.selected('modules/sd_models.py', {'read_metadata_from_safetensors'}, errors=NS(report=lambda *a, **kw: None))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'header.safetensors'
            for data in (b'bad', (100_000_001).to_bytes(8, 'little') + b'{}', (999).to_bytes(8, 'little') + b'{}'):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    mod.read_metadata_from_safetensors(path)
            header = b'{ "__metadata__": {"test": "value"} }'
            path.write_bytes(len(header).to_bytes(8, 'little') + header)
            self.assertEqual(mod.read_metadata_from_safetensors(path), {'test': 'value'})

    def test_unknown_model_is_not_silently_sd1(self):
        env = dict(config_default='SD1', config_sdxl='XL', config_sd2='SD2', config_sd2v='SD2v')
        mod = h.selected('modules/sd_models_config.py', {'guess_model_config_from_state_dict'}, **env)
        with self.assertRaises(ValueError):
            mod.guess_model_config_from_state_dict({'unrelated': torch.ones(2)}, 'unknown')
        with self.assertRaisesRegex(ValueError, 'Incomplete SDXL'):
            mod.guess_model_config_from_state_dict({'conditioner.embedders.1.model.ln_final.weight': torch.ones(2)}, 'partial-xl')
        self.assertEqual(mod.guess_model_config_from_state_dict({
            'model.diffusion_model.input_blocks.0.0.weight': torch.zeros(1, 4, 3, 3),
            'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight': torch.ones(2, 768)}, 'sd1'), 'SD1')

    def test_lora_metadata_does_not_mislabel_sd3_flux_or_arbitrary_metadata_as_sd1(self):
        mod = h.selected('extensions-builtin/Lora/network.py', {'SdVersion', 'NetworkOnDisk'}, enum=enum)
        obj = mod.NetworkOnDisk.__new__(mod.NetworkOnDisk)
        for metadata, expected in (({'ss_base_model_version': 'sd3'}, 'SD3'),
                                   ({'modelspec.architecture': 'flux.1-dev/lora'}, 'Flux'),
                                   ({'modelspec.architecture': 'stable-diffusion-xl-v1-base/lora'}, 'SDXL'),
                                   ({'arbitrary': 'metadata'}, 'Unknown')):
            obj.metadata = metadata
            self.assertEqual(obj.detect_version().name, expected)

    def test_explicit_checkpoint_yaml_retains_priority(self):
        mod = h.selected('modules/sd_models_config.py', {'find_checkpoint_config', 'find_checkpoint_config_near_filename'},
            os=os, guess_model_config_from_state_dict=lambda *a: self.fail('explicit config was ignored'))
        with tempfile.TemporaryDirectory() as tmp:
            filename = Path(tmp) / 'custom.safetensors'
            filename.with_suffix('.yaml').write_text('model: explicit', encoding='utf-8')
            self.assertEqual(mod.find_checkpoint_config({}, NS(filename=str(filename))), str(filename.with_suffix('.yaml')))

    def test_sd2_v_marker_skips_probe_and_probe_failure_cleans_up(self):
        mod = h.selected('modules/sd_models_config.py', {'guess_model_config_from_state_dict'},
            config_sd2v='v', is_using_v_parameterization_for_sd2=lambda sd: self.fail('unnecessary forward'))
        self.assertEqual(mod.guess_model_config_from_state_dict({
            'model.diffusion_model.input_blocks.0.0.weight': torch.zeros(1, 4, 3, 3),
            'cond_stage_model.model.transformer.resblocks.0.attn.in_proj_weight': torch.zeros(3, 1024),
            'v_pred': torch.tensor(1)}, ''), 'v')
        modules = {name: ModuleType(name) for name in ('ldm', 'ldm.modules', 'ldm.modules.diffusionmodules', 'ldm.modules.diffusionmodules.openaimodel')}
        modules['ldm'].modules = modules['ldm.modules']
        modules['ldm.modules'].diffusionmodules = modules['ldm.modules.diffusionmodules']
        modules['ldm.modules.diffusionmodules'].openaimodel = modules['ldm.modules.diffusionmodules.openaimodel']
        class FakeUNet:
            def __init__(self, **kwargs):
                pass
            def eval(self):
                pass
            def load_state_dict(self, *a, **kw):
                raise RuntimeError('probe load failed')
        modules['ldm.modules.diffusionmodules.openaimodel'].UNetModel = FakeUNet
        cleaned = []
        mod = h.selected('modules/sd_models_config.py', {'is_using_v_parameterization_for_sd2'},
            torch=torch, devices=NS(device='cpu', torch_gc=lambda: cleaned.append(True)),
            sd_disable_initialization=NS(DisableInitialization=contextlib.nullcontext))
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, 'probe load failed'):
            mod.is_using_v_parameterization_for_sd2({})
        self.assertEqual(cleaned, [True])

    def test_ti_preflight_prevents_full_checkpoint_load_and_errors_are_deduplicated(self):
        ti = compatibility.CompatibilityTests().ti()
        model = NS(cond_stage_model=torch.nn.Module())
        opened = []
        mod = h.selected('modules/textual_inversion/textual_inversion.py', {'EmbeddingDatabase'},
            os=os, logging=__import__('logging'), Embedding=ti.Embedding,
            shared=NS(sd_model=model), cache=NS(cache=lambda name: {}),
            safetensors=NS(safe_open=lambda *a, **kw: opened.append(True)),
            errors=NS(report=lambda *a, **kw: self.fail('known format mismatch printed traceback')))
        db = mod.EmbeddingDatabase()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint.safetensors'
            header = json.dumps({'a': {'shape': [1000, 1000]}, 'b': {'shape': [1000, 1000]}}).encode()
            path.write_bytes(len(header).to_bytes(8, 'little') + header)
            with self.assertLogs(level='WARNING') as messages:
                db.load_from_dir(NS(path=tmp))
                db.load_from_dir(NS(path=tmp))
            self.assertEqual(opened, [])
            self.assertEqual(len(messages.output), 1)
            self.assertEqual(db.file_errors[str(path)]['status'], 'INCOMPATIBLE_ARCH')

    def test_create_sdxl_embedding_saves_separate_encoders_and_reloads(self):
        ti = compatibility.CompatibilityTests().ti()
        class Encoder(torch.nn.Module):
            def __init__(self, key, width, value):
                super().__init__()
                self.layer = torch.nn.Embedding(8, width)
                self.layer.textual_inversion_key = key
                self.value = value
            def encode_embedding_init_text(self, text, count):
                return torch.full((count + 1, self.layer.embedding_dim), self.value)
        class Conditioner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedders = torch.nn.ModuleList([Encoder('clip_l', 4, 2.), Encoder('clip_g', 6, 3.)])
            def forward(self, *args):
                pass
            def encode_embedding_init_text(self, text, count):
                return torch.cat([e.encode_embedding_init_text(text, count) for e in self.embedders], dim=1)
        with tempfile.TemporaryDirectory() as tmp:
            ti.shared.sd_model = NS(cond_stage_model=Conditioner())
            ti.shared.cmd_opts = NS(embeddings_dir=tmp)
            ti.devices.autocast = contextlib.nullcontext
            creator = h.selected('modules/textual_inversion/textual_inversion.py', {'create_embedding'},
                shared=ti.shared, devices=ti.devices, Embedding=ti.Embedding, torch=torch, os=os)
            filename = creator.create_embedding('new-xl', 2, False)
            saved = torch.load(filename, weights_only=True)
            self.assertNotIn('string_to_param', saved)
            restored = ti.create_embedding_from_data(saved, 'new-xl', expected_shapes={'clip_l': 4, 'clip_g': 6})
            torch.testing.assert_close(restored.vec['clip_l'], torch.full((2, 4), 2.))
            torch.testing.assert_close(restored.vec['clip_g'], torch.full((2, 6), 3.))

    def test_old_emphasis_dual_span_preserves_neighbour_tokens(self):
        ti = compatibility.CompatibilityTests().ti()
        embedding = ti.create_embedding_from_data({'clip_l': torch.ones(2, 4), 'clip_g': torch.ones(2, 6)}, 'dual')
        mod = h.selected('modules/sd_hijack_clip_old.py', {'process_text_old'}, shared=NS(opts=NS(emphasis='None')))
        model = NS(id_start=1, id_end=2, wrapped=NS(max_length=77), tokenize=lambda _: [[333, 7, 8]], token_mults={},
            hijack=NS(embedding_db=NS(find_embedding_at_position=lambda tokens, i: (embedding, 1) if i == 0 else (None, None))))
        result = mod.process_text_old(model, ['dual words'])
        self.assertEqual(result[1][0][:6], [1, 0, 0, 7, 8, 2])
        self.assertEqual(len(result[4][0]), 1)

    def test_ti_watcher_ignores_readme_not_embeddings(self):
        mod = h.selected('modules/textual_inversion/textual_inversion.py', {'DirWithTextualInversionEmbeddings'}, os=os)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'one.pt').write_bytes(b'embedding')
            watcher = mod.DirWithTextualInversionEmbeddings(tmp)
            watcher.update()
            (root / 'readme.txt').write_text('unrelated', encoding='utf-8')
            self.assertFalse(watcher.has_changed())
            (root / 'one.pt').write_bytes(b'changed embedding')
            self.assertTrue(watcher.has_changed())

    def test_png_cache_replacement_same_mtime_is_reloaded_and_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'embedding.png'
            path.write_bytes(b'old')
            opened, closed = [], []
            class ImageStub:
                def __init__(self, filename):
                    opened.append(filename)
                    self.text = {'sd-ti-embedding': json.dumps({'name': Path(filename).read_text()})}
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    closed.append(True)
            mod = h.selected('modules/textual_inversion/textual_inversion.py', {'EmbeddingDatabase'},
                os=os, Image=NS(open=ImageStub), embedding_from_b64=json.loads,
                shared=NS(opts=NS(textual_inversion_image_embedding_data_cache=True)),
                cache=NS(cache=lambda name: {}), errors=NS(report=lambda *a, **kw: None))
            db = mod.EmbeddingDatabase()
            self.assertEqual(db.read_embedding_from_image(str(path), 'fallback')[1], 'old')
            before = path.stat()
            path.write_bytes(b'replaced')
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            self.assertEqual(db.read_embedding_from_image(str(path), 'fallback')[1], 'replaced')
            db.read_embedding_from_image(str(path), 'fallback')
            self.assertEqual(len(opened), 2)
            self.assertEqual(len(closed), 2)

    def test_ti_traversal_cycle_is_pruned_and_order_shared(self):
        # OS-independent cycle simulation; realpath reveals the revisited root.
        calls = []
        def walk(*args, **kwargs):
            dirs = ['loop']
            yield 'root', dirs, ['z.pt', 'readme.txt', 'a.safetensors']
            loop_dirs = ['loop']
            yield 'root/loop', loop_dirs, ['z.pt']
            calls.append(loop_dirs)
        with patch.object(asset_files.os, 'walk', walk), patch.object(asset_files.os.path, 'realpath', return_value='root'):
            files = list(asset_files.embedding_files('root'))
        self.assertEqual([os.path.basename(p) for p in files], ['a.safetensors', 'z.pt'])
        self.assertEqual(calls, [[]])

    def test_ti_header_preflight_rejects_checkpoint_and_unequal_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'header.safetensors'
            def write(header):
                data = json.dumps(header).encode()
                path.write_bytes(len(data).to_bytes(8, 'little') + data)
            write({'weight1': {'shape': [100, 100]}, 'weight2': {'shape': [100, 100]}})
            with self.assertRaisesRegex(ValueError, 'not a supported textual'):
                asset_files.preflight_embedding(path, {})
            write({'clip_l': {'shape': [2, 4], 'dtype': 'F32'}, 'clip_g': {'shape': [3, 6], 'dtype': 'F32'}})
            with self.assertRaisesRegex(ValueError, 'UNSUPPORTED_DUAL_ROWS'):
                asset_files.preflight_embedding(path, {'clip_l': 4, 'clip_g': 6})


if __name__ == '__main__':
    unittest.main(verbosity=2)
