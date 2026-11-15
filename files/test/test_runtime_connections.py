"""Desired-contract tests of cross-module state. CPU only; no model downloads.

Production AST: registry/cache/reuse/lowvram; model transfers are CPU or injected
failures. A failing test means an unresolved contract, not a passing bug snapshot.
"""
import ast
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


def selected(path, names, methods=None, **env):
    tree = ast.parse((ROOT / path).read_bytes(), path)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    if methods:
        for n in nodes:
            if isinstance(n, ast.ClassDef):
                n.bases = []
                n.decorator_list = []
                n.body = [m for m in n.body if isinstance(m, ast.FunctionDef) and m.name in methods]
    code = ast.Module([ast.ImportFrom('__future__', [ast.alias('annotations')], 0), *nodes], [])
    exec(compile(ast.fix_missing_locations(code), path, 'exec'), env)
    return NS(**env)


class ConnectionTests(unittest.TestCase):
    def test_region_context_is_restored_after_nested_failure(self):
        roles = ContextVar('audit_roles', default=None)
        shape = ContextVar('audit_shape', default=None)
        contexts = selected('modules/region_utils.py', {'region_attention_context', 'region_query_context'},
                            contextmanager=contextmanager, _attention_roles=roles, _query_shape=shape)
        with contexts.region_attention_context([0, 1]), contexts.region_query_context([8, 12]):
            with self.assertRaises(RuntimeError):
                with contexts.region_attention_context([1]), contexts.region_query_context([4, 6]):
                    raise RuntimeError('injected attention failure')
            self.assertEqual(roles.get(), (0, 1))
            self.assertEqual(shape.get(), (8, 12))
        self.assertIsNone(roles.get())
        self.assertIsNone(shape.get())

    def test_extension_callback_failure_is_reported_and_later_callbacks_run(self):
        events = []

        def bad(model):
            raise RuntimeError('injected extension callback failure')

        callbacks = [NS(callback=bad), NS(callback=lambda model: events.append(model))]
        reported = []
        dispatcher = selected('modules/script_callbacks.py', {'model_loaded_callback'},
                              ordered_callbacks=lambda name: callbacks,
                              report_exception=lambda callback, name: reported.append(callback))
        model = object()
        dispatcher.model_loaded_callback(model)
        self.assertEqual(reported, [callbacks[0]])
        self.assertEqual(events, [model])

    def registry(self):
        hijack = NS(fixes=None)
        devices = NS(cond_cast_unet=lambda v: v, torch_npu_set_device=lambda: None)
        consumer = selected('modules/sd_hijack.py', {'EmbeddingsWithFixes'}, torch=torch, devices=devices)

        class Conditioner(torch.nn.Module):
            def __init__(self, width):
                super().__init__()
                self.token_embedding = consumer.EmbeddingsWithFixes(torch.nn.Embedding(4, width), hijack)
                self.width = width

            def tokenize(self, texts):
                return [[1] for text in texts]

            def encode_embedding_init_text(self, text, count):
                return torch.zeros(count, self.width)

        first = NS(cond_stage_model=Conditioner(4), sd_checkpoint_info=NS(filename='a', title='a'))
        second = NS(cond_stage_model=Conditioner(6), sd_checkpoint_info=NS(filename='b', title='b'))
        state = NS(sd_model=first, loaded_sd_models=[first, second])

        class Shared:
            opts = NS(textual_inversion_print_at_load=False, sd_checkpoints_keep_in_cpu=False, sd_checkpoints_limit=2)

            @property
            def sd_model(self):
                return state.sd_model

        shared = Shared()
        ti = selected('modules/textual_inversion/textual_inversion.py', {'Embedding', 'EmbeddingDatabase'},
                      torch=torch, shared=shared, devices=devices, cache=NS(cache=lambda key: {}))
        db = ti.EmbeddingDatabase()
        db.embedding_dirs['virtual'] = NS(has_changed=lambda: False, snapshot=lambda: 1)
        db.expected_shape = 4
        original = ti.Embedding(torch.ones(1, 4), 'token')
        db.register_embedding(original, first)
        loads = []

        def load(directory):
            loads.append(shared.sd_model)
            width = shared.sd_model.cond_stage_model.width
            db.register_embedding(ti.Embedding(torch.ones(1, width), 'token'), shared.sd_model)

        db.load_from_dir = load  # disk reader only; actual scan/registry logic retained

        def publish(model, already_loaded=False):
            state.sd_model = model

        state.set_sd_model = publish
        switch = selected('modules/sd_models.py', {'reuse_model_from_already_loaded', 'discard_failed_model'},
                          shared=shared, model_data=state, devices=NS(fp8=False),
                          model_has_fp8_weights=lambda model: False,
                          send_model_to_device=lambda model: None,
                          sd_vae=NS(reload_vae_weights=lambda model: None),
                          SkipWritingToConfig=NS(skip=True))
        return shared, first, second, db, loads, switch, hijack

    def test_explicit_registry_refresh_revalidates_for_new_model(self):
        shared, first, second, db, loads, switch, hijack = self.registry()
        timer = NS(record=lambda *args: None, summary=lambda: 'test')
        switch.reuse_model_from_already_loaded(first, second.sd_checkpoint_info, timer)
        db.load_textual_inversion_embeddings(force_reload=True)
        self.assertEqual(db.expected_shape, 6)
        self.assertEqual(db.word_embeddings['token'].vec.shape[1], 6)

    def test_cached_checkpoint_switch_revalidates_unchanged_embedding_directory(self):
        shared, first, second, db, loads, switch, hijack = self.registry()
        timer = NS(record=lambda *args: None, summary=lambda: 'test')
        switch.reuse_model_from_already_loaded(first, second.sd_checkpoint_info, timer)
        self.assertIs(shared.sd_model, second)
        db.load_textual_inversion_embeddings()  # actual process_images_inner call
        stale = db.word_embeddings['token']
        hijack.fixes = [[(0, stale)]]
        if stale.vec.shape[1] != 6:
            # Confirm this is not harmless metadata: the real consumer fails.
            with self.assertRaises(RuntimeError):
                second.cond_stage_model.token_embedding(torch.tensor([[0, 1, 2]]))
        self.assertEqual(db.expected_shape, 6, 'registry retained previous model width')

    def conditioning(self):
        shared, first, second, db, loads, switch, hijack = self.registry()
        opts = NS(CLIP_stop_at_last_layers=1, sdxl_crop_left=0, sdxl_crop_top=0,
                  fp8_storage='Disable', cache_fp16_weight=False, emphasis='Original', use_old_scheduling=False,
                  textual_inversion_print_at_load=False)
        shared.opts = opts
        processor = selected('modules/processing.py', {'StableDiffusionProcessing'},
                             methods={'cached_params', 'get_conds_with_caching'}, shared=shared, opts=opts,
                             devices=NS(autocast=nullcontext), model_hijack=NS(embedding_db=db)).StableDiffusionProcessing()
        processor.width = processor.height = 512
        processor.extra_generation_params = {}
        cache = [None, None]
        calls = []

        def encode(*args):
            calls.append(1)
            return db.word_embeddings['token'].vec.clone()

        def compute():
            return processor.get_conds_with_caching(encode, ['token'], 20, [cache], {})

        return db, calls, compute, opts, processor

    def test_unchanged_conditioning_uses_cache(self):
        db, calls, compute, opts, processor = self.conditioning()
        first = compute()
        self.assertIs(compute(), first)
        self.assertEqual(len(calls), 1)

    def test_clip_skip_change_invalidates_conditioning(self):
        db, calls, compute, opts, processor = self.conditioning()
        compute()
        opts.CLIP_stop_at_last_layers = 2
        compute()
        self.assertEqual(len(calls), 2)

    def test_refreshed_embedding_invalidates_persistent_conditioning(self):
        db, calls, compute, opts, processor = self.conditioning()
        compute()
        # Actual API refresh + database scan, with just the disk reader replaced.
        # Same model/name but a different, newly registered embedding object.
        embedding = db.word_embeddings['token']
        replacement = type(embedding)(torch.full_like(embedding.vec, 7), 'token')
        shared = processor.cached_params.__func__.__globals__['shared']
        db.load_from_dir = lambda directory: db.register_embedding(replacement, shared.sd_model)
        api = selected('modules/api/api.py', {'Api'}, methods={'refresh_embeddings'},
                       sd_hijack=NS(model_hijack=NS(embedding_db=db))).Api()
        api.queue_lock = nullcontext()
        api.refresh_embeddings()
        self.assertIs(db.word_embeddings['token'], replacement)
        output = compute()
        torch.testing.assert_close(output, replacement.vec)
        self.assertEqual(len(calls), 2)

    def test_unit_emphasis_preserves_finite_zero_mean_conditioning(self):
        emphasis = selected('modules/sd_emphasis.py', {'Emphasis', 'EmphasisOriginal'}, torch=torch)
        value = emphasis.EmphasisOriginal()
        value.z = torch.tensor([[[-1., 1.], [-2., 2.]]])
        original = value.z.clone()
        value.multipliers = torch.ones(1, 2)
        value.after_transformers()
        self.assertTrue(torch.isfinite(value.z).all().item(), 'unit weighting turned finite conditioning into NaN')
        torch.testing.assert_close(value.z, original)

    def test_no_norm_preserves_finite_zero_mean_conditioning(self):
        emphasis = selected('modules/sd_emphasis.py', {'Emphasis', 'EmphasisOriginal', 'EmphasisOriginalNoNorm'}, torch=torch)
        value = emphasis.EmphasisOriginalNoNorm()
        value.z = torch.tensor([[[-1., 1.], [-2., 2.]]])
        original = value.z.clone()
        value.multipliers = torch.ones(1, 2)
        value.after_transformers()
        torch.testing.assert_close(value.z, original)

    def lowvram(self, fail=False):
        class Stage(torch.nn.Linear):
            def encode(self, x):
                return self(x)

            def decode(self, x):
                return self(x)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first_stage_model = Stage(2, 2)
                self.cond_stage_model = torch.nn.Module()
                self.cond_stage_model.transformer = torch.nn.Linear(2, 2)
                self.model = torch.nn.Linear(2, 2)

            def to(self, *args, **kwargs):
                if fail:
                    raise RuntimeError('injected device transfer failure')
                return super().to(*args, **kwargs)

        model = Model()
        originals = model.first_stage_model, model.model, model.cond_stage_model.transformer
        low = selected('modules/lowvram.py', {'setup_for_low_vram'}, torch=torch,
                       cpu=torch.device('cpu'), devices=NS(device=torch.device('cpu')), module_in_gpu=None)
        return low, model, originals

    def test_successful_lowvram_setup_preserves_attached_modules(self):
        low, model, original = self.lowvram()
        low.setup_for_low_vram(model, use_medvram=True)
        self.assertEqual((model.first_stage_model, model.model, model.cond_stage_model.transformer), original)
        self.assertTrue(model.lowvram)

    def test_failed_lowvram_setup_restores_detached_modules(self):
        low, model, original = self.lowvram(fail=True)
        with self.assertRaisesRegex(RuntimeError, 'injected device'):
            low.setup_for_low_vram(model, use_medvram=True)
        self.assertEqual((model.first_stage_model, model.model, model.cond_stage_model.transformer), original)
        self.assertFalse(getattr(model, 'lowvram', False))


if __name__ == '__main__':
    unittest.main(verbosity=2)

