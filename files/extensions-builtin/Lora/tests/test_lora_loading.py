"""Numerical regression tests of actual loader AST with NumPy stand-ins, NOT Torch.

Run: python -B -m unittest discover -s extensions-builtin/Lora/tests -v
The stand-ins test our dispatch/math/lifecycle, not PyTorch/CUDA integration.
"""
from __future__ import annotations

import ast
import __future__
from collections import namedtuple
from contextlib import nullcontext
from dataclasses import dataclass
import importlib.abc
import importlib.util
import json
import logging
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lora_keys  # noqa: E402
import lora_runtime  # noqa: E402
sys.path.pop(0)


class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('Real Torch imports are forbidden in this test suite')


class Shape(tuple):
    def numel(self):
        return int(np.prod(self))


class Tensor:
    def __init__(self, data):
        self.array = np.array(data)

    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def device(self):
        return 'cpu'

    def size(self, dim=None):
        return Shape(self.shape) if dim is None else self.shape[dim]

    def to(self, device=None, dtype=None, copy=False):
        return Tensor(self.array.astype(dtype or self.dtype, copy=True))

    def detach(self):
        return self

    def clone(self):
        return Tensor(self.array.copy())

    def copy_(self, other):
        self.array[...] = other.array
        return self

    def reshape(self, *shape):
        return Tensor(self.array.reshape(*shape))

    def transpose(self, a, b):
        return Tensor(self.array.swapaxes(a, b))

    def item(self):
        return self.array.item()

    def requires_grad_(self, value):
        return self

    def __getitem__(self, index):
        return Tensor(self.array[index])

    def __matmul__(self, other):
        return Tensor(self.array @ other.array)

    def __add__(self, other):
        return Tensor(self.array + (other.array if isinstance(other, Tensor) else other))

    def __sub__(self, other):
        return Tensor(self.array - (other.array if isinstance(other, Tensor) else other))

    def __mul__(self, other):
        return Tensor(self.array * (other.array if isinstance(other, Tensor) else other))


def embedding(indices, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    assert np.issubdtype(indices.dtype, np.integer), 'Embedding indices were incorrectly cast to floating point'
    if max_norm is not None:
        for index in np.unique(indices.array):
            norm = np.linalg.norm(weight.array[index], ord=norm_type)
            if norm > max_norm:
                weight.array[index] *= max_norm / (norm + 1e-7)
    return Tensor(weight.array[indices.array])


def linear(inputs, weight, bias=None):
    return Tensor(inputs.array @ weight.array.T + (bias.array if bias is not None else 0))


class Embedding:
    def __init__(self, rows=5, columns=3, max_norm=None):
        self.weight = Tensor(np.arange(rows * columns, dtype=np.float64).reshape(rows, columns) / 10)
        self.padding_idx, self.max_norm, self.norm_type = None, max_norm, 2.0
        self.scale_grad_by_freq, self.sparse = False, False
        self.network_layer_name = '0_transformer_text_model_embeddings_position_embedding'

    def forward(self, indices):
        return embedding(indices, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

    def _load_from_state_dict(self, replacement):
        self.weight.copy_(replacement)


class Linear:
    def __init__(self, ins=3, outs=5, bias=False):
        self.weight = Tensor(np.zeros((outs, ins)))
        self.bias = None

    def forward(self, inputs):
        return linear(inputs, self.weight, self.bias)

    __call__ = forward


def ast_module(name, filename, wanted, namespace):
    tree = ast.parse((ROOT / filename).read_bytes())
    body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted]
    module = ModuleType(name)
    module.__dict__.update(namespace)
    sys.modules[name] = module
    exec(compile(ast.Module(body=body, type_ignores=[]), str(ROOT / filename), 'exec', flags=__future__.annotations.compiler_flag), module.__dict__)
    return module


class LoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = NoTorch()
        sys.meta_path.insert(0, cls.guard)

    @classmethod
    def tearDownClass(cls):
        sys.meta_path.remove(cls.guard)
        assert 'torch' not in sys.modules

    def setUp(self):
        nn = NS(Embedding=Embedding, Linear=Linear, Conv2d=type('Conv2d', (), {}), MultiheadAttention=type('MHA', (), {}),
                LayerNorm=type('LayerNorm', (), {}), GroupNorm=type('GroupNorm', (), {}),
                modules=NS(linear=NS(NonDynamicallyQuantizableLinear=type('NDQL', (), {}))), functional=NS(embedding=embedding))
        self.torch = NS(nn=nn, no_grad=nullcontext)
        self.shared = NS(opts=NS(lora_functional=False, lora_gpu_backup=False), state=NS(sampling_step=0, sampling_steps=10))
        self.devices = NS(cpu='cpu', device='cpu', dtype=np.float64, cond_cast_unet=lambda _: (_ for _ in ()).throw(AssertionError('Embedding used U-Net input cast')))
        modules = NS(models=NS(sd3=NS(mmdit=NS(QkvLinear=type('QkvLinear', (), {})))))
        self.base = ast_module('_lora_test_network', 'network.py', {'Network', 'NetworkActivation', 'NetworkModule', 'ModuleType'}, {
            'nn': nn, 'F': NS(linear=linear), 'modules': modules, 'shared': self.shared, 'dataclass': dataclass, 'lora_runtime': lora_runtime,
        })
        self.addCleanup(sys.modules.pop, '_lora_test_network', None)
        self.base.NetworkWeights = namedtuple('NetworkWeights', 'network_key sd_key w sd_module')
        helpers = ast_module('_lora_test_helpers', 'lyco_helpers.py', {'rebuild_conventional'}, {})
        self.addCleanup(sys.modules.pop, '_lora_test_helpers', None)
        self.lora = ast_module('_lora_test_lora', 'network_lora.py', {'ModuleTypeLora', 'NetworkModuleLoraEmbedding', 'NetworkModuleLora'}, {
            'torch': self.torch, 'network': self.base, 'lora_keys': lora_keys, 'devices': self.devices, 'modules': modules, 'lyco_helpers': helpers,
        })
        self.addCleanup(sys.modules.pop, '_lora_test_lora', None)
        names = {'load_network', 'store_weights_backup', 'restore_weights_backup', 'network_restore_weights_from_backup',
                 'network_apply_weights', '_network_apply_weights', 'network_reset_cached_weight', 'network_Embedding_forward',
                 'network_Embedding_load_state_dict', 'network_forward'}
        self.runtime = ast_module('_lora_test_runtime', 'networks.py', names, {
            'torch': self.torch, 'shared': self.shared, 'devices': self.devices, 'modules': modules,
            'loaded_networks': [], 'network': self.base, 'lora_keys': lora_keys, 'logging': logging,
            'originals': NS(Embedding_forward=Embedding.forward, Embedding_load_state_dict=Embedding._load_from_state_dict),
            're_x_proj': re.compile(r'(.*)_([qkv]_proj)$'), 'convert_diffusers_name_to_compvis': lambda name, sd2: name,
            'network_file_signature': lambda _: (123000000000, 100), 'extra_network_lora': NS(errors={}),
        })
        self.addCleanup(sys.modules.pop, '_lora_test_runtime', None)
        self.layer = Embedding()
        self.indices = Tensor(np.array([[0, 2, 4]], dtype=np.int64))

    def activation(self, layer=None, *, transpose=False, peft=False, dyn=None, multiplier=0.7, extra=None):
        layer = layer or self.layer
        rows, cols = layer.weight.shape
        up = Tensor(np.arange((cols if transpose else rows) * 2, dtype=float).reshape(-1, 2) / 20 + 0.1)
        down = Tensor(np.arange(2 * (rows if transpose else cols), dtype=float).reshape(2, -1) / 30 + 0.2)
        parts = {'alpha': Tensor(4.), 'lora_embedding_B' if peft else 'lora_up.weight': up,
                 'lora_embedding_A' if peft else 'lora_down.weight': down, **(extra or {})}
        net = self.base.Network('test', NS(filename='test.safetensors'))
        weights = self.base.NetworkWeights('test.alpha', layer.network_layer_name, parts, layer)
        adapter = self.lora.ModuleTypeLora().create_module(net, weights)
        net.modules[layer.network_layer_name] = adapter
        activation = self.base.NetworkActivation(net, 'test', te_multiplier=multiplier, unet_multiplier=99., dyn_dim=dyn)
        return activation, up.array.copy(), down.array.copy()

    def forward(self, activations, functional=False, layer=None):
        self.runtime.loaded_networks = activations
        self.shared.opts.lora_functional = functional
        return self.runtime.network_Embedding_forward(layer or self.layer, self.indices).array

    def test_direct_and_transposed_numeric_parity_with_dyn_alpha_and_te(self):
        for transpose in (False, True):
            for functional in (False, True):
                for dyn in (None, 1, 9):
                    with self.subTest(transpose=transpose, functional=functional, dyn=dyn):
                        self.layer = Embedding()
                        base = self.layer.weight.array.copy()
                        act, up, down = self.activation(transpose=transpose, dyn=dyn)
                        rank = dyn or 2
                        delta = up[:, :rank] @ down[:rank]
                        if transpose:
                            delta = delta.T
                        expected = (base + delta * 2 * 0.7)[self.indices.array]
                        np.testing.assert_allclose(self.forward([act], functional), expected)
                        if functional:
                            np.testing.assert_array_equal(self.layer.weight.array, base)

    def test_disable_and_merge_functional_mode_switch_restores_base(self):
        base = self.layer.weight.array.copy()
        act, _, _ = self.activation()
        expected = self.forward([act])
        np.testing.assert_allclose(self.forward([act], functional=True), expected)
        np.testing.assert_array_equal(self.layer.weight.array, base)
        self.forward([act])
        np.testing.assert_array_equal(self.forward([], functional=True), base[self.indices.array])
        self.forward([act])
        np.testing.assert_array_equal(self.forward([]), base[self.indices.array])
        self.assertFalse(hasattr(self.layer, 'bias'))

    def test_unrelated_lora_and_plain_embedding_are_passthrough(self):
        base = self.layer.weight.array.copy()
        unrelated = self.base.NetworkActivation(self.base.Network('other', NS(filename='other')), 'other')
        np.testing.assert_array_equal(self.forward([unrelated]), base[self.indices.array])
        self.assertFalse(hasattr(self.layer, 'network_weights_backup'))
        np.testing.assert_array_equal(self.forward([], True), base[self.indices.array])

    def test_repeated_payload_and_te_change_do_not_accumulate(self):
        act, up, down = self.activation()
        base = self.layer.weight.array.copy()
        other = self.base.NetworkActivation(act.network, 'alias', te_multiplier=-0.2)
        wanted = (base + (up @ down) * 2 * (0.7 - 0.2))[self.indices.array]
        np.testing.assert_allclose(self.forward([act, other]), wanted)
        np.testing.assert_allclose(self.forward([act, other]), wanted)
        act.te_multiplier = 0.1
        np.testing.assert_allclose(self.forward([act]), (base + (up @ down) * 0.2)[self.indices.array])
        np.testing.assert_array_equal(self.forward([]), base[self.indices.array])

    def test_hires_weight_and_block_mask_apply_to_embedding(self):
        act, _, _ = self.activation()
        base = self.layer.weight.array.copy()
        act.is_hires, act.hr_te_multiplier = True, 0.
        np.testing.assert_array_equal(self.forward([act]), base[self.indices.array])
        act.hr_te_multiplier = 1.
        act.block_weights = NS(value_for=lambda *args, **kwargs: 0.)
        np.testing.assert_array_equal(self.forward([act]), base[self.indices.array])

    def test_max_norm_functional_matches_merge_without_mutating_base(self):
        act, _, _ = self.activation()
        self.layer.max_norm = 0.5
        base = self.layer.weight.array.copy()
        functional = self.forward([act], True)
        np.testing.assert_array_equal(self.layer.weight.array, base)
        np.testing.assert_allclose(self.forward([act]), functional)

    def test_checkpoint_reload_clears_old_backup(self):
        act, _, _ = self.activation()
        self.forward([act])
        replacement = Tensor(np.ones((5, 3)))
        self.runtime.network_Embedding_load_state_dict(self.layer, replacement)
        np.testing.assert_array_equal(self.forward([]), replacement.array[self.indices.array])
        self.assertIsNone(self.layer.network_weights_backup)

    def test_peft_embedding_pair_has_explicit_transpose(self):
        act, up, down = self.activation(transpose=True, peft=True)
        base = self.layer.weight.array.copy()
        np.testing.assert_allclose(self.forward([act], True), (base + (up @ down).T * 1.4)[self.indices.array])

    def test_unsupported_embedding_formats_and_invalid_dyn_fail(self):
        for extra in ({'dora_scale': Tensor([1, 1, 1])}, {'lora_mid.weight': Tensor([[1.]])}, {'bias': Tensor([1, 2, 3])}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.activation(extra=extra)
        for dyn in (0, -1, 1.5, True):
            with self.subTest(dyn=dyn), self.assertRaises(ValueError):
                act, _, _ = self.activation(dyn=dyn)
                self.forward([act], True)

    def test_ab_alias_retains_metadata_and_original_dict(self):
        # Exercise the real factory but intercept the regular Linear constructor.
        original = {'lora_A.weight': object(), 'lora_B.weight': object(), 'alpha': 7, 'scale': 0.5, 'dora_scale': object()}
        captured = []
        self.lora.NetworkModuleLora = lambda net, weights: captured.append(weights) or 'ok'
        weights = self.base.NetworkWeights('linear', 'linear', dict(original), Linear())
        self.assertEqual(self.lora.ModuleTypeLora().create_module(None, weights), 'ok')
        self.assertEqual(weights.w, original)
        for key in ('alpha', 'scale', 'dora_scale'):
            self.assertIs(captured[0].w[key], original[key])

    def fake_load(self, state, mapping, diffusers_map=None):
        self.shared.sd_model = NS(network_layer_mapping=mapping)
        if diffusers_map is not None:
            self.shared.sd_model.diffusers_weight_map = diffusers_map
        self.runtime.sd_models = NS(read_state_dict=lambda _: state)
        captured = []
        def create(net, weights):
            captured.append(weights)
            if 'lora_up.weight' in weights.w and 'lora_down.weight' in weights.w:
                return NS(network_to_device=lambda _: None)
            return None
        self.runtime.module_types = [NS(create_module=create)]
        self.runtime.textual_inversion = NS(
            Embedding=NS(encoder_shapes=lambda model: {'clip_l': 4, 'clip_g': 6}),
            create_embedding_from_data=lambda data, name, filename, expected_shapes: NS(data=data, expected_shapes=expected_shapes))
        self.runtime.BundledTIHash = lambda name: name
        result = self.runtime.load_network(NS(name='fixture', filename='fixture.safetensors'))
        return result, captured

    def test_loader_groups_790_dotted_layers_separately(self):
        paths = json.loads((ROOT / 'tests/ceb_reported_layers.json').read_text(encoding='utf-8'))
        self.assertEqual(len(paths), 790)
        for prefix in ('lora_unet', 'diffusion_model'):
            state, mapping = {}, {}
            for layer in paths:
                for suffix in ('lora_down.weight', 'lora_up.weight'):
                    state[f'{prefix}.{layer}.{suffix}'] = object()
                mapping['diffusion_model_' + layer.replace('.', '_')] = object()
            net, captured = self.fake_load(state, mapping)
            self.assertEqual(len(net.modules), 790)
            self.assertTrue(all(set(w.w) == {'lora_up.weight', 'lora_down.weight'} for w in captured))

    def test_regular_linear_ab_constructor_keeps_alpha_scale(self):
        # Construct the actual regular LoRA adapter, not just an intercepted factory.
        layer = Linear(3, 5)
        data = {'lora_A.weight': Tensor(np.ones((2, 3))), 'lora_B.weight': Tensor(np.ones((5, 2))),
                'alpha': Tensor(8.), 'scale': Tensor(0.125)}
        weights = self.base.NetworkWeights('linear.alpha', 'linear', data, layer)
        adapter = self.lora.ModuleTypeLora().create_module(None, weights)
        self.assertEqual(adapter.alpha, 8.)
        self.assertEqual(adapter.calc_scale(), 0.125)
        self.assertNotIn('lora_up.weight', data)
        act = NS(dyn_dim=1, is_text_encoder_key=lambda key: False, effective_multiplier=lambda te: 0.5,
                 layer_multiplier=lambda *args: 1.)
        delta, bias = adapter.calc_updown(layer.weight, act)
        np.testing.assert_allclose(delta.array, np.ones((5, 3)) * 0.0625)
        self.assertIsNone(bias)

    def test_dtype_conversion_and_bias_matrix_preserve_factors(self):
        act, up, down = self.activation(extra={'bias': Tensor(np.ones((5, 3)) * 0.25)})
        adapter = act.modules[self.layer.network_layer_name]
        self.devices.dtype = np.float32
        adapter.network_to_device('cpu')
        self.assertEqual(adapter.up.dtype, np.dtype('float32'))
        base = self.layer.weight.array.copy()
        expected = (base + (up @ down + 0.25) * 1.4)[self.indices.array]
        np.testing.assert_allclose(self.forward([act], True), expected, rtol=1e-6)
        np.testing.assert_allclose(self.forward([act]), expected, rtol=1e-6)
        np.testing.assert_allclose(adapter.up.array, up, rtol=1e-6)
        np.testing.assert_allclose(adapter.down.array, down, rtol=1e-6)

    def test_patches_register_and_undo_embedding_hooks(self):
        originals = {}
        for name in ('Linear', 'Embedding', 'Conv2d', 'GroupNorm', 'LayerNorm', 'MultiheadAttention'):
            cls = getattr(self.torch.nn, name)
            for field in ('forward', '_load_from_state_dict'):
                if not hasattr(cls, field):
                    setattr(cls, field, lambda *args, **kwargs: None)
                originals[cls, field] = getattr(cls, field)
            for suffix in ('forward', 'load_state_dict'):
                method = f'network_{name}_{suffix}'
                if not hasattr(self.runtime, method):
                    setattr(self.runtime, method, lambda *args, **kwargs: None)

        # Exercise the matching production registry, including batch preflight.
        spec = importlib.util.spec_from_file_location('_lora_test_patch_registry', ROOT.parents[1] / 'modules/patches.py')
        registry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(registry)

        patch_module = ast_module('_lora_test_patches', 'lora_patches.py', {'LoraPatches'}, {
            'torch': self.torch, 'networks': self.runtime, 'patches': registry,
        })
        self.addCleanup(sys.modules.pop, '_lora_test_patches', None)
        try:
            patcher = patch_module.LoraPatches()
            self.assertIs(Embedding.forward, self.runtime.network_Embedding_forward)
            self.runtime.originals = patcher
            act, _, _ = self.activation()
            self.runtime.loaded_networks = [act]
            base = self.layer.weight.array.copy()
            self.layer.forward(self.indices)
            self.runtime.loaded_networks = []
            self.layer.forward(self.indices)
            np.testing.assert_array_equal(self.layer.weight.array, base)
            patcher.undo()
            self.assertIs(Embedding.forward, originals[Embedding, 'forward'])
            self.assertIs(Embedding._load_from_state_dict, originals[Embedding, '_load_from_state_dict'])
        finally:
            for (cls, field), original in originals.items():
                setattr(cls, field, original)

    def test_generic_functional_disable_restores_previous_merge(self):
        self.layer.network_weights_backup = Tensor(np.zeros((5, 3)))
        self.layer.network_current_names = ('old',)
        self.runtime.loaded_networks = []
        output = self.runtime.network_forward(self.layer, self.indices, Embedding.forward)
        np.testing.assert_array_equal(output.array, np.zeros((1, 3, 3)))
        self.assertIsNone(self.layer.network_weights_backup)

    def test_diffusers_map_keeps_alpha_on_its_layer(self):
        state = {f'unet.block.0.{part}': object() for part in ('alpha', 'lora_up.weight', 'lora_down.weight')}
        _, captured = self.fake_load(state, {'actual_layer': object()}, {'unet.block.0': 'actual_layer'})
        self.assertEqual(set(captured[0].w), {'alpha', 'lora_up.weight', 'lora_down.weight'})

    def test_duplicate_aliases_fail_instead_of_overwriting(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.fake_load({'lora_unet.input.0.alpha': object(), 'lora_unet_input_0.alpha': object()}, {'diffusion_model_input_0': object()})

    def test_no_matched_layers_is_reported(self):
        with self.assertRaisesRegex(ValueError, 'matched no layers'):
            self.fake_load({'missing.alpha': object()}, {})

    def test_bundled_textual_inversion_preserves_multiple_parameters(self):
        a, b = object(), object()
        state = {'bundle_emb.word.string_to_param.clip_l': a, 'bundle_emb.word.string_to_param.clip_g': b}
        net, captured = self.fake_load(state, {}, {'unrelated': 'other'})
        self.assertFalse(captured)
        self.assertEqual(net.bundle_embeddings['word'].data['string_to_param'], {'clip_l': a, 'clip_g': b})
        self.assertEqual(net.bundle_embeddings['word'].expected_shapes, {'clip_l': 4, 'clip_g': 6})


class FormatTests(unittest.TestCase):
    def test_every_supported_suffix_with_dots_and_underscores(self):
        for layer in ('lora_unet.input_blocks.0.0', 'lora_unet_input_blocks_0_0'):
            for part in lora_keys.WEIGHT_PARTS:
                with self.subTest(layer=layer, part=part):
                    self.assertEqual(lora_keys.split_network_key(layer + '.' + part), (layer, part))

    def test_bundle_and_legacy_extra_field(self):
        self.assertEqual(lora_keys.split_network_key('bundle_emb.word.string_to_param.clip_l'), ('bundle_emb', 'word.string_to_param.clip_l'))
        self.assertEqual(lora_keys.split_network_key('lora_layer.custom.data'), ('lora_layer', 'custom.data'))

    def test_invalid_key_and_shape_rejected(self):
        for key in (None, '', 'no_suffix', '.alpha'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                lora_keys.split_network_key(key)
        for shapes in (((5, 2), (4, 3), (5, 3)), ((5, 2), (2, 7), (5, 3)), ((3, 2), (2, 3), (3, 3)), ((5, 2, 1), (2, 3), (5, 3))):
            with self.subTest(shapes=shapes), self.assertRaises(ValueError):
                lora_keys.embedding_layout(*shapes)
        self.assertTrue(lora_keys.embedding_layout((3, 2), (2, 3), (3, 3), peft=True))

    def test_text_encoder_and_oft_candidates(self):
        self.assertIn('0_transformer_text_model_embeddings_position_embedding', lora_keys.layer_candidates('lora_te1.text_model.embeddings.position_embedding', lambda x: x))
        self.assertIn('transformer_text_model_embeddings_position_embedding', lora_keys.layer_candidates('lora_te.text_model.embeddings.position_embedding', lambda x: x))
        self.assertIn('diffusion_model_input_blocks_0_0', lora_keys.layer_candidates('oft_unet.input_blocks.0.0', lambda x: x))


if __name__ == '__main__':
    unittest.main()
