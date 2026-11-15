"""Regression contracts from the v8 follow-up audit; real CPU Torch."""
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import test_runtime_stability as harness
import test_runtime_compatibility as compatibility

torch = harness.torch


class BiasEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.h = harness.RuntimeTests()
        self.h.setUp()
        self.full = harness.selected('extensions-builtin/Lora/network_full.py', {'NetworkModuleFull'}, network=self.h.base)

    def adapter(self, layer, projected=False, partial=False):
        activation = self.h.activation(layer)
        activation.modules.clear()
        suffixes = ('q', 'k', 'v', 'out') if isinstance(layer, torch.nn.MultiheadAttention) else ('q', 'k', 'v') if projected else ('',)
        for suffix in suffixes:
            key = layer.network_layer_name + ('_' + suffix + '_proj' if suffix else '')
            shape = (4, 4) if projected else tuple(layer.weight.shape)
            data = {'diff': torch.zeros(shape)}
            if not partial or suffix == 'q':
                data['diff_b'] = torch.ones(shape[0])
            weights = self.h.base.NetworkWeights(key, key, data, layer)
            activation.modules[key] = self.full.NetworkModuleFull(activation.network, weights)
        return activation

    def test_direct_bias_repeat_weight_change_stack_disable(self):
        for kind in ('linear', 'conv'):
            for original_bias in (False, True):
                with self.subTest(kind=kind, original_bias=original_bias):
                    layer = torch.nn.Linear(3, 2, bias=original_bias) if kind == 'linear' else torch.nn.Conv2d(3, 2, 1, bias=original_bias)
                    old = layer.bias.detach().clone() if original_bias else None
                    weight = layer.weight.detach().clone()
                    first, second = self.adapter(layer), self.adapter(layer)
                    first.te_multiplier = 0.25
                    second.te_multiplier = 0.5
                    self.h.runtime.loaded_networks[:] = [first, second]
                    for _ in range(2):
                        self.h.runtime.network_apply_weights(layer)
                        torch.testing.assert_close(layer.bias, (old if old is not None else 0) + torch.full((2,), 0.75))
                    first.te_multiplier = 1
                    self.h.runtime.network_apply_weights(layer)
                    torch.testing.assert_close(layer.bias, (old if old is not None else 0) + torch.full((2,), 1.5))
                    self.h.runtime.loaded_networks.clear()
                    self.h.runtime.network_apply_weights(layer)
                    torch.testing.assert_close(layer.weight, weight)
                    if original_bias:
                        torch.testing.assert_close(layer.bias, old)
                    else:
                        self.assertIsNone(layer.bias)

    def test_projection_bias_all_partial_dtype_restore(self):
        for kind in ('mha', 'packed'):
            for original_bias in (False, True):
                for partial in (False, True):
                    with self.subTest(kind=kind, original_bias=original_bias, partial=partial):
                        layer = (torch.nn.MultiheadAttention(4, 1, bias=original_bias) if kind == 'mha' else
                                 self.h.modules.models.sd3.mmdit.QkvLinear(4, 12, bias=original_bias)).to(torch.float16)
                        old_bias = layer.in_proj_bias if kind == 'mha' else layer.bias
                        baseline = old_bias.detach().clone() if old_bias is not None else torch.zeros(12, dtype=torch.float16)
                        out_bias = layer.out_proj.bias.detach().clone() if kind == 'mha' and original_bias else torch.zeros(4, dtype=torch.float16)
                        activation = self.adapter(layer, projected=True, partial=partial)
                        self.h.runtime.loaded_networks[:] = [activation]
                        for strength in (1, 1, 0.5):
                            activation.te_multiplier = strength
                            self.h.runtime.network_apply_weights(layer)
                            current = layer.in_proj_bias if kind == 'mha' else layer.bias
                            expected = baseline.clone()
                            expected[:4 if partial else 12] += strength
                            torch.testing.assert_close(current, expected)
                            self.assertEqual(current.dtype, torch.float16)
                            if kind == 'mha' and not partial:
                                torch.testing.assert_close(layer.out_proj.bias, out_bias + strength)
                        self.h.runtime.loaded_networks.clear()
                        self.h.runtime.network_apply_weights(layer)
                        current = layer.in_proj_bias if kind == 'mha' else layer.bias
                        if original_bias:
                            torch.testing.assert_close(current, baseline)
                        else:
                            self.assertIsNone(current)
                            if kind == 'mha':
                                self.assertIsNone(layer.out_proj.bias)

    def test_projection_failure_restores_bias_and_allows_retry(self):
        for kind in ('mha', 'packed'):
            layer = torch.nn.MultiheadAttention(4, 1, bias=False) if kind == 'mha' else self.h.modules.models.sd3.mmdit.QkvLinear(4, 12, bias=False)
            good, bad = self.adapter(layer, projected=True), self.adapter(layer, projected=True)
            self.h.runtime.loaded_networks[:] = [good, bad]
            target = bad.modules[layer.network_layer_name + '_v_proj']
            with patch.object(target, 'calc_updown', side_effect=RuntimeError('failure after first adapter')):
                with self.assertRaisesRegex(RuntimeError, 'failure after first adapter'):
                    self.h.runtime.network_apply_weights(layer)
            self.assertIsNone(layer.in_proj_bias if kind == 'mha' else layer.bias)
            self.assertEqual(layer.network_current_names, ())
            self.h.runtime.loaded_networks[:] = [good]
            self.h.runtime.network_apply_weights(layer)
            torch.testing.assert_close(layer.in_proj_bias if kind == 'mha' else layer.bias, torch.ones(12))

    def test_invalid_projection_bias_fails_before_publishing(self):
        layer = self.h.modules.models.sd3.mmdit.QkvLinear(4, 12, bias=False)
        activation = self.adapter(layer, projected=True)
        activation.modules[layer.network_layer_name + '_k_proj'].ex_bias = torch.ones(1)
        self.h.runtime.loaded_networks[:] = [activation]
        with self.assertRaisesRegex(ValueError, 'Invalid Q/K/V'):
            self.h.runtime.network_apply_weights(layer)
        self.assertIsNone(layer.bias)
        self.assertEqual(layer.network_current_names, ())

    def test_float8_bias_accumulates_then_casts_once(self):
        for projected in (False, True):
            layer = self.h.modules.models.sd3.mmdit.QkvLinear(4, 12, bias=True) if projected else torch.nn.Linear(4, 4)
            layer.to(torch.float8_e4m3fn)
            original = layer.bias.float().clone()
            first, second = self.adapter(layer, projected), self.adapter(layer, projected)
            first.te_multiplier = 0.125
            second.te_multiplier = 0.25
            self.h.runtime.loaded_networks[:] = [first, second]
            for _ in range(2):
                self.h.runtime.network_apply_weights(layer)
                torch.testing.assert_close(layer.bias.float(), (original + 0.375).to(torch.float8_e4m3fn).float())
            self.h.runtime.loaded_networks.clear()
            self.h.runtime.network_apply_weights(layer)
            torch.testing.assert_close(layer.bias.float(), original)

    def model(self, widths):
        conditioner = torch.nn.ModuleList()
        for key, width in widths.items():
            wrapper = torch.nn.Module()
            wrapper.textual_inversion_key = key
            wrapper.wrapped = torch.nn.Embedding(5, width)
            conditioner.append(wrapper)
        return NS(cond_stage_model=conditioner)

    def test_encoder_dimensions_reject_wrong_split_before_device_transfer(self):
        ti = compatibility.CompatibilityTests().ti()
        shapes = ti.Embedding.encoder_shapes(self.model({'clip_l': 768, 'clip_g': 1280}))
        ti.devices.device = 'meta'  # invalid input must fail before this transfer
        with self.assertRaisesRegex(ValueError, 'clip_l: expected width 768, got 1024'):
            ti.create_embedding_from_data({'clip_l': torch.zeros(1, 1024), 'clip_g': torch.zeros(1, 1024)}, 'bad', expected_shapes=shapes)
        ti.devices.device = 'cpu'
        good = ti.create_embedding_from_data({'clip_l': torch.zeros(2, 768), 'clip_g': torch.zeros(2, 1280)}, 'good', expected_shapes=shapes)
        self.assertEqual((good.vectors, good.shape), (2, 2048))
        with self.assertRaisesRegex(ValueError, 'separate encoder'):
            ti.create_embedding_from_data({'string_to_param': {'*': torch.zeros(1, 2048)}}, 'bad-single', expected_shapes=shapes)

    def test_database_does_not_register_wrong_split(self):
        ti = compatibility.CompatibilityTests().ti()
        model = self.model({'clip_l': 768, 'clip_g': 1280})
        data = {'clip_l': torch.zeros(1, 1024), 'clip_g': torch.zeros(1, 1024)}
        dbmod = harness.selected('modules/textual_inversion/textual_inversion.py', {'EmbeddingDatabase'},
            os=os, torch=NS(load=lambda *a, **k: data), Embedding=ti.Embedding,
            create_embedding_from_data=ti.create_embedding_from_data, shared=NS(sd_model=model), cache=NS(cache=lambda n: {}))
        db = dbmod.EmbeddingDatabase()
        db.expected_shape = 2048
        with patch.object(db, 'register_embedding') as register:
            with self.assertRaisesRegex(ValueError, 'expected width 768'):
                db.load_from_file('unused.pt', 'bad.pt')
            register.assert_not_called()
        self.assertEqual(db.ids_lookup, {})
        self.assertEqual(db.word_embeddings, {})

    def test_formats_reject_nonfinite_empty_integer_and_overflow(self):
        ti = compatibility.CompatibilityTests().ti()
        invalid = [torch.full((1, 4), v) for v in (float('nan'), float('inf'), -float('inf'))]
        invalid += [torch.empty(0, 4), torch.empty(1, 0), torch.ones(1, 4, dtype=torch.int64), torch.full((1, 4), 1e100, dtype=torch.float64)]
        for value in invalid:
            for data in ({'string_to_param': {'*': value}}, {'concept': value}, {'clip_l': value, 'clip_g': torch.ones(1, 6)}):
                with self.subTest(shape=tuple(value.shape), format=tuple(data)):
                    with self.assertRaises(ValueError):
                        ti.create_embedding_from_data(data, 'invalid')
        for data in ({}, {'clip_l': torch.ones(1, 4)}, {'clip_g': torch.ones(1, 6)}):
            with self.assertRaises(ValueError):
                ti.create_embedding_from_data(data, 'invalid')

    def test_classic_and_dynamic_encoder_widths(self):
        ti = compatibility.CompatibilityTests().ti()
        for widths in ({'clip_l': 4}, {'clip_l': 7, 'clip_g': 9}):
            shapes = ti.Embedding.encoder_shapes(self.model(widths))
            data = {'concept': torch.ones(4)} if len(widths) == 1 else {key: torch.ones(2, width) for key, width in widths.items()}
            embedding = ti.create_embedding_from_data(data, 'valid', expected_shapes=shapes)
            self.assertEqual(embedding.shape, sum(widths.values()))
        with self.assertRaisesRegex(ValueError, 'expected width 5'):
            ti.create_embedding_from_data({'concept': torch.ones(4)}, 'changed-model', expected_shapes={'clip_l': 5})


if __name__ == '__main__':
    unittest.main()
