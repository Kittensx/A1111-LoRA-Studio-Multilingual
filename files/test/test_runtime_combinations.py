"""Prompt -> activation -> real Torch deltas; cross-feature regressions."""
from types import SimpleNamespace as NS
import unittest

import test_runtime_stability as h
import lora_schema
import lyco_helpers

torch = h.torch


class CombinationTests(unittest.TestCase):
    def setUp(self):
        self.h = h.RuntimeTests()
        self.h.setUp()
        self.lora = h.selected('extensions-builtin/Lora/network_lora.py',
            {'ModuleTypeLora', 'NetworkModuleLora', 'NetworkModuleLoraEmbedding'},
            torch=torch, network=self.h.base, modules=self.h.modules,
            devices=self.h.devices, lora_keys=h.lora_keys, lyco_helpers=lyco_helpers)

    def activation(self, tag, net=None, hires=False):
        spec = lora_schema.parse_lora_tag(tag)
        name, te, unet, dyn, settings = spec.to_runtime_values()
        net = net or self.h.base.Network(name, NS(filename=name + '.safetensors'))
        return self.h.base.NetworkActivation(net, name, te, unet, dyn, **settings, is_hires=hires)

    def test_hires_scalar_replaces_base_curve_not_only_its_fallback(self):
        a = self.activation('<lora:a:te=0.2@0,0.8@1:unet=0.5@5,1@9:hr=0.3>', hires=True)
        for step in (0, 5, 7, 9, 19):
            with self.subTest(step=step):
                self.h.shared.state.sampling_step = step
                self.assertAlmostEqual(a.effective_multiplier(False), .3)
                self.assertAlmostEqual(a.effective_multiplier(True), .3)

    def test_te_curve_is_initial_not_a_leftover_sampler_step(self):
        a = self.activation('<lora:a:te=0.2@0,0.8@1:unet=0.5@5,1@9:start=5:stop=9>')
        for step in (0, 5, 7, 9, 19):
            self.h.shared.state.sampling_step = step
            self.assertAlmostEqual(a.effective_multiplier(True), .2)

    def test_short_classic_masks_zero_omitted_known_blocks_like_old_lbw(self):
        for preset in ('NONE', 'MIDD', 'OUTS'):
            a = self.activation(f'<lora:a:1:lbw={preset}>')
            self.assertEqual(a.layer_multiplier('diffusion_model_input_blocks_0_0', False), 0)
            self.assertEqual(a.layer_multiplier('diffusion_model_output_blocks_0_0', False), 0)
        # MIDD is a multi-block legacy preset, not a synonym for M00 only.
        self.assertEqual(a.layer_multiplier('unmapped_layer', False), 1)
        midd = self.activation('<lora:a:1:lbw=MIDD>')
        self.assertEqual(midd.layer_multiplier('diffusion_model_middle_block_1_attn1_to_q', False), 1)
        self.assertEqual(midd.layer_multiplier('diffusion_model_input_blocks_5_1', False), 1)

    def test_mixed_units_range_compares_resolved_steps(self):
        a = self.activation('<lora:a:1:start=5:stop=0.8>')
        self.h.shared.state.sampling_steps = 21
        for step, expected in ((4, 0), (5, 1), (16, 1), (17, 0)):
            self.h.shared.state.sampling_step = step
            self.assertEqual(a.effective_multiplier(False), expected)

    def test_rank_truncates_cp_factors_and_keeps_alpha_scale(self):
        layer = torch.nn.Conv2d(2, 2, 3, padding=1, bias=False)
        key = 'diffusion_model_middle_block_0'
        a = self.activation('<lora:a:1:dyn=1>')
        weights = {'lora_up.weight': torch.ones(2, 2, 1, 1),
                   'lora_down.weight': torch.ones(2, 2, 1, 1),
                   'lora_mid.weight': torch.ones(2, 2, 3, 3), 'alpha': torch.tensor(2.)}
        module = self.lora.ModuleTypeLora().create_module(a.network, self.h.base.NetworkWeights(key, key, weights, layer))
        delta, _ = module.calc_updown(layer.weight, a)
        torch.testing.assert_close(delta, torch.ones_like(layer.weight))

    def test_sdxl_openclip_embedding_uses_te_and_not_unet_gate(self):
        a = self.activation('<lora:a:te=0.2:unet=0.9:start=5:stop=9>')
        for key in ('1_model_token_embedding', '0_transformer_text_model_embeddings_position_embedding', 'model_token_embedding'):
            self.assertTrue(a.is_text_encoder_key(key))
            self.assertEqual(a.cache_identity(key)[3], .2)
        self.assertFalse(a.is_text_encoder_key('diffusion_model_middle_block_1_transformer_blocks_0_attn1_to_q'))

    def test_invalid_mixed_window_is_reported_after_resolving_and_negative_rejected(self):
        a = self.activation('<lora:a:1:start=0.8:stop=5>')
        with self.assertRaisesRegex(ValueError, 'resolves to step'):
            a.effective_multiplier(False)
        with self.assertRaisesRegex(ValueError, 'non-negative'):
            self.activation('<lora:a:1:step=-1-5>')

    def test_1_2_4_occurrences_curves_masks_rank_hires_merge_and_functional(self):
        # Deliberately share one immutable payload between occurrences 1 and 3.
        tags = ['<lora:shared:te=0.2:unet=1:start=0.2:stop=0.8:lbw=MIDD:dyn=1>',
                '<lora:second:te=0.4:unet=0.5@5,1@9:lbw=ALL:hrunet=0.3>',
                '<lora:shared:te=0.7:unet=-0.25:lbw=MIDD:lbwe=ATTNDEEPOFF>',
                '<lora:fourth:0.8:lbw=NONE:lbwe=ATTNDEEPON:dyn=1>']
        keys = ['transformer_text_model_encoder_layers_0_self_attn_q_proj',
                '1_model_token_embedding',
                'diffusion_model_middle_block_1_transformer_blocks_0_attn1_to_q',
                'diffusion_model_middle_block_0_resnet',
                'diffusion_model_input_blocks_0_0']
        self.h.shared.state.sampling_steps = 11
        for count in (1, 2, 4):
            for hires in (False, True):
                for key in keys:
                    with self.subTest(count=count, hires=hires, key=key):
                        layer = torch.nn.Linear(2, 2, bias=False)
                        layer.network_layer_name = key
                        base = layer.weight.detach().clone()
                        payloads = {}
                        activations = []
                        for tag in tags[:count]:
                            a = self.activation(tag, hires=hires)
                            a.network = payloads.setdefault(a.name, a.network)
                            if key not in a.modules:
                                factors = {'lora_up.weight': torch.ones(2, 2), 'lora_down.weight': torch.ones(2, 2)}
                                a.modules[key] = self.lora.ModuleTypeLora().create_module(a.network, self.h.base.NetworkWeights(key, key, factors, layer))
                            activations.append(a)
                        self.h.runtime.loaded_networks[:] = activations
                        x = torch.tensor([[1., 2.]])
                        for step in range(11):
                            self.h.shared.state.sampling_step = step
                            is_te, is_attn = key.startswith(('transformer', '1_model')), 'attn1' in key
                            omitted = key.endswith('input_blocks_0_0')
                            curve = .5 if step <= 5 else 1 if step >= 9 else .5 + (step - 5) / 8
                            expected = [(.2 if is_te else float(2 <= step <= 8)) * (0 if omitted else 1),
                                        2 * (.4 if is_te else .3 if hires else curve) * (0 if omitted else 1),
                                        2 * (.7 if is_te else -.25) * (0 if omitted or is_attn else 1),
                                        .8 * (1 if is_attn else 0)]
                            delta = sum(expected[:count])
                            self.h.runtime.network_apply_weights(layer)
                            torch.testing.assert_close(layer.weight, base + delta)
                            self.h.runtime.network_apply_weights(layer)
                            torch.testing.assert_close(layer.weight, base + delta)
                            # Restore before exercising the actual factorized/fallback forwards.
                            self.h.runtime.network_restore_weights_from_backup(layer)
                            self.h.runtime.network_reset_cached_weight(layer)
                            y = torch.nn.functional.linear(x, base)
                            for a in activations:
                                y = a.modules[key].forward(x, y, a)
                            torch.testing.assert_close(y, torch.nn.functional.linear(x, base + delta))
                            self.h.runtime.loaded_networks.reverse()
                            self.h.runtime.network_apply_weights(layer)
                            torch.testing.assert_close(layer.weight, base + delta)
                            self.h.runtime.loaded_networks.reverse()
                        self.h.runtime.loaded_networks.clear()
                        self.h.runtime.network_apply_weights(layer)
                        torch.testing.assert_close(layer.weight, base)
                        for a in activations:
                            torch.testing.assert_close(a.modules[key].up_model.weight, torch.ones(2, 2))
                            torch.testing.assert_close(a.modules[key].down_model.weight, torch.ones(2, 2))


if __name__ == '__main__':
    unittest.main()
