"""Actual Torch merge/restore and secondary low-VRAM failure regressions."""
import logging
import re
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import test_runtime_stability as h
import test_runtime_connections as connections

torch = h.torch


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.h = h.RuntimeTests()
        self.h.setUp()

    def test_wrapped_embedding_apply_repeat_disable_and_ti(self):
        fixture = self.h
        lora = h.selected('extensions-builtin/Lora/network_lora.py',
                          {'ModuleTypeLora', 'NetworkModuleLoraEmbedding', 'NetworkModuleLora'},
                          torch=torch, network=fixture.base, modules=fixture.modules, devices=fixture.devices, lora_keys=h.lora_keys)
        fixes = NS(fixes=None)
        wrapped_cls = h.selected('modules/sd_hijack.py', {'EmbeddingsWithFixes'}, torch=torch,
                                devices=NS(cond_cast_unet=lambda value: value)).EmbeddingsWithFixes
        child = torch.nn.Embedding(8, 4)
        wrapped = wrapped_cls(child, fixes)
        text = torch.nn.Module()
        text.token_embedding = wrapped
        model = NS(is_sdxl=True, conditioner=NS(embedders=[NS(wrapped=text)]), model=torch.nn.Module())
        fixture.runtime.assign_network_names_to_compvis_modules(model)
        key = '0_token_embedding'
        self.assertIs(model.network_layer_mapping[key], child)
        self.assertEqual(child.network_layer_name, key)
        self.assertIs(model.network_layer_mapping[key + '_wrapped'], child)
        base = child.weight.detach().clone()
        net = fixture.base.Network('test', NS(filename='test.safetensors'))
        factors = {'lora_up.weight': torch.ones(8, 2), 'lora_down.weight': torch.ones(2, 4)}
        net.modules[key] = lora.ModuleTypeLora().create_module(net, fixture.base.NetworkWeights(key, key, factors, wrapped))
        activation = fixture.base.NetworkActivation(net, 'test')
        ids = torch.tensor([[0, 1, 2]])
        with patch.object(torch.nn.Embedding, 'forward', fixture.runtime.network_Embedding_forward):
            fixture.runtime.loaded_networks[:] = [activation]
            first = wrapped(ids)
            torch.testing.assert_close(first, base[ids] + 2)
            torch.testing.assert_close(wrapped(ids), first)
            fixes.fixes = [[(0, NS(vec=torch.full((1, 4), 7.)))]]
            torch.testing.assert_close(wrapped(ids)[:, 1], torch.full((1, 4), 7.))
            activation.te_multiplier = activation.unet_multiplier = .5
            torch.testing.assert_close(wrapped(ids), base[ids] + 1)
            fixture.runtime.loaded_networks.clear()
            torch.testing.assert_close(wrapped(ids), base[ids])
        torch.testing.assert_close(child.weight, base)

    def test_actual_xl_upsampler_loader_merge_and_restore(self):
        fixture = self.h
        key = 'diffusion_model_output_blocks_2_2_conv'
        layer = torch.nn.Conv2d(4, 4, 3, padding=1, bias=False)
        layer.network_layer_name = key
        fixture.shared.sd_model = NS(network_layer_mapping={key: layer}, lora_model_token=object())
        lora = h.selected('extensions-builtin/Lora/network_lora.py', {'ModuleTypeLora', 'NetworkModuleLora', 'NetworkModuleLoraEmbedding'},
                          torch=torch, network=fixture.base, modules=fixture.modules, devices=fixture.devices,
                          lora_keys=h.lora_keys)
        prefix = 'lora_unet_up_blocks_0_upsamplers_0_conv.'
        state = {prefix + 'lora_up.weight': torch.ones(4, 2, 1, 1),
                 prefix + 'lora_down.weight': torch.ones(2, 4, 3, 3), prefix + 'alpha': torch.tensor(2.)}
        loader = h.selected('extensions-builtin/Lora/networks.py', {'load_network', 'convert_diffusers_name_to_compvis'},
                            torch=torch, network=fixture.base, shared=fixture.shared, devices=fixture.devices,
                            sd_models=NS(read_state_dict=lambda filename: state), network_file_signature=lambda filename: ('test', 1, 1, 1, 1, 1),
                            lora_keys=h.lora_keys, module_types=[lora.ModuleTypeLora()], re=re,
                            re_compiled={}, re_digits=re.compile(r'\d+'), re_x_proj=re.compile(r'(.*)_([qkv]_proj)$'),
                            suffix_conversion={}, logger=logging.getLogger('test'))
        net = loader.load_network(NS(name='test', filename='test.safetensors'))
        self.assertEqual(list(net.modules), [key])
        original = layer.weight.detach().clone()
        # Conventional full-rank reconstruction requires the unmodified helper.
        import lyco_helpers
        lora.lyco_helpers = lyco_helpers
        fixture.runtime.loaded_networks[:] = [fixture.base.NetworkActivation(net, 'test')]
        fixture.runtime.network_apply_weights(layer)
        torch.testing.assert_close(layer.weight, original + 2)
        fixture.runtime.loaded_networks.clear()
        fixture.runtime.network_apply_weights(layer)
        torch.testing.assert_close(layer.weight, original)
        fixture.shared.sd_model.network_layer_mapping['diffusion_model_output_blocks_2_1_conv'] = torch.nn.Conv2d(4, 4, 3)
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            loader.load_network(NS(name='test', filename='test.safetensors'))

    def test_second_lowvram_transfer_rolls_back_hooks_methods_and_fields(self):
        low, model, _ = connections.ConnectionTests().lowvram()
        diff = torch.nn.Module()
        diff.input_blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        diff.output_blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        diff.middle_block = torch.nn.Linear(2, 2)
        diff.time_embed = torch.nn.Linear(2, 2)
        model.model.diffusion_model = diff
        before = tuple(diff.children())
        encode = model.first_stage_model.encode
        with patch.object(model.model, 'to', side_effect=RuntimeError('second transfer')):
            with self.assertRaisesRegex(RuntimeError, 'second transfer'):
                low.setup_for_low_vram(model, use_medvram=False)
        self.assertEqual(tuple(diff.children()), before)
        self.assertEqual(model.first_stage_model.encode, encode)
        self.assertNotIn('encode', vars(model.first_stage_model))
        self.assertTrue(all(not module._forward_pre_hooks for module in model.modules()))
        self.assertFalse(getattr(model, 'lowvram', False))
        low.setup_for_low_vram(model, use_medvram=False)
        self.assertTrue(model.lowvram)

    def test_emphasis_normal_values_dtype_and_unit_weights(self):
        cls = h.selected('modules/sd_emphasis.py', {'Emphasis', 'EmphasisOriginal'}, torch=torch).EmphasisOriginal
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for weights in (torch.ones(1, 2), torch.tensor([[.5, 1.5]])):
                value = cls()
                original = torch.tensor([[[1., 2.], [3., 4.]]], dtype=dtype)
                value.z, value.multipliers = original, weights
                expected = original * weights[..., None]
                expected *= original.float().mean() / expected.float().mean()
                value.after_transformers()
                torch.testing.assert_close(value.z, expected)
                self.assertEqual(value.z.dtype, expected.dtype)


if __name__ == '__main__':
    unittest.main()
