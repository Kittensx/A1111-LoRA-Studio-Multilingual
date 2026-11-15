"""Reproduce reported loader routes without the user's safetensors/checkpoint.

Regression contracts for wrapped Embedding and XL upsampler mapping.
Synthetic payloads do not identify the unknown tensor in the remote log.
"""
from pathlib import Path
from collections import namedtuple
from dataclasses import dataclass
import re
import sys
from types import SimpleNamespace as NS
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'test'))
import test_runtime_stability as h  # noqa: E402
torch = h.torch


class LoadingContracts(unittest.TestCase):
    def setUp(self):
        class QkvLinear(torch.nn.Linear):
            pass
        self.harness = NS(modules=NS(models=NS(sd3=NS(mmdit=NS(QkvLinear=QkvLinear)))),
                          devices=NS(device=torch.device('cpu'), dtype=torch.float32))
        fixture = self.harness
        fixture.base = h.selected('extensions-builtin/Lora/network.py',
            {'Network', 'NetworkActivation', 'NetworkModule', 'ModuleType'},
            nn=torch.nn, F=torch.nn.functional, modules=fixture.modules,
            dataclass=dataclass, lora_runtime=h.lora_runtime)
        fixture.base.NetworkWeights = namedtuple('NetworkWeights', 'network_key sd_key w sd_module')
        fixture.runtime = h.selected('extensions-builtin/Lora/networks.py',
            {'assign_network_names_to_compvis_modules'}, networks_in_memory={}, loaded_networks=[], torch=torch)
        self.lora = h.selected('extensions-builtin/Lora/network_lora.py',
            {'ModuleTypeLora', 'NetworkModuleLora', 'NetworkModuleLoraEmbedding'},
            torch=torch, network=fixture.base, modules=fixture.modules,
            devices=fixture.devices, lora_keys=h.lora_keys)
        # sd_hijack is a host file, not replaced by either v8 or v9.
        self.wrapper = h.selected(str(PROJECT / 'modules/sd_hijack.py'), {'EmbeddingsWithFixes'}, torch=torch).EmbeddingsWithFixes

    def construct(self, layer):
        fixture = self.harness
        key = '0_transformer_text_model_embeddings_token_embedding'
        weights = fixture.base.NetworkWeights(key + '.lora_up.weight', key,
            {'lora_up.weight': torch.ones(4, 2), 'lora_down.weight': torch.ones(2, 8)}, layer)
        net = fixture.base.Network('test', NS(filename='synthetic.safetensors'))
        return self.lora.ModuleTypeLora().create_module(net, weights)

    def test_bare_embedding_adapter_loads(self):
        self.assertIsInstance(self.construct(torch.nn.Embedding(8, 4)), self.lora.NetworkModuleLoraEmbedding)

    def test_actual_hijacked_embedding_mapping_loads_adapter(self):
        text = torch.nn.Module()
        text.transformer = torch.nn.Module()
        text.transformer.text_model = torch.nn.Module()
        text.transformer.text_model.embeddings = torch.nn.Module()
        wrapped = self.wrapper(torch.nn.Embedding(8, 4), NS(fixes=None))
        text.transformer.text_model.embeddings.token_embedding = wrapped
        model = NS(is_sdxl=True, conditioner=NS(embedders=[NS(wrapped=text)]), model=torch.nn.Module())
        self.harness.runtime.assign_network_names_to_compvis_modules(model)
        target = model.network_layer_mapping['0_transformer_text_model_embeddings_token_embedding']
        adapter = self.construct(target)
        self.assertIsInstance(adapter, self.lora.NetworkModuleLoraEmbedding)

    def candidates(self):
        conversion = h.selected('extensions-builtin/Lora/networks.py', {'convert_diffusers_name_to_compvis'},
            re=re, re_compiled={}, re_digits=re.compile(r'\d+'), suffix_conversion={})
        return h.lora_keys.layer_candidates('lora_unet_up_blocks_0_upsamplers_0_conv',
            lambda key: conversion.convert_diffusers_name_to_compvis(key, False))

    def test_sd1_first_upsampler_candidate_preserved(self):
        self.assertIn('diffusion_model_output_blocks_2_1_conv', self.candidates())

    def test_xl_first_upsampler_after_attention_can_be_resolved(self):
        # XL output block contains resnet, attention, upsampler in this order.
        # Unlike SD1, first upsampling stage already has attention.
        self.assertIn('diffusion_model_output_blocks_2_2_conv', self.candidates())


if __name__ == '__main__':
    unittest.main(verbosity=2)

