"""Track unresolved v14 export routes, not a claim of adapter compatibility.

Based on the supplied independent audit's miniature encoder trees. Expectations
for unsupported routes must change with a separately tested resolver, not merely
with a prefix substitution. No real checkpoint or user LoRA is loaded here.
"""
import re
from types import SimpleNamespace as NS
import unittest

import test_runtime_stability as h

torch = h.torch


class KnownRouteTests(unittest.TestCase):
    def test_supported_control_and_four_unimplemented_export_routes(self):
        runtime = h.selected('extensions-builtin/Lora/networks.py',
            {'convert_diffusers_name_to_compvis', 'assign_network_names_to_compvis_modules'},
            re=re, re_digits=re.compile(r'\d+'), re_compiled={},
            suffix_conversion={'attentions': {}, 'resnets': {}}, torch=torch,
            networks_in_memory={}, loaded_networks=[])
        clip = torch.nn.Module()
        clip.model = torch.nn.Module()
        clip.model.transformer = torch.nn.Module()
        clip.model.transformer.resblocks = torch.nn.ModuleList([torch.nn.Module()])
        clip.model.transformer.resblocks[0].mlp = torch.nn.Module()
        clip.model.transformer.resblocks[0].mlp.c_fc = torch.nn.Linear(2, 2)
        clip.model.text_projection = torch.nn.Parameter(torch.ones(2, 2))
        xl = NS(is_sdxl=True, conditioner=NS(embedders=[NS(wrapped=torch.nn.Module()), NS(wrapped=clip)]), model=torch.nn.Module())
        runtime.assign_network_names_to_compvis_modules(xl)
        cond = torch.nn.Module()
        cond.clip_g = torch.nn.Module()
        cond.clip_g.transformer = torch.nn.Module()
        text = cond.clip_g.transformer.text_model = torch.nn.Module()
        text.encoder = torch.nn.Module()
        text.encoder.layers = torch.nn.ModuleList([torch.nn.Module()])
        text.encoder.layers[0].mlp = torch.nn.Module()
        text.encoder.layers[0].mlp.fc1 = torch.nn.Linear(2, 2)
        text.embeddings = torch.nn.Module()
        text.embeddings.token_embedding = torch.nn.Embedding(10, 2)
        sd3 = NS(is_sdxl=False, cond_stage_model=cond, model=torch.nn.Module())
        runtime.assign_network_names_to_compvis_modules(sd3)
        for key, model, supported in (
            ('lora_te2_text_model_encoder_layers_0_mlp_fc1', xl, True),
            ('lora_te2_text_projection', xl, False),
            ('text_encoders.clip_g.transformer.text_model.embeddings.token_embedding', sd3, False),
            ('text_encoders.clip_g.transformer.text_model.encoder.layers.0.mlp.fc1', sd3, False),
            ('text_encoders.clip_g.transformer.text_model.encoder.layers.0.mlp.fc1', xl, False),
        ):
            with self.subTest(key=key, sdxl=model.is_sdxl):
                candidates = h.lora_keys.layer_candidates(key, lambda flat: runtime.convert_diffusers_name_to_compvis(flat, False))
                self.assertEqual(any(candidate in model.network_layer_mapping for candidate in candidates), supported)
        self.assertFalse(any('text_projection' in key for key in xl.network_layer_mapping))


if __name__ == '__main__':
    unittest.main()
