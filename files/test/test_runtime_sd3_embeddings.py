"""Regression contracts for actual SD3 embedding consumers.

Real Torch and production classes selected by AST; no WebUI/model downloads.
Run with the existing D: Torch Python. Does not modify project or model files.
"""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


def selected(path, names, **env):
    tree = ast.parse(path.read_bytes(), str(path))
    body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert {node.name for node in body} == set(names)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), env)
    return NS(**env)


class EmbeddingContracts(unittest.TestCase):
    def setUp(self):
        self.ti = selected(ROOT / 'modules/textual_inversion/textual_inversion.py',
                           {'Embedding', 'create_embedding_from_data'}, torch=torch,
                           devices=NS(device=torch.device('cpu')))
        self.hijack = NS(fixes=None)
        self.layers = selected(ROOT / 'modules/sd_hijack.py',
                               {'EmbeddingsWithFixes', 'TextualInversionEmbeddings'},
                               torch=torch, model_hijack=self.hijack,
                               devices=NS(cond_cast_unet=lambda value: value))
        self.l = self.layers.TextualInversionEmbeddings(4, 768, textual_inversion_key='clip_l')
        self.g = self.layers.TextualInversionEmbeddings(4, 1280, textual_inversion_key='clip_g')
        self.model = NS(cond_stage_model=torch.nn.ModuleList([self.l, self.g]))
        self.data = {'clip_l': torch.ones(1, 768), 'clip_g': torch.ones(1, 1280)}

    def test_real_sd3_consumers_accept_dual_vectors(self):
        embedding = self.ti.create_embedding_from_data(self.data, 'valid', expected_shapes=None)
        for layer in (self.l, self.g):
            self.hijack.fixes = [[(0, embedding)]]
            output = layer(torch.tensor([[0, 1, 2]]))
            torch.testing.assert_close(output[:, 1], self.data[layer.textual_inversion_key])

    def test_sd3_width_discovery(self):
        self.assertEqual(self.ti.Embedding.encoder_shapes(self.model), {'clip_l': 768, 'clip_g': 1280})

    def test_valid_sd3_import_with_actual_consumers(self):
        embedding = self.ti.create_embedding_from_data(
            self.data, 'valid', expected_shapes=self.ti.Embedding.encoder_shapes(self.model))
        self.assertEqual(embedding.shape, 2048)

    def test_sd3_rejects_undivided_vector_before_registration(self):
        with self.assertRaises(ValueError):
            self.ti.create_embedding_from_data(
                {'string_to_param': {'*': torch.ones(1, 2048)}}, 'invalid',
                expected_shapes=self.ti.Embedding.encoder_shapes(self.model))



if __name__ == '__main__':
    unittest.main(verbosity=2)

