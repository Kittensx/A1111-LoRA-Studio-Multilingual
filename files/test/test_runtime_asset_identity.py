"""Cross-cache regressions with valid tiny safetensors, no safetensors dependency.

Production catalogue/load/cache functions and real Torch factors are exercised.
Only tensor file decoding is substituted with a small stdlib fixture decoder;
this is not a test of safetensors' mmap implementation or a full WebUI boot.
"""
from concurrent.futures import ThreadPoolExecutor
import enum
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import test_runtime_stability as h
from modules.asset_files import read_safetensors_header

torch = h.torch


def write_adapter(path, value, architecture='sd1 ', alias='old_alias', declared_hash=None):
    metadata = {'ss_base_model_version': architecture, 'ss_output_name': alias}
    if declared_hash:
        metadata['sshs_model_hash'] = declared_hash
    header = {'__metadata__': metadata,
              'transformer_test.lora_up.weight': {'dtype': 'F32', 'shape': [2, 1], 'data_offsets': [0, 8]},
              'transformer_test.lora_down.weight': {'dtype': 'F32', 'shape': [1, 2], 'data_offsets': [8, 16]}}
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' ' * ((-len(encoded)) % 8)
    path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + struct.pack('<4f', value, value, 1., 1.))


def read_adapter(filename):
    data = Path(filename).read_bytes()
    length = struct.unpack('<Q', data[:8])[0]
    header = json.loads(data[8:8 + length])
    result = {}
    for name, spec in header.items():
        if name == '__metadata__':
            continue
        start, end = spec['data_offsets']
        values = struct.unpack('<' + 'f' * ((end - start) // 4), data[8 + length + start:8 + length + end])
        result[name] = torch.tensor(values).reshape(spec['shape'])
    return result


class AssetIdentityTests(unittest.TestCase):
    def setUp(self):
        self.stores = {}
        self.shared = NS(cmd_opts=NS(no_hashing=False))
        self.cache = h.selected('modules/cache.py', {'cached_data_for_file'}, os=os,
            cache=lambda section: self.stores.setdefault(section, {}), dump_cache=lambda: None)
        self.hashes = h.selected('modules/hashes.py',
            {'sha256_from_cache', 'sha256', 'calculate_sha256', 'addnet_hash_safetensors', 'partial_hash_from_cache'},
            hashlib=hashlib, os=os, shared=self.shared,
            cache=lambda section: self.stores.setdefault(section, {}), dump_cache=lambda: None,
            errors=NS(report=lambda *a, **kw: None))

    def replace(self, path, value=2., architecture='sdxl', alias='new_alias'):
        stat = path.stat()
        other = path.with_name('replacement.tmp')
        write_adapter(other, value, architecture, alias)
        self.assertEqual(stat.st_size, other.stat().st_size)
        os.utime(other, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        os.replace(other, path)
        self.assertEqual(stat.st_mtime_ns, path.stat().st_mtime_ns)

    def test_legacy_cache_migrates_then_same_mtime_replacement_invalidates_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.safetensors'
            write_adapter(path, 1.)
            for section in ('hashes', 'hashes-addnet'):
                self.stores[section] = {'title': {'mtime': path.stat().st_mtime, 'sha256': 'legacy'}}
            self.stores['metadata'] = {'title': {'mtime': path.stat().st_mtime, 'value': {'wrong': True}}}
            def metadata():
                return read_safetensors_header(path)['__metadata__']
            first = self.cache.cached_data_for_file('metadata', 'title', path, metadata)
            self.assertEqual(first['ss_base_model_version'], 'sd1 ')
            old_hashes = [self.hashes.sha256(path, 'title', addnet) for addnet in (False, True)]
            self.assertNotIn('legacy', old_hashes)
            # Unchanged files should hit the migrated cache without reading.
            self.assertEqual(self.cache.cached_data_for_file('metadata', 'title', path,
                lambda: self.fail('unexpected read')), first)
            self.replace(path)
            second = self.cache.cached_data_for_file('metadata', 'title', path, metadata)
            self.assertEqual(second['ss_base_model_version'], 'sdxl')
            for index, addnet in enumerate((False, True)):
                self.assertIsNone(self.hashes.sha256_from_cache(path, 'title', addnet))
                self.assertNotEqual(self.hashes.sha256(path, 'title', addnet), old_hashes[index])

    def test_same_title_different_path_and_backward_mtime_are_cache_misses(self):
        with tempfile.TemporaryDirectory() as tmp:
            one, two = Path(tmp) / 'a', Path(tmp) / 'b'
            one.write_bytes(b'aaaa')
            two.write_bytes(b'bbbb')
            timestamp = one.stat().st_mtime_ns
            os.utime(two, ns=(timestamp, timestamp))
            for path, expected in ((one, b'aaaa'), (two, b'bbbb'), (one, b'aaaa')):
                self.assertEqual(self.cache.cached_data_for_file('meta', 'same', path, path.read_bytes), expected)
                self.assertEqual(self.hashes.sha256(path, 'same'), hashlib.sha256(expected).hexdigest())
            one.write_bytes(b'cccc')
            os.utime(one, ns=(timestamp, timestamp - 10_000_000_000))
            self.assertEqual(self.cache.cached_data_for_file('meta', 'same', one, one.read_bytes), b'cccc')

    def test_partial_hash_and_no_hashing_use_new_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model'
            path.write_bytes(bytes(0x100000) + b'aaaa')
            old = self.hashes.partial_hash_from_cache(path)
            self.hashes.sha256(path, 'model')
            stat = path.stat()
            other = Path(tmp) / 'new'
            other.write_bytes(bytes(0x100000) + b'bbbb')
            os.utime(other, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            os.replace(other, path)
            self.assertNotEqual(self.hashes.partial_hash_from_cache(path), old)
            self.shared.cmd_opts.no_hashing = True
            self.assertIsNone(self.hashes.sha256(path, 'model'))

    def test_changed_during_metadata_or_hash_read_never_publishes_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.safetensors'
            write_adapter(path, 1.)
            def metadata():
                value = read_safetensors_header(path)
                self.replace(path)
                return value
            with self.assertRaisesRegex(RuntimeError, 'changed while'):
                self.cache.cached_data_for_file('metadata', 'changing', path, metadata)
            self.assertNotIn('changing', self.stores['metadata'])
            def calculate(_):
                self.replace(path, value=3.)
                return 'must-not-be-cached'
            with patch.object(self.hashes, 'calculate_sha256', calculate), self.assertRaisesRegex(RuntimeError, 'changed while'):
                self.hashes.sha256(path, 'changing')
            self.assertNotIn('changing', self.stores['hashes'])

    def fixture(self, root):
        fixture = h.RuntimeTests()
        fixture.setUp()
        shared = fixture.shared
        shared.cmd_opts.lora_dir = str(root)
        shared.cmd_opts.lyco_dir_backcompat = str(root / 'absent')
        shared.opts.lora_not_found_warning_console = False
        shared.opts.lora_not_found_gradio_warning = False
        shared.walk_files = lambda directory, **kw: sorted(str(p) for p in Path(directory).rglob('*.safetensors'))
        layer = torch.nn.Linear(2, 2, bias=False)
        layer.network_layer_name = 'transformer_test'
        shared.sd_model = NS(network_layer_mapping={'transformer_test': layer}, lora_model_token=object())
        descriptor = h.selected('extensions-builtin/Lora/network.py', {'NetworkOnDisk', 'SdVersion'}, os=os, enum=enum,
            metadata_tags_order={}, cache=self.cache, hashes=self.hashes,
            sd_models=NS(read_metadata_from_safetensors=lambda path: read_safetensors_header(path).get('__metadata__', {})),
            errors=NS(display=lambda e, *a: self.fail(str(e))))
        fixture.base.NetworkOnDisk = descriptor.NetworkOnDisk
        lora = h.selected('extensions-builtin/Lora/network_lora.py',
            {'ModuleTypeLora', 'NetworkModuleLora', 'NetworkModuleLoraEmbedding'}, torch=torch,
            network=fixture.base, modules=fixture.modules, devices=fixture.devices,
            lora_keys=h.lora_keys, lyco_helpers=h.import_file('extensions-builtin/Lora/lyco_helpers.py'))
        errors = []
        runtime = h.selected('extensions-builtin/Lora/networks.py',
            {'network_file_signature', 'network_cache_key', 'load_network', 'load_networks',
             'network_relative_name', 'register_network_alias', 'register_network_canonical_name',
             'get_network_on_disk', 'process_network_files', 'list_available_networks', 'update_available_networks_by_names'},
            os=os, shared=shared, network=fixture.base, torch=torch, devices=fixture.devices,
            sd_models=NS(read_state_dict=read_adapter), module_types=[lora.ModuleTypeLora()],
            convert_diffusers_name_to_compvis=lambda value, *a: value,
            lora_keys=h.lora_keys, re_x_proj=re.compile(r'(.*)_([qkv]_proj)$'),
            available_networks={}, available_network_aliases={}, forbidden_network_aliases={},
            available_network_hash_lookup={}, loaded_networks=[], networks_in_memory={},
            errors=NS(display=lambda error, *a: errors.append(str(error)), report=lambda *a, **kw: errors.append(str(a))),
            logger=logging.getLogger('audit'), logging=logging,
            sd_hijack=NS(model_hijack=NS(embedding_db=NS(ensure_model_compatibility=lambda: None), comments=[])),
            purge_networks_from_memory=lambda: None, finalize_fp16_cache=lambda model: None)
        return runtime, layer, errors

    def test_catalogue_and_two_activations_refresh_metadata_hash_alias_and_weights(self):
        for explicit_refresh in (False, True):
            with self.subTest(explicit_refresh=explicit_refresh), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = root / 'adapter.safetensors'
                write_adapter(path, 1.)
                runtime, layer, errors = self.fixture(root)
                with patch.dict(sys.modules, {'networks': runtime}):
                    runtime.list_available_networks()
                    runtime.load_networks(['old_alias'])
                    first = runtime.loaded_networks[0]
                    old_hash = first.network_on_disk.shorthash
                    self.assertEqual(first.network_on_disk.sd_version.name, 'SD1')
                    self.replace(path)
                    if explicit_refresh:
                        runtime.list_available_networks()
                    runtime.load_networks(['adapter'])
                    second = runtime.loaded_networks[0]
                    self.assertIsNot(first.network.payload_identity, second.network.payload_identity)
                    self.assertEqual(second.network_on_disk.sd_version.name, 'SDXL')
                    self.assertEqual(second.network_on_disk.alias, 'new_alias')
                    self.assertIsNone(runtime.get_network_on_disk('old_alias'))
                    self.assertIs(runtime.get_network_on_disk('new_alias'), second.network_on_disk)
                    self.assertNotIn(old_hash, runtime.available_network_hash_lookup)
                    self.assertNotEqual(old_hash, second.network_on_disk.shorthash)
                    self.assertEqual(second.network.file_signature, second.network_on_disk.file_signature)
                    delta, _ = second.modules['transformer_test'].calc_updown(layer.weight, second)
                    torch.testing.assert_close(delta, torch.full_like(layer.weight, 2.))
                    runtime.load_networks(['new_alias'])
                    self.assertIs(runtime.loaded_networks[0].network, second.network)
                    self.assertFalse(errors, errors)

    def test_real_file_replaced_between_header_payload_and_activation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'adapter.safetensors'
            write_adapter(path, 1.)
            runtime, _, errors = self.fixture(root)
            entered, release = threading.Event(), threading.Event()
            def read_with_writer(filename):
                tensors = read_adapter(filename)
                entered.set()
                if not release.wait(5):
                    raise RuntimeError('writer timeout')
                return tensors
            with patch.dict(sys.modules, {'networks': runtime}), ThreadPoolExecutor(max_workers=1) as pool:
                runtime.list_available_networks()
                runtime.sd_models.read_state_dict = read_with_writer
                task = pool.submit(runtime.load_networks, ['adapter'])
                try:
                    self.assertTrue(entered.wait(5))
                    self.replace(path)
                finally:
                    release.set()
                task.result(timeout=5)
                self.assertEqual(runtime.loaded_networks, [])
                self.assertEqual(runtime.networks_in_memory, {})
                self.assertTrue(any('changed while reading' in error for error in errors))
                runtime.sd_models.read_state_dict = read_adapter
                runtime.load_networks(['adapter'])
                self.assertEqual(runtime.loaded_networks[0].network_on_disk.sd_version.name, 'SDXL')

    def test_declared_metadata_hash_is_not_used_as_verified_payload_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'adapter.safetensors'
            write_adapter(path, 1., declared_hash='f' * 64)
            runtime, _, errors = self.fixture(root)
            with patch.dict(sys.modules, {'networks': runtime}):
                runtime.list_available_networks()
                runtime.load_networks(['adapter'])
                descriptor = runtime.loaded_networks[0].network_on_disk
                self.assertNotEqual(descriptor.hash, 'f' * 64)
                self.assertEqual(descriptor.hash, hashlib.sha256(struct.pack('<4f', 1., 1., 1., 1.)).hexdigest())
                self.assertFalse(errors, errors)

    def test_file_disappearing_during_index_refresh_fails_without_none_dereference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'adapter.safetensors'
            write_adapter(path, 1.)
            runtime, _, errors = self.fixture(root)
            with patch.dict(sys.modules, {'networks': runtime}):
                runtime.list_available_networks()
                runtime.load_networks(['adapter'])
                self.replace(path)
                refresh = runtime.list_available_networks
                def remove_before_refresh():
                    path.unlink()  # Exact fixture file, not a user asset.
                    refresh()
                runtime.load_networks.__globals__['list_available_networks'] = remove_before_refresh
                runtime.load_networks(['adapter'])
                self.assertEqual(runtime.loaded_networks, [])
                self.assertEqual(runtime.networks_in_memory, {})
                self.assertTrue(any('index changed' in message for message in errors), errors)


if __name__ == '__main__':
    unittest.main()
