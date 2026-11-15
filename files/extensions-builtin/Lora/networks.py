from __future__ import annotations
import gradio as gr
import logging
import os
import re

import lora_patches
import lora_keys
import network
import network_lora
import network_glora
import network_hada
import network_ia3
import network_lokr
import network_full
import network_norm
import network_oft

import torch
from typing import Union

from modules import shared, devices, sd_models, errors, scripts, sd_hijack
import modules.textual_inversion.textual_inversion as textual_inversion
import modules.models.sd3.mmdit

from lora_logger import logger

module_types = [
    network_lora.ModuleTypeLora(),
    network_hada.ModuleTypeHada(),
    network_ia3.ModuleTypeIa3(),
    network_lokr.ModuleTypeLokr(),
    network_full.ModuleTypeFull(),
    network_norm.ModuleTypeNorm(),
    network_glora.ModuleTypeGLora(),
    network_oft.ModuleTypeOFT(),
]


re_digits = re.compile(r"\d+")
re_x_proj = re.compile(r"(.*)_([qkv]_proj)$")
re_compiled = {}

suffix_conversion = {
    "attentions": {},
    "resnets": {
        "conv1": "in_layers_2",
        "conv2": "out_layers_3",
        "norm1": "in_layers_0",
        "norm2": "out_layers_0",
        "time_emb_proj": "emb_layers_1",
        "conv_shortcut": "skip_connection",
    }
}


def convert_diffusers_name_to_compvis(key, is_sd2):
    def match(match_list, regex_text):
        regex = re_compiled.get(regex_text)
        if regex is None:
            regex = re.compile(regex_text)
            re_compiled[regex_text] = regex

        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []

    if match(m, r"lora_unet_conv_in(.*)"):
        return f'diffusion_model_input_blocks_0_0{m[0]}'

    if match(m, r"lora_unet_conv_out(.*)"):
        return f'diffusion_model_out_2{m[0]}'

    if match(m, r"lora_unet_time_embedding_linear_(\d+)(.*)"):
        return f"diffusion_model_time_embed_{m[0] * 2 - 2}{m[1]}"

    if match(m, r"lora_unet_down_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_mid_block_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[0], {}).get(m[2], m[2])
        return f"diffusion_model_middle_block_{1 if m[0] == 'attentions' else m[1] * 2}_{suffix}"

    if match(m, r"lora_unet_up_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_output_blocks_{m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv"):
        return f"diffusion_model_input_blocks_{3 + m[0] * 3}_0_op"

    if match(m, r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv"):
        return f"diffusion_model_output_blocks_{2 + m[0] * 3}_{2 if m[0]>0 else 1}_conv"

    if match(m, r"lora_te_text_model_encoder_layers_(\d+)_(.+)"):
        if is_sd2:
            if 'mlp_fc1' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
            elif 'mlp_fc2' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
            else:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

        return f"transformer_text_model_encoder_layers_{m[0]}_{m[1]}"

    if match(m, r"lora_te2_text_model_encoder_layers_(\d+)_(.+)"):
        if 'mlp_fc1' in m[1]:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
        elif 'mlp_fc2' in m[1]:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
        else:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

    return key


def assign_network_names_to_compvis_modules(sd_model):
    network_layer_mapping = {}
    assigned = set()

    def add_layer(name, module):
        if type(module).__name__ == 'EmbeddingsWithFixes' and isinstance(getattr(module, 'wrapped', None), torch.nn.Embedding):
            module = module.wrapped
        network_layer_mapping[name] = module
        # The .wrapped child is an alias, not a different adapter target.
        if id(module) not in assigned:
            module.network_layer_name = name
            assigned.add(id(module))

    if sd_model.is_sdxl:
        for i, embedder in enumerate(sd_model.conditioner.embedders):
            if not hasattr(embedder, 'wrapped'):
                continue

            for name, module in embedder.wrapped.named_modules():
                network_name = f'{i}_{name.replace(".", "_")}'
                add_layer(network_name, module)
    else:
        cond_stage_model = getattr(sd_model.cond_stage_model, 'wrapped', sd_model.cond_stage_model)

        for name, module in cond_stage_model.named_modules():
            network_name = name.replace(".", "_")
            add_layer(network_name, module)

    for name, module in sd_model.model.named_modules():
        network_name = name.replace(".", "_")
        add_layer(network_name, module)

    sd_model.network_layer_mapping = network_layer_mapping
    sd_model.lora_model_token = object()
    # Adapters contain sd_module references and cannot outlive this binding.
    networks_in_memory.clear()
    loaded_networks.clear()


class BundledTIHash(str):
    def __init__(self, hash_str):
        self.hash = hash_str

    def __str__(self):
        return self.hash if shared.opts.lora_bundled_ti_to_infotext else ''


def network_file_signature(filename):
    """One generation identity shared by payload, metadata and hash caches."""
    from modules.asset_files import file_signature
    signature = file_signature(filename)
    if signature is None:
        raise FileNotFoundError(filename)
    return signature


def network_cache_key(network_on_disk):
    """Aliases must share one payload cache entry for the same file."""
    return os.path.normcase(os.path.abspath(network_on_disk.filename))


def load_network(network_on_disk):
    net = network.Network(network_on_disk.name, network_on_disk)
    net.file_signature = network_file_signature(network_on_disk.filename)
    net.mtime = net.file_signature[4] / 1_000_000_000
    if getattr(network_on_disk, 'file_signature', net.file_signature) != net.file_signature:
        raise OSError(f'LoRA descriptor is stale: {network_on_disk.filename}; refresh and retry')

    sd = sd_models.read_state_dict(network_on_disk.filename)
    if network_file_signature(network_on_disk.filename) != net.file_signature:
        raise RuntimeError(f'LoRA changed while reading: {network_on_disk.filename}; retry after the write completes')

    # this should not be needed but is here as an emergency fix for an unknown error people are experiencing in 1.2.0
    if not hasattr(shared.sd_model, 'network_layer_mapping'):
        assign_network_names_to_compvis_modules(shared.sd_model)
    if not hasattr(shared.sd_model, 'lora_model_token'):
        shared.sd_model.lora_model_token = object()
    net.model_token = shared.sd_model.lora_model_token

    keys_failed_to_match = {}
    is_sd2 = 'model_transformer_resblocks' in shared.sd_model.network_layer_mapping
    if hasattr(shared.sd_model, 'diffusers_weight_map'):
        diffusers_weight_map = shared.sd_model.diffusers_weight_map
    elif hasattr(shared.sd_model, 'diffusers_weight_mapping'):
        diffusers_weight_map = {}
        for k, v in shared.sd_model.diffusers_weight_mapping():
            diffusers_weight_map[k] = v
        shared.sd_model.diffusers_weight_map = diffusers_weight_map
    else:
        diffusers_weight_map = None

    matched_networks = {}
    bundle_embeddings = {}

    for key_network, weight in sd.items():

        key_network_without_network_parts, network_part = lora_keys.split_network_key(key_network)

        if key_network_without_network_parts == "bundle_emb":
            emb_name, vec_name = network_part.split(".", 1)
            emb_dict = bundle_embeddings.get(emb_name, {})
            if vec_name.split('.')[0] == 'string_to_param':
                _, k2 = vec_name.split('.', 1)
                emb_dict.setdefault('string_to_param', {})[k2] = weight
            else:
                emb_dict[vec_name] = weight
            bundle_embeddings[emb_name] = emb_dict
            continue

        candidates = lora_keys.layer_candidates(
            key_network_without_network_parts,
            lambda name: convert_diffusers_name_to_compvis(name, is_sd2),
            diffusers_weight_map,
        )
        if key_network_without_network_parts.replace('.', '_') == 'lora_unet_up_blocks_0_upsamplers_0_conv':
            targets = {id(shared.sd_model.network_layer_mapping[candidate]) for candidate in candidates
                       if candidate in shared.sd_model.network_layer_mapping}
            if len(targets) > 1:
                raise ValueError(f"Ambiguous LoRA upsampler mapping for {key_network!r}: {candidates}")
        sd_module = None
        key = candidates[0]
        for candidate in candidates:
            sd_module = shared.sd_model.network_layer_mapping.get(candidate)
            if sd_module is None:
                projection = re_x_proj.fullmatch(candidate)
                if projection:
                    sd_module = shared.sd_model.network_layer_mapping.get(projection.group(1))
            if sd_module is not None:
                key = candidate
                break

        if sd_module is None:
            keys_failed_to_match[key_network] = key
            continue

        # Keep explicit .wrapped aliases on the same binding as the canonical
        # embedding name. Projection keys deliberately remain distinct.
        if isinstance(sd_module, torch.nn.Embedding):
            key = getattr(sd_module, 'network_layer_name', key)

        if key not in matched_networks:
            matched_networks[key] = network.NetworkWeights(network_key=key_network, sd_key=key, w={}, sd_module=sd_module)

        if network_part in matched_networks[key].w:
            raise ValueError(f"Duplicate LoRA tensor after name normalization: {key_network!r} -> {key}.{network_part}")
        matched_networks[key].w[network_part] = weight

    for key, weights in matched_networks.items():
        net_module = None
        for nettype in module_types:
            net_module = nettype.create_module(net, weights)
            if net_module is not None:
                break

        if net_module is None:
            parts = list(weights.w)
            raise ValueError(f"Unsupported LoRA layer {key!r} ({type(weights.sd_module).__name__}); "
                             f"weight keys: {', '.join(parts[:12])}" + (f" ... ({len(parts)} total)" if len(parts) > 12 else ""))

        net.modules[key] = net_module
        net_module.network_to_device(devices.device)

    if not net.modules and not bundle_embeddings:
        raise ValueError(f"LoRA {network_on_disk.name!r} matched no layers of the active checkpoint; check its architecture and tensor names")

    embeddings = {}
    for emb_name, data in bundle_embeddings.items():
        embedding = textual_inversion.create_embedding_from_data(
            data, emb_name, filename=network_on_disk.filename + "/" + emb_name,
            expected_shapes=textual_inversion.Embedding.encoder_shapes(shared.sd_model))
        embedding.loaded = None
        embedding.shorthash = BundledTIHash(net.name)
        embeddings[emb_name] = embedding

    net.bundle_embeddings = embeddings

    if keys_failed_to_match:
        net.unmatched_keys = tuple(keys_failed_to_match)
        logger.warning("LoRA %s: %d unmatched tensor keys; %d matched modules. Examples: %s",
                       net.name, len(keys_failed_to_match), len(net.modules), ', '.join(list(keys_failed_to_match)[:5]))

    if network_file_signature(network_on_disk.filename) != net.file_signature:
        raise RuntimeError(f'LoRA changed while preparing tensors: {network_on_disk.filename}; retry after the write completes')
    return net


def purge_networks_from_memory():
    while len(networks_in_memory) > shared.opts.lora_in_memory_limit and len(networks_in_memory) > 0:
        name = next(iter(networks_in_memory))
        networks_in_memory.pop(name, None)

    devices.torch_gc()


def load_networks(names, te_multipliers=None, unet_multipliers=None, dyn_dims=None, activation_settings=None):
    model = shared.sd_model
    if not hasattr(model, 'network_layer_mapping'):
        assign_network_names_to_compvis_modules(model)
    if not hasattr(model, 'lora_model_token'):
        model.lora_model_token = object()
    emb_db = sd_hijack.model_hijack.embedding_db
    # Rebind the registry before unregistering/adding bundled TI for this model.
    emb_db.ensure_model_compatibility()
    already_loaded = {}
    processed_embeddings = set()

    for activation in loaded_networks:
        net = activation.network
        already_loaded[network_cache_key(net.network_on_disk)] = net
        for emb_name, embedding in net.bundle_embeddings.items():
            if embedding.loaded:
                emb_db.register_embedding_by_name(None, shared.sd_model, emb_name)

    loaded_networks.clear()

    unavailable_networks = [name for name in names if get_network_on_disk(name) is None]

    if unavailable_networks:
        update_available_networks_by_names(unavailable_networks)

    networks_on_disk = [get_network_on_disk(name) for name in names]
    if any(x is None for x in networks_on_disk):
        list_available_networks()

        networks_on_disk = [get_network_on_disk(name) for name in names]

    failed_to_load_networks = []

    for i, (network_on_disk, name) in enumerate(zip(networks_on_disk, names)):
        net = None

        if network_on_disk is not None:
            cache_key = network_cache_key(network_on_disk)
            filename = network_on_disk.filename
            try:
                file_signature = network_file_signature(network_on_disk.filename)
                if getattr(network_on_disk, 'file_signature', file_signature) != file_signature:
                    # Refresh the index as a whole so retired aliases and hashes
                    # cannot point at the replacement's old descriptor.
                    previous = network_on_disk
                    list_available_networks()
                    network_on_disk = get_network_on_disk(previous.name)
                    if network_on_disk is None or network_cache_key(network_on_disk) != cache_key:
                        raise OSError(f'LoRA index changed for {name}; select its current canonical name')
                    file_signature = network_file_signature(network_on_disk.filename)
                    if network_on_disk.file_signature != file_signature:
                        raise OSError(f'LoRA keeps changing: {network_on_disk.filename}; retry later')
            except OSError as e:
                networks_in_memory.pop(cache_key, None)
                errors.display(e, f"reading network {filename}")
                failed_to_load_networks.append(name)
                continue

            net = already_loaded.get(cache_key)
            if net is None:
                net = networks_in_memory.get(cache_key)

            if net is None or file_signature != net.file_signature or net.model_token is not model.lora_model_token:
                try:
                    net = load_network(network_on_disk)

                    networks_in_memory.pop(cache_key, None)
                    networks_in_memory[cache_key] = net
                except Exception as e:
                    errors.display(e, f"loading network {network_on_disk.filename}")
                    failed_to_load_networks.append(name)
                    continue

            # A refresh creates a fresh NetworkOnDisk entry. The payload may
            # still be safely reused by its absolute filename, but its hash
            # and canonical display name must reflect the current entry.
            net.name = network_on_disk.name
            net.network_on_disk = network_on_disk

            already_loaded[cache_key] = net

            try:
                network_on_disk.read_hash()
                if network_file_signature(network_on_disk.filename) != net.file_signature:
                    raise OSError(f'LoRA changed before activation: {network_on_disk.filename}; retry later')
            except (OSError, RuntimeError) as e:
                networks_in_memory.pop(cache_key, None)
                errors.display(e, f'verifying network {network_on_disk.filename}')
                failed_to_load_networks.append(name)
                continue

        if net is None:
            failed_to_load_networks.append(name)
            logging.info(f"Couldn't find network with name {name}")
            continue

        settings = activation_settings[i] if activation_settings and i < len(activation_settings) else {}
        activation = network.NetworkActivation(
            network=net,
            mentioned_name=name,
            te_multiplier=te_multipliers[i] if te_multipliers else 1.0,
            unet_multiplier=unet_multipliers[i] if unet_multipliers else 1.0,
            dyn_dim=dyn_dims[i] if dyn_dims else None,
            **settings,
        )
        loaded_networks.append(activation)
        if getattr(net, 'unmatched_keys', ()):
            message = f'LoRA {name}: {len(net.unmatched_keys)} tensor keys did not match this checkpoint; see console.'
            if message not in sd_hijack.model_hijack.comments:
                sd_hijack.model_hijack.comments.append(message)

        # Repeated use of the same payload must not register bundled textual
        # inversions a second time.
        if id(net) in processed_embeddings:
            continue
        processed_embeddings.add(id(net))
        for emb_name, embedding in net.bundle_embeddings.items():
            if embedding.loaded is None and emb_name in emb_db.word_embeddings:
                logger.warning(
                    f'Skip bundle embedding: "{emb_name}"'
                    ' as it was already loaded from embeddings folder'
                )
                continue

            embedding.loaded = False
            if emb_db.expected_shape == -1 or emb_db.expected_shape == embedding.shape:
                embedding.loaded = True
                emb_db.register_embedding(embedding, shared.sd_model)
            else:
                emb_db.skipped_embeddings[name] = embedding

    if failed_to_load_networks:
        lora_not_found_message = f'LoRA missing or failed to load: {", ".join(failed_to_load_networks)} (see console for details)'
        sd_hijack.model_hijack.comments.append(lora_not_found_message)
        if shared.opts.lora_not_found_warning_console:
            print(f'\n{lora_not_found_message}\n')
        if shared.opts.lora_not_found_gradio_warning:
            gr.Warning(lora_not_found_message)

    model.lora_conditioning_signature = tuple(
        (activation.network.payload_identity, activation.te_multiplier,
         activation.hr_te_multiplier, activation.te_schedule, activation.hr_te_schedule,
         activation.dyn_dim, repr(activation.block_weights), repr(activation.elemental_weights),
         activation.is_hires, activation.model_profile)
        for activation in loaded_networks
    )
    purge_networks_from_memory()

    finalize_fp16_cache(shared.sd_model)


def allowed_layer_without_weight(layer):
    if isinstance(layer, torch.nn.LayerNorm) and not layer.elementwise_affine:
        return True

    return False


def finalize_fp16_cache(sd_model):
    """Transfer cached fp16 weights from model._fp16_cache to active LoRA modules."""
    cache = getattr(sd_model, '_fp16_cache', None)
    if cache is None:
        return
    if not loaded_networks:
        return
    for net in loaded_networks:
        for _name, lora_mod in net.modules.items():
            sd_mod = getattr(lora_mod, 'sd_module', None)
            if sd_mod is None:
                continue
            item = cache.pop(id(sd_mod), None)
            if item is not None:
                weight, bias = item
                sd_mod.fp16_weight = weight
                if bias is not None:
                    sd_mod.fp16_bias = bias
    # Keep entries for layers no active LoRA has touched yet. Clearing the
    # entire cache here makes a later LoRA that targets different layers fall
    # back to FP8 weights. Claimed entries were already popped above.
    if not cache:
        del sd_model._fp16_cache


def store_weights_backup(weight):
    if weight is None:
        return None

    if getattr(shared.opts, 'lora_gpu_backup', False):
        return weight.detach().clone()

    return weight.to(devices.cpu, copy=True)


def restore_weights_backup(obj, field, weight):
    if weight is None:
        setattr(obj, field, None)
        return

    with torch.no_grad():
        getattr(obj, field).copy_(weight)


def network_restore_weights_from_backup(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.MultiheadAttention]):
    weights_backup = getattr(self, "network_weights_backup", None)
    bias_backup = getattr(self, "network_bias_backup", None)

    if weights_backup is None and bias_backup is None:
        return

    if weights_backup is not None:
        if isinstance(self, torch.nn.MultiheadAttention):
            restore_weights_backup(self, 'in_proj_weight', weights_backup[0])
            restore_weights_backup(self.out_proj, 'weight', weights_backup[1])
        else:
            restore_weights_backup(self, 'weight', weights_backup)

    if isinstance(self, torch.nn.MultiheadAttention):
        restore_weights_backup(self.out_proj, 'bias', bias_backup)
        if hasattr(self, 'network_in_proj_bias_backup'):
            restore_weights_backup(self, 'in_proj_bias', self.network_in_proj_bias_backup)
    elif hasattr(self, 'bias'):
        restore_weights_backup(self, 'bias', bias_backup)


def network_merge_projection_bias(base_bias, biases, weight):
    """Pack Q/K/V output deltas; a missing projection bias contributes zero."""
    present = [bias for bias in biases if bias is not None]
    if not present:
        return base_bias
    width = weight.shape[0] // 3
    if any(tuple(bias.shape) != (width,) for bias in present):
        raise ValueError(f'Invalid Q/K/V output bias: expected ({width},) for each projection')
    delta = torch.cat([bias.to(weight.device) if bias is not None else
                       torch.zeros(width, device=weight.device, dtype=present[0].dtype) for bias in biases])
    # Float8 storage cannot participate in ordinary mixed-dtype addition.
    if base_bias is not None and base_bias.element_size() == 1:
        base_bias = base_bias.float()
    return delta if base_bias is None else base_bias + delta


def network_set_bias(module, field, value, weight):
    if value is None:
        return
    current = getattr(module, field)
    if current is None:
        setattr(module, field, torch.nn.Parameter(value.to(device=weight.device, dtype=weight.dtype), requires_grad=False))
    else:
        current.copy_(value.to(device=current.device, dtype=current.dtype))


def network_apply_weights(self):
    """A failed layer merge must not leave a partial combination marked applied."""
    try:
        _network_apply_weights(self)
    except Exception:
        network_restore_weights_from_backup(self)
        self.network_current_names = ()
        raise


def _network_apply_weights(self):
    """
    Applies the currently selected set of networks to the weights of torch layer self.
    If weights already have this particular set of networks applied, does nothing.
    If not, restores original weights from backup and alters weights according to networks.
    """

    network_layer_name = getattr(self, 'network_layer_name', None)
    if network_layer_name is None:
        return

    current_names = getattr(self, "network_current_names", ())
    # Curves and block weights are evaluated for this layer only.  Including
    # their effective value here makes merge mode refresh exactly when that
    # layer's result changes, without mutating the cached LoRA payload.
    keys = (network_layer_name,)
    if isinstance(self, (torch.nn.MultiheadAttention, modules.models.sd3.mmdit.QkvLinear)):
        keys += tuple(network_layer_name + suffix for suffix in ('_q_proj', '_k_proj', '_v_proj', '_out_proj'))
    active = [net for net in loaded_networks if any(key in net.modules for key in keys)]
    if isinstance(self, modules.models.sd3.mmdit.QkvLinear):
        direct = any(network_layer_name in net.modules for net in active)
        projected = any(any(key in net.modules for key in keys[1:]) for net in active)
        if direct and projected:
            raise ValueError(f"Mixed full and projected QKV adapters are unsupported for {network_layer_name}")
    # Packed attention applies adapters by projection key, whose lbwe values
    # can change independently while the parent's multiplier stays unchanged.
    wanted_names = tuple(
        tuple((key, net.cache_identity(key)) for key in keys if key in net.modules)
        for net in active
    )
    if not active:
        if getattr(self, 'network_weights_backup', None) is not None:
            network_restore_weights_from_backup(self)
            network_reset_cached_weight(self)
        return

    weights_backup = getattr(self, "network_weights_backup", None)
    if weights_backup is None and wanted_names != ():
        if current_names != () and not allowed_layer_without_weight(self):
            raise RuntimeError(f"{network_layer_name} - no backup weights found and current weights are not unchanged")

        if isinstance(self, torch.nn.MultiheadAttention):
            weights_backup = (store_weights_backup(self.in_proj_weight), store_weights_backup(self.out_proj.weight))
            bias_backup = store_weights_backup(self.out_proj.bias)
            in_proj_bias_backup = store_weights_backup(self.in_proj_bias)
        else:
            weights_backup = store_weights_backup(self.weight)
            bias_backup = store_weights_backup(getattr(self, 'bias', None))

        # Weight backup is the initialization marker. A saved None bias is a
        # valid original state, not an invitation to back up a merged bias.
        self.network_bias_backup = bias_backup
        if isinstance(self, torch.nn.MultiheadAttention):
            self.network_in_proj_bias_backup = in_proj_bias_backup
        self.network_weights_backup = weights_backup

    if current_names != wanted_names:
        network_restore_weights_from_backup(self)
        merged_weight = None
        merged_bias = None

        for net in active:
            module = net.modules.get(network_layer_name, None)
            if module is not None and hasattr(self, 'weight'):
                try:
                    with torch.no_grad():
                        if merged_weight is None:
                            base = getattr(self, 'fp16_weight', None)
                            merged_weight = (base if base is not None else self.weight).clone().to(self.weight.device)
                            base_bias = getattr(self, 'fp16_bias', getattr(self, 'bias', None))
                            merged_bias = base_bias.clone().to(self.weight.device) if base_bias is not None else None
                        weight, bias = merged_weight, merged_bias
                        updown, ex_bias = module.calc_updown(weight, net)

                        if isinstance(self, torch.nn.Embedding) and (tuple(updown.shape) != tuple(weight.shape) or ex_bias is not None):
                            raise ValueError(f"Invalid Embedding LoRA delta for {network_layer_name}: expected {tuple(weight.shape)} without output bias")

                        if len(weight.shape) == 4 and weight.shape[1] == 9:
                            # inpainting model. zero pad updown to make channel[1]  4 to 9
                            updown = torch.nn.functional.pad(updown, (0, 0, 0, 0, 0, 5))

                        merged_weight = weight.to(dtype=updown.dtype) + updown
                        if ex_bias is not None and hasattr(self, 'bias'):
                            if bias is not None and bias.element_size() == 1:
                                bias = bias.float()
                            merged_bias = ex_bias if bias is None else bias + ex_bias
                except RuntimeError as e:
                    logging.debug(f"Network {net.name} layer {network_layer_name}: {e}")
                    extra_network_lora.errors[net.name] = extra_network_lora.errors.get(net.name, 0) + 1
                    raise

                continue

            module_q = net.modules.get(network_layer_name + "_q_proj", None)
            module_k = net.modules.get(network_layer_name + "_k_proj", None)
            module_v = net.modules.get(network_layer_name + "_v_proj", None)
            module_out = net.modules.get(network_layer_name + "_out_proj", None)

            if isinstance(self, torch.nn.MultiheadAttention) and module_q and module_k and module_v and module_out:
                try:
                    with torch.no_grad():
                        # Send "real" orig_weight into MHA's lora module
                        qw, kw, vw = self.in_proj_weight.chunk(3, 0)
                        updown_q, bias_q = module_q.calc_updown(qw, net)
                        updown_k, bias_k = module_k.calc_updown(kw, net)
                        updown_v, bias_v = module_v.calc_updown(vw, net)
                        del qw, kw, vw
                        updown_qkv = torch.vstack([updown_q, updown_k, updown_v])
                        updown_out, ex_bias = module_out.calc_updown(self.out_proj.weight, net)
                        in_bias = network_merge_projection_bias(self.in_proj_bias, (bias_q, bias_k, bias_v), self.in_proj_weight)
                        out_bias = self.out_proj.bias
                        if ex_bias is not None:
                            if tuple(ex_bias.shape) != (self.out_proj.weight.shape[0],):
                                raise ValueError('Invalid attention output bias shape')
                            out_bias = ex_bias if out_bias is None else out_bias + ex_bias

                        self.in_proj_weight += updown_qkv
                        self.out_proj.weight += updown_out
                        network_set_bias(self, 'in_proj_bias', in_bias, self.in_proj_weight)
                        network_set_bias(self.out_proj, 'bias', out_bias, self.out_proj.weight)

                except RuntimeError as e:
                    logging.debug(f"Network {net.name} layer {network_layer_name}: {e}")
                    extra_network_lora.errors[net.name] = extra_network_lora.errors.get(net.name, 0) + 1
                    raise

                continue

            if isinstance(self, modules.models.sd3.mmdit.QkvLinear) and module_q and module_k and module_v:
                try:
                    with torch.no_grad():
                        # Accumulate all adapters before the final storage cast,
                        # just like the direct Linear path. In-place Float8 +
                        # float is unsupported and ignores the precise cache.
                        if merged_weight is None:
                            base = getattr(self, 'fp16_weight', None)
                            merged_weight = (base if base is not None else self.weight).clone().to(self.weight.device)
                            base_bias = getattr(self, 'fp16_bias', self.bias)
                            merged_bias = base_bias.clone().to(self.weight.device) if base_bias is not None else None
                        qw, kw, vw = merged_weight.chunk(3, 0)
                        updown_q, bias_q = module_q.calc_updown(qw, net)
                        updown_k, bias_k = module_k.calc_updown(kw, net)
                        updown_v, bias_v = module_v.calc_updown(vw, net)
                        merged_bias = network_merge_projection_bias(merged_bias, (bias_q, bias_k, bias_v), merged_weight)
                        del qw, kw, vw
                        updown_qkv = torch.vstack([updown_q, updown_k, updown_v])
                        merged_weight = merged_weight.to(dtype=updown_qkv.dtype) + updown_qkv

                except RuntimeError as e:
                    logging.debug(f"Network {net.name} layer {network_layer_name}: {e}")
                    extra_network_lora.errors[net.name] = extra_network_lora.errors.get(net.name, 0) + 1
                    raise

                continue

            raise RuntimeError(f"Network {net.name} layer {network_layer_name}: incomplete projections or unsupported operation")

        if merged_weight is not None:
            with torch.no_grad():
                self.weight.copy_(merged_weight.to(dtype=self.weight.dtype))
                if merged_bias is not None and hasattr(self, 'bias'):
                    network_set_bias(self, 'bias', merged_bias, self.weight)
        self.network_current_names = wanted_names


def network_forward(org_module, input, original_forward):
    """
    Old way of applying Lora by executing operations during layer's forward.
    Stacking many loras this way results in big performance degradation.
    """

    if getattr(org_module, "network_weights_backup", None) is not None:
        network_restore_weights_from_backup(org_module)
        network_reset_cached_weight(org_module)

    if len(loaded_networks) == 0:
        return original_forward(org_module, input)

    input = devices.cond_cast_unet(input)

    y = original_forward(org_module, input)

    network_layer_name = getattr(org_module, 'network_layer_name', None)
    for lora in loaded_networks:
        module = lora.modules.get(network_layer_name, None)
        if module is None:
            continue

        y = module.forward(input, y, lora)

    return y


def network_reset_cached_weight(self: Union[torch.nn.Conv2d, torch.nn.Linear]):
    self.network_current_names = ()
    self.network_weights_backup = None
    self.network_bias_backup = None
    if hasattr(self, 'network_in_proj_bias_backup'):
        del self.network_in_proj_bias_backup


def network_Embedding_forward(self, input):
    """Embedding indices stay integer; functional mode uses a temporary matrix."""
    layer = getattr(self, 'network_layer_name', None)
    active = [net for net in loaded_networks if layer in net.modules]
    has_backup = getattr(self, 'network_weights_backup', None) is not None
    if not active and not has_backup:
        return originals.Embedding_forward(self, input)

    if not shared.opts.lora_functional:
        network_apply_weights(self)
        return originals.Embedding_forward(self, input)

    # Also restore after merge -> functional -> no LoRA. The early return must
    # not leave a previously merged adapter on the base embedding matrix.
    if has_backup:
        network_restore_weights_from_backup(self)
        network_reset_cached_weight(self)
    if not active:
        return originals.Embedding_forward(self, input)

    with torch.no_grad():
        weight = self.weight
        for activation in active:
            delta, bias = activation.modules[layer].calc_updown(weight, activation)
            if tuple(delta.shape) != tuple(weight.shape) or bias is not None:
                raise ValueError(f"Invalid Embedding LoRA delta for {layer}: {tuple(delta.shape)}; expected {tuple(weight.shape)} without output bias")
            weight = (weight.to(dtype=delta.dtype) + delta).to(dtype=self.weight.dtype)
        # max_norm mutates its weight argument; this is now a fresh tensor,
        # not the base weight or a cached LoRA tensor.
        return torch.nn.functional.embedding(
            input, weight, self.padding_idx, self.max_norm, self.norm_type,
            self.scale_grad_by_freq, self.sparse,
        )


def network_Embedding_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)
    return originals.Embedding_load_state_dict(self, *args, **kwargs)


def network_Linear_forward(self, input):
    if shared.opts.lora_functional:
        return network_forward(self, input, originals.Linear_forward)

    network_apply_weights(self)

    return originals.Linear_forward(self, input)


def network_Linear_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)

    return originals.Linear_load_state_dict(self, *args, **kwargs)


def network_Conv2d_forward(self, input):
    if shared.opts.lora_functional:
        return network_forward(self, input, originals.Conv2d_forward)

    network_apply_weights(self)

    return originals.Conv2d_forward(self, input)


def network_Conv2d_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)

    return originals.Conv2d_load_state_dict(self, *args, **kwargs)


def network_GroupNorm_forward(self, input):
    if shared.opts.lora_functional:
        return network_forward(self, input, originals.GroupNorm_forward)

    network_apply_weights(self)

    return originals.GroupNorm_forward(self, input)


def network_GroupNorm_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)

    return originals.GroupNorm_load_state_dict(self, *args, **kwargs)


def network_LayerNorm_forward(self, input):
    if shared.opts.lora_functional:
        return network_forward(self, input, originals.LayerNorm_forward)

    network_apply_weights(self)

    return originals.LayerNorm_forward(self, input)


def network_LayerNorm_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)

    return originals.LayerNorm_load_state_dict(self, *args, **kwargs)


def network_MultiheadAttention_forward(self, *args, **kwargs):
    network_apply_weights(self)

    return originals.MultiheadAttention_forward(self, *args, **kwargs)


def network_MultiheadAttention_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)

    return originals.MultiheadAttention_load_state_dict(self, *args, **kwargs)


def network_relative_name(filename, directory):
    relative_path = os.path.relpath(filename, directory)
    return os.path.splitext(relative_path)[0].replace('\\', '/')


def register_network_alias(alias, entry):
    """Register an alias only when it identifies exactly one network."""
    if not alias or alias == entry.name:
        return

    alias_key = alias.casefold()
    conflicting_aliases = [key for key, value in available_network_aliases.items() if key.casefold() == alias_key and value is not entry]
    has_canonical_conflict = any(key.casefold() == alias_key and value is not entry for key, value in available_networks.items())
    if conflicting_aliases or has_canonical_conflict:
        forbidden_network_aliases[alias_key] = 1
        for conflicting_alias in conflicting_aliases:
            available_network_aliases.pop(conflicting_alias, None)
        return

    if alias_key not in forbidden_network_aliases:
        available_network_aliases[alias] = entry


def register_network_canonical_name(entry):
    """Make the canonical path resolvable and retire a conflicting old alias."""
    name_key = entry.name.casefold()
    conflicting_aliases = [key for key, value in available_network_aliases.items() if key.casefold() == name_key and value is not entry]
    if conflicting_aliases:
        forbidden_network_aliases[name_key] = 1
        for conflicting_alias in conflicting_aliases:
            available_network_aliases.pop(conflicting_alias, None)
    available_network_aliases[entry.name] = entry


def get_network_on_disk(name):
    # Canonical relative paths always win over aliases. Ambiguous aliases must
    # be rejected rather than silently selecting the last file seen on disk.
    entry = available_networks.get(name)
    if entry is not None:
        return entry
    if name.casefold() in forbidden_network_aliases:
        return None
    return available_network_aliases.get(name)


def process_network_files(names: list[str] | None = None):
    directories = [
        (shared.cmd_opts.lora_dir, "Lora"),
        (shared.cmd_opts.lyco_dir_backcompat, "LyCORIS"),
    ]
    candidates = []
    for directory, source in directories:
        for filename in shared.walk_files(directory, allowed_extensions=[".pt", ".ckpt", ".safetensors"]):
            if not os.path.isdir(filename):
                candidates.append((filename, network_relative_name(filename, directory), source))

    name_counts = {}
    for _filename, relative_name, _source in candidates:
        name_counts[relative_name.casefold()] = name_counts.get(relative_name.casefold(), 0) + 1

    for filename, relative_name, source in candidates:
        if os.path.isdir(filename):
            continue
        name = relative_name if name_counts[relative_name.casefold()] == 1 else f"{source}/{relative_name}"
        legacy_name = os.path.splitext(os.path.basename(filename))[0]
        # if names is provided, only load networks with names in the list
        if names and name not in names and legacy_name not in names:
            continue
        try:
            entry = network.NetworkOnDisk(name, filename)
        except OSError:  # should catch FileNotFoundError and PermissionError etc.
            errors.report(f"Failed to load network {name} from {filename}", exc_info=True)
            continue

        available_networks[name] = entry
        register_network_canonical_name(entry)
        register_network_alias(legacy_name, entry)
        register_network_alias(entry.alias, entry)


def update_available_networks_by_names(names: list[str]):
    # A full refresh detects newly introduced alias collisions. A partial scan
    # could otherwise make the first matching file win until the next refresh.
    list_available_networks()


def list_available_networks():
    available_networks.clear()
    available_network_aliases.clear()
    forbidden_network_aliases.clear()
    available_network_hash_lookup.clear()
    forbidden_network_aliases.update({"none": 1, "addams": 1})

    os.makedirs(shared.cmd_opts.lora_dir, exist_ok=True)

    process_network_files()


re_network_name = re.compile(r"(.*)\s*\([0-9a-fA-F]+\)")


def infotext_pasted(infotext, params):
    if "AddNet Module 1" in [x[1] for x in scripts.scripts_txt2img.infotext_fields]:
        return  # if the other extension is active, it will handle those fields, no need to do anything

    added = []

    for k in params:
        if not k.startswith("AddNet Model "):
            continue

        num = k[13:]

        if params.get("AddNet Module " + num) != "LoRA":
            continue

        name = params.get("AddNet Model " + num)
        if name is None:
            continue

        m = re_network_name.match(name)
        if m:
            name = m.group(1)

        multiplier = params.get("AddNet Weight A " + num, "1.0")

        added.append(f"<lora:{name}:{multiplier}>")

    if added:
        params["Prompt"] += "\n" + "".join(added)


originals: lora_patches.LoraPatches = None

extra_network_lora = None

available_networks = {}
available_network_aliases = {}
loaded_networks = []
loaded_bundle_embeddings = {}
networks_in_memory = {}
available_network_hash_lookup = {}
forbidden_network_aliases = {}

list_available_networks()
