from __future__ import annotations
import os
from collections import namedtuple
from dataclasses import dataclass
import enum

import torch.nn as nn
import torch.nn.functional as F

from modules import sd_models, cache, errors, hashes, shared
import modules.models.sd3.mmdit
import lora_runtime

NetworkWeights = namedtuple('NetworkWeights', ['network_key', 'sd_key', 'w', 'sd_module'])

metadata_tags_order = {"ss_sd_model_name": 1, "ss_resolution": 2, "ss_clip_skip": 3, "ss_num_train_images": 10, "ss_tag_frequency": 20}


class SdVersion(enum.Enum):
    Unknown = 1
    SD1 = 2
    SD2 = 3
    SDXL = 4
    SD3 = 5
    Flux = 6


class NetworkOnDisk:
    def __init__(self, name, filename):
        from modules.asset_files import file_signature
        self.file_signature = file_signature(filename)
        if self.file_signature is None:
            raise FileNotFoundError(filename)
        self.name = name
        self.filename = filename
        self.metadata = {}
        self.is_safetensors = os.path.splitext(filename)[1].lower() == ".safetensors"

        def read_metadata():
            metadata = sd_models.read_metadata_from_safetensors(filename)

            return metadata

        if self.is_safetensors:
            try:
                self.metadata = cache.cached_data_for_file('safetensors-metadata', "lora/" + self.name, filename, read_metadata)
            except Exception as e:
                errors.display(e, f"reading lora {filename}")

        if self.metadata:
            m = {}
            for k, v in sorted(self.metadata.items(), key=lambda x: metadata_tags_order.get(x[0], 999)):
                m[k] = v

            self.metadata = m

        self.alias = self.metadata.get('ss_output_name', self.name)

        self.hash = None
        self.shorthash = None
        self.set_hash(
            hashes.sha256_from_cache(self.filename, "lora/" + self.name, use_addnet_hash=self.is_safetensors) or
            ''
        )

        self.sd_version = self.detect_version()
        if file_signature(filename) != self.file_signature:
            self.set_hash('')
            raise OSError(f'LoRA changed while reading metadata: {filename}; refresh after the write completes')

    def detect_version(self):
        architecture = str(self.metadata.get('ss_base_model_version', '') or self.metadata.get('modelspec.architecture', '')).casefold()
        if 'flux' in architecture:
            return SdVersion.Flux
        if architecture.startswith(('sd3', 'stable-diffusion-v3')):
            return SdVersion.SD3
        if architecture.startswith(('sdxl', 'stable-diffusion-xl')):
            return SdVersion.SDXL
        elif str(self.metadata.get('ss_v2', "")).casefold() == "true" or architecture.startswith(('sd2', 'stable-diffusion-v2')):
            return SdVersion.SD2
        elif architecture.startswith(('sd1', 'stable-diffusion-v1')) or str(self.metadata.get('ss_v2', '')).casefold() == 'false':
            return SdVersion.SD1

        return SdVersion.Unknown

    def set_hash(self, v):
        import networks
        previous = getattr(self, 'shorthash', None)
        if previous and networks.available_network_hash_lookup.get(previous) is self:
            networks.available_network_hash_lookup.pop(previous, None)
        self.hash = v
        self.shorthash = self.hash[0:12]

        if self.shorthash:
            networks.available_network_hash_lookup[self.shorthash] = self

    def read_hash(self):
        from modules.asset_files import file_signature
        if file_signature(self.filename) != self.file_signature:
            self.set_hash('')
            raise OSError(f'LoRA descriptor is stale: {self.filename}; refresh and retry')
        value = hashes.sha256(self.filename, "lora/" + self.name, use_addnet_hash=self.is_safetensors) or ''
        if file_signature(self.filename) != self.file_signature:
            self.set_hash('')
            raise OSError(f'LoRA changed while hashing: {self.filename}; refresh and retry')
        self.set_hash(value)

    def get_alias(self):
        import networks
        if shared.opts.lora_preferred_name == "Filename" or self.alias.casefold() in networks.forbidden_network_aliases:
            return self.name
        else:
            return self.alias


class Network:  # LoraModule
    def __init__(self, name, network_on_disk: NetworkOnDisk):
        self.name = name
        self.network_on_disk = network_on_disk
        self.modules = {}
        self.payload_identity = object()
        self.model_token = None
        self.bundle_embeddings = {}
        self.file_signature = None
        """Shared asset identity when the payload was loaded."""
        self.mtime = None
        """Compatibility timestamp for scripts that inspected loaded_loras."""


@dataclass
class NetworkActivation:
    """One use of a cached network in the current prompt.

    A network payload can appear more than once in a prompt with different
    multipliers. Keeping those values here, rather than on ``Network``, avoids
    aliases in ``loaded_networks`` mutating one another.
    """

    network: Network
    mentioned_name: str
    te_multiplier: float = 1.0
    unet_multiplier: float = 1.0
    dyn_dim: int | None = None
    te_schedule: tuple[tuple[float, float], ...] | None = None
    unet_schedule: tuple[tuple[float, float], ...] | None = None
    hr_te_multiplier: float | None = None
    hr_unet_multiplier: float | None = None
    hr_te_schedule: tuple[tuple[float, float], ...] | None = None
    hr_unet_schedule: tuple[tuple[float, float], ...] | None = None
    start_step: float | None = None
    stop_step: float | None = None
    block_weights: lora_runtime.BlockWeights | None = None
    elemental_weights: lora_runtime.ElementalWeights | None = None
    model_profile: str = "auto"
    is_hires: bool = False

    # These accessors preserve the small public surface exposed by lora.py for
    # third-party scripts that inspect ``loaded_loras``.
    @property
    def name(self):
        return self.network.name

    @property
    def network_on_disk(self):
        return self.network.network_on_disk

    @property
    def modules(self):
        return self.network.modules

    @property
    def bundle_embeddings(self):
        return self.network.bundle_embeddings

    @property
    def mtime(self):
        return self.network.mtime

    @mtime.setter
    def mtime(self, value):
        self.network.mtime = value

    @staticmethod
    def is_text_encoder_key(sd_key):
        key = str(sd_key)
        if key.startswith('diffusion_model_'):
            return False
        # SDXL's embedder index belongs to the text namespace. OpenCLIP token
        # embeddings have no "transformer" in their name but must use TE too.
        prefix, separator, _ = key.partition('_')
        return (bool(separator) and prefix.isdigit()) or key.startswith(
            ('model_token_embedding', 'model_positional_embedding')
        ) or 'transformer' in key[:20]

    def _uses_hires_weights(self):
        # This is captured when the tag is activated.  Reading a global flag
        # here is too late: Hires.fix builds its text conditioning before the
        # ``before_hr`` script callback runs.
        return self.is_hires

    def effective_multiplier(self, is_text_encoder):
        use_hires = self._uses_hires_weights()
        if is_text_encoder:
            static = self.hr_te_multiplier if use_hires and self.hr_te_multiplier is not None else self.te_multiplier
            curve = self.hr_te_schedule if use_hires and self.hr_te_multiplier is not None else self.te_schedule
        else:
            static = self.hr_unet_multiplier if use_hires and self.hr_unet_multiplier is not None else self.unet_multiplier
            curve = self.hr_unet_schedule if use_hires and self.hr_unet_multiplier is not None else self.unet_schedule

        # Text conditioning is created once before denoising.  A start/stop
        # interval can therefore only safely gate the U-Net contribution;
        # the text encoder receives the curve's initial value.
        if is_text_encoder:
            return float(curve[0][1] if curve else static)
        if not lora_runtime.step_is_active(self.start_step, self.stop_step, shared.state.sampling_step, shared.state.sampling_steps):
            return 0.0

        return lora_runtime.evaluate_weight_spec(curve, static, shared.state.sampling_step, shared.state.sampling_steps)

    def block_multiplier(self, sd_key, is_text_encoder):
        if self.block_weights is None:
            return 1.0
        return self.block_weights.value_for(
            sd_key,
            is_text_encoder=is_text_encoder,
            model_profile=self.model_profile,
        )

    def elemental_multiplier(self, sd_key, is_text_encoder):
        if self.elemental_weights is None:
            return None
        return self.elemental_weights.value_for(
            sd_key,
            is_text_encoder=is_text_encoder,
            model_profile=self.model_profile,
        )

    def layer_multiplier(self, sd_key, is_text_encoder):
        """Return block weight, overridden only by a matching lbwe rule."""
        elemental = self.elemental_multiplier(sd_key, is_text_encoder)
        return self.block_multiplier(sd_key, is_text_encoder) if elemental is None else elemental

    def cache_identity(self, sd_key):
        """Return the effective state that can change a merged model layer."""
        is_text_encoder = self.is_text_encoder_key(sd_key)
        return (
            self.network_on_disk.filename,
            self.network.payload_identity,
            self.network.model_token,
            self.effective_multiplier(is_text_encoder),
            self.layer_multiplier(sd_key, is_text_encoder),
            self.dyn_dim,
        )


class ModuleType:
    def create_module(self, net: Network, weights: NetworkWeights) -> Network | None:
        return None


class NetworkModule:
    def __init__(self, net: Network, weights: NetworkWeights):
        self.network = net
        self.network_key = weights.network_key
        self.sd_key = weights.sd_key
        self.sd_module = weights.sd_module

        if isinstance(self.sd_module, modules.models.sd3.mmdit.QkvLinear):
            s = self.sd_module.weight.shape
            self.shape = (s[0] // 3, s[1])
        elif getattr(self.sd_module, 'weight', None) is not None:
            self.shape = self.sd_module.weight.shape
        elif isinstance(self.sd_module, nn.MultiheadAttention):
            # For now, only self-attn use Pytorch's MHA
            # So assume all qkvo proj have same shape
            self.shape = self.sd_module.out_proj.weight.shape
        else:
            self.shape = None

        if self.shape is None:
            raise ValueError(f"LoRA layer {self.network_key!r} mapped to {self.sd_key!r}: "
                             f"unsupported weightless target {type(self.sd_module).__name__}")

        self.ops = None
        self.extra_kwargs = {}
        if isinstance(self.sd_module, nn.Conv2d):
            self.ops = F.conv2d
            self.extra_kwargs = {
                'stride': self.sd_module.stride,
                'padding': self.sd_module.padding
            }
        elif isinstance(self.sd_module, nn.Linear):
            self.ops = F.linear
        elif isinstance(self.sd_module, nn.LayerNorm):
            self.ops = F.layer_norm
            self.extra_kwargs = {
                'normalized_shape': self.sd_module.normalized_shape,
                'eps': self.sd_module.eps
            }
        elif isinstance(self.sd_module, nn.GroupNorm):
            self.ops = F.group_norm
            self.extra_kwargs = {
                'num_groups': self.sd_module.num_groups,
                'eps': self.sd_module.eps
            }

        self.dim = None
        self.bias = weights.w.get("bias")
        self.alpha = weights.w["alpha"].item() if "alpha" in weights.w else None
        self.scale = weights.w["scale"].item() if "scale" in weights.w else None

        self.dora_scale = weights.w.get("dora_scale", None)
        self.dora_norm_dims = len(self.shape) - 1

    def multiplier(self, activation: NetworkActivation):
        is_text_encoder = activation.is_text_encoder_key(self.sd_key)
        return activation.effective_multiplier(is_text_encoder) * activation.layer_multiplier(self.sd_key, is_text_encoder)

    def calc_scale(self):
        if self.scale is not None:
            return self.scale
        if self.dim is not None and self.alpha is not None:
            return self.alpha / self.dim

        return 1.0

    def apply_weight_decompose(self, updown, orig_weight):
        # Match the device/dtype
        orig_weight = orig_weight.to(updown.dtype)
        dora_scale = self.dora_scale.to(device=orig_weight.device, dtype=updown.dtype)
        updown = updown.to(orig_weight.device)

        merged_scale1 = updown + orig_weight
        merged_scale1_norm = (
            merged_scale1.transpose(0, 1)
            .reshape(merged_scale1.shape[1], -1)
            .norm(dim=1, keepdim=True)
            .reshape(merged_scale1.shape[1], *[1] * self.dora_norm_dims)
            .transpose(0, 1)
        )

        dora_merged = (
            merged_scale1 * (dora_scale / merged_scale1_norm)
        )
        final_updown = dora_merged - orig_weight
        return final_updown

    def finalize_updown(self, updown, orig_weight, output_shape, activation: NetworkActivation, ex_bias=None):
        if self.bias is not None:
            updown = updown.reshape(self.bias.shape)
            updown += self.bias.to(orig_weight.device, dtype=updown.dtype)
            updown = updown.reshape(output_shape)

        if len(output_shape) == 4:
            updown = updown.reshape(output_shape)

        if orig_weight.size().numel() == updown.size().numel():
            updown = updown.reshape(orig_weight.shape)

        if ex_bias is not None:
            ex_bias = ex_bias * self.multiplier(activation)

        updown = updown * self.calc_scale()

        if self.dora_scale is not None:
            updown = self.apply_weight_decompose(updown, orig_weight)

        return updown * self.multiplier(activation), ex_bias

    def network_to_device(self, device):
        if self.bias is not None:
            self.bias = self.bias.to(device)
        if self.dora_scale is not None:
            self.dora_scale = self.dora_scale.to(device)

    def calc_updown(self, target, activation: NetworkActivation):
        raise NotImplementedError()

    def forward(self, x, y, activation: NetworkActivation):
        """A general forward implementation for all modules"""
        if self.ops is None:
            raise NotImplementedError()
        else:
            updown, ex_bias = self.calc_updown(self.sd_module.weight, activation)
            return y + self.ops(x, weight=updown, bias=ex_bias, **self.extra_kwargs)

