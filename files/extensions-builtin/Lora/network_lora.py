import torch

import lyco_helpers
import lora_keys
import modules.models.sd3.mmdit
import network
from modules import devices


class ModuleTypeLora(network.ModuleType):
    def create_module(self, net: network.Network, weights: network.NetworkWeights):
        target = weights.sd_module
        if type(target).__name__ == 'EmbeddingsWithFixes' and isinstance(getattr(target, 'wrapped', None), torch.nn.Embedding):
            weights = weights._replace(sd_module=target.wrapped)
        if isinstance(weights.sd_module, torch.nn.Embedding):
            pairs = (
                ("lora_up.weight", "lora_down.weight", False),
                ("lora_B.weight", "lora_A.weight", False),
                ("lora_embedding_B", "lora_embedding_A", True),
                ("lora_embedding_B.weight", "lora_embedding_A.weight", True),
            )
            matches = [(up, down, peft) for up, down, peft in pairs if up in weights.w and down in weights.w]
            if len(matches) > 1:
                raise ValueError(f"Multiple Embedding LoRA factor formats in {weights.network_key}")
            if matches:
                up, down, peft = matches[0]
                return NetworkModuleLoraEmbedding(net, weights, up, down, peft=peft)

        if all(x in weights.w for x in ["lora_up.weight", "lora_down.weight"]):
            return NetworkModuleLora(net, weights)

        if all(x in weights.w for x in ["lora_A.weight", "lora_B.weight"]):
            # Do not discard alpha/scale/DoRA/bias or mutate the shared input.
            w = dict(weights.w, **{"lora_up.weight": weights.w["lora_B.weight"], "lora_down.weight": weights.w["lora_A.weight"]})
            weights = weights._replace(w=w)

            return NetworkModuleLora(net, weights)

        return None


class NetworkModuleLoraEmbedding(network.NetworkModule):
    def __init__(self, net, weights, up_key, down_key, *, peft=False):
        super().__init__(net, weights)
        if "lora_mid.weight" in weights.w or self.dora_scale is not None:
            raise ValueError(f"Embedding CP/DoRA format needs a dedicated adapter; refusing partial load: {weights.network_key}")
        up, down = weights.w[up_key], weights.w[down_key]
        self.transpose_delta = lora_keys.embedding_layout(up.shape, down.shape, self.shape, peft=peft)
        if self.bias is not None and tuple(self.bias.shape) != tuple(self.shape):
            raise ValueError(f"Embedding LoRA bias must be a matrix matching {tuple(self.shape)}")
        self.up = up.detach()
        self.down = down.detach()
        self.dim = down.shape[0]

    def calc_updown(self, orig_weight, activation):
        up, down = self.up.to(orig_weight.device), self.down.to(orig_weight.device)
        rank = activation.dyn_dim
        if rank is not None:
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
                raise ValueError("Embedding LoRA dynamic rank must be a positive integer")
            up, down = up[:, :rank], down[:rank, :]
        delta = up @ down
        if self.transpose_delta:
            delta = delta.transpose(0, 1)
        if tuple(delta.shape) != tuple(orig_weight.shape):
            raise ValueError(f"Embedding LoRA delta {tuple(delta.shape)} does not match {tuple(orig_weight.shape)}")
        return self.finalize_updown(delta, orig_weight, orig_weight.shape, activation)

    def network_to_device(self, device):
        super().network_to_device(device)
        self.up = self.up.to(device=device, dtype=devices.dtype)
        self.down = self.down.to(device=device, dtype=devices.dtype)


class NetworkModuleLora(network.NetworkModule):
    def __init__(self,  net: network.Network, weights: network.NetworkWeights):
        super().__init__(net, weights)

        self.up_model = self.create_module(weights.w, "lora_up.weight")
        self.down_model = self.create_module(weights.w, "lora_down.weight")
        self.mid_model = self.create_module(weights.w, "lora_mid.weight", none_ok=True)

        self.dim = weights.w["lora_down.weight"].shape[0]

    def create_module(self, weights, key, none_ok=False):
        weight = weights.get(key)

        if weight is None and none_ok:
            return None

        is_linear = type(self.sd_module) in [torch.nn.Linear, torch.nn.modules.linear.NonDynamicallyQuantizableLinear, torch.nn.MultiheadAttention, modules.models.sd3.mmdit.QkvLinear]
        is_conv = type(self.sd_module) in [torch.nn.Conv2d]

        if is_linear:
            weight = weight.reshape(weight.shape[0], -1)
            module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        elif is_conv and key in ("lora_down.weight", "dyn_up"):
            if len(weight.shape) == 2:
                weight = weight.reshape(weight.shape[0], -1, 1, 1)

            if weight.shape[2] != 1 or weight.shape[3] != 1:
                module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], self.sd_module.kernel_size, self.sd_module.stride, self.sd_module.padding, bias=False)
            else:
                module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
        elif is_conv and key == "lora_mid.weight":
            module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], self.sd_module.kernel_size, self.sd_module.stride, self.sd_module.padding, bias=False)
        elif is_conv and key in ("lora_up.weight", "dyn_down"):
            module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
        else:
            raise AssertionError(f'Lora layer {self.network_key} matched a layer with unsupported type: {type(self.sd_module).__name__}')

        with torch.no_grad():
            if weight.shape != module.weight.shape:
                weight = weight.reshape(module.weight.shape)
            module.weight.copy_(weight)

        module.weight.requires_grad_(False)

        return module

    def calc_updown(self, orig_weight, activation):
        up = self.up_model.weight.to(orig_weight.device)
        down = self.down_model.weight.to(orig_weight.device)
        rank = activation.dyn_dim
        if rank is not None and (isinstance(rank, bool) or not isinstance(rank, int) or rank < 1):
            raise ValueError("LoRA dynamic rank must be a positive integer")

        output_shape = [up.size(0), down.size(1)]
        if self.mid_model is not None:
            # cp-decomposition
            mid = self.mid_model.weight.to(orig_weight.device)
            if rank is not None:
                up, down, mid = up[:, :rank], down[:rank], mid[:rank, :rank]
            updown = lyco_helpers.rebuild_cp_decomposition(up, down, mid)
            output_shape += mid.shape[2:]
        else:
            if len(down.shape) == 4:
                output_shape += down.shape[2:]
            updown = lyco_helpers.rebuild_conventional(up, down, output_shape, activation.dyn_dim)

        return self.finalize_updown(updown, orig_weight, output_shape, activation)

    def forward(self, x, y, activation):
        # The factorized path is faster, but cannot express CP decomposition,
        # a LoRA bias, DoRA's weight decomposition, or dynamic rank. Use the
        # general path for those cases so functional and merged modes remain
        # equivalent.
        if self.mid_model is None and self.bias is None and self.dora_scale is None and activation.dyn_dim is None:
            return y + self.up_model(self.down_model(x)) * self.multiplier(activation) * self.calc_scale()

        return super().forward(x, y, activation)

    def network_to_device(self, device):
        self.up_model.to(device=device, dtype=devices.dtype)
        self.down_model.to(device=device, dtype=devices.dtype)
        if self.mid_model is not None:
            self.mid_model.to(device=device, dtype=devices.dtype)


