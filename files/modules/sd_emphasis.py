from __future__ import annotations
import torch


class Emphasis:
    """Emphasis class decides how to death with (emphasized:1.1) text in prompts"""

    name: str = "Base"
    description: str = ""

    tokens: list[list[int]]
    """tokens from the chunk of the prompt"""

    multipliers: torch.Tensor
    """tensor with multipliers, once for each token"""

    z: torch.Tensor
    """output of cond transformers network (CLIP)"""

    def after_transformers(self):
        """Called after cond transformers network has processed the chunk of the prompt; this function should modify self.z to apply the emphasis"""

        pass


class EmphasisNone(Emphasis):
    name = "None"
    description = "disable the mechanism entirely and treat (:.1.1) as literal characters"


class EmphasisIgnore(Emphasis):
    name = "Ignore"
    description = "treat all empasised words as if they have no emphasis"


class EmphasisOriginal(Emphasis):
    name = "Original"
    description = "the original emphasis implementation"

    def after_transformers(self):
        # Accumulate in float32: fp16 sums/ratios can overflow even for finite inputs.
        original = self.z
        original_mean = original.float().mean()
        self.z = self.z * self.multipliers.reshape(self.multipliers.shape + (1,)).expand(self.z.shape)

        # restoring original mean is likely not correct, but it seems to work well to prevent artifacts that happen otherwise
        weighted = self.z.float()
        new_mean = weighted.mean()
        threshold = torch.finfo(torch.float32).eps * weighted.abs().mean().clamp_min(1.0)
        usable = new_mean.abs() > threshold
        ratio = original_mean / torch.where(usable, new_mean, torch.ones_like(new_mean))
        ratio = torch.where(usable & torch.isfinite(ratio), ratio, torch.ones_like(ratio))
        normalized = (weighted * ratio).to(self.z.dtype)
        # Cancellation has no meaningful mean-restoring scale. Keep the weighted
        # values (not zeros); do not create Inf on the cast back to fp16 either.
        self.z = torch.where(torch.isfinite(normalized).all(), normalized, self.z)


class EmphasisOriginalNoNorm(EmphasisOriginal):
    name = "No norm"
    description = "same as original, but without normalization (seems to work better for SDXL)"

    def after_transformers(self):
        self.z = self.z * self.multipliers.reshape(self.multipliers.shape + (1,)).expand(self.z.shape)


def get_current_option(emphasis_option_name):
    return next(iter([x for x in options if x.name == emphasis_option_name]), EmphasisOriginal)


def get_options_descriptions():
    return ", ".join(f"{x.name}: {x.description}" for x in options)


options = [
    EmphasisNone,
    EmphasisIgnore,
    EmphasisOriginal,
    EmphasisOriginalNoNorm,
]
