"""Typed, torch-free schema for native LoRA prompt tags.

The WebUI parser still supplies ``ExtraNetworkParams`` to the LoRA backend.
This module owns the conversion at that boundary, and also exposes a small tag
parser/serializer for the future Studio browser model.  It intentionally does
not import WebUI, Gradio or torch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable

import lora_runtime


SCHEMA_VERSION = 1
SUPPORTED_TAG_TYPES = ("lora", "lyco", "loractl")
_KNOWN_NAMED_ARGUMENTS = frozenset({
    "te", "unet", "dyn", "hr", "hrte", "hrunet", "start", "stop", "step", "lbw", "w", "lbwe", "profile",
})
_TAG_PATTERN = re.compile(r"^<(?P<tag_type>[A-Za-z0-9_-]+):(?P<body>[^<>]*)>$")


def _finite_number(value, description):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must be a number, got {value!r}") from error

    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite, got {value!r}")
    return result


def _format_number(value):
    value = float(value)
    if value == 0:
        value = 0.0
    return format(value, ".12g")


@dataclass(frozen=True)
class WeightSpec:
    """A scalar weight or a normalized piecewise-linear curve."""

    value: float
    curve: tuple[tuple[float, float], ...] | None = None

    @classmethod
    def parse(cls, raw, default=1.0):
        value, curve = lora_runtime.parse_weight_spec(raw, default)
        return cls(value=float(value), curve=tuple(curve) if curve else None)

    def serialize(self):
        if not self.curve:
            return _format_number(self.value)
        return ",".join(f"{_format_number(weight)}@{_format_number(position)}" for position, weight in self.curve)

    def to_contract(self):
        if not self.curve:
            return {"kind": "scalar", "value": self.value}
        return {
            "kind": "curve",
            "points": [{"position": position, "value": value} for position, value in self.curve],
        }


@dataclass(frozen=True)
class StepRange:
    start: float | None = None
    stop: float | None = None

    def to_contract(self):
        return {"start": self.start, "stop": self.stop}


@dataclass(frozen=True)
class BlockWeightSpec:
    """Prompt spelling plus the validated runtime table it represents."""

    raw: str
    weights: lora_runtime.BlockWeights

    def to_contract(self):
        # The text is the portable contract; values are intentionally evaluated
        # by the runtime so custom preset availability remains explicit.
        return {"raw": self.raw}


@dataclass(frozen=True)
class ElementalWeightSpec:
    """Prompt spelling plus validated, per-module elemental rules."""

    raw: str
    weights: lora_runtime.ElementalWeights

    def to_contract(self):
        # As with lbw, only the portable named preset is exposed to Studio.
        # The runtime owns the rule evaluation and never mutates LoRA payloads.
        return {"raw": self.raw}


@dataclass(frozen=True)
class ModelProfile:
    """A named block-layout profile; detection is intentionally a later step."""

    profile_id: str
    display_name: str
    block_ids: tuple[str, ...] = ()
    supported_controls: tuple[str, ...] = ("weights", "curves", "timing", "hires")


MODEL_PROFILES = {
    "auto": ModelProfile("auto", "Automatic (safe raw fallback)"),
    "classic-12": ModelProfile("classic-12", "Classic 12-block layout", lora_runtime.BLOCK_IDS_12),
    "classic-17": ModelProfile("classic-17", "Classic 17-block layout", lora_runtime.BLOCK_IDS_17),
    "classic-20": ModelProfile("classic-20", "Classic 20-block layout", lora_runtime.BLOCK_IDS_20),
    "classic-26": ModelProfile("classic-26", "Classic 26-block layout", lora_runtime.BLOCK_IDS_26),
    "flux-61": ModelProfile("flux-61", "Flux 61-block layout", lora_runtime.BLOCK_IDS_FLUX),
    "dit-61": ModelProfile("dit-61", "Flat DiT 61-block layout", lora_runtime.BLOCK_IDS_DIT),
}


def get_model_profile(profile_id):
    profile_id = str(profile_id or "auto").strip().casefold()
    try:
        return MODEL_PROFILES[profile_id]
    except KeyError as error:
        choices = ", ".join(MODEL_PROFILES)
        raise ValueError(f"unknown LoRA model profile {profile_id!r}; expected one of {choices}") from error


@dataclass(frozen=True)
class LoraActivationSpec:
    """One immutable LoRA occurrence in a prompt.

    Unknown arguments are stored in encounter order.  This is important for a
    future Studio editor: editing a known field must not discard syntax added
    by another extension or by a newer WebUI release.
    """

    network_name: str
    tag_type: str = "lora"
    text_encoder: WeightSpec = WeightSpec(1.0)
    unet: WeightSpec = WeightSpec(1.0)
    hires_text_encoder: WeightSpec | None = None
    hires_unet: WeightSpec | None = None
    dyn_dim: int | None = None
    active_range: StepRange = StepRange()
    block_weights: BlockWeightSpec | None = None
    elemental_weights: ElementalWeightSpec | None = None
    model_profile: str = "auto"
    passthrough_named: tuple[tuple[str, str], ...] = ()
    passthrough_positional: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        name = str(self.network_name).strip()
        if not name:
            raise ValueError("LoRA tag needs a network name as its first argument")
        if self.tag_type not in SUPPORTED_TAG_TYPES:
            raise ValueError(f"unsupported LoRA tag type {self.tag_type!r}")
        if not str(self.model_profile).strip():
            raise ValueError("LoRA model profile must not be empty")

    def to_contract(self):
        return {
            "schema_version": self.schema_version,
            "tag_type": self.tag_type,
            "network_name": self.network_name,
            "weights": {"te": self.text_encoder.to_contract(), "unet": self.unet.to_contract()},
            "hires": {
                "te": self.hires_text_encoder.to_contract() if self.hires_text_encoder else None,
                "unet": self.hires_unet.to_contract() if self.hires_unet else None,
            },
            "dyn_dim": self.dyn_dim,
            "active_range": self.active_range.to_contract(),
            "block_weights": self.block_weights.to_contract() if self.block_weights else None,
            "elemental_weights": self.elemental_weights.to_contract() if self.elemental_weights else None,
            "model_profile": self.model_profile,
            "passthrough_named": [{"name": name, "value": value} for name, value in self.passthrough_named],
            "passthrough_positional": list(self.passthrough_positional),
        }

    def to_runtime_values(self):
        """Return the legacy loader values without exposing parser details."""
        settings = {
            "te_schedule": self.text_encoder.curve,
            "unet_schedule": self.unet.curve,
            "hr_te_multiplier": self.hires_text_encoder.value if self.hires_text_encoder else None,
            "hr_unet_multiplier": self.hires_unet.value if self.hires_unet else None,
            "hr_te_schedule": self.hires_text_encoder.curve if self.hires_text_encoder else None,
            "hr_unet_schedule": self.hires_unet.curve if self.hires_unet else None,
            "start_step": self.active_range.start,
            "stop_step": self.active_range.stop,
            "block_weights": self.block_weights.weights if self.block_weights else None,
        }
        # Keep the old loader contract byte-for-byte for ordinary tags.  The
        # optional fields are supplied only when a tag actually uses the new
        # elemental/profile controls; NetworkActivation has safe defaults.
        if self.elemental_weights is not None:
            settings["elemental_weights"] = self.elemental_weights.weights
        if self.model_profile != "auto":
            settings["model_profile"] = self.model_profile
        return self.network_name, self.text_encoder.value, self.unet.value, self.dyn_dim, settings


def _split_items(items: Iterable[object]):
    values = [str(item) for item in items]
    if not values or not values[0].strip():
        raise ValueError("LoRA tag needs a network name as its first argument")

    positional = [values[0]]
    named_items = []
    for item in values[1:]:
        key, separator, value = item.partition("=")
        if separator and key.strip():
            named_items.append((key.strip(), value))
        else:
            positional.append(item)

    return positional, named_items


def parse_lora_items(items, tag_type="lora", custom_presets="", elemental_presets=""):
    """Parse WebUI ExtraNetwork item strings into :class:`LoraActivationSpec`."""
    tag_type = str(tag_type).casefold()
    if tag_type not in SUPPORTED_TAG_TYPES:
        raise ValueError(f"unsupported LoRA tag type {tag_type!r}")

    positional, named_items = _split_items(items)
    # ExtraNetworkParams has always used the last named value.  Preserve that
    # execution behavior while retaining all unknown values for round-tripping.
    named = {}
    for key, value in named_items:
        named[key] = value

    name, te, unet, dyn_dim, settings = _parse_runtime_options(positional, named, custom_presets)
    hr_te = _weight_from_runtime(settings["hr_te_multiplier"], settings["hr_te_schedule"])
    hr_unet = _weight_from_runtime(settings["hr_unet_multiplier"], settings["hr_unet_schedule"])
    block_raw = named.get("lbw", named.get("w"))
    block_weights = BlockWeightSpec(str(block_raw), settings["block_weights"]) if settings["block_weights"] is not None else None
    elemental_raw = named.get("lbwe")
    elemental = lora_runtime.parse_elemental_weights(elemental_raw, elemental_presets)
    elemental_weights = ElementalWeightSpec(str(elemental_raw), elemental) if elemental is not None else None
    profile = str(named.get("profile", "auto")).strip().casefold() or "auto"

    return LoraActivationSpec(
        network_name=name,
        tag_type=tag_type,
        text_encoder=WeightSpec(te, settings["te_schedule"]),
        unet=WeightSpec(unet, settings["unet_schedule"]),
        hires_text_encoder=hr_te,
        hires_unet=hr_unet,
        dyn_dim=dyn_dim,
        active_range=StepRange(settings["start_step"], settings["stop_step"]),
        block_weights=block_weights,
        elemental_weights=elemental_weights,
        model_profile=profile,
        passthrough_named=tuple((key, value) for key, value in named_items if key not in _KNOWN_NAMED_ARGUMENTS),
        passthrough_positional=tuple(positional[4:]),
    )


def _parse_runtime_options(positional, named, custom_presets):
    """Compatibility adapter while the pure runtime helpers migrate modules."""
    # Do not duplicate the legacy parser here: the existing runtime semantics
    # remain the single source of truth until their implementation is moved
    # from lora_advanced.py into lora_runtime.py in a later small change.
    return lora_runtime.parse_activation_options(positional, named, custom_presets)


def _weight_from_runtime(value, curve):
    return None if value is None else WeightSpec(float(value), tuple(curve) if curve else None)


def parse_lora_tag(tag, custom_presets="", elemental_presets=""):
    """Parse one complete ``<lora:...>``/``<lyco:...>``/``<loractl:...>`` tag."""
    match = _TAG_PATTERN.fullmatch(str(tag).strip())
    if not match:
        raise ValueError("expected a complete <lora:...> tag")

    tag_type = match.group("tag_type").casefold()
    body = match.group("body")
    return parse_lora_items(body.split(":"), tag_type=tag_type, custom_presets=custom_presets, elemental_presets=elemental_presets)


def serialize_lora_tag(spec: LoraActivationSpec, canonical_tag_type=False):
    """Serialize a canonical tag, retaining aliases and unknown arguments by default."""
    if not isinstance(spec, LoraActivationSpec):
        raise TypeError("spec must be a LoraActivationSpec")

    tag_type = "lora" if canonical_tag_type else spec.tag_type
    # A future positional argument can be meaningful only relative to the
    # legacy name/TE/U-Net/dyn slots.  Keep those slots positional in that
    # uncommon case; otherwise a parse -> serialize -> parse cycle would move
    # the unknown value into the TE slot and silently lose it.
    if spec.passthrough_positional:
        items = [
            spec.network_name,
            spec.text_encoder.serialize(),
            spec.unet.serialize(),
            "" if spec.dyn_dim is None else str(spec.dyn_dim),
        ]
    else:
        items = [
            spec.network_name,
            f"te={spec.text_encoder.serialize()}",
            f"unet={spec.unet.serialize()}",
        ]
        if spec.dyn_dim is not None:
            items.append(f"dyn={spec.dyn_dim}")
    if spec.hires_text_encoder is not None:
        items.append(f"hrte={spec.hires_text_encoder.serialize()}")
    if spec.hires_unet is not None:
        items.append(f"hrunet={spec.hires_unet.serialize()}")
    if spec.active_range.start is not None:
        items.append(f"start={_format_number(spec.active_range.start)}")
    if spec.active_range.stop is not None:
        items.append(f"stop={_format_number(spec.active_range.stop)}")
    if spec.block_weights is not None:
        items.append(f"lbw={spec.block_weights.raw}")
    if spec.elemental_weights is not None:
        items.append(f"lbwe={spec.elemental_weights.raw}")
    if spec.model_profile != "auto":
        items.append(f"profile={spec.model_profile}")
    items.extend(spec.passthrough_positional)
    items.extend(f"{key}={value}" for key, value in spec.passthrough_named)
    return f"<{tag_type}:{':'.join(items)}>"


def parse_extra_network_params(params, custom_presets="", elemental_presets=""):
    """Adapter for ``modules.extra_networks.ExtraNetworkParams`` without importing it."""
    return parse_lora_items(params.items, custom_presets=custom_presets, elemental_presets=elemental_presets)
