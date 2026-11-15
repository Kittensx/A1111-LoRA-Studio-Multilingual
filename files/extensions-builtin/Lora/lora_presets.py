"""Pure, versioned schema and validation for LoRA Studio control presets.

The persistent representation is a small JSON document with fully typed
fields.  It deliberately never stores a loose prompt fragment: a preset is
validated before it can be written to WebUI settings or applied by Studio.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import lora_runtime
from lora_schema import BlockWeightSpec, ElementalWeightSpec, StepRange, WeightSpec


PRESET_SCHEMA_VERSION = 3
LEGACY_PRESET_SCHEMA_VERSION = 1
HIRES_PRESET_SCHEMA_VERSION = 2
MAX_PRESETS = 100


@dataclass(frozen=True)
class HiresOverrides:
    """An explicit Hires.fix part of a v2+ preset.

    ``None`` at the preset level has a different, deliberate meaning: it is
    the migrated representation of a v1 preset, whose Hires values were not
    part of the snapshot and must therefore be left unchanged when applied.
    """

    text_encoder: WeightSpec | None = None
    unet: WeightSpec | None = None

    def to_contract(self):
        return {
            "te": self.text_encoder.to_contract() if self.text_encoder else None,
            "unet": self.unet.to_contract() if self.unet else None,
        }

    @classmethod
    def from_contract(cls, value):
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("LoRA preset Hires overrides must be an object or null")
        if "te" not in value or "unet" not in value:
            raise ValueError("LoRA preset Hires overrides need te and unet fields")
        return cls(
            text_encoder=_weight_from_contract(value["te"], "Hires text encoder"),
            unet=_weight_from_contract(value["unet"], "Hires U-Net"),
        )


@dataclass(frozen=True)
class LoraControlPreset:
    preset_id: str
    title: str
    model_profile: str = "auto"
    text_encoder: WeightSpec | None = None
    unet: WeightSpec | None = None
    dyn_dim: int | None = None
    active_range: StepRange | None = None
    block_weights: BlockWeightSpec | None = None
    elemental_weights: ElementalWeightSpec | None = None
    hires: HiresOverrides | None = None
    favorite: bool = False
    schema_version: int = PRESET_SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != PRESET_SCHEMA_VERSION:
            raise ValueError(f"unsupported LoRA preset schema version {self.schema_version!r}")
        if not isinstance(self.preset_id, str) or not self.preset_id.strip():
            raise ValueError("LoRA preset needs a stable preset_id")
        if len(self.preset_id) > 160:
            raise ValueError("LoRA preset id must be 160 characters or fewer")
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("LoRA preset needs a title")
        if len(self.title) > 120:
            raise ValueError("LoRA preset title must be 120 characters or fewer")
        if not isinstance(self.model_profile, str) or not self.model_profile.strip():
            raise ValueError("LoRA preset model profile must be a non-empty string")
        if not isinstance(self.favorite, bool):
            raise ValueError("LoRA preset favorite must be a boolean")

    def to_contract(self):
        return {
            "schema_version": self.schema_version,
            "preset_id": self.preset_id,
            "title": self.title,
            "model_profile": self.model_profile,
            "weights": {
                "te": self.text_encoder.to_contract() if self.text_encoder else None,
                "unet": self.unet.to_contract() if self.unet else None,
            },
            "dyn_dim": self.dyn_dim,
            "active_range": self.active_range.to_contract() if self.active_range else None,
            "block_weights": self.block_weights.to_contract() if self.block_weights else None,
            "elemental_weights": self.elemental_weights.to_contract() if self.elemental_weights else None,
            # ``null`` is intentionally preserved for a migrated v1 preset:
            # applying it leaves the draft's existing Hires overrides alone.
            "hires": self.hires.to_contract() if self.hires else None,
            "favorite": self.favorite,
        }

    @classmethod
    def from_contract(cls, value, custom_block_presets="", custom_elemental_presets=""):
        """Validate one JSON-compatible preset and return its typed form."""
        if not isinstance(value, dict):
            raise ValueError("LoRA preset must be an object")
        source_version = value.get("schema_version")
        if source_version not in (LEGACY_PRESET_SCHEMA_VERSION, HIRES_PRESET_SCHEMA_VERSION, PRESET_SCHEMA_VERSION):
            raise ValueError(f"unsupported LoRA preset schema version {value.get('schema_version')!r}")

        preset_id = value.get("preset_id")
        title = value.get("title")
        if not isinstance(preset_id, str) or not isinstance(title, str):
            raise ValueError("LoRA preset id and title must be strings")

        weights = value.get("weights", {})
        if not isinstance(weights, dict):
            raise ValueError("LoRA preset weights must be an object")

        dyn_dim = value.get("dyn_dim")
        if dyn_dim is not None and (isinstance(dyn_dim, bool) or not isinstance(dyn_dim, int)):
            raise ValueError("LoRA preset dynamic rank must be an integer or null")

        active_range = _step_range_from_contract(value.get("active_range"))
        block_weights = _block_weights_from_contract(value.get("block_weights"), custom_block_presets)
        if source_version < PRESET_SCHEMA_VERSION:
            # v1/v2 have no elemental field, so their migrated v3 snapshot
            # simply carries no elemental override.
            elemental_weights = None
        else:
            if "elemental_weights" not in value:
                raise ValueError("LoRA preset schema v3 needs an elemental_weights field")
            elemental_weights = _elemental_weights_from_contract(value.get("elemental_weights"), custom_elemental_presets)
        model_profile = value.get("model_profile", "auto")
        if not isinstance(model_profile, str):
            raise ValueError("LoRA preset model profile must be a string")
        if source_version == LEGACY_PRESET_SCHEMA_VERSION:
            # Version 1 had no Hires field.  Do not invent an explicit
            # all-null override: that would clear Hires values that applying
            # the same v1 preset used to preserve.
            hires = None
        else:
            if "hires" not in value:
                raise ValueError("LoRA preset schema v2 needs a hires field")
            hires = HiresOverrides.from_contract(value["hires"])
        favorite = value.get("favorite", False)
        if not isinstance(favorite, bool):
            raise ValueError("LoRA preset favorite must be a boolean")

        return cls(
            preset_id=preset_id.strip(),
            title=title.strip(),
            # Profile identifiers are editor metadata, not execution code.
            # Preserve an unknown future identifier so an older Studio can
            # back up and restore it without turning it into "auto".
            model_profile=model_profile.strip().casefold() or "auto",
            text_encoder=_weight_from_contract(weights.get("te"), "text encoder"),
            unet=_weight_from_contract(weights.get("unet"), "U-Net"),
            dyn_dim=dyn_dim,
            active_range=active_range,
            block_weights=block_weights,
            elemental_weights=elemental_weights,
            hires=hires,
            favorite=favorite,
        )


def _weight_from_contract(value, label):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"LoRA preset {label} weight must be an object or null")

    kind = value.get("kind")
    if kind == "scalar":
        if "value" not in value:
            raise ValueError(f"LoRA preset {label} scalar needs a value")
        return WeightSpec.parse(value.get("value"))
    if kind != "curve":
        raise ValueError(f"LoRA preset {label} weight kind must be scalar or curve")

    points = value.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError(f"LoRA preset {label} curve needs at least one point")
    try:
        raw = ",".join(f"{point['value']}@{point['position']}" for point in points if isinstance(point, dict))
    except KeyError as error:
        raise ValueError(f"LoRA preset {label} curve points need value and position") from error
    if raw.count("@") != len(points):
        raise ValueError(f"LoRA preset {label} curve points must be objects")
    return WeightSpec.parse(raw)


def _step_range_from_contract(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("LoRA preset active range must be an object or null")
    start, stop = lora_runtime.parse_step_range(value.get("start"), value.get("stop"))
    return StepRange(start, stop)


def _block_weights_from_contract(value, custom_block_presets):
    if value is None:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("raw"), str):
        raise ValueError("LoRA preset block weights must contain a raw string")
    raw = value["raw"].strip()
    if not raw:
        raise ValueError("LoRA preset block weights cannot be empty")
    weights = lora_runtime.parse_block_weights(raw, custom_block_presets)
    return BlockWeightSpec(raw, weights)


def _elemental_weights_from_contract(value, custom_elemental_presets):
    if value is None:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("raw"), str):
        raise ValueError("LoRA preset elemental weights must contain a raw string")
    raw = value["raw"].strip()
    if not raw:
        raise ValueError("LoRA preset elemental weights cannot be empty")
    weights = lora_runtime.parse_elemental_weights(raw, custom_elemental_presets)
    return ElementalWeightSpec(raw, weights)


def parse_preset_library(value, custom_block_presets="", custom_elemental_presets=""):
    """Validate the JSON-compatible preset-library envelope.

    The current document is ``{"schema_version": 3, "presets": [...]}``.
    Version 1 is migrated in memory with ``hires: null`` (which preserves
    the draft Hires values); version 2 receives
    ``elemental_weights: null`` (no elemental override).
    Keeping an envelope allows migrations without treating a user settings
    value as executable or prompt syntax.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("LoRA preset library is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("LoRA preset library must be an object")
    if value.get("schema_version") not in (LEGACY_PRESET_SCHEMA_VERSION, HIRES_PRESET_SCHEMA_VERSION, PRESET_SCHEMA_VERSION):
        raise ValueError(f"unsupported LoRA preset library schema version {value.get('schema_version')!r}")
    presets = value.get("presets")
    if not isinstance(presets, list):
        raise ValueError("LoRA preset library presets must be a list")
    if len(presets) > MAX_PRESETS:
        raise ValueError(f"LoRA preset library supports at most {MAX_PRESETS} presets")

    typed = tuple(LoraControlPreset.from_contract(item, custom_block_presets, custom_elemental_presets) for item in presets)
    identifiers = [preset.preset_id.casefold() for preset in typed]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("LoRA preset identifiers must be unique")
    return typed


def serialize_preset_library(presets):
    """Return the canonical compact JSON form stored in WebUI settings."""
    typed = tuple(presets)
    if len(typed) > MAX_PRESETS:
        raise ValueError(f"LoRA preset library supports at most {MAX_PRESETS} presets")
    if not all(isinstance(preset, LoraControlPreset) for preset in typed):
        raise TypeError("LoRA preset library must contain LoraControlPreset values")
    identifiers = [preset.preset_id.casefold() for preset in typed]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("LoRA preset identifiers must be unique")
    return json.dumps({
        "schema_version": PRESET_SCHEMA_VERSION,
        "presets": [preset.to_contract() for preset in typed],
    }, ensure_ascii=False, separators=(",", ":"))
