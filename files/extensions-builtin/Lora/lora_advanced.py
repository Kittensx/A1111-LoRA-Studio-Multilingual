"""Pure helpers for advanced LoRA prompt controls.

This module deliberately has no WebUI or torch imports.  Keeping parsing and
weight selection here makes the syntax testable without loading a model and
prevents advanced prompt options from mutating a cached LoRA payload.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re


BLOCK_IDS_12 = ("BASE", "IN04", "IN05", "IN07", "IN08", "M00", "OUT00", "OUT01", "OUT02", "OUT03", "OUT04", "OUT05")
BLOCK_IDS_17 = ("BASE", "IN01", "IN02", "IN04", "IN05", "IN07", "IN08", "M00", "OUT03", "OUT04", "OUT05", "OUT06", "OUT07", "OUT08", "OUT09", "OUT10", "OUT11")
BLOCK_IDS_20 = ("BASE", "IN00", "IN01", "IN02", "IN03", "IN04", "IN05", "IN06", "IN07", "IN08", "M00", "OUT00", "OUT01", "OUT02", "OUT03", "OUT04", "OUT05", "OUT06", "OUT07", "OUT08")
BLOCK_IDS_26 = ("BASE", *(f"IN{i:02}" for i in range(12)), "M00", *(f"OUT{i:02}" for i in range(12)))
BLOCK_IDS_FLUX = ("CLIP", "T5", "IN", *(f"D{i:02}" for i in range(19)), *(f"S{i:02}" for i in range(38)), "OUT")
BLOCK_IDS_DIT = ("CLIP", "T5", "IN", *(f"B{i:02}" for i in range(57)), "OUT")
BLOCK_IDS_BY_LENGTH = {
    len(BLOCK_IDS_12): BLOCK_IDS_12,
    len(BLOCK_IDS_17): BLOCK_IDS_17,
    len(BLOCK_IDS_20): BLOCK_IDS_20,
    len(BLOCK_IDS_26): BLOCK_IDS_26,
    len(BLOCK_IDS_FLUX): BLOCK_IDS_FLUX,
}


DEFAULT_BLOCK_PRESETS = """\
NONE:0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
ALL:1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
INS:1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0
IND:1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0
INALL:1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0
MIDD:1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0
OUTD:1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0
OUTS:1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1
OUTALL:1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1
ALL0.5:0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
FLUXALL:""" + ",".join(["1"] * len(BLOCK_IDS_FLUX))

# The portable defaults from the old Block Weight extension.  They are data,
# not code: a prompt can select one by name with ``lbwe=ATTNDEEPON``.  Keeping
# these rules in the native runtime means they work without that extension's
# global monkeypatches or in-place edits to cached LoRA tensors.
DEFAULT_ELEMENTAL_PRESETS = """\
ATTNDEEPON:IN05-OUT05:attn:1

ATTNDEEPOFF:IN05-OUT05:attn:0

PROJDEEPOFF:IN05-OUT05:proj:0

XYZ:::1"""


def _as_finite_number(value, description):
    try:
        result = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{description} must be a number, got {value!r}") from e

    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite, got {value!r}")

    return result


def parse_weight_spec(value, default=1.0):
    """Return ``(initial_value, curve)`` for ``0.2@0,1@0.5`` syntax.

    A scalar retains normal LoRA semantics.  A curve has ``weight@position``
    points; positions in the inclusive 0..1 range are fractions of sampling
    progress, otherwise they are absolute sampler steps.  Duplicate positions
    are intentionally resolved by the last occurrence, which makes prompt
    editing predictable.
    """
    if value is None or str(value).strip() == "":
        return float(default), None

    raw = str(value).strip()
    if "@" not in raw and "~" not in raw:
        return _as_finite_number(raw, "LoRA weight"), None

    points = {}
    for item in re.split(r"[,;]", raw):
        item = item.strip()
        if not item:
            continue

        fields = re.split(r"[@~]", item)
        if len(fields) != 2 or not fields[0].strip() or not fields[1].strip():
            raise ValueError(f"invalid LoRA weight curve item {item!r}; use weight@position")

        weight = _as_finite_number(fields[0].strip(), "curve weight")
        position = _as_finite_number(fields[1].strip(), "curve position")
        points[position] = weight

    if not points:
        raise ValueError("LoRA weight curve has no points")

    curve = tuple((position, points[position]) for position in sorted(points))
    return curve[0][1], curve


def evaluate_weight_spec(curve, fallback, sampling_step, sampling_steps):
    """Linearly interpolate a parsed curve at the current sampling step."""
    if not curve:
        return float(fallback)

    positions = tuple(point[0] for point in curve)
    if all(0.0 <= position <= 1.0 for position in positions):
        denominator = max(int(sampling_steps) - 1, 1)
        current = float(sampling_step) / denominator
    else:
        current = float(sampling_step)

    if current <= positions[0]:
        return curve[0][1]
    if current >= positions[-1]:
        return curve[-1][1]

    for (left_position, left_weight), (right_position, right_weight) in zip(curve, curve[1:]):
        if current <= right_position:
            progress = (current - left_position) / (right_position - left_position)
            return left_weight + (right_weight - left_weight) * progress

    return curve[-1][1]


def parse_step_range(start=None, stop=None, value=None):
    """Parse ``start``, ``stop`` or the shorthand ``step=start-stop``."""
    if value is not None and str(value).strip():
        number = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
        match = re.fullmatch(r"\s*" + number + r"\s*[-~,;]\s*" + number + r"\s*", str(value))
        if match is None:
            raise ValueError("step must use start-stop, for example step=0.2-0.8")
        parts = match.groups()
        if start is None:
            start = parts[0]
        if stop is None:
            stop = parts[1]

    parsed_start = None if start is None or str(start).strip() == "" else _as_finite_number(start, "LoRA start")
    parsed_stop = None if stop is None or str(stop).strip() == "" else _as_finite_number(stop, "LoRA stop")
    if any(v is not None and v < 0 for v in (parsed_start, parsed_stop)):
        raise ValueError("LoRA start and stop must be non-negative")
    if parsed_start is not None and parsed_stop is not None and (parsed_start <= 1) == (parsed_stop <= 1) and parsed_start > parsed_stop:
        raise ValueError("LoRA start must not be greater than stop")
    return parsed_start, parsed_stop


def step_is_active(start, stop, sampling_step, sampling_steps):
    """Whether an optional inclusive activation interval includes this step."""
    if start is None and stop is None:
        return True

    denominator = max(int(sampling_steps) - 1, 1)

    def as_step(value):
        return value * denominator if 0.0 <= value <= 1.0 else value

    first = None if start is None else as_step(start)
    last = None if stop is None else as_step(stop)
    if first is not None and last is not None and first > last:
        raise ValueError(f"LoRA start resolves to step {first:g}, after stop {last:g} for {sampling_steps} steps")
    step = float(sampling_step)
    return (first is None or step >= first) and (last is None or step <= last)


def _preset_map(text):
    result = {}
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        name, weights = line.split(":", 1)
        name = name.strip()
        if name and weights.strip():
            result[name.casefold()] = weights.strip()
    return result


def _elemental_preset_map(text):
    """Return named elemental directives separated by blank lines.

    A preset's first ``:`` separates its name from its rules.  The remaining
    rules use the old, documented ``blocks:modules:weight`` spelling and may
    be separated with either commas or newlines.  A later definition wins, as
    it does for regular block-weight presets.
    """
    result = {}
    for group in re.split(r"\r?\n[ \t]*\r?\n", str(text or "")):
        name, separator, directives = group.strip().partition(":")
        name = name.strip()
        if name and separator and directives.strip():
            result[name.casefold()] = directives.strip()
    return result


def _block_ids_for_elemental_profile(profile, sd_key):
    """Choose only the block *naming* used by an elemental selector.

    ``auto`` needs a local decision because the same Studio tag may be used
    with classic or Flux-like models.  The key patterns are the same patterns
    used by :func:`block_id_for_key`; this never inspects model objects.
    """
    profile = str(profile or "auto").strip().casefold()
    if profile == "dit-61":
        return BLOCK_IDS_DIT
    if profile == "flux-61":
        return BLOCK_IDS_FLUX
    if profile.startswith("classic-"):
        return BLOCK_IDS_26

    key = str(sd_key).replace(".", "_").casefold()
    if any(marker in key for marker in ("double_blocks", "doubleblocks", "single_blocks", "singleblocks")):
        return BLOCK_IDS_FLUX
    # A flat DiT stack deliberately has no input/output-block spelling.
    classic = any(marker in key for marker in ("input_blocks", "middle_block", "output_blocks", "encoder_layers", "transformer_resblocks"))
    if re.search(r"(?:^|_)(?:layers|blocks)_\d+_", key) and not classic:
        return BLOCK_IDS_DIT
    return BLOCK_IDS_26


def _expand_elemental_terms(selector, block_ids):
    """Expand inclusive ``IN05-OUT05`` ranges in a selector safely."""
    terms = []
    for term in str(selector or "").strip().split():
        if "-" not in term:
            terms.append(term.casefold())
            continue
        left, right = (part.strip().upper() for part in term.split("-", 1))
        if left not in block_ids or right not in block_ids:
            raise ValueError(f"unknown elemental block range {term!r}")
        start, stop = sorted((block_ids.index(left), block_ids.index(right)))
        terms.extend(block.casefold() for block in block_ids[start:stop + 1])
    return tuple(terms)


def _parse_elemental_selector(selector, block_ids, label):
    """Return ``(is_negative, terms)`` for old ``NOT`` selector syntax."""
    text = str(selector or "").strip()
    words = text.split()
    negative = bool(words and words[0].casefold() == "not")
    if negative:
        text = " ".join(words[1:])
    # Block ranges must be expanded against the active model layout.  Keep
    # their source spelling here: ``IN05-OUT05`` is valid for classic U-Nets
    # but intentionally not for Flux; validating it against every layout at
    # prompt-parse time would reject a perfectly valid classic prompt.
    terms = tuple(term.casefold() for term in text.split())
    return negative, terms


def _validate_elemental_block_ranges(terms):
    """Reject malformed ranges while allowing classic/Flux/DiT spellings.

    A valid classic preset such as ``IN05-OUT05`` is not applicable to a
    Flux or flat-DiT key.  That is a non-match at runtime, not an error.  A
    range which does not exist in *any* supported layout is instead rejected
    while parsing the tag, before a sampling callback can see it.
    """
    layouts = (BLOCK_IDS_12, BLOCK_IDS_17, BLOCK_IDS_20, BLOCK_IDS_26, BLOCK_IDS_FLUX, BLOCK_IDS_DIT)
    for term in terms:
        if "-" not in term:
            continue
        left, right = (part.strip().upper() for part in term.split("-", 1))
        if not any(left in layout and right in layout for layout in layouts):
            raise ValueError(f"unknown elemental block range {term!r}")


@dataclass(frozen=True)
class ElementalRule:
    """A per-module override from the former Block Weight ``lbwe`` syntax."""

    block_negative: bool
    block_terms: tuple[str, ...]
    element_negative: bool
    element_terms: tuple[str, ...]
    value: float

    def matches(self, key, block_id, block_ids):
        haystack = (str(key).replace(".", "_") + str(block_id or "")).casefold()

        def selected(negative, terms):
            if not terms:
                # An empty positive selector means "all"; ``NOT`` on its
                # own also meant all in the source extension.
                return True
            found = any(term in haystack for term in terms)
            return not found if negative else found

        try:
            block_terms = _expand_elemental_terms(" ".join(self.block_terms), block_ids)
        except ValueError:
            # Named legacy presets can describe a classic IN/OUT range.  On
            # a Flux/DiT profile that range simply has no target modules.
            return False
        return selected(self.block_negative, block_terms) and selected(self.element_negative, self.element_terms)


@dataclass(frozen=True)
class ElementalWeights:
    """Immutable named elemental rules selected by ``lbwe=NAME``.

    ``value_for`` returns ``None`` when no rule matched.  That distinction is
    essential: normal ``lbw`` values remain in effect for untouched modules;
    a matching elemental rule replaces the block value, exactly as the older
    extension documented.
    """

    raw: str
    rules: tuple[ElementalRule, ...]

    def value_for(self, sd_key, is_text_encoder=False, model_profile="auto"):
        block_ids = _block_ids_for_elemental_profile(model_profile, sd_key)
        block_id = block_id_for_key(
            sd_key,
            is_text_encoder=is_text_encoder,
            count=len(block_ids),
            model_profile=model_profile,
        )
        result = None
        for rule in self.rules:
            if rule.matches(sd_key, block_id, block_ids):
                # Last matching rule wins, matching the original extension.
                result = rule.value
        return result


def parse_elemental_weights(value, custom_presets=""):
    """Resolve and validate a named ``lbwe`` preset without loading a model."""
    if value is None or not str(value).strip():
        return None

    raw = str(value).strip()
    presets = _elemental_preset_map(DEFAULT_ELEMENTAL_PRESETS)
    presets.update(_elemental_preset_map(custom_presets))
    directives = presets.get(raw.casefold())
    if directives is None:
        names = ", ".join(sorted(presets)) or "(none)"
        raise ValueError(f"unknown LoRA elemental preset {raw!r}; available: {names}")

    # The named preset itself has already been removed.  Every remaining item
    # must have exactly three fields: blocks, module substring(s), multiplier.
    rules = []
    for item in re.split(r"[,\r\n]", directives):
        item = item.strip()
        if not item:
            continue
        fields = item.split(":")
        if len(fields) != 3:
            raise ValueError(f"invalid elemental rule {item!r}; use blocks:modules:weight")
        block_negative, block_terms = _parse_elemental_selector(fields[0], (), "block")
        _validate_elemental_block_ranges(block_terms)
        element_negative, element_terms = _parse_elemental_selector(fields[1], (), "element")
        rules.append(ElementalRule(
            block_negative=block_negative,
            block_terms=block_terms,
            element_negative=element_negative,
            element_terms=element_terms,
            value=_as_finite_number(fields[2].strip(), "LoRA elemental weight"),
        ))

    if not rules:
        raise ValueError(f"LoRA elemental preset {raw!r} has no rules")
    return ElementalWeights(raw=raw, rules=tuple(rules))


@dataclass(frozen=True)
class BlockWeights:
    """A validated LoRA block-weight table without model tensors."""

    ids: tuple[str, ...]
    values: tuple[float, ...]

    def value_for(self, sd_key, is_text_encoder=False, model_profile="auto"):
        block = block_id_for_key(
            sd_key,
            is_text_encoder=is_text_encoder,
            count=len(self.values),
            model_profile=model_profile,
        )
        if block is None:
            return 1.0

        try:
            return self.values[self.ids.index(block)]
        except ValueError:
            # Legacy short classic masks are expanded to 26 entries with zero,
            # not one. Otherwise NONE/MIDD leak into omitted Conv/ResNet blocks.
            if len(self.values) in (12, 17, 20, 26) and block in BLOCK_IDS_26:
                return 0.0
            # A 61-value table can be used for a flat DiT stack as well as
            # Flux.  In the former case B00 begins at the first model block.
            if len(self.values) == len(BLOCK_IDS_FLUX) and block.startswith("B"):
                index = 3 + int(block[1:])
                return self.values[index] if index < len(self.values) - 1 else 1.0
            return 1.0


def parse_block_weights(value, custom_presets=""):
    """Parse ``lbw=PRESET`` or a 12/17/20/26/61-number block list."""
    if value is None or str(value).strip() == "":
        return None

    raw = str(value).strip()
    presets = _preset_map(DEFAULT_BLOCK_PRESETS)
    presets.update(_preset_map(custom_presets))
    raw = presets.get(raw.casefold(), raw)

    parts = [part.strip() for part in raw.replace("\n", ",").split(",")]
    if any(not part for part in parts):
        raise ValueError("LoRA block weights contain an empty entry; blank is not zero")
    ids = BLOCK_IDS_BY_LENGTH.get(len(parts))
    if ids is None:
        accepted = ", ".join(str(x) for x in sorted(BLOCK_IDS_BY_LENGTH))
        raise ValueError(f"LoRA block weights need {accepted} values, got {len(parts)}")

    return BlockWeights(ids=ids, values=tuple(_as_finite_number(part, "LoRA block weight") for part in parts))


def parse_activation_options(positional, named, custom_presets=""):
    """Parse one LoRA tag into normal multipliers and activation-only state.

    This is intentionally independent from WebUI's ``ExtraNetworkParams`` so
    syntax can be validated without importing torch or a model.
    """
    if not positional:
        raise ValueError("LoRA tag needs a network name as its first argument")

    name = positional[0]
    te_raw = named.get("te", positional[1] if len(positional) > 1 else 1.0)
    te_multiplier, te_schedule = parse_weight_spec(te_raw)

    unet_raw = named.get("unet", positional[2] if len(positional) > 2 else te_raw)
    unet_multiplier, unet_schedule = parse_weight_spec(unet_raw, te_multiplier)

    dyn_raw = named.get("dyn", positional[3] if len(positional) > 3 else None)
    dyn_dim = int(dyn_raw) if dyn_raw is not None and str(dyn_raw).strip() else None

    hr_raw = named.get("hr")
    hr_te_raw = named.get("hrte", hr_raw)
    hr_unet_raw = named.get("hrunet", hr_raw)
    hr_te_multiplier, hr_te_schedule = parse_weight_spec(hr_te_raw, te_multiplier) if hr_te_raw is not None else (None, None)
    hr_unet_multiplier, hr_unet_schedule = parse_weight_spec(hr_unet_raw, unet_multiplier) if hr_unet_raw is not None else (None, None)

    start_step, stop_step = parse_step_range(named.get("start"), named.get("stop"), named.get("step"))
    block_weights = parse_block_weights(named.get("lbw", named.get("w")), custom_presets)

    return name, te_multiplier, unet_multiplier, dyn_dim, {
        "te_schedule": te_schedule,
        "unet_schedule": unet_schedule,
        "hr_te_multiplier": hr_te_multiplier,
        "hr_unet_multiplier": hr_unet_multiplier,
        "hr_te_schedule": hr_te_schedule,
        "hr_unet_schedule": hr_unet_schedule,
        "start_step": start_step,
        "stop_step": stop_step,
        "block_weights": block_weights,
    }


def block_id_for_key(sd_key, is_text_encoder=False, count=26, model_profile="auto"):
    """Map the WebUI layer key to a classic/Flux/DiT block identifier."""
    key = str(sd_key).replace(".", "_").casefold()
    model_profile = str(model_profile or "auto").strip().casefold()

    def text_block():
        if is_text_encoder or "t5" in key or "llm_adapter" in key or "txtfusion" in key or "text_embedder" in key:
            return "T5" if ("t5" in key or "llm_adapter" in key or "txtfusion" in key) else "CLIP"
        if "clip" in key or "transformer" in key and "text" in key:
            return "CLIP"
        return None

    def terminal_block():
        if "final_layer" in key or "final_linear" in key or "final_norm" in key or key.endswith("_last"):
            return "OUT"
        if "_in" in key or "input" in key:
            return "IN"
        return None

    def flat_dit_block():
        # Do not read the ``blocks`` part of a Flux ``double_blocks`` or
        # ``single_blocks`` key as a flat DiT block number.
        if any(marker in key for marker in ("double_blocks", "doubleblocks", "single_blocks", "singleblocks")):
            return None
        match = re.search(r"(?:^|_)(?:transformer_)?(?:layers|blocks)_(\d+)_", key)
        if match:
            return f"B{int(match.group(1)):02}"
        return terminal_block()

    if model_profile == "dit-61":
        return text_block() or flat_dit_block()

    if count == len(BLOCK_IDS_FLUX):
        text = text_block()
        if text:
            return text

        match = re.search(r"(?:double_blocks|doubleblocks)_(\d+)_", key)
        if match:
            return f"D{int(match.group(1)):02}"
        match = re.search(r"(?:single_blocks|singleblocks)_(\d+)_", key)
        if match:
            return f"S{int(match.group(1)):02}"
        # With an explicit Flux profile do not reinterpret a flat DiT key.
        # The ``auto`` fallback retains compatibility with old 61-value tags.
        if model_profile != "flux-61":
            flat = flat_dit_block()
            if flat:
                return flat
        return terminal_block()

    if is_text_encoder or "encoder" in key or "transformer_resblocks" in key or "embedders" in key:
        return "BASE"

    match = re.search(r"diffusion_model_input_blocks_(\d+)_", key)
    if match:
        return f"IN{int(match.group(1)):02}"
    if "diffusion_model_middle_block_" in key:
        return "M00"
    match = re.search(r"diffusion_model_output_blocks_(\d+)_", key)
    if match:
        return f"OUT{int(match.group(1)):02}"
    return None
