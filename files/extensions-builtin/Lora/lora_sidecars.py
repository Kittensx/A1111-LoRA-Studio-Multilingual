"""Bounded, read-only inspection of WebUI LoRA sidecar files.

The browser supplies only an Extra Networks LoRA name. The API resolves that
name first and passes its filename here, so this module never accepts a path
from a client. It recognises the same sibling names as WebUI's Extra Networks
page: ``name.txt``, ``name.description.txt`` and user metadata ``name.json``.

Sidecars are untrusted display hints. Only a small allow-list of fields is
returned and no files are written, executed, or interpreted as prompt syntax.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


SIDECAR_SCHEMA_VERSION = 1
MAX_SIDECAR_BYTES = 256 * 1024
MAX_DESCRIPTION_CHARACTERS = 12_000
MAX_ACTIVATION_TEXT_CHARACTERS = 1_000


class SidecarUnavailable(ValueError):
    """The indexed LoRA file itself is not available for safe inspection."""


def _model_base(filename):
    try:
        model_file = Path(filename).resolve(strict=True)
    except OSError as error:
        raise SidecarUnavailable(f"could not resolve the indexed LoRA file: {error}") from error
    if not model_file.is_file():
        raise SidecarUnavailable("the indexed LoRA path is not a file")
    return model_file.with_suffix("")


def _sibling_file(base, suffix):
    """Return one existing regular sidecar in the LoRA's resolved directory."""
    candidate = base.with_name(base.name + suffix)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        return None
    if resolved.parent != base.parent or not resolved.is_file():
        return None
    return resolved


def _read_bounded_text(path, warnings, label):
    try:
        with path.open("rb") as file:
            raw = file.read(MAX_SIDECAR_BYTES + 1)
    except OSError as error:
        warnings.append(f"Could not read {label}: {error}")
        return None
    if len(raw) > MAX_SIDECAR_BYTES:
        warnings.append(f"Ignored {label} because it exceeds {MAX_SIDECAR_BYTES // 1024} KiB.")
        return None
    return raw.decode("utf-8", errors="replace")


def _display_text(value, limit):
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return ""
    return text if len(text) <= limit else f"{text[:limit]}…"


def _read_user_metadata(base, warnings):
    path = _sibling_file(base, ".json")
    if path is None:
        return None
    raw = _read_bounded_text(path, warnings, "user metadata JSON")
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        warnings.append("Ignored user metadata JSON because it is not valid JSON.")
        return None
    if not isinstance(value, dict):
        warnings.append("Ignored user metadata JSON because its root is not an object.")
        return None
    return value


def _read_description_text(base, warnings):
    for suffix in (".txt", ".description.txt"):
        path = _sibling_file(base, suffix)
        if path is None:
            continue
        value = _read_bounded_text(path, warnings, "description text")
        if value is None:
            continue
        return _display_text(value, MAX_DESCRIPTION_CHARACTERS), "description-text"
    return None, None


def _preferred_weight(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def inspect_lora_sidecars(filename):
    """Return safe display fields from sidecars next to one indexed LoRA.

    ``user_metadata_json`` wins for description when it contains a string,
    matching Extra Networks' user-metadata precedence. A blank JSON
    description deliberately suppresses the text-file fallback, also matching
    the existing editor's behaviour.
    """
    base = _model_base(filename)
    warnings = []
    user_metadata = _read_user_metadata(base, warnings)
    text_description, text_source = _read_description_text(base, warnings)

    if isinstance(user_metadata, dict) and isinstance(user_metadata.get("description"), str):
        description = _display_text(user_metadata["description"], MAX_DESCRIPTION_CHARACTERS)
        description_source = "user-metadata-json"
    else:
        description = text_description
        description_source = text_source

    activation_text = None
    preferred_weight = None
    if isinstance(user_metadata, dict):
        activation_text = _display_text(user_metadata.get("activation text"), MAX_ACTIVATION_TEXT_CHARACTERS)
        preferred_weight = _preferred_weight(user_metadata.get("preferred weight"))

    return {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "description": description,
        "description_source": description_source,
        "activation_text": activation_text,
        "preferred_weight": preferred_weight,
        "warnings": warnings,
    }
