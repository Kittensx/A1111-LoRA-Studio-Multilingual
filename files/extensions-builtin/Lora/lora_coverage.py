"""Read-only LoRA file-key coverage for the Studio profile diagnostics.

This module deliberately reads only the JSON header of a ``.safetensors``
file.  It never imports torch, mmap-loads a tensor, unpickles ``.pt``/``.ckpt``
files, loads a LoRA network, or inspects the active checkpoint.  The report is
therefore evidence about *keys in the selected LoRA file*, not a compatibility
claim about the currently loaded model.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import struct

import lora_runtime


COVERAGE_SCHEMA_VERSION = 1
MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024


class CoverageUnavailable(ValueError):
    """The selected file cannot be safely inspected by the header reader."""


def read_safetensors_tensor_keys(filename):
    """Return tensor names from a bounded safetensors header, without tensors."""
    path = Path(filename)
    if path.suffix.casefold() != ".safetensors":
        raise CoverageUnavailable("file-key inspection is available only for .safetensors files")

    try:
        with path.open("rb") as file:
            raw_length = file.read(8)
            if len(raw_length) != 8:
                raise CoverageUnavailable("file is too small to be a safetensors file")
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length < 2 or header_length > MAX_SAFETENSORS_HEADER_BYTES:
                raise CoverageUnavailable("safetensors header has an unsafe size")
            raw_header = file.read(header_length)
    except OSError as error:
        raise CoverageUnavailable(f"could not read file header: {error}") from error

    if len(raw_header) != header_length:
        raise CoverageUnavailable("safetensors header is truncated")
    try:
        header = json.loads(raw_header)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CoverageUnavailable("file does not contain a valid safetensors JSON header") from error
    if not isinstance(header, dict):
        raise CoverageUnavailable("safetensors header must be an object")

    return tuple(sorted(key for key, value in header.items() if key != "__metadata__" and isinstance(key, str) and isinstance(value, dict)))


def _module_key(tensor_key):
    """Remove a common LoRA tensor-part suffix without decoding tensor data."""
    text = str(tensor_key)
    for marker in (".lora_down.", ".lora_up.", ".lora_A.", ".lora_B.", ".hada_", ".oft_", ".ia3_", ".dora_scale"):
        index = text.find(marker)
        if index >= 0:
            return text[:index]
    return text.partition(".")[0]


def _is_text_key(key):
    text = str(key).casefold().replace(".", "_")
    return any(token in text for token in (
        "lora_te", "text_encoder", "transformer_text", "transformer_resblocks",
        "text_embedder", "llm_adapter", "txtfusion", "clip_",
    ))


def _classic_block_id(key):
    text = str(key).casefold().replace(".", "_")
    if _is_text_key(text):
        return "BASE"

    direct = re.search(r"diffusion_model_input_blocks_(\d+)_", text)
    if direct:
        return f"IN{int(direct.group(1)):02}"
    if "diffusion_model_middle_block_" in text:
        return "M00"
    direct = re.search(r"diffusion_model_output_blocks_(\d+)_", text)
    if direct:
        return f"OUT{int(direct.group(1)):02}"

    match = re.search(r"lora_unet_down_blocks_(\d+)_(?:attentions|resnets)_(\d+)_", text)
    if match:
        return f"IN{1 + int(match.group(1)) * 3 + int(match.group(2)):02}"
    match = re.search(r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv", text)
    if match:
        return f"IN{3 + int(match.group(1)) * 3:02}"
    if "lora_unet_conv_in" in text:
        return "IN00"
    if "lora_unet_mid_block_" in text:
        return "M00"
    match = re.search(r"lora_unet_up_blocks_(\d+)_(?:attentions|resnets)_(\d+)_", text)
    if match:
        return f"OUT{int(match.group(1)) * 3 + int(match.group(2)):02}"
    match = re.search(r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv", text)
    if match:
        return f"OUT{2 + int(match.group(1)) * 3:02}"
    return None


def _flux_block_id(key):
    text = str(key).casefold().replace(".", "_")
    if _is_text_key(text):
        return "T5" if any(token in text for token in ("t5", "llm_adapter", "txtfusion", "text_embedder")) else "CLIP"
    match = re.search(r"(?:double_transformer_blocks|double_blocks|doubleblocks)_(\d+)_", text)
    if match:
        return f"D{int(match.group(1)):02}"
    match = re.search(r"(?:single_transformer_blocks|single_blocks|singleblocks)_(\d+)_", text)
    if match:
        return f"S{int(match.group(1)):02}"
    if any(token in text for token in ("final_layer", "final_linear", "final_norm")):
        return "OUT"
    if "_in" in text or "input" in text:
        return "IN"
    return None


def _dit_block_id(key):
    """Map the flat transformer stack used by the ``dit-61`` profile."""
    text = str(key).casefold().replace(".", "_")
    if _is_text_key(text):
        return "T5" if any(token in text for token in ("t5", "llm_adapter", "txtfusion", "text_embedder")) else "CLIP"
    # A Flux double/single-block key is deliberately not also a DiT layer.
    if any(token in text for token in ("double_transformer_blocks", "single_transformer_blocks", "double_blocks", "single_blocks")):
        return None
    match = re.search(r"(?:^|_)(?:transformer_)?(?:layers|blocks)_(\d+)_", text)
    if match:
        return f"B{int(match.group(1)):02}"
    if any(token in text for token in ("final_layer", "final_linear", "final_norm")):
        return "OUT"
    if "_in" in text or "input" in text:
        return "IN"
    return None


def _profile_coverage(profile_id, block_ids, module_keys):
    mapper = {
        "flux-61": _flux_block_id,
        "dit-61": _dit_block_id,
    }.get(profile_id, _classic_block_id)
    block_set = set(block_ids)
    blocks = {block_id: 0 for block_id in block_ids}
    recognized = 0
    profile_mismatches = 0
    unclassified = 0
    for module_key in module_keys:
        block_id = mapper(module_key)
        if block_id is None:
            unclassified += 1
        elif block_id in block_set:
            recognized += 1
            blocks[block_id] += 1
        else:
            profile_mismatches += 1

    matched_blocks = [block_id for block_id, count in blocks.items() if count]
    return {
        "profile_id": profile_id,
        "matched_block_ids": matched_blocks,
        "missing_block_ids": [block_id for block_id in block_ids if not blocks[block_id]],
        "block_module_counts": {block_id: count for block_id, count in blocks.items() if count},
        "matched_module_count": recognized,
        "profile_mismatch_module_count": profile_mismatches,
        "unclassified_module_count": unclassified,
        "total_block_count": len(block_ids),
    }


def _suggest_profile(coverages):
    scored = []
    for coverage in coverages:
        matched = len(coverage["matched_block_ids"])
        total = coverage["total_block_count"]
        score = matched / total if total else 0.0
        scored.append((score, coverage["matched_module_count"], matched, coverage))
    if not scored:
        return {"profile_id": None, "confidence": "none", "reason": "No known profile block pattern was found in the file keys."}
    score, modules, matched, best = max(scored, key=lambda item: item[:3])
    if best is None or not matched:
        return {"profile_id": None, "confidence": "none", "reason": "No known profile block pattern was found in the file keys."}
    equally_scored = [coverage for candidate_score, candidate_modules, candidate_matched, coverage in scored if (candidate_score, candidate_modules, candidate_matched) == (score, modules, matched)]
    if len(equally_scored) > 1:
        names = ", ".join(coverage["profile_id"] for coverage in equally_scored)
        return {
            "profile_id": None,
            "candidate_profile_ids": [coverage["profile_id"] for coverage in equally_scored],
            "confidence": "ambiguous",
            "reason": f"File keys fit multiple layouts equally: {names}.",
        }
    if matched >= 4 and score >= 0.5:
        confidence = "high"
    elif matched >= 2 and score >= 0.15:
        confidence = "medium"
    else:
        confidence = "low"
    return {
        "profile_id": best["profile_id"],
        "confidence": confidence,
        "reason": f"{matched} of {best['total_block_count']} layout blocks have recognised file keys.",
    }


def coverage_from_tensor_keys(tensor_keys):
    """Create a deterministic, read-only block coverage report from key names."""
    tensor_keys = tuple(str(key) for key in tensor_keys if str(key).strip())
    module_keys = tuple(sorted({_module_key(key) for key in tensor_keys}))
    profiles = []
    for profile_id, block_ids in (
        ("classic-12", lora_runtime.BLOCK_IDS_12),
        ("classic-17", lora_runtime.BLOCK_IDS_17),
        ("classic-20", lora_runtime.BLOCK_IDS_20),
        ("classic-26", lora_runtime.BLOCK_IDS_26),
        ("flux-61", lora_runtime.BLOCK_IDS_FLUX),
        ("dit-61", lora_runtime.BLOCK_IDS_DIT),
    ):
        profiles.append(_profile_coverage(profile_id, block_ids, module_keys))
    return {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "source": "safetensors-header-keys",
        "tensor_key_count": len(tensor_keys),
        "module_key_count": len(module_keys),
        "profiles": profiles,
        "suggestion": _suggest_profile(profiles),
        "notice": "This is file-key coverage only; it does not prove compatibility with the active checkpoint.",
    }


def inspect_safetensors_file(filename):
    """Safely inspect one LoRA file's header and return its coverage report."""
    return coverage_from_tensor_keys(read_safetensors_tensor_keys(filename))
