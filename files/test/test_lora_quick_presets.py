"""Check browser quick recipes against the native parser without Torch."""
import json
import pathlib
import shutil
import subprocess
import sys
from itertools import product

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "extensions-builtin" / "Lora"))
import lora_schema  # noqa: E402
import lora_runtime  # noqa: E402
sys.path.pop(0)


def test_every_builtin_quick_recipe_roundtrips_through_native_schema():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    script = """
const model = require(process.argv[1]);
require(process.argv[2]);
const presets = require(process.argv[3]);
const base = model.parseTag('<lora:test:te=0.6:unet=0.8:dyn=16:hr=0.3:plugin=keep>');
process.stdout.write(JSON.stringify(presets.listQuickPresets().map(choice => {
    const spec = presets.applyQuickPreset(base, choice);
    return {id: choice.id, spec, tag: model.serializeTag(spec)};
})));
"""
    result = subprocess.run(
        [node, "-e", script, *(str(ROOT / "javascript" / name) for name in (
            "loraStudioModel.js", "loraStudioBlocks.js", "loraStudioPresets.js",
        ))], capture_output=True, text=True, check=True,
    )
    rows = json.loads(result.stdout)
    for row in rows:
        native = lora_schema.parse_lora_tag(row["tag"])
        assert native.to_contract() == row["spec"], row["id"]
        assert lora_schema.serialize_lora_tag(native) == row["tag"], row["id"]

    fade = next(row for row in rows if row["id"] == "mode:fade-in")
    spec = lora_schema.parse_lora_tag(fade["tag"])
    _, te, unet, _, settings = spec.to_runtime_values()
    assert te == 0.6
    assert settings["hr_unet_multiplier"] == 0.3
    assert lora_runtime.evaluate_weight_spec(settings["unet_schedule"], unet, 0, 21) == 0
    assert lora_runtime.evaluate_weight_spec(settings["unet_schedule"], unet, 7, 21) == 0.8


def test_combined_controls_match_native_parser_including_empty_and_mixed_slots():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    tags = [
        "<lora:test:" + ":".join(part for part in parts if part) + ">"
        for parts in product(
            ("", "te=0.3", "te=0@0,0.6@1"),
            ("", "unet=0", "unet=-0.8", "unet=0.2@0,0.8@1"),
            ("", "dyn=1", "dyn=16"),
            ("", "hr=0.4", "hrte=0.2:hrunet=0@0,0.6@1"),
            ("", "start=0.2:stop=0.8:lbw=OUTS:lbwe=ATTNDEEPOFF:plugin=keep"),
        )
    ]
    tags += [
        "<lora:test:te=:unet=:dyn=>", "<lora:test:::16>",
        "<lora:test:0.3:unet=:dyn=16:hr=>",
        "<lora:test:0.3:-0.8:16>", "<lora:test:9:8:4:te=0.3:unet=-0.8:dyn=16>",
        "<lora:test:dyn=16:unet=0.8:te=0.3>",
        "<lora:test:0.7:TE=0.3:UNET=0.8:DYN=16>",
    ]
    script = """
const model = require(process.argv[1]);
const tags = JSON.parse(require('fs').readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(tags.map(tag => {
    const spec = model.parseTag(tag);
    return {spec, tag: model.serializeTag(spec)};
})));
"""
    result = subprocess.run(
        [node, "-e", script, str(ROOT / "javascript" / "loraStudioModel.js")],
        input=json.dumps(tags), capture_output=True, text=True, check=True,
    )
    rows = json.loads(result.stdout)
    assert len(rows) == len(tags) == 223
    for tag, row in zip(tags, rows):
        native = lora_schema.parse_lora_tag(tag)
        assert native.to_contract() == row["spec"], tag
        assert lora_schema.serialize_lora_tag(native) == row["tag"], tag
        assert lora_schema.parse_lora_tag(row["tag"]).to_runtime_values() == native.to_runtime_values(), tag
