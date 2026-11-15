import json
import pathlib
import shutil
import struct
import subprocess
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
LORA_DIR = ROOT / "extensions-builtin" / "Lora"
CONTRACT_PATH = ROOT / "test" / "lora_tag_contract.json"
sys.path.insert(0, str(LORA_DIR))
import lora_presets  # noqa: E402
import lora_schema  # noqa: E402
import lora_advanced  # noqa: E402
import lora_coverage  # noqa: E402
import lora_sidecars  # noqa: E402
sys.path.pop(0)


def read_contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_python_schema_matches_tag_contract():
    contract = read_contract()
    assert contract["schema_version"] == lora_schema.SCHEMA_VERSION

    for case in contract["cases"]:
        spec = lora_schema.parse_lora_tag(case["tag"])
        assert spec.to_contract() == case["normalized"], case["id"]
        assert lora_schema.serialize_lora_tag(spec) == case["serialized"], case["id"]
        reparsed = lora_schema.parse_lora_tag(case["serialized"])
        assert reparsed.to_contract() == case["normalized"], case["id"]


def test_runtime_adapter_retains_existing_extra_network_semantics():
    spec = lora_schema.parse_lora_items(
        ["portrait", "0@0,1@0.5", "unet=0.8", "dyn=32", "hr=0.4", "step=0.2-0.8", "lbw=OUTS"],
    )

    name, te, unet, dyn, settings = spec.to_runtime_values()
    assert (name, te, unet, dyn) == ("portrait", 0.0, 0.8, 32)
    assert settings["te_schedule"] == ((0.0, 0.0), (0.5, 1.0))
    assert settings["hr_te_multiplier"] == settings["hr_unet_multiplier"] == 0.4
    assert settings["start_step"] == 0.2
    assert settings["stop_step"] == 0.8
    assert settings["block_weights"].value_for("diffusion_model_output_blocks_8_0", False) == 1


def test_runtime_adapter_passes_elemental_override_only_when_a_tag_requests_it():
    spec = lora_schema.parse_lora_tag(
        "<lora:portrait:lbw=OUTS:lbwe=ATTNDEEPOFF:profile=classic-26>"
    )

    _, _, _, _, settings = spec.to_runtime_values()
    assert settings["model_profile"] == "classic-26"
    assert settings["elemental_weights"].value_for(
        "diffusion_model_input_blocks_5_1_transformer_blocks_0_attn_to_q",
        model_profile=settings["model_profile"],
    ) == 0


@pytest.mark.parametrize("items", [
    ["plain", "0.7"],
    ["curve", "0@0,1@1", "unet=0.8", "dyn=16"],
    ["hires", "te=0.3", "unet=0.6", "hrte=0.2", "hrunet=0.4", "start=2", "stop=8"],
    ["blocks", "0.5", "lbw=OUTS", "plugin=preserve"],
])
def test_schema_runtime_adapter_matches_previous_advanced_parser(items):
    positional = []
    named = {}
    for item in items:
        fields = item.split("=", 2)
        if len(fields) == 2:
            named[fields[0]] = fields[1]
        else:
            positional.append(item)

    expected = lora_advanced.parse_activation_options(positional, named)
    actual = lora_schema.parse_lora_items(items).to_runtime_values()

    assert actual == expected


def test_unknown_arguments_and_tag_aliases_survive_schema_round_trip():
    spec = lora_schema.parse_lora_tag("<loractl:style:0.5:0.5:16:future-pos:plugin=a:plugin=b>")

    assert spec.tag_type == "loractl"
    assert spec.passthrough_positional == ("future-pos",)
    assert spec.passthrough_named == (("plugin", "a"), ("plugin", "b"))
    assert lora_schema.serialize_lora_tag(spec).endswith(":future-pos:plugin=a:plugin=b>")
    assert lora_schema.serialize_lora_tag(spec, canonical_tag_type=True).startswith("<lora:")


def test_profiles_and_preset_schema_are_typed_without_persistence():
    assert lora_schema.get_model_profile("flux-61").block_ids[0] == "CLIP"
    assert lora_schema.get_model_profile("dit-61").block_ids[3] == "B00"
    with pytest.raises(ValueError, match="unknown LoRA model profile"):
        lora_schema.get_model_profile("imaginary")

    preset = lora_presets.LoraControlPreset(
        preset_id="portrait-soft",
        title="Portrait soft",
        model_profile="classic-26",
        text_encoder=lora_schema.WeightSpec.parse("0.4"),
        unet=lora_schema.WeightSpec.parse("0.8"),
        active_range=lora_schema.StepRange(0.2, 0.8),
    )
    assert preset.to_contract()["weights"]["unet"] == {"kind": "scalar", "value": 0.8}


def test_safetensors_header_coverage_is_torch_free_and_never_claims_checkpoint_compatibility(tmp_path):
    keys = ["transformer_text_model_encoder_layers_0"]
    keys.extend(f"diffusion_model_input_blocks_{index}_0" for index in (1, 2, 4, 5, 7, 8))
    keys.append("diffusion_model_middle_block_0")
    keys.extend(f"diffusion_model_output_blocks_{index}_0" for index in range(3, 12))
    header = {"__metadata__": {"ss_base_model_version": "sd_v1"}}
    header.update({key: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for key in keys})
    path = tmp_path / "coverage.safetensors"
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)

    report = lora_coverage.inspect_safetensors_file(path)
    classic_17 = next(item for item in report["profiles"] if item["profile_id"] == "classic-17")
    assert report["tensor_key_count"] == len(keys)
    assert report["suggestion"] == {
        "profile_id": "classic-17",
        "confidence": "high",
        "reason": "17 of 17 layout blocks have recognised file keys.",
    }
    assert classic_17["missing_block_ids"] == []
    assert classic_17["matched_module_count"] == 17
    assert "does not prove compatibility" in report["notice"]

    with pytest.raises(lora_coverage.CoverageUnavailable, match="only for .safetensors"):
        lora_coverage.inspect_safetensors_file(tmp_path / "unsafe.ckpt")


def test_file_key_coverage_distinguishes_flux_from_flat_dit_layouts():
    report = lora_coverage.coverage_from_tensor_keys([
        "transformer.double_transformer_blocks.0.attn.to_q.lora_A.weight",
        "transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight",
        "transformer.final_layer.adaLN_modulation.lora_A.weight",
    ])

    assert report["suggestion"] == {
        "profile_id": "flux-61",
        "confidence": "low",
        "reason": "3 of 61 layout blocks have recognised file keys.",
    }

    dit = lora_coverage.coverage_from_tensor_keys([
        "transformer.blocks.0.attn.to_q.lora_A.weight",
        "transformer.layers.1.mlp.fc1.lora_A.weight",
        "transformer.final_layer.adaLN_modulation.lora_A.weight",
    ])
    assert dit["suggestion"] == {
        "profile_id": "dit-61",
        "confidence": "low",
        "reason": "3 of 61 layout blocks have recognised file keys.",
    }


def test_file_key_coverage_reports_an_unsupported_layout_without_guessing():
    report = lora_coverage.coverage_from_tensor_keys([
        "vendor_adapter.custom_stack.layer_0.lora_A.weight",
        "vendor_adapter.custom_stack.layer_1.lora_B.weight",
    ])

    assert report["suggestion"] == {
        "profile_id": None,
        "confidence": "none",
        "reason": "No known profile block pattern was found in the file keys.",
    }
    assert all(item["matched_module_count"] == 0 for item in report["profiles"])


def test_lora_sidecars_are_bounded_allowlisted_and_follow_webui_description_precedence(tmp_path):
    model = tmp_path / "portrait.safetensors"
    model.write_bytes(b"not loaded")
    (tmp_path / "portrait.txt").write_text("Text description", encoding="utf-8")
    (tmp_path / "portrait.description.txt").write_text("Lower-priority description", encoding="utf-8")
    (tmp_path / "portrait.json").write_text(json.dumps({
        "description": "User metadata description",
        "activation text": "portrait, studio light",
        "preferred weight": 0.65,
        "notes": "must never leave the local sidecar",
        "unknown": "must not leave the local sidecar",
    }), encoding="utf-8")

    report = lora_sidecars.inspect_lora_sidecars(model)
    assert report == {
        "schema_version": 1,
        "description": "User metadata description",
        "description_source": "user-metadata-json",
        "activation_text": "portrait, studio light",
        "preferred_weight": 0.65,
        "warnings": [],
    }

    (tmp_path / "portrait.json").write_text("not json", encoding="utf-8")
    fallback = lora_sidecars.inspect_lora_sidecars(model)
    assert fallback["description"] == "Text description"
    assert fallback["description_source"] == "description-text"
    assert fallback["activation_text"] is None
    assert fallback["preferred_weight"] is None
    assert fallback["warnings"] == ["Ignored user metadata JSON because it is not valid JSON."]

    (tmp_path / "portrait.json").write_bytes(b"x" * (lora_sidecars.MAX_SIDECAR_BYTES + 1))
    oversized = lora_sidecars.inspect_lora_sidecars(model)
    assert oversized["description"] == "Text description"
    assert oversized["warnings"] == ["Ignored user metadata JSON because it exceeds 256 KiB."]


def test_browser_model_matches_shared_contract():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed; Python contract remains covered")

    script = """
const fs = require('fs');
const model = require(process.argv[1]);
const contract = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const result = contract.cases.map((item) => {
    const spec = model.parseTag(item.tag);
    return {id: item.id, normalized: spec, serialized: model.serializeTag(spec)};
});
process.stdout.write(JSON.stringify(result));
"""
    result = subprocess.run(
        [node, "-e", script, str(ROOT / "javascript" / "loraStudioModel.js"), str(CONTRACT_PATH)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    expected = [
        {"id": case["id"], "normalized": case["normalized"], "serialized": case["serialized"]}
        for case in read_contract()["cases"]
    ]
    assert actual == expected


def test_browser_model_replaces_only_the_selected_tag_occurrence():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed; browser span model is not runnable")

    script = """
const model = require(process.argv[1]);
const prompt = 'one <lora:duplicate:0.2> two <loractl:duplicate:0.8:dyn=8> three';
const tags = model.findTags(prompt);
const replacement = model.parseTag('<loractl:duplicate:te=1:unet=1:dyn=8>');
let staleRejected = false;
try { model.replaceTagSpan('changed ' + prompt, tags[1], '<lora:x:1>'); } catch (_) { staleRejected = true; }
process.stdout.write(JSON.stringify({
    count: tags.length,
    first: tags[0].tag,
    edited: model.replaceTagSpan(prompt, tags[1], model.serializeTag(replacement)),
    staleRejected,
}));
"""
    result = subprocess.run(
        [node, "-e", script, str(ROOT / "javascript" / "loraStudioModel.js")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "count": 2,
        "first": "<lora:duplicate:0.2>",
        "edited": "one <lora:duplicate:0.2> two <loractl:duplicate:te=1:unet=1:dyn=8> three",
        "staleRejected": True,
    }


def test_saved_legacy_prompt_tags_migrate_without_touching_surrounding_text():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed; browser prompt migration is not runnable")

    prompt = (
        "cinematic, grainy <lora:portrait:0.55> and "
        "<lora:film:0.4:0.8:16> with "
        "<loractl:motion:0@0,1@1:unet=0.7:hr=0.35:step=0.2-0.8:plugin=keep> "
        "plus <lyco:detail:0.25:custom=a:custom=b>; keep <not-a-lora:x> unchanged"
    )
    legacy_tags = [
        "<lora:portrait:0.55>",
        "<lora:film:0.4:0.8:16>",
        "<loractl:motion:0@0,1@1:unet=0.7:hr=0.35:step=0.2-0.8:plugin=keep>",
        "<lyco:detail:0.25:custom=a:custom=b>",
    ]
    expected_serialized = [lora_schema.serialize_lora_tag(lora_schema.parse_lora_tag(tag)) for tag in legacy_tags]

    script = """
const model = require(process.argv[1]);
const prompt = process.argv[2];
const tags = model.findTags(prompt);
const serialized = tags.map((entry) => model.serializeTag(entry.spec));
const rewritten = model.replaceTagSpan(prompt, tags[2], serialized[2]);
process.stdout.write(JSON.stringify({tags: tags.map((entry) => entry.tag), serialized, rewritten}));
"""
    result = subprocess.run(
        [node, "-e", script, str(ROOT / "javascript" / "loraStudioModel.js"), prompt],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    assert actual["tags"] == legacy_tags
    assert actual["serialized"] == expected_serialized
    assert actual["rewritten"] == prompt.replace(legacy_tags[2], expected_serialized[2], 1)
    assert "<not-a-lora:x>" in actual["rewritten"]


def test_browser_model_rejects_the_same_incomplete_step_range_as_native_parser():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed; browser parser is not runnable")

    with pytest.raises(ValueError, match="step must use start-stop"):
        lora_schema.parse_lora_tag("<lora:broken:step=0.2- >")

    script = """
const model = require(process.argv[1]);
let rejected = false;
try { model.parseTag('<lora:broken:step=0.2- >'); } catch (error) { rejected = /step must use start-stop/.test(error.message); }
process.stdout.write(JSON.stringify({rejected}));
"""
    result = subprocess.run(
        [node, "-e", script, str(ROOT / "javascript" / "loraStudioModel.js")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"rejected": True}


def test_browser_model_number_serialization_matches_native_schema():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed; browser serializer is not runnable")

    values = ["0.30000000000000004", "0.00001", "0.0000001", "100000000000000000000"]
    expected = [
        lora_schema.serialize_lora_tag(lora_schema.parse_lora_tag(f"<lora:precision:te={value}:unet={value}>"))
        for value in values
    ]
    script = """
const model = require(process.argv[1]);
const values = JSON.parse(process.argv[2]);
process.stdout.write(JSON.stringify(values.map((value) => {
    const spec = model.parseTag(`<lora:precision:te=${value}:unet=${value}>`);
    return model.serializeTag(spec);
})));
"""
    result = subprocess.run(
        [node, "-e", script, str(ROOT / "javascript" / "loraStudioModel.js"), json.dumps(values)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected


def test_browser_metadata_helpers_extract_safe_deterministic_trigger_suggestions():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed; metadata helper is not runnable")

    item = {
        "name": "portrait/adapter",
        "alias": "portrait-adapter",
        "metadata": {
            "ssmd_display_name": "Portrait Adapter",
            "modelspec.description": "Read-only model description",
            "ss_base_model_version": "sdxl_base_v1-0",
            "sshs_model_hash": "1234567890abcdef",
            "ss_tag_frequency": '{"dataset": {"blue eyes": 12, "smile": 4}}',
            "trigger_words": "portrait, detailed face, portrait",
            "modelspec.tags": "photo; portrait",
            "broken": "{not-json",
        },
    }
    script = """
const metadata = require(process.argv[1]);
const item = JSON.parse(process.argv[2]);
process.stdout.write(JSON.stringify(metadata.describeLora(item)));
"""
    result = subprocess.run(
        [node, "-e", script, str(ROOT / "javascript" / "loraStudioMetadata.js"), json.dumps(item)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "title": "Portrait Adapter",
        "description": "Read-only model description",
        "baseModel": "sdxl_base_v1-0",
        "hash": "1234567890abcdef",
        "triggers": ["blue eyes", "smile", "detailed face", "photo", "portrait"],
    }


def test_browser_curve_helpers_match_native_piecewise_linear_semantics():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed; curve helper is not runnable")

    script = """
const curves = require(process.argv[1]);
const source = {kind: 'curve', points: [
    {position: 0, value: 0}, {position: 0.5, value: 1}, {position: 1, value: 0},
]};
let absoluteRejected = false;
try { curves.insertNormalizedMidpoint([{position: 2, value: 0}, {position: 4, value: 1}]); } catch (error) { absoluteRejected = /absolute sampler steps/.test(error.message); }
const chart = curves.chart(source, 5);
process.stdout.write(JSON.stringify({
    interpolation: curves.valueAt(source.points, 0.25),
    normalized: chart.x.normalized,
    samples: chart.samples.map((point) => point.value),
    midpoint: curves.insertNormalizedMidpoint(source.points),
    duplicate: curves.normalizePoints([{position: 0, value: 0.2}, {position: 0, value: 0.7}]),
    absoluteRejected,
}));
"""
    result = subprocess.run(
        [node, "-e", script, str(ROOT / "javascript" / "loraStudioCurves.js")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "interpolation": 0.5,
        "normalized": True,
        "samples": [0, 0.5, 1, 0.5, 0],
        "midpoint": [
            {"position": 0, "value": 0},
            {"position": 0.25, "value": 0.5},
            {"position": 0.5, "value": 1},
            {"position": 1, "value": 0},
        ],
        "duplicate": [{"position": 0, "value": 0.7}],
        "absoluteRejected": True,
    }


def test_lora_studio_scripts_can_load_before_ui_declares_opts():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed; browser load order is not runnable")

    script = """
const fs = require('fs');
const vm = require('vm');
global.window = global;
global.document = {
    addEventListener() {},
    getElementsByTagName() { return []; },
    querySelector() { return null; },
};
global.MutationObserver = function() { this.observe = function() {}; };
function classic(file) {
    vm.runInThisContext(fs.readFileSync(file, 'utf8'), {filename: file});
}
for (const file of process.argv.slice(1)) classic(file);
if (typeof opts !== 'object' || !uiLoadedCallbacks.length) throw new Error('Studio scripts did not register');
process.stdout.write(JSON.stringify({ok: true}));
"""
    files = [
        ROOT / "script.js",
        ROOT / "javascript" / "loraStudioBlocks.js",
        ROOT / "javascript" / "loraStudioCurves.js",
        ROOT / "javascript" / "loraStudioMetadata.js",
        ROOT / "javascript" / "loraStudioModel.js",
        ROOT / "javascript" / "loraStudioPresets.js",
        ROOT / "javascript" / "loraStudioView.js",
        ROOT / "javascript" / "ui.js",
    ]
    result = subprocess.run(
        [node, "-e", script, *(str(file) for file in files)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"ok": True}


def test_studio_mounts_below_the_complete_top_row_not_in_its_flex_columns():
    source = (ROOT / "javascript" / "loraStudioView.js").read_text(encoding="utf-8")

    assert 'querySelector(`#${tabname}_toprow`) || promptContainer' in source
    assert 'mountTarget.insertAdjacentElement("afterend", root)' in source
    assert 'promptContainer.insertAdjacentElement("afterend", root)' not in source


def test_prompt_tag_quick_tune_uses_the_existing_draft_transaction():
    source = (ROOT / "javascript" / "loraStudioView.js").read_text(encoding="utf-8")

    assert "function openQuickTuneForTagAtCaret" in source
    assert 'const draft = draftFor(tabname, "replace")' in source
    assert 'button("Apply", "lora-studio-quick-tune-apply", () => applyDraft(tabname))' in source
    assert 'button("Cancel", "lora-studio-quick-tune-cancel", () => discardDraft(tabname))' in source
    assert 'textarea.addEventListener("contextmenu", (event) => openQuickTuneForTagAtCaret(tabname, event))' in source


def test_browser_block_profiles_and_builtin_presets_match_native_runtime():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed; block editor contract is not runnable")

    script = """
require(process.argv[1]);
const blocks = require(process.argv[2]);
const profiles = blocks.listProfiles().map((profile) => ({id: profile.id, blockIds: profile.blockIds}));
const resolved = blocks.resolveBlockWeights('OUTS');
process.stdout.write(JSON.stringify({
    profiles,
    exact: blocks.parseExactBlockWeights('0, 0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1'),
    outs: resolved.values,
    groups: blocks.groupBlocks('classic-17'),
    serialized: blocks.serializeExactBlockWeights(Array(12).fill(0.30000000000000004)),
}));
"""
    result = subprocess.run(
        [
            node, "-e", script,
            str(ROOT / "javascript" / "loraStudioModel.js"),
            str(ROOT / "javascript" / "loraStudioBlocks.js"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    assert actual["profiles"] == [
        {"id": profile_id, "blockIds": list(profile.block_ids)}
        for profile_id, profile in lora_schema.MODEL_PROFILES.items()
    ]
    assert actual["exact"] == [0, 0.5, *([1] * 10)]
    assert actual["outs"] == list(lora_advanced.parse_block_weights("OUTS").values)
    assert actual["groups"] == [
        {"label": "Text / base", "blockIds": ["BASE"]},
        {"label": "Input", "blockIds": ["IN01", "IN02", "IN04", "IN05", "IN07", "IN08"]},
        {"label": "Middle", "blockIds": ["M00"]},
        {"label": "Output", "blockIds": ["OUT03", "OUT04", "OUT05", "OUT06", "OUT07", "OUT08", "OUT09", "OUT10", "OUT11"]},
    ]
    assert actual["serialized"] == ",".join(["0.3"] * 12)


def test_versioned_preset_library_validates_and_round_trips_without_torch():
    value = {
        "schema_version": 1,
        "presets": [{
            "schema_version": 1,
            "preset_id": "portrait-soft",
            "title": "Portrait soft",
            "model_profile": "classic-17",
            "weights": {
                "te": {"kind": "scalar", "value": 0.35},
                "unet": {"kind": "curve", "points": [{"position": 0, "value": 0.4}, {"position": 1, "value": 0.8}]},
            },
            "dyn_dim": 16,
            "active_range": {"start": 0.2, "stop": 0.8},
            "block_weights": {"raw": "OUTS"},
        }],
    }
    presets = lora_presets.parse_preset_library(value)
    assert presets[0].unet.serialize() == "0.4@0,0.8@1"
    assert presets[0].block_weights.weights.values == lora_advanced.parse_block_weights("OUTS").values
    assert presets[0].hires is None
    assert lora_presets.parse_preset_library(lora_presets.serialize_preset_library(presets))[0].to_contract() == presets[0].to_contract()
    migrated = json.loads(lora_presets.serialize_preset_library(presets))
    assert migrated["schema_version"] == 3
    assert migrated["presets"][0]["schema_version"] == 3
    assert migrated["presets"][0]["hires"] is None
    assert migrated["presets"][0]["elemental_weights"] is None
    assert migrated["presets"][0]["favorite"] is False

    v2 = {
        "schema_version": 2,
        "presets": [{
            **value["presets"][0],
            "schema_version": 2,
            "hires": {"te": {"kind": "scalar", "value": 0.25}, "unet": None},
            "favorite": True,
        }],
    }
    assert lora_presets.parse_preset_library(v2)[0].hires.text_encoder.serialize() == "0.25"
    assert lora_presets.parse_preset_library(v2)[0].favorite is True
    with pytest.raises(ValueError, match="v2 needs a hires field"):
        lora_presets.parse_preset_library({"schema_version": 2, "presets": [{**value["presets"][0], "schema_version": 2}]})
    with pytest.raises(ValueError, match="favorite must be a boolean"):
        lora_presets.parse_preset_library({"schema_version": 2, "presets": [{**v2["presets"][0], "favorite": "yes"}]})

    v3 = {
        "schema_version": 3,
        "presets": [{
            **v2["presets"][0],
            "schema_version": 3,
            "elemental_weights": {"raw": "ATTNDEEPOFF"},
        }],
    }
    assert lora_presets.parse_preset_library(v3)[0].elemental_weights.raw == "ATTNDEEPOFF"
    with pytest.raises(ValueError, match="v3 needs an elemental_weights field"):
        lora_presets.parse_preset_library({"schema_version": 3, "presets": [{**v2["presets"][0], "schema_version": 3}]})

    future_profile = {**value, "presets": [{**value["presets"][0], "model_profile": "vendor-next-99"}]}
    assert lora_presets.parse_preset_library(future_profile)[0].model_profile == "vendor-next-99"

    duplicate = {**value, "presets": [value["presets"][0], {**value["presets"][0], "title": "Duplicate"}]}
    with pytest.raises(ValueError, match="identifiers must be unique"):
        lora_presets.parse_preset_library(duplicate)
    with pytest.raises(ValueError, match="curve needs at least one point"):
        lora_presets.LoraControlPreset.from_contract({**value["presets"][0], "weights": {"te": {"kind": "curve", "points": []}, "unet": None}})
    with pytest.raises(ValueError, match="120 characters"):
        lora_presets.LoraControlPreset.from_contract({**value["presets"][0], "title": "x" * 121})


def test_browser_preset_snapshot_is_accepted_by_native_preset_schema():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed; browser preset contract is not runnable")

    script = """
const model = require(process.argv[1]);
const presets = require(process.argv[2]);
const source = model.parseTag('<lora:source:te=0.3:unet=0.7:hrte=0.2:hrunet=0.4:dyn=16:start=0.2:stop=0.8:lbw=OUTS:lbwe=ATTNDEEPOFF:profile=classic-17>');
const preset = presets.createPreset('Studio snapshot', source, 'studio-snapshot');
const applied = presets.applyPreset(model.parseTag('<lora:target:1:hrte=0.9:hrunet=0.8>'), preset);
const restored = presets.parseEnvelope(JSON.stringify(presets.envelope([preset])));
const legacy = {...preset, schema_version: 1}; delete legacy.model_profile; delete legacy.hires;
const restoredLegacy = presets.parseEnvelope({schema_version: 1, presets: [legacy]});
const appliedLegacy = presets.applyPreset(model.parseTag('<lora:target:1:hrte=0.9:hrunet=0.8>'), legacy);
const explicitInherit = {...preset, hires: {te: null, unet: null}};
const appliedExplicitInherit = presets.applyPreset(model.parseTag('<lora:target:1:hrte=0.9:hrunet=0.8>'), explicitInherit);
const blankProfile = {...preset, model_profile: '   '};
const restoredBlankProfile = presets.parseEnvelope({schema_version: 2, presets: [blankProfile]});
let duplicateRejected = false;
try { presets.parseEnvelope({schema_version: 2, presets: [preset, {...preset, title: 'Duplicate'}]}); } catch (error) { duplicateRejected = /identifiers must be unique/.test(error.message); }
let badFavoriteRejected = false;
try { presets.parseEnvelope({schema_version: 2, presets: [{...preset, favorite: 'yes'}]}); } catch (error) { badFavoriteRejected = /favorite must be true or false/.test(error.message); }
process.stdout.write(JSON.stringify({preset, applied, restored, restoredLegacy, appliedLegacy, appliedExplicitInherit, restoredBlankProfile, duplicateRejected, badFavoriteRejected}));
"""
    result = subprocess.run(
        [node, "-e", script, str(ROOT / "javascript" / "loraStudioModel.js"), str(ROOT / "javascript" / "loraStudioPresets.js")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    preset = lora_presets.LoraControlPreset.from_contract(actual["preset"])
    assert preset.title == "Studio snapshot"
    assert preset.to_contract()["block_weights"] == {"raw": "OUTS"}
    assert preset.to_contract()["elemental_weights"] == {"raw": "ATTNDEEPOFF"}
    assert actual["applied"]["network_name"] == "target"
    assert actual["applied"]["weights"]["unet"] == {"kind": "scalar", "value": 0.7}
    assert actual["applied"]["hires"] == {
        "te": {"kind": "scalar", "value": 0.2},
        "unet": {"kind": "scalar", "value": 0.4},
    }
    assert actual["preset"]["favorite"] is False
    assert actual["restored"] == {"schema_version": 3, "presets": [actual["preset"]]}
    assert actual["restoredLegacy"]["presets"][0]["model_profile"] == "auto"
    assert actual["restoredLegacy"]["presets"][0]["hires"] is None
    assert actual["appliedLegacy"]["hires"] == {
        "te": {"kind": "scalar", "value": 0.9},
        "unet": {"kind": "scalar", "value": 0.8},
    }
    assert actual["appliedExplicitInherit"]["hires"] == {"te": None, "unet": None}
    assert actual["restoredBlankProfile"]["presets"][0]["model_profile"] == "auto"
    assert actual["duplicateRejected"] is True
    assert actual["badFavoriteRejected"] is True
