import pathlib
import sys

import pytest


LORA_DIR = pathlib.Path(__file__).resolve().parents[1] / "extensions-builtin" / "Lora"
sys.path.insert(0, str(LORA_DIR))
import lora_advanced  # noqa: E402
sys.path.pop(0)


def test_weight_curve_uses_fractional_sampling_positions():
    initial, curve = lora_advanced.parse_weight_spec("0@0, 1@1")

    assert initial == 0
    assert lora_advanced.evaluate_weight_spec(curve, initial, 0, 5) == 0
    assert lora_advanced.evaluate_weight_spec(curve, initial, 2, 5) == 0.5
    assert lora_advanced.evaluate_weight_spec(curve, initial, 4, 5) == 1


def test_weight_curve_supports_absolute_steps_and_last_duplicate_wins():
    initial, curve = lora_advanced.parse_weight_spec("0.1@2;0.25@2;1@6")

    assert initial == 0.25
    assert lora_advanced.evaluate_weight_spec(curve, initial, 4, 20) == 0.625


def test_step_range_accepts_fractional_and_absolute_values():
    start, stop = lora_advanced.parse_step_range(value="0.25-0.75")

    assert not lora_advanced.step_is_active(start, stop, 0, 5)
    assert lora_advanced.step_is_active(start, stop, 1, 5)
    assert lora_advanced.step_is_active(start, stop, 2, 5)
    assert not lora_advanced.step_is_active(start, stop, 4, 5)
    assert lora_advanced.step_is_active(2, 4, 3, 10)


def test_block_weights_map_classic_flux_and_dit_layer_keys():
    classic = lora_advanced.parse_block_weights(",".join(str(x) for x in range(26)))
    flux = lora_advanced.parse_block_weights(",".join(str(x) for x in range(61)))

    assert classic.value_for("diffusion_model_input_blocks_04_1_transformer_blocks_0", False) == 5
    assert classic.value_for("diffusion_model_middle_block_1", False) == 13
    assert classic.value_for("transformer_text_model_encoder_layers_1", True) == 0
    assert flux.value_for("double_blocks_3_img_attn_qkv", False) == 6
    assert flux.value_for("single_blocks_4_linear1", False) == 26
    assert flux.value_for("layers_8_self_attn", False) == 11
    assert flux.value_for("layers_8_self_attn", False, model_profile="dit-61") == 11
    assert flux.value_for("double_blocks_3_img_attn_qkv", False, model_profile="dit-61") == 1


def test_block_weight_presets_and_validation():
    assert lora_advanced.parse_block_weights("NONE").value_for("diffusion_model_middle_block_0") == 0
    assert lora_advanced.parse_block_weights("MINE", "MINE:1,1,1,1,1,1,1,1,1,1,1,1").values[-1] == 1

    with pytest.raises(ValueError, match="weight@position"):
        lora_advanced.parse_weight_spec("0@")
    with pytest.raises(ValueError, match="need 12"):
        lora_advanced.parse_block_weights("1,1")


def test_elemental_presets_override_only_matching_modules_without_mutating_block_tables():
    elemental = lora_advanced.parse_elemental_weights("ATTNDEEPOFF")

    attention_key = "diffusion_model_input_blocks_5_1_transformer_blocks_0_attn_to_q"
    resnet_key = "diffusion_model_input_blocks_5_0_in_layers_2"
    assert elemental.value_for(attention_key) == 0
    assert elemental.value_for(resnet_key) is None

    custom = lora_advanced.parse_elemental_weights(
        "FOCUS", "FOCUS:IN05-OUT05:attn:0.25,IN05:attn_to_q:0.5"
    )
    assert custom.value_for(attention_key) == 0.5  # last matching rule wins
    assert custom.value_for("diffusion_model_output_blocks_5_1_transformer_blocks_0_attn_to_q") == 0.25
    assert custom.value_for("diffusion_model_output_blocks_5_0_resnets_0") is None
    assert elemental.value_for("transformer.layers.8.self_attn", model_profile="dit-61") is None

    with pytest.raises(ValueError, match="unknown LoRA elemental preset"):
        lora_advanced.parse_elemental_weights("MISSING")
    with pytest.raises(ValueError, match="unknown elemental block range"):
        lora_advanced.parse_elemental_weights("BROKEN", "BROKEN:NOT_A_BLOCK-IN05:attn:0")


def test_activation_options_keep_prompt_state_separate_from_payload_values():
    name, te, unet, dyn, settings = lora_advanced.parse_activation_options(
        ["portrait", "0@0,1@0.5", "0.8", "32"],
        {"hr": "0.4@0,1@1", "step": "0.2-0.8", "lbw": "OUTS"},
    )

    assert (name, te, unet, dyn) == ("portrait", 0, 0.8, 32)
    assert settings["te_schedule"] == ((0.0, 0.0), (0.5, 1.0))
    assert settings["hr_te_schedule"] == settings["hr_unet_schedule"]
    assert (settings["start_step"], settings["stop_step"]) == (0.2, 0.8)
    assert settings["block_weights"].value_for("diffusion_model_output_blocks_8_0") == 1

    with pytest.raises(ValueError, match="network name"):
        lora_advanced.parse_activation_options([], {})
