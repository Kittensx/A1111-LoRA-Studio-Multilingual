import pytest

from modules import extra_networks


def test_parse_prompts_accepts_identical_extra_networks():
    prompts, data = extra_networks.parse_prompts([
        "portrait <lora:style:0.7:unet=0.9>",
        "landscape <lora:style:0.7:unet=0.9>",
    ])

    assert prompts == ["portrait ", "landscape "]
    assert data["lora"][0].items == ["style", "0.7", "unet=0.9"]


def test_parse_prompts_rejects_mixed_extra_networks_within_batch():
    with pytest.raises(RuntimeError, match="batch size 1"):
        extra_networks.parse_prompts([
            "portrait <lora:style-a:0.7>",
            "portrait <lora:style-b:0.7>",
        ])
