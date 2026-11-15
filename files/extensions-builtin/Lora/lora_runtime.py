"""Pure LoRA runtime evaluation API.

This is the stable boundary used by ``network.py``.  The implementation still
lives in :mod:`lora_advanced` during the gradual migration, but callers no
longer need to know that transitional detail.  This module deliberately has
no WebUI or torch imports.
"""

from lora_advanced import (  # noqa: F401
    BLOCK_IDS_12,
    BLOCK_IDS_17,
    BLOCK_IDS_20,
    BLOCK_IDS_26,
    BLOCK_IDS_BY_LENGTH,
    BLOCK_IDS_DIT,
    BLOCK_IDS_FLUX,
    DEFAULT_BLOCK_PRESETS,
    DEFAULT_ELEMENTAL_PRESETS,
    BlockWeights,
    ElementalWeights,
    block_id_for_key,
    evaluate_weight_spec,
    parse_activation_options,
    parse_block_weights,
    parse_elemental_weights,
    parse_step_range,
    parse_weight_spec,
    step_is_active,
)

__all__ = [
    "BLOCK_IDS_12",
    "BLOCK_IDS_17",
    "BLOCK_IDS_20",
    "BLOCK_IDS_26",
    "BLOCK_IDS_BY_LENGTH",
    "BLOCK_IDS_DIT",
    "BLOCK_IDS_FLUX",
    "DEFAULT_BLOCK_PRESETS",
    "DEFAULT_ELEMENTAL_PRESETS",
    "BlockWeights",
    "ElementalWeights",
    "block_id_for_key",
    "evaluate_weight_spec",
    "parse_activation_options",
    "parse_block_weights",
    "parse_elemental_weights",
    "parse_step_range",
    "parse_weight_spec",
    "step_is_active",
]
