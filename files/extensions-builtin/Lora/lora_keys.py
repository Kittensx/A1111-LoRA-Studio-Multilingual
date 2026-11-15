"""Torch-free state-dict naming and embedding-layout validation."""

# Match a complete known suffix, not the first dot in a dotted module path.
# Keep this list aligned with the native network_* implementations.
WEIGHT_PARTS = tuple(sorted({
    "lora_up.weight", "lora_down.weight", "lora_mid.weight",
    "lora_A.weight", "lora_B.weight", "lora_embedding_A", "lora_embedding_B",
    "lora_embedding_A.weight", "lora_embedding_B.weight",
    "a1.weight", "a2.weight", "b1.weight", "b2.weight",
    "hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b", "hada_t1", "hada_t2",
    "lokr_w1", "lokr_w1_a", "lokr_w1_b", "lokr_w2", "lokr_w2_a", "lokr_w2_b", "lokr_t2",
    "oft_blocks", "oft_diag", "rescale", "w_norm", "b_norm", "diff", "diff_b",
    "weight", "on_input", "alpha", "scale", "bias", "dora_scale",
}, key=len, reverse=True))


def split_network_key(key):
    if not isinstance(key, str):
        raise ValueError(f"LoRA state-dict key must be a string, got {type(key).__name__}")
    if key.startswith("bundle_emb."):
        return "bundle_emb", key[len("bundle_emb."):]
    for part in WEIGHT_PARTS:
        suffix = "." + part
        if key.endswith(suffix) and len(key) > len(suffix):
            return key[:-len(suffix)], part
    # Preserve unknown auxiliary fields on legacy underscore-named layers.
    # An unrecognised dotted format is not guessed into a supported format.
    layer, separator, part = key.partition(".")
    if not layer or not separator or not part:
        raise ValueError(f"Unrecognised LoRA tensor key: {key!r}")
    return layer, part


def layer_candidates(layer, convert, diffusers_map=None):
    """Ordered, whole-path candidates; never replace dots inside weight suffixes."""
    flat = layer.replace(".", "_")
    mapped = (diffusers_map or {}).get(layer, (diffusers_map or {}).get(flat))
    candidates = []

    def add(value):
        if value and value not in candidates:
            candidates.append(value)

    if mapped is not None:
        add(mapped)
        add(mapped.replace(".", "_"))
    else:
        add(convert(flat))
    # The first SD1 upsampler follows a ResBlock; in SDXL it follows
    # ResBlock + attention. Resolve against actual modules, not filename hints.
    if flat == "lora_unet_up_blocks_0_upsamplers_0_conv":
        add("diffusion_model_output_blocks_2_2_conv")
    add(layer)
    add(flat)
    for prefix, replacements in (
        ("lora_unet_", ("diffusion_model_",)),
        ("oft_unet_", ("diffusion_model_",)),
        ("lora_te1_text_model_", ("0_transformer_text_model_", "transformer_text_model_")),
        ("lora_te_text_model_", ("transformer_text_model_",)),
    ):
        if flat.startswith(prefix):
            for replacement in replacements:
                add(replacement + flat[len(prefix):])
    return candidates


def embedding_layout(up_shape, down_shape, target_shape, *, peft=False):
    """Return whether B@A must be transposed. Never reshape a wrong orientation."""
    up, down, target = tuple(up_shape), tuple(down_shape), tuple(target_shape)
    if any(len(shape) != 2 or min(shape) <= 0 for shape in (up, down, target)):
        raise ValueError(f"Embedding LoRA requires non-empty 2D matrices: up={up}, down={down}, target={target}")
    if up[1] != down[0]:
        raise ValueError(f"Embedding LoRA rank mismatch: up={up}, down={down}")
    product = (up[0], down[1])
    direct, transposed = product == target, product[::-1] == target
    if peft:
        if transposed:
            return True  # Explicit lora_embedding_A/B uses PEFT's transpose convention.
    elif direct != transposed:
        return transposed
    elif direct:
        raise ValueError("Ambiguous square embedding LoRA; an explicit export convention is required")
    raise ValueError(f"Embedding LoRA shape mismatch: up={up}, down={down}, target={target}, peft={peft}")
