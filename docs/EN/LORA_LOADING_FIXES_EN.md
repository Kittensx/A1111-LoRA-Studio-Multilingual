# LoRA loader / Embedding compatibility fixes — 2026-10-30

## What changed

- `lora_keys.py` splits recognised tensor suffixes from the right. Dotted
  CompVis module paths no longer collapse into the parent UNet group.
  Native underscore paths, diffusers mappings, alpha/scale, LoHa/LoKr/OFT/
  GLoRA suffixes and bundled textual-inversion names are retained.
- Normalized duplicate tensor slots are rejected rather than silently overwritten.
- `NetworkModuleLoraEmbedding` handles ordinary up/down and A/B factors on
  `torch.nn.Embedding`. It validates rank and both matrix dimensions; it accepts
  direct and transposed products when dimensions uniquely identify the layout.
  Explicit `lora_embedding_A/B` uses the PEFT transpose convention.
- `Embedding.forward` and `_load_from_state_dict` are registered alongside the
  other native layer hooks. Merge mode uses the existing backup/cache machinery;
  functional mode applies adapters to a temporary embedding matrix and performs
  an index lookup. Integer token/position indices are never cast as U-Net data.
- TE/Hires strength, block/elemental masks, alpha/scale, dynamic rank, repeated
  activations and additive weight-matrix bias use the existing activation contract.
- Functional mode preserves the base matrix even with `max_norm`; its
  renormalization operates on a temporary combined matrix. The normal unmodified
  embedding path is preserved when no adapter targets that layer.
- Switching merge -> functional with no active LoRA restores the previous
  merged weights before the early return. This also fixes the generic functional
  layer path. Checkpoint state loading clears old adapter backup/cache state.
- A/B normalization no longer deletes alpha, scale, DoRA or bias metadata from
  regular LoRAs. The source dictionary is not mutated.
- Multiple `bundle_emb.*.string_to_param.*` parameters are accumulated rather
  than replacing one another. Ordinary external Textual Inversion loading and
  prompt/tokenization code are not rewritten by this patch.
- No matched layers and duplicate normalized weights fail explicitly; unsupported
  groups produce bounded diagnostics. Failed loads enter the missing/failed-LoRA
  summary as well as the detailed console error.

## Deliberate limits

This is not universal support for every exported embedding adapter. Embedding
CP/LoCon-mid and DoRA formats are explicitly rejected pending a tested format
contract. An ambiguous square up/down embedding layout is rejected; the explicit
PEFT embedding keys remove that ambiguity. Malformed shapes are not reshaped to
hide an orientation mismatch, and unsupported layers are not silently skipped.

The reported CEB log contains 790 layer paths / 1,580 up/down keys. Those actual
paths are in `tests/ceb_reported_layers.json`. Because the log omits the original
prefix and all tensor shapes, the regression test reconstructs both `lora_unet.`
and `diffusion_model.` inputs; it is not a load test of the actual safetensors file.
Likewise, the user's Velvette safetensors header/weights were unavailable. Its
actual export must still be checked on the user's existing WebUI installation.

## Tests (no Torch import)

`python -B -m unittest discover -s extensions-builtin/Lora/tests -v`

The suite uses installed NumPy stand-ins and selected **actual production AST**
functions/classes, protected by a real-Torch import blocker. It verifies parsing,
factor math, dtype conversion, dispatch, patch registration/undo, repeated calls,
mode changes, rank, masks, Hires, max_norm, reload, and disabling adapters. It does
not substitute for real PyTorch/CUDA, full model lifecycle or image generation.
No dependency is installed by these tests.

## Reference semantics

- [PyTorch embedding lookup and in-place max_norm behavior](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.embedding.html)
- [PEFT embedding adapter layout](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py)

No source code from those projects was vendored; the implementation follows this
repository's activation and weight-backup contract. `lyco_helpers.py` is unchanged.
