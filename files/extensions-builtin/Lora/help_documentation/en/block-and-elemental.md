# Block weights and elemental module overrides

Fine-grained controls for compatible LoRA architectures.

Block weights let you supply weights for groups of LoRA blocks instead of one global value. A block layout profile controls which blocks are shown.

- Start with a known layout compatible with your model.
- Named block presets may be selected from built-ins or configured entries.
- **Elemental overrides** target matching module substrings, and can take precedence over block weights.
- The exact block editor helps review and change individual values.

**Compatibility matters:** not every LoRA architecture exposes the same blocks or module names. Unsupported or mismatched presets may fail validation or have no useful effect.
