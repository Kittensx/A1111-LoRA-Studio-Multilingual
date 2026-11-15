# Getting started with LoRA Studio

A quick path from an installed LoRA to an editable prompt tag.

LoRA Studio is a visual editor for LoRA tags in **txt2img** and **img2img** prompts. It does not create or train a LoRA model.

## First steps
- Install or copy your LoRA model to the A1111 LoRA location, then refresh the LoRA list if necessary.
- Open the LoRA Studio panel under the prompt.
- Search for a LoRA in **Add LoRA**. Select **Add as draft**.
- Adjust Text Encoder (TE) and U-Net weights or apply a preset.
- Review the tag preview and click **Apply to prompt**.

## Important safety behavior
Editing a draft does **not** change your prompt until you apply it. If the prompt changes while a draft is open, cancel/reopen the draft rather than overwriting someone else's edits.

This package is experimental and not yet validated for every A1111 release or GPU configuration.
