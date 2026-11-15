# TE, U-Net weights and dynamic rank

Understand what the main sliders and DYN field control.

**TE** controls the weight applied to the text-encoder portion of supported LoRA models. **U-Net** controls the main denoiser weight. Not all architectures support both components in the same way.

- Start with moderate weights, then test your output. Higher values are not automatically better.
- **Quick controls** are convenient for simple changes.
- **Exact weights** allow direct numeric entry.
- **DYN (dynamic rank)** requests a supported rank limit; Off retains full rank. Some adapter types ignore DYN.

Weight changes can affect the generated image. Keep a copy of prompts when comparing settings.
