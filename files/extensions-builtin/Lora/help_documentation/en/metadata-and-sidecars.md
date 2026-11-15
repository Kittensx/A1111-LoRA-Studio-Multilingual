# LoRA metadata and sidecar files

Where descriptions, triggers and model information come from.

LoRA Studio can inspect the selected safetensors metadata header and nearby sidecar files.

- **Metadata** reads the bounded JSON header from a `.safetensors` file. It does not load model tensors.
- **Sidecars** may include `.txt`, `.description.txt`, or `.json` files adjacent to a LoRA file.
- Suggested trigger words are optional hints and should be verified against the LoRA's actual training information.

Inspecting a file does not silently insert tags or modify the prompt. Treat third-party descriptions as untrusted text.
