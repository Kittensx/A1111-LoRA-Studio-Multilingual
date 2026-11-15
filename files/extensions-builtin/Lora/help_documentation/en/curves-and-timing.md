# Curves and activation timing

Use step-aware LoRA schedules and separate Hires.fix overrides.

LoRA Studio supports an activation window and, on supported paths, piecewise-linear weight curves.

## Activation window
**Start** and **Stop** gate when a LoRA is active in a sampling timeline. Blank values mean no additional override. Exact behavior depends on the active runtime and sampler.

## Curves
Use the curve editor to control how TE or U-Net weight changes over time. Preview the curve, edit points, and check the serialized prompt tag before applying.

## Hires.fix
The Hires.fix section can override TE and U-Net values for the high-resolution pass, where supported. Unset values inherit the normal weights.

Test advanced schedules on a small job before using them in a longer workflow.
