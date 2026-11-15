# Advanced LoRA prompt controls

The builtin LoRA extension accepts the normal tag and the compatibility aliases
`lyco` and `loractl`.  All aliases use the same safe implementation:

```text
<lora:model-name:1:unet=0.8>
<loractl:model-name:0@0,1@0.35,0.4@1>
```

## Weight curves and timing

Use `weight@position` (or `weight~position`) points separated by commas or
semicolons.  If all positions are between `0` and `1`, they mean progress
through the sampling run; otherwise they are sampler steps.  Values are linearly
interpolated and clamp to the first/last point outside the curve.

```text
# Ramp the entire LoRA in over the first third of sampling.
<lora:portrait:0@0,1@0.33>

# Separate text-encoder and U-Net curves.
<lora:portrait:te=1@0,0.7@1:unet=0@0,1@0.3>

# Active only from 20% through 80% (absolute sampler steps also work).
<lora:portrait:1:start=0.2:stop=0.8>
<lora:portrait:1:step=4-18>
```

The interval gates the U-Net contribution.  Text conditioning is computed once
before denoising, so its curve supplies the initial value rather than being
silently recomputed after the conditioning already exists.

`hr`, `hrte`, and `hrunet` select a scalar or curve for the Hires.fix pass.
`hr` supplies both values, while the other two override one side:

```text
<lora:detail:0.7:hr=1.0:hrunet=0.8>
```

## Block weights

Add `lbw=` (or its compact alias `w=`) with a preset or a comma-separated
table.  Tables with 12, 17, 20, 26, or 61 numbers are accepted.  The 61-value
form applies to Flux and flat DiT stacks; unknown layer families are left at
weight `1` instead of being accidentally disabled.

```text
<lora:detail:0.8:lbw=OUTS>
<lora:detail:0.8:lbw=1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0>
```

Builtin presets are `NONE`, `ALL`, `INS`, `IND`, `INALL`, `MIDD`, `OUTD`,
`OUTS`, `OUTALL`, `ALL0.5`, and `FLUXALL`.  Additional presets can be entered
in **Settings → Extra Networks → Additional LoRA block-weight presets**, one
`NAME:comma,separated,weights` line at a time.

The LoRA-card gear menu exposes the same settings, including curves, Hires.fix
weights, timing, and a block-weight/preset field.  A curve disables its matching
numeric slider so editing a slider cannot silently erase it.

## Elemental module overrides

`lbwe=NAME` selects a named, per-module override after `lbw` has selected its
normal block weight.  The matching elemental rule replaces the block weight for
that module only; modules that do not match keep their ordinary `lbw` value.
This is useful when a LoRA's attention or projection layers make a style too
strong while its other layers remain useful.

```text
# Disable attention modules from IN05 through OUT05, after applying OUTS.
<lora:detail:0.8:lbw=OUTS:lbwe=ATTNDEEPOFF:profile=classic-26>
```

Builtin names are `ATTNDEEPON`, `ATTNDEEPOFF`, and `PROJDEEPOFF`.  Define more
in **Settings → Extra Networks → Additional LoRA elemental presets**, with one
named group per blank line:

```text
SOFT_DEEP_ATTENTION:IN05-OUT05:attn:0.35

ONLY_QUERY:IN05:attn_to_q:0.5,IN05:attn_to_k:0
```

Each rule is `blocks:module-substrings:weight`.  Multiple terms in either
selector are space-separated; ranges are inclusive; `NOT` negates a selector;
and the last matching rule wins.  Matching is case-insensitive and uses module
key substrings, so it is intentionally model/profile-dependent.  Studio offers
named choices but preserves the raw name; the native parser validates it when
the prompt is applied.

## Performance and safety

With normal merged LoRA mode, a curve can require a safe weight refresh at each
step where its value changes.  This is expected.  Functional LoRA mode reads
the current multiplier during the forward pass instead; it can be more useful
for rapidly changing LoRAs but may be slower overall.

All per-prompt controls belong to a `NetworkActivation`, not the cached LoRA
file.  Thus two tags for the same file can use different weights without
changing each other or contaminating the next generation.
