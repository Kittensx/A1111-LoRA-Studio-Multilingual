# LoRA: Presets and Pluggable Features

## Current stage v12 — 2026-11-12, external audits

Findings from November 8 were compared against v11, not only historical reports.
Fixed: late API import, missing REGION install operation, VAE exact matching and
duplicates, VAE/checkpoint file version and forced disk read, bounded metadata
header, unknown checkpoint fallback, TI traversal/image cache/preflight and
diagnostics, SDXL Create/save/reload, old emphasis span, same-key Options
callback convergence, and SD3/Flux metadata labels. Installed smoke was added:
install onto an expanded old base → test installed code → byte-for-byte rollback.

Full findings and limitations: EXTERNAL_AUDIT_EN.md in the v12 package.
Run actual Gradio/CUDA/WAI smoke first; then ModelFamily/asset-ID/index and
patch-ownership facade; later memory budget/VAE tiles. Independent dual token
counts, SDXL TI trainer, per-step TE, mixed-LoRA batch, and history/undo are not
implemented. Acceptance with actual user safetensors and external hooks remains
open. v10/v11 packages are preserved. Historical items below do not waive this gate.

## Current stage v11 — 2026-11-11

Confirmed interactions were fixed for Hires scalar/base curve, TE initial
value, SDXL OpenCLIP key routing, short LBW masks, MID elemental matching,
mixed start/stop units, and CP dynamic rank. Profile is now included in the
conditioning key. UI: user tables instead of built-ins on override, safe
projection of classic block IDs, block sliders with exact input, absolute
curve midpoint, validation of empty/unknown masks, and adaptation to the panel's
own width. 112 CPU runtime, 59 parser/API/config (+13 subtests), 22 Node,
and 66 browser-fixture assertions PASS. This is neither a full Gradio/CUDA run
nor coverage of every combination. Short LBW now sets missing known blocks
to zero like the original LBW: an intentional change from v10. MIDD is not
just M00. TE curves remain initial-only; Start/Stop affects U-Net only.

Next sequence: assemble/verify v11 and rollback → real SDXL/WAI smoke with
1/2/4 LoRAs, Hires/TI/VAE, and cancellation → A/B against source extensions →
history/undo and further UI simplification. Per-block curves, dynamic TE,
automatic profile selection, and new adapters are not declared ready.
Full report in v11: COMBINATIONS_AUDIT_EN.md. Historical plans below do not
mean this acceptance gate is complete.

## v10 addition — 2026-11-06

Fixes for confirmed V9-1–5, C1–4, and L1–2 were applied: SD3 consumer geometry,
wrapped Embedding LoRA, SDXL upsampler, TI/model/conditioning cache consistency,
atomic writes and deferred publication of presets, lowvram rollback, zero-mean
emphasis, and malformed Origin. 104 CPU runtime and 54 parser/API/config tests
PASS. Final review also eliminated holding the config lock during callbacks:
a model/VAE switch must not block settings changes originating from the
generation queue. Details: RUNTIME_STABILITY_FIXES.md. Real user WAI/LoRA files,
CUDA/Gradio/refiner integration still require acceptance. No new UX features
are declared complete; actual smoke comes next, followed by UI U1/history/blocks.
Historical items below do not supersede these limits.

Updated: 2026-10-29. This plan continues LORA_STUDIO_ARCHITECTURE.md.

## Current v9 addition, 2026-11-05

After the follow-up audit, A1–A5 were fixed: protection of all LoRA API routes,
repeat merge of added bias, Q/K/V bias application/restoration, per-encoder
TI validation, and early rejection of NaN/Inf/invalid matrices.
81 CPU/AST runtime and six API methods pass; the GPU/Gradio gate remains.
See RUNTIME_STABILITY_FIXES.md. v8 VAE code was not rewritten; its partial-load
quarantine was independently retested.

Next usability step: compact Quick tune with always-accessible actions and
focus restoration on close → block sliders → draft history. The UI has not
changed yet. Preserve isolation by occurrence/tab, unknown arguments,
incomplete-input validation, and explicit Apply/Cancel. Integration acceptance
is also required for auth in real Gradio/behind reverse proxy and recovery
from failures in a working GPU environment. Concurrent/failing config writes
remain a separate open item; A1 is not a security audit of the whole server.
Architectural innovations come later.

## Priority clarification as of 2026-11-04

This document describes feature evolution, not completed stabilization.
Before stages 2–4 there is a mandatory gate: load/failure tests, LoRA+Embedding,
and actual SD1.5/SDXL acceptance in the browser. v3–v6 fixes and limitations
are described in RUNTIME_STABILITY_FIXES.md. Fifty-six CPU/AST runtime tests
do not replace generation, checking third-party hooks, or image quality.

Immediate order: close reproducible runtime defects and explicitly identify
unsupported formats → pass GPU/UI smoke on a working installation → verify
the package and rollback → port individual ideas from other WebUIs. Compare
against source extensions using fixed prompt/seed/model hashes; do not change
basic TE/U-Net contribution under the guise of interface improvements.

Addition 2026-11-05: v7 fixes specific CPU scenarios involving metadata/
extra_state, dtype during meta loading, partially written TI/optimizer, and
FP8 cached selection. Total 69 runtime tests; the required acceptance above
still applies. Before new architecture, projection output-bias contracts,
independent VAE reload, external backend/low-VRAM hooks, and real GPU/UI runs
remain. Atomicity of `.pt`+`.optim`, custom-state migrations, and third-party
quantizers are not claimed.

## Correction after direct Studio testing, 2026-11-05

Independent VAE reload is protected from partially publishing a model on
failure; 72 CPU/AST runtime tests pass. This closes the local case, not the
entire gate. Studio passed 55 checks on a real browser HTML fixture, including
contrast in four modes. Gradio/GPU was not run. Older notes below saying
browser verification was unavailable refer to an earlier run, not these results.

Source review of Weight Helper / Block Weight / LoRA Control confirmed the
loss of convenient workflows. Before expanding architecture, add a separate
UX increment:

1. Sliders + exact numbers + temporary block disable directly in Quick tune;
   shared component with full Studio, with unchanged mathematical semantics.
2. Draft-history Undo/Redo and pinned variants; isolation by tab and tag
   occurrence. Existing presets do not replace edit history.
3. Compact movable popup, collapsible presets, accessible Apply/Cancel.

This is a plan, not controls already implemented. Preserve Cancel, unknown
fields, incomplete-input validation, stale-prompt protection, and duplicate
LoRA names. Then separately accept preview/trigger workflows and interactive
curves. Recorded runtime graphs and XYZ/Effective Block experiments are later
modules, after the runtime gate. A configured SVG chart is not a measured
runtime trace. Do not import old global hooks just to reproduce an interface.

Acceptance: actual select/edit/restore-variant steps, 1366×768 and narrow
windows, themes/zoom/keyboard, and then real Gradio. A passing event test
does not prove usability. Detailed comparative report: UX_AUDIT_EN.md in the
v8 install package (not installed into WebUI).

## Interface goal

Choose a LoRA, adjust weight, and select a ready-made mode without needing
to know lbw/lbwe. Exact parameters remain available. New modes are attached
through shared contracts and verified independently of the interface.

## Stages and completion conditions

1. **Unified preset selection — first increment implemented; visual acceptance still pending.**
   - One catalog for Quick tune and full Studio: simple modes, legacy
     block/elemental presets, and user-saved settings snapshots.
   - Describe affected parameters before applying; selecting an item writes
     nothing. Use preset updates the draft; Apply updates the selected tag.
   - Common modes are calculated relative to current weights; repeated use
     must not inadvertently reduce a weight over and over.
   - Block presets change only blocks; elemental presets change only module
     overrides; a saved Studio preset restores a full snapshot. Users can see
     the difference. Existing overrides remain visible.
   - Explicitly incompatible layout families are blocked. Auto does not mean
     compatibility with the loaded checkpoint has been verified.
   - Saving is available near quick settings; import, delete, and library
     management stay in an expandable section.
   - Tests: duplicate LoRAs, Cancel/Apply, stale drafts, curves, user presets,
     profiles, and concurrent views.

2. **Calculation modules and architecture adapters — planned.**
   - ArchitectureAdapter describes known layer names, components, and supported
     operations. Start with classic models, then individually verified
     architectures. The DiT name and length 61 do not establish support for
     an arbitrary transformer model.
   - ControlModule declares ID/version, input parameters, affected components,
     and combination rules. Start with existing TE/U-Net weights, schedule,
     Hires, lbw, and lbwe.
   - Validated parameters form an immutable plan for one LoRA application;
     loaded tensor payloads remain separate.
   - Evaluation order is fixed: component/phase weight, schedule, active
     interval, block mask with elemental override. The last term replaces
     block weight for matching layers, rather than multiplying it again.
   - Merge/functional share one evaluator and identical cache identity.
     Verify on existing formats, including special DoRA/OFT paths.
   - An empty set of extra functions preserves normal behavior.

3. **Settings through UI without long tags — planned.**
   - Visible mode on a particular LoRA occurrence; txt2img/img2img independent.
   - Backend receives a versioned settings snapshot per request and saves
     it in generation metadata. Preset name alone is insufficient: its
     contents may change after the image was saved.
   - Until that path is built, existing tags remain. Do not introduce a hidden
     global preset that cannot be reconstructed from an output.
   - One agreed priority order: tag parameters, occurrence-specific parameters,
     explicitly enabled user defaults.
   - Restore, batch, Hires, and API must recover the identical snapshot.

4. **Result verification and experimental features — planned.**
   - Compare normal path, source extensions, and new modes with identical
     checkpoint/LoRA/hash, prompt, negative prompt, seed, sampler, steps,
     CFG, resolution, and Hires. Preserve actual parameters.
   - Distinguish computational compatibility, speed/VRAM, and subjective quality.
     One successful frame does not prove a general improvement.
   - Regional application and more complex mixing are separate modules after
     base-path acceptance; they may require activation access absent from
     simple weight multipliers.

## First-stage decisions

- Implement catalog and transformations as pure functions in
  javascript/loraStudioPresets.js. UI uses a shared contract rather than
  duplicating formulas in event handlers. This is an initial small step toward
  modularity, not a claim of a complete arbitrary-plugin runtime API.
- Do not change existing v3 library schema: quick mode becomes an already-
  supported settings snapshot, which can be saved as a personal preset.
- Do not copy old monkeypatches or modify pre-patch/check directories.
- Do not install dependencies or Torch. First-stage checks: Node, existing
  pure Python tests, isolated browser UI fixture. Image testing needs a
  separately available working environment.

## Verification status

- Shared Quick tune / Studio catalog, snapshot saving, and regular DOM field
  styles are implemented. DYN validation, synchronization of quick/exact
  weights, scalar/curve transitions, protection of incomplete fields, and
  draft ownership were fixed.
- Browser parsing of empty TE in a tag was fixed: like backend, it now
  receives weight 1 rather than NaN. This is not a new generation semantic.
- Automated checks: 37 Python tests, including 223 parameter combinations
  against the native parser; 17 Node tests including 37 UI event assertions.
  Syntax checks covered eight modified/new JS files. Torch import was blocked
  during Python checks. No dependencies were installed.
- Event fixture uses a minimal simulated DOM, not a browser engine. Visual
  checking of local HTML was blocked by browser-tool policy; no bypass was
  attempted. Themes, window sizes, and Gradio integration need acceptance
  in working WebUI using the smoke checklist.
- Stages 2–4 remain unimplemented. Stage 1 does not demonstrate image-quality,
  speed, or VRAM improvements. Generation and tensor paths were not run.

## How TE, U-Net, and DYN combine

`<lora:name:te=0.5:unet=0.8:dyn=16>` sets three independent parameters:
strength for the text encoder, strength for U-Net, and a rank limit in
LoRA layers supporting it. DYN is not a third weight multiplier or a step count.
Conventional LoRA uses the first rank components of matrices; this neither
guarantees quality nor works universally on all adapter types. Without `dyn`,
rank is not limited. UI explicitly displays Off / full rank.

- Keys in the tag are case-sensitive: `te`, `unet`, `dyn`, not `TE/UNET/DYN`.
  Unknown keys are preserved; Studio warns about lookalike uppercase keys.
- Positional form `:0.5:0.8:16` means TE, U-Net, DYN. Named arguments override
  the corresponding positional argument. Without U-Net, it inherits TE,
  including the curve; UI can subsequently edit them separately.
- `hr`, `hrte`, `hrunet` set separate Hires weights. An absent value in UI
  means inherit, not zero. DYN remains a LoRA-application setting; the current
  contract has no separate Hires DYN.
- `lbw` sets block weights; `lbwe` replaces selected block multipliers on
  matching modules. They do not erase TE/U-Net/DYN. Actual effect depends on
  architecture, available LoRA layers, and preset contents.
- An empty **required editor field** blocks Apply. This differs from the
  historical grammar for an empty value **in the tag**: the editor must not
  silently reuse an old value while the user is still typing a new one.
