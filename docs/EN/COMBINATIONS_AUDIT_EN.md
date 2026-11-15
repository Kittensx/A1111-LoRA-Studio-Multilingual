# LoRA Studio v11: Combinations, Source Extensions, and Verification Limits

Date: 2026-11-11. This is a report on specific fixes, not a declaration that
the entire plan is complete.
Working project: `D:\Users\Asus\Music\Musify\test\stable-diffusion-webui-dev`.
Backup before this pass: `D:\AI-tools\lora-audit\before-20261110-combinations`.

## What was actually wrong

1. An explicit fixed Hires weight did not disable the base curve. Now `hrunet=0.3`
   replaces `unet=0.5@5,1@9` on the second pass; `hrte` acts independently.
2. The TE curve depended on the residual global step number. Now its first
   point is used: full text re-encoding on every step has NOT been implemented.
3. Certain SDXL OpenCLIP keys with an encoder index, such as
   `1_model_token_embedding`, were incorrectly treated as U-Net and received
   U-Net weight/timing.
4. Known blocks omitted from a short classic LBW table received 1 instead of 0.
   Checked against the original Block Weight `to26()`. Unknown keys retain
   fallback 1. This intentionally changes results compared with v10.
5. A classic middle-block transformer could be misclassified as flat DiT,
   preventing IN05–OUT05 elemental rules from reaching the intended MID attention.
6. Mixed units such as `start=5:stop=0.8` were compared as plain numbers.
   Resolved boundaries are now compared; negative boundaries are prohibited,
   shorthand retains the minus sign, and scientific notation is supported.
7. DYN was ignored for CP decomposition. Now up/down/mid are truncated, while
   the original alpha/rank scale is preserved. This does not promise DYN for
   every LyCORIS implementation.
8. Block profiles were omitted from the conditioning signature; now included.
9. The UI showed the built-in table even when the user had overridden MIDD.
   It now displays the user's table; an invalid override is not silently
   replaced with the built-in one.
10. Converting a short mask into the visual classic table was inconvenient.
    Mapping is now by ID; missing blocks are zero. A conversion that would
    lose nonzero blocks is not silently performed.
11. For absolute steps, the add-curve-point button was unavailable. Now an
    interpolated point at 7 can be added between 5 and 9.
12. A narrow Studio in a wide window remained two-column. A container query
    based on Studio's own width was added; the media-query fallback remains.

## Findings from comparing the three source extensions again

- `extensions/check/sd-webui-loractl`: compared curve units, interpolation,
  Hires override, and the issue of parameters attached only to the LoRA name.
  Our settings remain separate for each occurrence of the same file.
- `extensions/check/sd-webui-lora-block-weight`: checked BLOCKIDS, MIDD/ALL
  tables, zero-padding of short masks, and precedence of elemental overrides.
- `extensions/check/sd-webui-weight-helper-main`: adopted principles of
  slider + exact value, expandable range, block correspondence, and explicit
  layout selection.

The UI now has individual block sliders alongside exact-value inputs; typing
3.5 expands the range instead of clipping to 2. Group operations remain.
An unknown LBW/LBWE preset or incomplete numeric table blocks Apply.
An empty input is not turned into zero or a hidden previous value.
LBWE rules are ultimately validated by the backend, not just the browser.
The three source extensions were not changed. This is not a complete port
of all their functionality.

## Unambiguous semantics of combinations

- With no extra arguments, a simple LoRA does not get a hidden preset.
- TE and U-Net are independent. DYN limits the rank of a supported layer;
  it is not a strength setting.
- LBW multiplies the selected block's contribution; matching LBWE replaces
  that particular block multiplier, after which TE/U-Net and U-Net start/stop
  still apply.
- `MIDD` is a multi-block legacy preset, not a synonym for the sole M00 block.
- `ALL` with 17 values does not mean all 26 blocks: omitted known blocks are 0.
  To remove the mask entirely, use All blocks / no module overrides.
- `0.5@5,1@9`: before step 5 the weight is 0.5; between 5 and 9 it is linearly
  interpolated; starting at 9 it is 1. A curve does NOT, by itself, disable
  LoRA outside that interval.
- If all curve positions are in 0..1, they represent fractions of the pass;
  otherwise all positions are treated as absolute steps. Units cannot be mixed
  among points in one curve.
- Start/Stop: 0..1 inclusive represents a fraction of the 0..N-1 range; 1
  means the end; values >1 are absolute zero-based steps. Boundaries are
  inclusive. This is not an exact copy of older LBW, in which 1 and boundary
  rounding could mean something else.
- Start/Stop limits U-Net, not already calculated TE conditioning.
- TE uses the first curve value. Per-block curves, fully dynamic TE conditioning,
  and automatic architecture-profile selection were not added at this stage.

## Verified

- 112 runtime tests with existing Torch CPU; production definitions through AST.
- New matrix: 1/2/4 occurrences, repeat of one payload, base/Hires, 5 key
  types, 11 steps = 330 timestep/layer scenarios. Independent expected-delta
  arithmetic, merge/repeated merge, functional forward, reverse order, disable,
  and unchanged source factors. The embedding-name test uses Linear; real
  Embedding/TI is covered separately by earlier runtime tests.
- CP DYN checked separately on a real Conv2d. The entire new matrix has no
  real safetensors loader: factors are small and synthetic.
- 59 parser/API/config tests + 13 subtests; 22 Node tests, including
  61 UI assertions.
- 66 browser UI assertions: events, two occurrences of one name, four LoRAs,
  empty/invalid fields, presets, curves, sliders, narrow column, and field
  contrast under light/dark themes with/without theme tokens. An isolated
  fixture was used and preset network responses were stubbed there;
  this is NOT full Gradio.

Nothing was downloaded or installed. Torch ran from the existing D: environment.
Known warnings: NumPy is missing from the Torch venv; pytest base_url is absent
when third-party plugins are disabled. Tests passed.

## What remains on the plan — and is not declared fixed without verification

1. Integration acceptance in real WebUI: SDXL/WAI, user LoRAs, Embeddings,
   VAE switch, Hires, img2img, batch, cancel/retry, external hooks. No complete
   local Gradio/CUDA environment with the required model is available; no new
   dependencies were installed. The CPU matrix sets steps manually and does
   not test the sampler lifecycle.
2. A/B comparisons using fixed seed/prompt/model hashes against the three
   extensions: UI checks and arithmetic do not prove better aesthetics or speed.
3. Draft history/undo, further simplification of long tooltips/navigation,
   and verification with actual third-party themes. The sliders must not be
   presented as completion of the UX plan.
4. New adapters, separate per-block curves, dynamic TE, and enhanced XYZ/plot
   functions are future changes after integration acceptance.

## Files changed relative to v10

Runtime: `extensions-builtin/Lora/network.py`, `network_lora.py`, `networks.py`,
`lora_advanced.py`. Outside builtin: `javascript/loraStudioModel.js`,
`loraStudioCurves.js`, `loraStudioBlocks.js`, `loraStudioView.js`, `style.css`.
Tests: new `test_runtime_combinations.py`, `test_lora_interaction_contract.py`,
`lora_interactions.test.cjs`; expanded `lora_studio_fixture.js`.
Documentation: `LORA_EVOLUTION_PLAN.md`, `RUNTIME_STABILITY_FIXES.md`.
No `modules/*`, `lyco_helpers.py`, user models, or user settings were rewritten
in this pass. The v11 package includes the earlier modules fixes from v10.

Install only the complete package after stopping WebUI; run verify/check first.
Nothing was automatically installed onto remote `/content/A1111`.
