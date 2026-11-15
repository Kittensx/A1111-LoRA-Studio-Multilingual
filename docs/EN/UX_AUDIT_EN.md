# LoRA Studio: Usability Review Against Three Source Extensions

Date: 2026-11-05. These are review findings, not a claim that the new UI is ready.

Additional pass on the same day: 55 browser checks passed again; scrolling to
actions in a narrow area was confirmed, and after Cancel, focus falls back to
BODY. Add a consistent focus return when closing. Runtime/API issues were also
found outside UX; the current shared work queue is in
`СВОДНЫЙ_АУДИТ_UI_LORA_VAE_EMBEDDINGS_2026-11-05.md`. API authentication and
bias/Embedding fixes must precede claims of release readiness; UI work remains.

## Main conclusion

Studio cannot yet be called a fully usable replacement for the source
extensions. Moving parameters into the native runtime and saving presets does
not replace fast editing workflows. Some features were ported but require more
actions; others are missing. A new calculation architecture alone will not
solve these problems.

## What was actually checked

- Current Studio code and sources in `extensions/check`: Weight Helper,
  LoRA Block Weight, LoRA Control. Originals were neither moved nor enabled.
- Current Studio was opened in a real browser using local
  `test/lora_studio_fixture.html`. Quick tune, Advanced controls, Classic
  17-block layout, and draft interactions were reviewed.
- **55 browser checks** passed: 51 state/event scenarios and four contrast
  checks (light/dark themes, with/without theme tokens).
- Separately, Node: 17 tests including 51 events/DOM assertions. These are NOT
  another 51 independent integration checks; the cases overlap with the browser
  fixture.

This was a static fixture with mocked APIs, not a running Gradio/WebUI.
The static server lacked webui-assets/css/sourcesanspro.css; a fallback font
was used. No exact pixel-size parity with production WebUI is claimed.
Source extensions were inspected in code, not run side by side during generation.
The old white-field background problem was not reproduced in this fixture.
Tests with other themes, scales, and CSS conflicts in installed WebUI are still
mandatory.

## Three major missing usability features

| Action | Original Weight Helper | Current Studio | Resolution |
|---|---|---|---|
| Quickly tune block contribution | Slider, number and block toggle in one compact editor | Full Studio → Advanced → profile → scroll; block inputs are text fields | Blocks in Quick tune: slider + exact number + temporary disable with restore |
| Return to a successful variant | Back/forward history, pin variant | Preset/favorites library is not a history of current edits | Draft history + pinned variants, separate from library |
| Adjust LoRA without losing sight of prompt and Apply | Draggable compact editor; reveal advanced parameters as needed | Large preset area on top; at tested window size Apply is below visible popup and requires internal scrolling; no drag | Short top bar, collapsible presets, always-accessible Apply/Cancel, drag/fit the window to the screen |

### Verifiable locations in the source code

- Weight Helper: `src/components/sections/common-section.ts` — range + number
  + checkbox, configurable step; `lbw-section.ts` — same controls for blocks.
- `header-section.ts` — window dragging, history and pinning;
  `src/shared/manager/history-manager.ts` — per-LoRA parameter history.
- `preview-section.ts` — image, metadata, trigger-word actions;
  `scripts/weight_helper.py` — UI scale, ranges/steps, preview size/position.
- Studio: `javascript/loraStudioView.js`, `renderExactBlockEditor` — block
  text fields; `renderQuickTune` — section order, buttons at content bottom.

## Other gaps compared with the sources

**LoRA context.** Weight Helper offers a preview image in its editor, metadata
and description actions, and insertion of trigger words into positive/negative
prompts. Studio already provides information, trigger-word suggestions, and
saved settings; it would be incorrect to say these do not exist. But Quick tune
lacks an equally compact visual context. Studio blocks inserting trigger words
while a draft is open: the user must Apply/Cancel first. A separate workflow is
needed to atomically confirm a tag and selected triggers without losing the
draft or overwriting someone else's prompt edit. Support for negative/other
editors should be checked separately without conflating it with actual negative
LoRA semantics in A1111.

**LoRA Block Weight.** `scripts/lora_block_weight.py` includes XYZ plot,
Effective Block Analyzer, X/Y/Z parameters, an image-comparison threshold,
preset editing/reloading, and block sliders. Having `lbw`, `lbwe`, and a
coverage table in Studio is not a substitute for running image-comparison
series. Experiments are a separate later module after runtime acceptance:
record seed, checkpoint/LoRA hashes, prompt, sampler, and Hires phase; preserve
results. Do not copy old global monkeypatches without compatibility review.

**LoRA Control.** `scripts/loractl.py` and `loractl/lib/plot.py` can plot
recorded per-step TE/U-Net multipliers from callbacks. Studio's SVG shows the
configured schedule, not observed runtime behavior. A separate opt-in trace
for multiple LoRAs, split by base/Hires, is needed. This is a diagnostic plot
of multipliers, not a measurement of visual effect or proof that TE conditioning
changed at each step. Do not change weight formulas to make the chart work.

**Curves.** Studio offers visualization and editable point strings; the presence
of SVG does not imply interactive point dragging. Follow-up: point dragging,
keyboard editing, step ranges/phases, TE/U-Net independence, preservation of
negative weights, and correct inheritance mode.

## Plan for small, safe changes

1. **Quick blocks.** Reuse one component in Quick tune/Studio. Slider and number
   share a draft; the range does not clip values already entered. Empty or
   incomplete input does not become 0. Do not guess unknown profiles from the
   name alone; retain explicit choice and diagnostics.
2. **Edit history.** Small separate state module, not more handlers inside the
   large View. One slider drag = one Undo step; Redo is truncated after a new
   branch. Drafts are distinguished by tab and occurrence, not just LoRA name.
   Pinned snapshots may be saved to the existing library.
3. **Compact window.** Reorder sections without changing tag grammar: LoRA/main
   weight at top, brief preset chooser nearby; expand blocks on demand;
   Apply/Cancel accessible without scrolling content. Mouse dragging is optional;
   keyboard operation and auto-fit to screen are required.
4. **Context/curves**, then **trace/experiments**, are separate tasks with their
   own contracts and tests. Do not tie the entire UX release to a rewrite of
   LoRA runtime or integration of ideas from other WebUIs.

### Invariants that must not be lost

- Apply changes only the selected occurrence; Cancel/Escape does not change prompt.
- Repeated names, unknown arguments, independent TE/U-Net/DYN/Hires, and elemental
  overrides are preserved. History does not restore someone else's prompt.
- If prompt changes externally, an old draft is not silently applied.
- txt2img/img2img do not share unfinished edits. Two editors must not write
  to the same draft simultaneously.
- Do not copy automatic saving on every click outside the window: it defeats
  explicit Apply/Cancel. Live preview requires a separate opt-in setting.

### UX acceptance

Test real actions, not only automated test counts: open tag → edit block →
compare previous version → confirm; save/restore a combination of
TE+U-Net+DYN+curve+Hires+lbwe; select a trigger with an open draft.
Test 1366×768, 1920×1080, and a narrow area around 800 px; light/dark theme,
125/150% scale, Tab/Shift-Tab/Escape, and Apply visibility. Then repeat in
real Gradio with a custom theme. Only then call Studio a usable replacement.

## What changed in this review

The UI has not yet been rewritten: this is an audit and work-priority update.
Separately, a critical fix was completed in `modules/sd_vae.py`: after a
failure during the model-mutating stage of VAE reload, a partially initialized
object is removed from the active model/cache. The original error is preserved;
resolving a VAE name before mutation does not remove a healthy model.
New `test/test_runtime_vae_failures.py`: 3 methods, including 7 failure points.
Total CPU/AST runtime suite: **72 PASS**. This is not a CUDA/generation test.
