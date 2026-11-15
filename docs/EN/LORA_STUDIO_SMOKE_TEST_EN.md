# LoRA Studio — manual Phase 4 smoke checklist

Run this only in an already configured WebUI installation. It requires no
model download or inference: do not generate an image for this checklist.
Record the WebUI revision, browser and any failure before changing code.

For an isolated DOM fixture without WebUI, open
`test/lora_studio_fixture.html` and press **Run UI checks**. The mock API saves
only in memory. It cannot verify Gradio integration or image quality.
`node --test test/lora_quick_presets.test.cjs test/lora_combinations.test.cjs test/lora_studio_dom.test.cjs`
checks pure recipes and event/state behaviour with a minimal simulated DOM;
it does not replace a real-browser layout/accessibility check.

The fixture also includes a deliberately conflicting forms reset (white
background/light text, loaded after style.css). In a real browser its Run UI
checks button tests native input/select/textarea text contrast in light/dark
themes with and without Gradio tokens. Node explicitly skips these CSS checks.

| Check | Procedure | Expected result |
| --- | --- | --- |
| Saved-prompt parsing | Open txt2img and img2img with `lora`, `lyco` and `loractl` tags, including repeated names. | Every occurrence is listed separately; unsupported tags and ordinary text stay untouched. |
| Draft safety | Edit a selected LoRA, then discard; edit again and Apply. | Discard changes nothing. Apply updates only the selected tag and rejects a stale prompt. |
| Advanced values | Set Hires override, timing, curve, raw block weight and a named elemental preset; apply then reopen the tag. | Values round-trip exactly, including blank/inherit Hires fields and `lbwe=NAME`. |
| TE/U-Net/DYN together | Try `<lora:name:te=0.3:unet=0.8:dyn=16>`, `:0.3:0.8:16`, then a U-Net curve with Hires and lbw/lbwe. Edit each control, Apply, reopen. | Separate weights and rank remain visible and independent; editing TE preserves the curve/rank/Hires/masks. Off removes dyn only. Uppercase parameter keys trigger a case-sensitive-name notice. |
| Unfinished input | Clear a required exact weight or a curve point; enter DYN 2.5, 0, -1 or an overflowing integer. Try Apply, a preset or Full Studio handoff. | No silent old value, zero conversion or rank clamp; error remains visible, prompt unchanged. Correcting DYN via a shortcut clears the error. |
| Scalar/curve mirrors | Change a quick TE weight, inspect Advanced, then enter a curve in Exact TE and switch to another control. | Exact/quick fields agree, the curve is shown as a curve; the focused exact field is not replaced on blur. An unfinished point in the other curve is preserved. |
| Other field focus | Type into a curve point, a block cell, raw lbw and lbwe. Click or Tab to the next field. | The active field survives input/blur; exact values, plots and preset selectors update without rebuilding the whole editor. |
| Empty block commands | Clear a cell; try Apply. Clear Set all / Set group and click Set. | A blank cell is invalid, not zero. Empty bulk/group commands change no weights. Correcting one field clears only real errors. |
| JSON text area | Open backup/restore; type or paste text, clear it, then rerender by discarding a LoRA draft. | Saved JSON is initially visible; edited text, including an intentional empty string, survives. The field has a readable placeholder and can be resized vertically. |
| Prompt shortcut | Put the text caret inside a LoRA tag and right-click it; change TE/U-Net, press Cancel/Escape, then reopen and Apply; right-click ordinary prompt text too. | The LoRA tag opens compact Quick tune controls. Cancel/Escape leave the prompt byte-for-byte unchanged; Apply updates only that tag; Full Studio keeps the same draft; ordinary text retains the browser context menu. |
| Block diagnostics | Inspect a known `.safetensors` LoRA, a Flux example, a flat-DiT example if available, and a file with no recognised layout. | Inspection is explicit and read-only; it never changes profile or prompt automatically; Flux `D/S` and flat DiT `B` layouts are reported separately; ambiguous/unknown layouts warn and preserve raw values. |
| Sidecars | Add sibling `.txt`/`.description.txt` and/or user `.json`, then press **Inspect local sidecars**. | The description precedence is JSON then `.txt` then `.description.txt`; only description, activation text and preferred weight appear; no prompt changes until the explicit insertion button is pressed. |
| Presets | Save a v3 preset with Hires and an elemental preset, favourite it, apply it, export/import it, then open a v1/v2 backup. | New values restore exactly; favourites persist; recent items remain browser-local; a v1 preset preserves existing Hires values. |
| Unified picker | In both Quick tune and Studio select a quick mode, block preset, module preset and saved preset; inspect the description before pressing Use preset. | Selection alone changes nothing; Use preset changes only the stated controls; Apply changes only the selected prompt occurrence. Existing module overrides remain visible when changing block masks. |
| Draft ownership | Open Quick tune while Studio is expanded; edit; use Full Studio; reselect the same LoRA or open another card. | The background Studio body is inactive while Quick tune is open. Transfer keeps the same draft; reselect/card handoff never silently discards it. |
| Theme and viewport | Check dark/light themes and narrow viewport, including an expanded save form and the card gear popup. | Controls inherit theme colours, popup fits the viewport and scrolls; preset selection, optional placeholders and exact weights remain readable. |
| Accessibility | Use Tab, Shift+Tab, Enter and Space through Studio, preset controls, disclosures, curve inputs and trigger chips. | All controls have a visible focus target and useful name; no focus trap; keyboard action equals mouse action. |
| Reload | Reload the browser, refresh Extra Networks, and change tabs. | No duplicate Studio panes; saved presets reload; local draft is never silently written to a prompt. |

If a test fails, save the source prompt, the exact LoRA tag, the browser
console error and the steps needed to reproduce it. Do not use an image result
as the only correctness signal: prompt and UI state are the acceptance record.

2026-10-29 field follow-up: 51 event/state assertions pass in the simulated
DOM (17 Node tests including the pure recipe/combination suites). UI controls
now have complete paired theme fallbacks, scoped forms-reset overrides,
explicit text/caret/placeholder colours and narrower-layout sizing. No live
WebUI tab was available in the browser tool, so actual browser rendering and
the user's reported white panel remain to be visually verified. No Torch,
model or generation process was launched.
