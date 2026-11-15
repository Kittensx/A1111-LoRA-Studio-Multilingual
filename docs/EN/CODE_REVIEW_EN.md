# A1111 LoRA Studio Pack v15 — code audit and ImageGen port assessment

## What this package does

This is an **A1111 WebUI modification pack**, not simply a LoRA file or a standalone program. It modifies the built-in LoRA extension and numerous A1111 core files, supplies a Studio UI and prompt/tag parser, patches loading/runtime behavior, and provides an offline installer with explicit checks, backup, idempotence and rollback.

User-facing functionality includes: separate text-encoder (TE) and U-Net LoRA strengths; piecewise-linear weight curves; start/stop activation windows; Hires.fix overrides; dynamic rank; 12/17/20/26/61-value block masks; named per-module (`lbwe`) overrides; multiple independently editable occurrences of one LoRA; editable LoRA tags, visual block/curve controls and presets; metadata and bounded sidecar inspection; and draft-only editing until Apply. The current package contains Python API and core processing modifications in addition to the visual editor.

## License finding

The archive contains `LICENSE-A1111.txt`, whose first lines specify **GNU Affero General Public License, version 3 (AGPL-3.0)**, with a copyright attribution to AUTOMATIC1111. This is licensing evidence for upstream A1111 code, not proof that every newly written file in this pack was separately and explicitly offered under a particular license. No separate top-level license/notice establishing ownership and terms of the Studio author's original additions was identified in this supplied archive. Preserve the license and upstream notices; seek provenance of the additions before redistributing a derivative port. Do not assume Apache-2.0/MIT compatibility for the copied code.

This is a technical source audit rather than legal advice. An AGPL-covered derivative distributed to others, or operated as a modified network service in circumstances covered by AGPL section 13, can have corresponding-source obligations. An independent reimplementation guided by interface concepts has different considerations from copying A1111-derived implementation.

## Notable code boundaries

| Area | Representative source | Responsibility |
| --- | --- | --- |
| Browser prompt model | `files/javascript/loraStudioModel.js` | Parsing, normalization, preserving unknown arguments, exact-span replacement |
| Browser studio | `files/javascript/loraStudioView.js` | txt2img/img2img DOM UI, drafts, controls, apply/discard |
| Browser presets and curves | `files/javascript/loraStudioPresets.js`, `loraStudioCurves.js`, `loraStudioBlocks.js` | Preset snapshot and pure UI calculations |
| Python schema | `files/extensions-builtin/Lora/lora_schema.py` | Validation and canonical LoRA activation data |
| Python runtime | `lora_runtime.py`, `lora_advanced.py`, `networks.py`, `network.py` | LoRA weight application, cache/routing, runtime changes |
| Model compatibility | `files/modules/processing.py`, `sd_models.py`, `sd_vae.py`, `textual_inversion/textual_inversion.py` | A1111-specific generation, load, embeddings, VAE behavior |
| API/security | `files/extensions-builtin/Lora/lora_studio_access.py`, `files/modules/api/api.py` | LoRA routes; A1111/Gradio authentication boundary |
| Installation | `install.py`, `manifest.json`, `patches/processing-hires.json` | Original/baseline matching, hash verification and rollback |

## Can it be adapted to ImageGen?

**Yes, as a designed port — not as a drop-in install.** The source is tightly coupled to A1111's Gradio app, A1111-specific `modules.*` imports, LoRA hooks, global `opts`, txt2img/img2img prompt IDs and model lifecycle. The installer deliberately rejects hosts that do not look like A1111, and its hash-based patch operations are not applicable to ImageGen.

A good ImageGen design would establish a `LoRA Studio Core` boundary with a typed activation specification, prompt parser/serializer, versioned preset store and pure curve/block functions. Then add an `ImageGen LoRA Adapter` that maps these specs onto ImageGen's actual inference backends, model-family capabilities, current registry/workspace state and GPU scheduling. The browser interface should be an ImageGen component rather than copied Gradio DOM mounting code. Preserve separate state per prompt/workspace and a single shared registry for reusable assets and presets.

Suggested port sequence: (1) audit code provenance and target ImageGen LoRA APIs; (2) match behavior with language-independent JSON contract fixtures; (3) port visual editor/draft workflow using ImageGen's existing frontend conventions; (4) implement one compatible model family and benchmark; (5) add SDXL/SD3.5 and additional families only where confirmed, with explicit unsupported notices; (6) verify repeated LoRAs, Hires, masks, VRAM switching and output equivalence with deterministic fixtures. A `Flux` metadata label or a 61-weight table is not evidence of complete Flux generation support.

## Verification and limits

The original Russian v15 README reports 142 CPU runtime checks, 64 parser/API/config checks plus 13 subtests, 22 Node tests and 62 UI assertions, as well as an install/rollback comparison. These are the *original author's reported results*, not GPU/Gradio acceptance. The author explicitly notes that real GPU inference, arbitrary user models and production use were not validated.

For the multilingual fork in this archive, the offline installer verification, original Node DOM suites, additional locale unit tests, and installer unit tests are rerun separately. The language modifications do not change Python inference functions; they do change the browser Studio view, Extra Networks popup, CSS, and installer manifest. All changes require real-browser checking before deployment.

## Multilingual implementation

The fork adds `loraStudioI18n.js` plus separate `loraStudioLocale*.js` catalogs for English, Russian, French, Spanish, Italian, Japanese, Simplified Chinese, Brazilian Portuguese, German and Korean. A selector appears in each Studio header; the choice persists in browser `localStorage` and is shared between txt2img/img2img. Text is localized only at UI call sites. Model names, LoRA prompt syntax, numerical values, filesystem paths, JSON preset identifiers and copied prompt text are deliberately unaffected. Unknown translation keys fall back to English.

**Translation coverage is currently partial:** 67 manually translated key strings in each non-English catalog, against 142 static English UI strings indexed in the initial extraction. Variable text, longer help text, backend diagnostics and other A1111 UI elements are not fully translated. This is an experimental multilingual source edition, not a declaration of full internationalization. See `I18N_GUIDE_EN.md` and `LOCALE_COVERAGE.json`.
