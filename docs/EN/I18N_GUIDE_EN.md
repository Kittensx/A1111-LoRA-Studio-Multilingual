# LoRA Studio multilingual UI — implementation and verification guide

This package is an **experimental A1111 LoRA Studio Pack v15 fork**. It is not an ImageGen integration and must not be installed into ImageGen. Keep its A1111 code and license notices separate from any future ImageGen codebase. Architectural isolation is not, by itself, proof of license compatibility; review the license obligations before distributing or connecting a future extension.

## Language support and scope

The LoRA Studio UI provides 10 selectable languages: English (`en`), Russian (`ru`), French (`fr`), Spanish (`es`), Italian (`it`), Japanese (`ja`), Simplified Chinese (`zh-CN`), Brazilian Portuguese (`pt-BR`), German (`de`) and Korean (`ko`). Selection is retained in browser storage at `loraStudio.language.v1`. The browser language is used initially when a supported language or fallback is recognized.

The release contains **195 text messages per locale** in the independent dictionaries. These include most LoRA Studio controls, instructional tooltips, status messages, and built-in quick-recipe titles/descriptions. English remains a safe fallback for text outside those catalogs, including some dynamic built-in descriptions, backend validation errors, and the surrounding A1111/Gradio UI. This count is **catalog coverage, not a guarantee that every possible displayed string has been localized**. Translation wording should still be reviewed by native speakers.

### Separation from LoRA data

- `files/javascript/loraStudioI18n.js`: common runtime: `register`, `t`, `setLocale`, `getLocale`, `canonical`, `supported`, and `missingKeys` diagnostics. Locale registration is deferred safely if dictionary scripts load before the engine.
- `files/javascript/loraStudioLocale*.js`: independent UTF-8 dictionaries with stable `studio.*` message IDs. Editing translations does not require changing the rendering or execution algorithms.
- `files/javascript/loraStudioView.js` and `files/javascript/extraNetworks.js`: select UI labels at display time. The Studio editor re-renders on a locale change while leaving the original prompt string and draft model untouched. Recognized status messages can be re-rendered in the new language.
- Names of LoRA models, raw prompt tags, raw masks, custom preset titles, user-entered strings, JSON and generation parameters **must not be translated**. Built-in preset display copy is translated only when the corresponding built-in mode is shown.

## Add or modify UI text

1. Add one stable `studio.*` key to the English dictionary, and the same key to all other language files (or accept English fallback until translated).
2. Use `t("studio.namespace.stable_key", {name: modelName})` for UI text, with placeholders like `{name}` unchanged in all translations. Use `textContent` rather than HTML insertion for untrusted interpolated values.
3. Avoid translating programmatic codes, model keys and user data. Translate presentation labels only.
4. Run the Node localization, model/preset and DOM tests. The Node DOM runner also verifies language switching preserves the prompt bytes.
5. Recompute the asset hashes in the existing `manifest.json`; the installer refuses modified assets with stale hashes.

## Offline checks

```sh
python install.py verify
python -m unittest discover -s tests -v
node --test files/test/*.test.cjs
```

For interactive checks, open `files/test/lora_studio_fixture.html` in a real Chromium-based browser with local-file access enabled, or run the optional `files/test/lora_studio_browser_automated.html` fixture. Inspect dark/light modes, Japanese and Chinese fonts, input-method composition, narrow widths, keyboard access, both txt2img/img2img and LoRA draft/preset preservation. The fixture does not use GPU inference.

**Verification boundaries:** Node event/DOM simulations and offline installer/unit checks do not prove appearance in a browser or successful operation inside a live A1111 release. Headless Chromium could not be started reliably in the authoring container, and no live A1111/Gradio/GPU installation or generation tests were performed. Installation modifies A1111 files; use the installer preflight/backup/rollback process against a compatible disposable test installation before any production use.
