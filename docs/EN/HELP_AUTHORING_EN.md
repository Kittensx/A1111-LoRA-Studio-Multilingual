# LoRA Studio runtime help: authoring guide

The **runtime** help system is independent of the historical technical documents in `docs/EN/` and `docs/RUS/`. Those older audits are not copied into A1111 by default. This directory contains contributor information; actual interactive articles live under `files/extensions-builtin/Lora/help_documentation/`.

## Layout

```text
files/extensions-builtin/Lora/
  lora_help_registry.py                # Read-only runtime scanner
  help_documentation/
    en/getting-started.md              # Canonical English fallback
    ru/getting-started.md              # Russian translation
    fr/getting-started.md              # Optional French translation
    registry/getting-started.json      # Optional cross-language metadata
```

The registry scans locale subfolders **each time an API request arrives**, so extra `.md` files placed into an installed A1111 help folder become discoverable after opening or reloading Help. New Python/JavaScript code must still be installed and A1111 restarted. The installer also distributes the bundled articles; an author can create additional local article files without repackaging.

## Create a new topic

1. Choose a unique lowercase slug such as `negative-weights` (letters, numbers, hyphens only).
2. Create `help_documentation/en/negative-weights.md`, starting with `# Title`, followed by a short summary paragraph and additional Markdown.
3. Translate it with **the same slug** into `ru/negative-weights.md`, `es/negative-weights.md`, or any other supported locale directory. A missing translation automatically falls back to `en`.
4. Optionally create `help_documentation/registry/negative-weights.json` for crosslinks, keywords, sort order, and UI context mappings.
5. Run `python run.py help-audit` before sharing the project.

### Optional JSON format

```json
{
  "order": 10,
  "contexts": ["studio.negative_weights"],
  "related": ["weights-and-rank", "curves-and-timing"],
  "tags": ["weights", "negative"]
}
```

No `.json` is necessary for an article that needs no related links or button mappings. The viewer derives translated titles and summaries from each Markdown file. The JSON is language-neutral; its `related` entries are **topic IDs**, not filenames or absolute paths. UI controls pass a stable `studio.*` context ID to the registry, which resolves it to whichever topic owns the context. Thus the article can be reassigned by changing JSON without changing JS.

## Runtime API

- `GET /sdapi/v1/lora-studio/help?locale=fr` returns the searchable topic registry, titles, summaries, context IDs and related IDs.
- `GET /sdapi/v1/lora-studio/help/getting-started?locale=ru` returns the selected article Markdown and fallback metadata.
- Both routes use the same authentication policy as existing A1111 LoRA Studio API routes. Neither endpoint changes prompts, presets or generation settings.

## Safety and limits

The help viewer supports a limited Markdown subset (headings, paragraphs, list items, bold, inline/code blocks and safe links). Raw HTML is always displayed as text, never executed. Keep articles below 64 KiB and JSON sidecars below 16 KiB. Article names must be safe slugs and all content must remain within their designated help subfolders. Do not put secrets or personal data into help files.

The interface has 10 translated languages. This release provides **nine complete help article topics in English and Russian**; the other eight interface locales currently fall back to English for article content. Native-speaker review of new translations is recommended.

## How to verify

```powershell
run.bat verify
run.bat help-audit
run.bat docs-audit
run.bat test
```

In an actual A1111 session, click the `?` beside a LoRA Studio control, try Search and Related topics, switch languages, and confirm the prompt has not changed. Live A1111/browser/GPU verification is still required for release readiness.
