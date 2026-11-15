# A1111 LoRA Studio Multilingual

**Unofficial community-maintained localization and development package for AUTOMATIC1111 LoRA Studio.**

This repository is the A1111 LoRA Studio Pack v15 code and installation package,
with a 10-language LoRA Studio UI, English/Russian paired documentation, tests,
and the original source archives for provenance. It is **not** an official
AUTOMATIC1111 release and is **not** integrated into ImageGen.

## Documentation

- [English — README](../docs/EN/README_EN.md) · [Русский — README](../docs/RUS/README_RUS.md)
- [English — localization guide](../docs/EN/I18N_GUIDE_EN.md) · [Русский — руководство по локализации](../docs/RUS/I18N_GUIDE_RUS.md)
- [English — code review and licensing notes](../docs/EN/CODE_REVIEW_EN.md) · [Русский — аудит кода](../docs/RUS/CODE_REVIEW_RUS.md)
- [English — runtime help authoring](../docs/EN/HELP_AUTHORING_EN.md) · [Русский — создание справки](../docs/RUS/HELP_AUTHORING_RUS.md)

Every project document in `docs/EN/` has a corresponding document in
`docs/RUS/`, sharing the same filename stem with `_EN.md` / `_RUS.md`.
The original snapshot and historical intermediate packages are preserved in `archive/`.

## Scope and compatibility

This is a **modification/installer package**, not an installable A1111 extension
that can be added through the Extensions tab. It modifies several A1111 core
files and its built-in LoRA extension. Install only into a compatible, backed-up
A1111 test installation, never Forge/SD.Next or ImageGen.

Stop A1111 completely and extract this repository outside the A1111 directory.
Perform an offline integrity check before inspecting the install plan:

```sh
python run.py verify
python run.py check /path/to/stable-diffusion-webui
```

The default installer copies **code and supporting runtime assets**, including the
new `extensions-builtin/Lora/help_documentation/` user-facing help articles.
Those articles power the in-app `?` buttons, searchable Help viewer and related
links. It does **not** install the six historical Markdown development reports;
those remain optional in `docs/EN/` and `docs/RUS/`.

Run `python run.py help-audit` to check user-facing articles and topic mappings,
and `python run.py docs-audit` to check the English/Russian development documents. To **optionally**
copy the six documents from either language folder using the same installer
backup and rollback mechanism, specify a language explicitly:

```sh
python run.py check /path/to/stable-diffusion-webui --docs-language EN
python run.py install /path/to/stable-diffusion-webui --docs-language EN
```

Use `RUS` instead of `EN` for Russian. Omitting `--docs-language` installs
only the **runtime** help articles, not the historical development reports. You can also run `python run.py docs-audit`,
`python run.py test`, and `python run.py smoke --help` for the paired-document
check, offline regression tests, and disposable-host integration test driver.

Do not blindly override `UNKNOWN` warnings. Read the installer output and
[English README](../docs/EN/README_EN.md) before any installation.

Offline tests:

```sh
python run.py test
```

The browser/Gradio/GPU integration and native-speaker translation review are
**not yet verified**. Localization catalogs cover 205 known UI messages per
language; this is not complete translation of AUTOMATIC1111 itself. The new
runtime help contains 9 article topics in English and Russian; articles in the
other 8 UI languages fall back to English until translated. Help is served
through read-only authenticated A1111 API routes. Adding an installed local
Markdown file in a supported locale folder does not require rebuilding the
application; refresh the Help viewer to discover it.

## Licensing and attribution

`LICENSE-A1111.txt` contains the **AGPL-3.0** license of the upstream A1111
code and must be retained with the affected source. This package includes
upstream A1111-derived files. The supplied snapshot did not separately
establish the copyright ownership or explicit licensing grant for every
LoRA Studio addition. We do **not** claim ownership of the original code or
assert that the entire package is available under a permissive license.
Review provenance and applicable redistribution obligations before
publishing, redistributing, or adapting the code. See the code review
under `docs/EN/` for the current findings.

**Contributions and corrections from the original developer are welcome.**
