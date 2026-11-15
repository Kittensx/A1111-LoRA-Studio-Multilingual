# A1111 LoRA Studio Pack v15 — Full v14 Audit and Confirmed Fixes

## Current status (2026-11-13)

Inconsistencies among LoRA payload/metadata/hash and two reproduced
`Options.set` races were fixed. Legacy cache migration, alias/descriptor
refresh, checks for file changes during reads, and HTTP 409 for conflicting
settings changes were added. While a callback is running, another change to
the same setting and saving an intermediate config are rejected: retry later.
Calculating a verified hash for the first time may take extra time.

Studio now clearly differentiates TE at encoding from per-step U-Net,
file-key coverage from actual application, and declared metadata hashes from
verified hashes. Existing defensive refusals for incompatible TI and
mixed-LoRA batches are preserved.

Full status of all 16 points in the supplied audit: **AUDIT_V15_EN.md**.
`text_projection`, `text_encoders.clip_g...` export, runtime coverage,
partial-encoder TI warnings, dual TI training, independent spans, and real
per-step TE remain unresolved. Support for all SD3/Flux adapters is not claimed.

On an actually installed temporary copy: **142 CPU runtime**, **64
parser/API/config + 13 subtests**, **22 Node / 62 UI assertions**. Repeat
check: 73/73 SAME; rollback restored 59 changed paths and all 73 paths matched
the original test base byte-for-byte. Installer: 26 PASS / 1 OS-symlink SKIP.
114 assets / 73 operations. GPU, actual Gradio, user models, and Drive/FUSE
were not tested. This is a candidate, not proven production readiness.

The working project on D: was updated. Nothing was installed to `/content/A1111`.
Extract v15 outside A1111 and fully stop WebUI before installing:

```bash
python install.py verify
python install.py check /content/A1111
python install.py install /content/A1111
```

On UNKNOWN, compare files first; **do not bypass checks blindly**. The installer
prints a backup ID and rollback command. Fully restart Python/WebUI after
installation. Older v8–v14 archives remain unchanged. Neither Torch nor other
dependencies are installed by this package.

Release history is preserved below. Earlier claims that Options was fully
resolved in v13 have been qualified by the new audit; v15 status above takes
precedence.

---

# A1111 LoRA Studio Pack v14 — In-Depth LoRA Cache Audit

## v14 audit (2026-11-12)

v14 includes v13 and fixes stale LoRA cache after same-size replacement with
preserved mtime, as well as rejection of a LoRA changed during reading.
Details, reproduction, and limits: **DEEP_AUDIT_V14_EN.md**.

Complete CPU matrix: 133 runtime, 62 parser/API/config (+13 subtests), 22 Node
(61 UI assertions). GPU/Gradio acceptance still not performed. v13 and older
archives were unchanged.

---

# A1111 LoRA Studio Pack v13 — Self-Audit and External Audit Fixes

## v13 audit (2026-11-12)

An independent follow-up audit of v12 found and fixed one Options concurrency
edge case: `run_callbacks=False` can no longer accidentally trigger a delayed
callback when the same setting is changed concurrently. A multithreaded
regression test was added. v12 and all earlier archives were preserved.
Details: **SELF_AUDIT_V13_EN.md**.

v13 contains all of v12; the remaining status and limitations below still apply.

---

# A1111 LoRA Studio Pack v12 — External Audit Fixes

## v12 status (2026-11-12)

Includes v11. Confirmed defects were fixed in late API import, REGION
installation, VAE selection/cache, forced checkpoint disk reads, SDXL TI
creation, old emphasis with dual embeddings, embedding traversal/cache, and
same-key Options callback ordering. Bounded safetensors-header reading,
explicit SD3/Flux LoRA labeling, and force-reload buttons in Settings →
Actions were added.

Full mapping of findings, resolutions, and open work: **EXTERNAL_AUDIT_EN.md**.
These are fixes for reproduced defects, not implementation of every
architectural idea. Independent CLIP-L/CLIP-G lengths, SDXL TI training,
per-step TE, a new memory manager, and VAE tiling were not implemented.
Incompatible embeddings are not automatically converted, truncated, or zero-padded.

Behavior changes: VAE Automatic uses an exact adjacent filename; duplicate
VAEs are shown with paths; unknown checkpoints lacking YAML are no longer
assumed to be SD1. A Flux label on LoRA does not add Flux generation support.

Verified after actual installation into a new local test copy: 131 Torch CPU
runtime, 61 parser/API/config (+13 subtests), 22 Node (61 UI assertions),
no-change repeat installation and byte-for-byte rollback.
Installer: 26 PASS / 1 OS-symlink SKIP. 110 assets, 70 operations.
Full WebUI/Gradio/CUDA and user models were not run here.

The working project on D: was updated. Nothing was installed on remote Colab.
v11 was preserved separately. Stop WebUI and extract v12 **outside** the A1111
directory:

```bash
python install.py verify
python install.py check /content/A1111
python install.py install /content/A1111
```

On UNKNOWN, stop and compare files; do not force blindly. The installer prints
backup and rollback commands. Fully restart Python/WebUI after installation.
The installer does not download dependencies.

Additionally:
`python tools/inspect_embedding_header.py /content/A1111/embeddings --clip-l 768 --clip-g 1280`
(on a single line). This checks headers, not tensor integrity or embedding
quality. Results and reproduction: VERIFICATION_EN.md.

Below is history, not current v12 metrics.

---

# A1111 LoRA Studio Pack v11 — LoRA Combinations and Block Controls

## v11 status (2026-11-11)

Includes v10 and fixes for Hires/curves, TE routing, LBW/LBWE/MID, mixed
start/stop, and CP DYN. Block sliders with exact input, visual insertion of a
point between absolute steps, correct display of custom presets, and panel-
width adaptation were added. Details: COMBINATIONS_AUDIT_EN.md.

**Behavior change:** short classic LBW tables now set missing known blocks to
zero, matching the original Block Weight extension. MIDD is a group, not M00.
ALL with 17 values does not mean all 26 blocks. To disable the mask, choose
All blocks / no module overrides. An ordinary tag without masks gets no new preset.

TE curves use their initial value; dynamic per-step TE re-encoding is not
implemented. Start/Stop applies only to U-Net; 1 means the end of the pass.
Per-block curves, history/undo, and a complete port of the three extensions
are not claimed. GPU/WAI/user safetensors and full Gradio need acceptance tests.

v10 is preserved separately. Installer downloads no dependencies. Stop WebUI,
extract the package **outside** target A1111, then run:

```bash
python install.py verify
python install.py check /content/A1111
python install.py install /content/A1111
```

Do not force blindly on UNKNOWN. Fully restart Python/WebUI after installation.
The installer prints rollback command and backup ID. The working project on D:
is already fixed; nothing was installed on remote Colab. Repeated package checks:
VERIFICATION_EN.md. Below is previous release history, not current v11 state.

## v10 status (2026-11-06)

Confirmed V9-1–5, C1–4, and L1–2 defects were fixed: SD3 Embeddings, wrapped
Embedding LoRA, first SDXL upsampler, TI/conditioning caches, safe preset
saving, low-VRAM rollback, zero mean in Emphasis, and invalid Origin.
Working project updated; this package also supports separate installation to
`/content/A1111`. It has NOT been installed automatically on remote Colab.

Checked from package: 104 CPU runtime, 54 parser/API/config, 23 NumPy loader;
installer 25 PASS / 1 SKIP. Node 17 PASS (51 UI assertions) in source project.
101 assets / 64 operations. Full CUDA/Gradio acceptance and the two user
safetensors were not run. v8/v9 preserved, lyco_helpers.py unchanged.
The new candidate is no guarantee of compatibility with every LoRA/checkpoint.

After stopping WebUI, from the extracted directory **outside** `/content/A1111`:

```bash
python install.py verify
python install.py check /content/A1111
python install.py install /content/A1111
```

On UNKNOWN/context mismatch, do not blindly force a replacement: compare the
relevant file. A full Python/WebUI restart is required after successful
installation. Backup and rollback command are printed by the installer.

Outside extensions-builtin, changes relative to v9 include modules/options.py,
modules/lowvram.py, modules/sd_emphasis.py,
modules/textual_inversion/textual_inversion.py, new modules/config_store.py,
and two additional targeted processing.py fragments. Presets/config are
written through a temporary file and replace. Serialization coordinates
Options API within one process; direct opts.data writes, other processes,
and competing whole-library edits are not protected by ETags. The UI is
unchanged; interface redesign/history is not claimed complete.

Earlier candidates and general commands are preserved below.

v9 includes v8 and fixes A1–A5 from the consolidated audit: secures LoRA API,
fixes repeated application of added bias and Q/K/V output bias, rejects invalid/
nonfinite TI vectors, and checks encoder widths separately. The UI is still
unchanged. The candidate needs acceptance on actual Gradio/GPU. New
lora_studio_access.py is required by lora_script.py; install the complete set.
Fully restart WebUI after installation; do not hot-reload hooks in a running
process. Historical v8 description remains for context; current checks are
in VERIFICATION_EN.md.

This candidate includes v7 plus protection from reusing a damaged model after
an independently failing VAE switch. This is not automatic rollback: explicitly
reload the checkpoint after such a failure. sd_vae.py requires the bundled
sd_models.py; do not move one file into an older A1111 on its own.
v1–v7 are preserved. UX_AUDIT_EN.md compares three source extensions and
outlines UI improvements. UI code was not rewritten in v8.
The full stabilization plan is not yet complete.

v8 contains all preceding Studio features and fixes to the meta-loader,
failure recovery, LoRA cache/combinations, manual_cast, patch manager, and
Textual Inversion. Details and limits:
`files/extensions-builtin/Lora/RUNTIME_STABILITY_FIXES.md`.

## Installation

Stop WebUI. Run from this directory outside target A1111:

```powershell
python install.py verify
python install.py check "<A1111_INSTALL_DIR>"
python install.py install "<A1111_INSTALL_DIR>"
```

On Colab use `/content/A1111` in place of the Windows path. After installation,
fully restart Python/WebUI; browser refresh is insufficient for hooks.

`check` does not change files. UNKNOWN means different source versions: inspect
differences first. `--allow-unrecognized` permits replacement of unknown full
files but does not disable context validation for targeted patches.

`processing.py` is modified in six context-aware fragments — Hires marker,
post-generation cleanup, two close methods, and two conditioning-cache additions.
The remaining REGION/TONEG code is preserved. Other sources listed in manifest
are replaced in full, so universal compatibility with any A1111, Forge/reForge/
SD.Next is not claimed.

The installer uses Python's standard library, does not download Torch or start
WebUI, and automatically saves originals being modified to
`.lora-studio-backups/<ID>`. Rollback commands:

```powershell
python install.py backups "<A1111_INSTALL_DIR>"
python install.py rollback "<A1111_INSTALL_DIR>" ID
```

## Tests

```powershell
python -B -m unittest discover -s tests -v
python -B files/test/test_runtime_stability.py
python -B files/test/test_runtime_recovery.py
python -B files/test/test_runtime_compatibility.py
python -B files/test/test_runtime_edgecases.py
python -B files/test/test_runtime_state_io.py
python -B files/test/test_runtime_vae_failures.py
python -B files/test/test_runtime_bias_embeddings.py
python -B files/test/test_lora_studio_access.py
```

Runtime suites use already installed Torch on CPU. A separate environment
`<HISTORICAL_AUDIT_ROOT>/venv` with `torch 2.10.0+cpu` was created earlier;
pip cache and temporary installation files also live on D:. It is a test
environment, not a complete A1111 venv.

test_lora_studio_access.py uses installed FastAPI/HTTPX, not Torch.
The installer installs no dependencies. On a server using --api-auth, valid
API credentials are allowed; Gradio with authentication also accepts a valid
session. If authentication is not configured, local mode stays open.
Browser writes from foreign Origins are rejected. Behind a reverse proxy,
Origin and external scheme/host need separate testing; do not disable auth
to work around a proxy error.

Actual generation, user checkpoints/LoRAs, and TI training were not run here.
Final acceptance requires the same SDXL in a working GPU environment.
Some FP8/TI and LoRA/manual_cast were tested on CPU; before porting ideas from
`<HISTORICAL_EXTENSIONS_DIR>`, actual third-party extensions, version matrix,
GPU/checkpoint, and real Gradio remain to be checked (the browser mock fixture
has already been tested).
SDXL TI training was not added: the trainer now reports a clear message
instead of failing late. Pickle protection remains enabled; the package updates
modules/safe.py, so compare any unknown version of that file first.
Do not copy lora_patches.py alone: it requires modules/patches.py from the
same candidate.
