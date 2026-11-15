# v15 Candidate Verification — 2026-11-13

Test base: local `stable-diffusion-webui-dev.zip` overlaid with local
`modules.zip`. This is not the same base composition as in the independent
Linux audit. New copy:
`<HISTORICAL_AUDIT_ROOT>/tmp/lora-v15-installed-kha0e4qa`.
Log: `evidence/v15-installed-smoke-accepted.log`.

- 142 Torch CPU runtime tests PASS, including eight new asset-identity
  scenarios and a separate test of five routes (four expected to be UNSUPPORTED).
- 64 parser/API/config tests PASS + 13 subtests PASS.
- 22 Node tests PASS, 62 UI assertions; no real browser/Gradio was run.
- Installer/header inspectors: 26 PASS, 1 SKIP (OS disallows creating symlinks).
- After installation 73/73 SAME; repeat planning requires no changes.
- Rollback of 59 actually modified/added paths; comparison of all 73 target
  paths with original bytes/absence succeeded.
- The 16 UNKNOWN entries are allowed ONLY in this fresh disposable test copy.
  The working installation still blocks unrecognized versions without an
  explicit flag.
- Ruff on changed Python and Node syntax PASS. `verify_package_v15.py` in the
  build workspace checks manifest integrity and previous releases.

Two intermediate smoke runs stopped at UI tests after curve titles were
renamed. Stable aria-labels were preserved and the title test updated with an
additional TE/UNet semantics check. The full repeat run above passed; failed
intermediate attempts are not presented as successful acceptance tests.

Caveats: tests use production AST and real Torch tensors but mock unavailable
WebUI services; new cache tests use dict instead of diskcache and a small stdlib
decoder instead of safetensors. Warnings about missing NumPy in the Torch venv
and old pytest `base_url` remain. GPU/quality/VRAM were not measured.
Exact SHA256 of the final ZIP appears in the external v15 receipt.

---

# v14 Candidate Verification — 2026-11-12

## In-depth LoRA cache verification

Real same-size replacement and file-changed-during-read scenarios were added
for the LoRA payload cache. See DEEP_AUDIT_V14_EN.md. v14 again passes all
install/rollback checks below; current source-project results: 133 runtime,
62 parser/API/config (+13 subtests), 22 Node/61 UI assertions.

---

# v13 Candidate Verification — 2026-11-12

## Additional independent audit

Added test for a concurrent data-only change to the same Options key during
a long-running callback. The test requires that the callback apply only to
the first call, while the second change with `run_callbacks=False` must not
trigger a hidden reload. Details and limits: SELF_AUDIT_V13_EN.md.

v13 again passes all installation/rollback checks listed below.

---

# v12 Candidate Verification — 2026-11-12

## Final test run on an installed copy

`tools/post_install_smoke.py` uses two already available local archives,
installs the package with normal install.py code, and then copies ONLY test
files. Runtime modules from reference/verification-only are not copied over
the installation. Nothing was downloaded; existing Torch CPU, pytest, and
Node were used.

- 131 Torch CPU/production-AST runtime tests PASS (including 19 new audit checks).
- 61 parser/API/config tests PASS, 13 subtests PASS.
- 22 Node tests PASS, 61 UI assertions in isolated DOM shim.
- Repeat install plan: all 70 operations SAME.
- Rollback: 56 modified files restored/removed; original bytes of all affected
  paths verified, including absence of added files after rollback.
- Preserved post-rollback test copy:
  `<HISTORICAL_AUDIT_ROOT>/tmp\lora-v12-installed-xggld_54`.
- Installer/inspector: 27 methods, 26 PASS / 1 SKIP (OS disallows symlink).
  "Installation interrupted" during the intentional-failure test is expected.
- `install.py check` of working project: 0 changes, 0 unrecognized; 110 assets,
  70 operations. Separate test rejects manifest missing a required REGION operation.
- Ruff on changed Python runtime/tests and package tools PASS; this is not
  repository-wide lint. `inspect_embedding_header.py --help` works without Torch.

Base: local Downloads/stable-diffusion-webui-dev.zip + source/modules.zip.
This base is not assumed to match the one from an outside audit. The 16 UNKNOWN
entries were allowed ONLY in the new test copy; the standard installer retains
UNKNOWN protection. During runtime tests, Torch's missing-NumPy warning and
checkpoint recovery messages in intentionally failing scenarios are expected.

Reproduction (from v12 root; supply your own existing paths):

```text
python -B tools/post_install_smoke.py --base-zip BASE.zip --modules-zip MODULES.zip --temp-root <HISTORICAL_AUDIT_ROOT>/tmp --torch-python <HISTORICAL_AUDIT_ROOT>/venv/Scripts/python.exe
python -B -m unittest discover -s tests
python -B install.py verify
```

The system Python used for smoke must already have pytest/FastAPI; separate
Python is used for Torch. The script does not install dependencies. The test
copy and backup remain on D:.

Full list of confirmations, compatibility changes, and limitations:
EXTERNAL_AUDIT_EN.md. Actual Gradio, CUDA/WAI, user TI/LoRA, image quality,
and VRAM were NOT checked in this run. The browser's 66 PASS below belong to
v11 and do not represent a new v12 browser run.

Historical results from previous versions follow.

---

# v11 Candidate Verification — 2026-11-11

## v11 source-project tests

112 Torch CPU/production-AST runtime tests PASS; 59 parser/API/config PASS and
13 subtests; 22 Node PASS with 61 UI assertions. The isolated fixture in a
real browser gave 66 PASS, including the panel's own width and dark/light
contrast without theme tokens. This is not full Gradio: the fixture mocks APIs
and CPU tests mock external services. The new matrix of 1/2/4 occurrences,
base/Hires, five keys, and 11 steps yields 330 scenarios. Reproduction details
and caveats: COMBINATIONS_AUDIT_EN.md.

## Repeat test run from the v11 package itself

- 112 Torch CPU runtime tests PASS.
- 59 parser/API/config tests PASS, 13 subtests PASS.
- 22 Node tests PASS, including 61 UI assertions in DOM shim.
- Installer and inspector: 25 PASS, 1 SKIP (OS cannot create symlink).
  Includes full installation/byte-for-byte rollback in a synthetic host on D:,
  interruption midway, and preservation of user changes on rollback.
- `install.py verify`: 104 assets PASS. Working-project `check`: 0 changes,
  0 unrecognized. 64 operations; install scope not expanded from v10.
- Ruff on four modified Python runtime and two new Python tests PASS.
  This is not repository-wide lint; ESLint was not installed.
- modules payload, processing patch, and lyco_helpers.py unchanged from v10.

Reproduction (from `files/`, with existing Node/Python/Torch/pytest):

```text
python -B -m unittest discover -s test -p "test_runtime_*.py"
node --test test/lora*.test.cjs
python -B -m pytest -q --noconftest -o addopts='' test/test_lora_interaction_contract.py test/test_config_store.py test/test_lora_studio_access.py test/test_lora_presets_transaction.py test/test_lora_advanced.py test/test_lora_schema.py test/test_extra_networks_parser.py test/test_lora_quick_presets.py
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` was used for pytest here; Torch CPU and
pytest ran in two different preexisting Python environments. From package root:
`python -B -m unittest discover -s tests -v`, `python install.py verify`.
For browser tests: serve `files/` with a local HTTP server, open
`test/lora_studio_fixture.html`, and click Run UI checks. Do not install
anything or start a public server solely for these tests.

The history of v10 and earlier packages follows.

# v10 Candidate Verification — 2026-11-06

## Current v10 run

- Source project and then package files: 104 runtime tests PASS, real Torch CPU.
- From package: 54 parser/API/config PASS (+13 subtests), 23 NumPy loader PASS.
- Final callback-lock regression: Options callback can wait for another thread
  updating Options. Config lock is not held while waiting for the GPU queue.
- Installer/inspector: 26 methods, 25 PASS / 1 OS-symlink SKIP; include synthetic
  full install/rollback. "Installation interrupted" in the failure-injection
  test is expected and does not mean an actual install failed.
- Node: 17 PASS, 51 UI assertions in source project; DOM shim, not Gradio.
- Earlier external reproducers: runtime/VAE 6 PASS, recovery 5 PASS,
  reported LoRA loading 4 PASS. SD3 standalone 5 PASS (including old baseline).
- Ruff on modified runtime/tests, except processing.py: PASS. processing.py
  has two preexisting F811 findings (sd_model/mask_blur), confirmed in backup.
  Other old full-modules findings remain; passing tests do not establish
  absence of other defects.
- Package verify PASS; working project check: changes=0, UNKNOWN=0.
- 101 assets / 64 operations. Source/reference processing snapshot and
  contextual patch agree; patch is idempotent. An old snapshot in
  files/modules/processing.py first caused a FAIL in the new cache test;
  packager was corrected and the entire package retested.

New tests exercise actual SD3/EmbeddingsWithFixes classes, both stages of
failing lowvram setup, LoRA merge/repeat/disable, real API refresh,
real temporary config files, and 24 concurrent Options transactions.
No existing safety assertions were weakened. Existing stubs gained
ensure_model_compatibility / set_and_save; model_hijack was added to the AST
environment. I/O fault injection moved from direct open to the new atomic-write
boundary, with actual checks of fsync/replace failure, unchanged file/memory,
and retry.

Not tested: particular Herrscher/IllustriousFlowmatch safetensors,
WAI-Illustrious-17 generation, CUDA/bfloat16/FP8 kernels, full Gradio/auth/proxy,
interprocess config writers, or arbitrary external callbacks. Nothing was
downloaded or installed; existing Torch was used only from D:.
The historical v9 results below remain for reference.

## Current v9 changes

A1–A5 were fixed in networks.py, scripts/lora_script.py, new
lora_studio_access.py, and modules/textual_inversion/textual_inversion.py.
A new bias helper was included in the existing AST harness without weakening
assertions. Nine CPU/AST and six in-process FastAPI methods were added.
Before fixes, reproducers returned five runtime FAIL and one API FAIL;
afterward, six runtime/VAE PASS and one API PASS.

Runtime: 81 PASS; parser/schema/presets: 37 PASS; API: 6 PASS; Node: 17 PASS
with 51 UI assertions. Float8 tested as CPU storage, not a CUDA kernel.
NaN/±Inf, float64→float32 overflow, bad dimensions, and partial SDXL pairs
are rejected. Widths tested on nonstandard dimensions, not just 768/1280.
Bias: Linear/Conv2d, MHA, packed QKV; original None/ordinary bias, fp16,
repeat, strength changes, multiple adapters, rollback/disable, separate Float8
bias test.

API: anonymous/wrong/malformed Basic, valid Basic, Gradio cookie, mixed mode,
forged/session-without-auth, no-auth, and cross-origin write. Real Gradio
was neither installed nor run; the pinned 3.41.2 auth/tokens contract was
checked against official sources, and app.auth/app.tokens were mocked in tests.
Do not claim verified integration with arbitrary Gradio versions or reverse proxies.

Backup: <HISTORICAL_AUDIT_ROOT>/before-20261105-a1-a5.
v8 and ZIP were preserved intact. UI, VAE implementation, defaults, and
lyco_helpers.py were unchanged. Nothing was installed. The independent TI test
was updated for the new early refusal: actual production import with real
encoder-width wrappers replaced the prebuilt stub, while the assertion that
registration does not happen remains.

Open issues: actual GPU/Gradio acceptance, U1/UI redesign, arbitrary external
hooks, and failure/concurrent writes of shared config. A1 authentication
is not a complete security audit of A1111. Historical sections below
refer to v8/v7.

## v9 package recheck

Final v9 package verification: 81 runtime PASS, 6 API PASS,
23 NumPy loading PASS, installer/inspector 25 PASS + 1 OS-symlink SKIP
(including synthetic install/rollback). 87 manifest assets, 60 operations;
project check: Changes 0 / UNKNOWN 0. Ruff on modified Python PASS with
project config. NumPy routing harness updated for expected_shapes with an
assertion that values are passed; earlier assertions retained. The older
independent v3 audit again reported 5 PASS.

## v8 history

## Current v8 addition

Among runtime files, modules/sd_vae.py changed. Before the fix, seven failure
points in standalone VAE reload left a partial model accessible; now the
object is removed from active/cache, the VAE marker is cleared, and the
original exception is preserved. A resolve error before changing the model
does not remove a healthy object. The previous weights are not automatically
restored: explicit checkpoint reload is needed.

New test_runtime_vae_failures.py: 3 methods covering seven mutation failure
stages, preflight, success/no-op, and a low-VRAM service-double branch.
Original runtime suite: 72 PASS. Node 17 PASS including 51 DOM assertions.
No new dependencies. Backup: <HISTORICAL_AUDIT_ROOT>/before-20261105-vae-ui.

Real browser at test/lora_studio_fixture.html: 55 PASS including four contrast
modes. This is a mock-API fixture, not Gradio. UX gaps against the original
Weight Helper / Block Weight / LoRA Control were found: UX_AUDIT_EN.md.
Plans were clarified; UI code, LoRA defaults, and lyco_helpers.py were unchanged.
Actual CUDA/Gradio, backend hooks, projection output bias, and custom migrations
remain untested. Older results below are v7 history, not claims that every old
test was rerun now.

## v8 package recheck

Final v8 recheck: 72 runtime PASS and 23 NumPy loading PASS from package;
installer 25 PASS + 1 OS-symlink SKIP (26 total), including synthetic
install/rollback. Sources: parser/schema/presets 37 PASS, independent
regression 5 PASS. 84 manifest assets, 59 operations; working-project check:
Changes 0 / UNKNOWN 0. Ruff on modified Python PASS with project configuration.
Old NumPy warnings in the D: venv and pytest base_url were not fixed by
installing dependencies. Static server lacked SourceSansPro CSS; screenshot
uses fallback font. This fixture does not establish exact real-Gradio dimensions.

## v7 history

## Current v7 changes and results

Three runtime files: modules/sd_disable_initialization.py, modules/sd_models.py,
modules/textual_inversion/textual_inversion.py. Added test_runtime_state_io.py
(13 tests); AST harnesses of three older suites were given dependencies for new
extracted helpers, without weakening existing assertions.

Before changes, new tests reproduced lost metadata version, dict extra_state
failure, tensor extra_state becoming meta, ignoring materialized parameter dtype,
and damage to previous .pt/.optim on serialization failure. Incorrect cached
FP8 decisions/global mode were also reproduced. A follow-up check of the new
save path found and fixed a legacy-sidecar fallback issue.

- Meta: metadata and extra_state preserved; dtype conversion applied on
  materialization; first_stage_model exception and assign=True checked.
- TI: both payloads are prepared in temporary files before replacement;
  an individual os.replace is atomic, the pair is not. Serialization failure
  preserves earlier files. Failure of the second replace leaves a valid
  new .pt and stale .optim, which is rejected by SHA256. A marker in the new
  .pt prohibits fallback to an old checksum-only .optim. A genuine legacy
  pair retains backward compatibility.
- FP8: actual target storage is used instead of another model's global mode;
  the exact checkpoint is reread on settings mismatch. Four desired/stored
  combinations, preflight failure, and upcasting real Float8 tensors before
  loading under global fp8=False were tested.

Working-project tests: 69 CPU/AST runtime PASS (56 old + 13 new), independent
audit 5 PASS, parser/schema/presets 37 PASS, Node 17 PASS with 51 DOM assertions,
Ruff on modified Python PASS. Torch 2.10.0+cpu/Python 3.12.10; prior warnings
about missing NumPy in the D: venv and pytest base_url remain.

Pre-change backup: <HISTORICAL_AUDIT_ROOT>/before-20261104-state-io.
v6 and its ZIP were unchanged. UI/LoRA defaults/lyco_helpers.py unchanged.
Nothing downloaded or installed. In-project plan updated.

Limitations: actual GPU generation, browser/Gradio, user models, training loop,
external backend hooks, and version matrix were not run. Optimizer SHA256 binding
checks correspondence, not a signature or replacement for safe-unpickle.
Concurrent writers and power-loss durability on every filesystem are not
guaranteed. Custom meta migrations before super, projection output bias, and
standalone VAE reload errors remain separate checks.

Repeat run from v7 package itself: 69 runtime PASS, 23 NumPy loading PASS,
installer/inspector 25 PASS and 1 SKIP (OS disallows symlink), 26 total.
Full payload installed and reverted byte-for-byte in a synthetic copy on D:.
82 manifest assets, 58 operations; working-project check: Changes 0, UNKNOWN 0.
Ruff PASS using source project's pyproject.toml configuration. Without that,
package-default lint flags three preexisting E721 findings intentionally
ignored by the project.

---

# History: v6 Candidate Verification — 2026-11-04

## v6 changes and reproduction

Before runtime changes, a new suite of seven methods produced seven failures
and two errors counting subtests. Causes were grouped into three fixes:

- modules/sd_models.py: do not leave a damaged cached checkpoint after
  transfer/VAE failure; checkpoint/hash settings update only after VAE success.
  Other cached models survive; automatic restoration of the former active
  object is not claimed.
- extensions-builtin/Lora/networks.py: projected QkvLinear LoRA accumulates
  adapters in a working weight, uses fp16 cache, and casts only once at end.
  Prior in-place Float8 + float failed. Disable/error restores storage backup.
- modules/sd_disable_initialization.py: rollback partial installation, raw
  classmethod descriptors, owner check, independent cleanup/retry, and preserve
  primary error. Separate nested contexts are supported; re-entering the same
  active instance is forbidden.

The test/test_runtime_edgecases.py suite was expanded to 11 methods. Checks
include all eight failure positions while installing meta-loader over
LoraPatches, restore failure and foreign owner, Class.method/Child.method,
two LoRAs with TE/DYN in packed FP8, cache/no-cache, late projection error,
and successful retry.

## v6 results

- 56 CPU/production-AST runtime tests PASS: 21 stability + 12 recovery +
  12 compatibility + 11 edgecases. Run both on working project and package.
- Older independent reaudit_v3_20261103.py: 5 methods / 7 scenarios PASS.
- 23 NumPy LoRA loading tests PASS.
- 37 schema/parser/preset tests PASS; old pytest base_url warning remains.
- 17 Node tests PASS including 51 DOM assertions. CSS/layout not tested in Node.
- Installer/inspector: 25 PASS, 1 SKIP (OS disallows symlink), 26 total.
  Entire payload installed/restored byte-for-byte in synthetic host on D:.
- Ruff on new/modified Python runtime/tests/build/install files PASS.
- verify/check: 81 assets, 58 operations; current project: Changes 0, UNKNOWN 0.

Runtime tests use real Torch 2.10.0+cpu/Python 3.12.10 but extract production
definitions with AST and mock missing WebUI services. Torch's warning about
missing NumPy in D: venv remains; NumPy tests ran separately in an existing
system Python. Nothing was installed.

## v6 sources and restoration

Sources already changed in the working project on D:. Pre-change snapshot:
<HISTORICAL_AUDIT_ROOT>/before-20261104-edgecases.
This is a separate direct backup; it is NOT an install.py journal ID.

v6 is based on v5. v5 and its ZIP were preserved. processing.py remains a
four-hunk contextual patch; other runtime operations are known complete files.
The package is not universal across arbitrary forks. Do not copy hooks on
their own without matching modules/patches.py. lyco_helpers.py unchanged.

LORA_EVOLUTION_PLAN.md was clarified: runtime and real acceptance are mandatory
before new architecture. UI/default weights did not change in this step.

## Open v6 checks

The machine has an RTX 3050 Ti Laptop 4096 MiB (driver 566.07), but CPU Torch
was used here. The tested project had no checkpoint .ckpt/.safetensors; WebUI
was not listening on 7860/7861/7862. Disks outside the project were not
scanned for models.

Not performed: actual SD1.5/SDXL/CEB/Velvette, CUDA forward/FP8, VRAM,
generation with external attention extensions, actual Gradio UI fields/colors/
responsiveness, and Torch/transformers version matrix. SDXL TI training
is unsupported.

Other open boundaries: arbitrary projection output bias; extra_state and
metadata in destructive meta loader; caches on FP8 setting change; standalone
VAE reload failures; low-VRAM/backend lifecycle; atomic TI/optimizer save.
Not all risks have been reproduced; none are declared fixed or broken without
testing. Unknown formats must not be accepted with silent data loss.

v6 is a candidate with locally validated fixes, not the end of the full plan
or proof of improved image style/quality.

---

# History: v5 Verification (State Before v6)

## Changes added since v4

- modules/safe.py: allow exact ZIP names .format_version and
  .storage_alignment written by installed Torch. No wildcards for arbitrary
  files or fallback disabling pickle checks were added.
- modules/textual_inversion/textual_inversion.py: SDXL clip_l/clip_g separated
  from metadata; shapes/vector counts validated; reloadable save and checksum
  of both matrices implemented. Classic TI format/checksum unchanged.
- SDXL TI training was not implemented. The old single-encoder trainer now
  refuses explicitly before changing job state or preparing dataset, rather
  than failing late with AttributeError.
- modules/devices.py: cleanup covers partial manual_cast installation failure,
  preserves forward inheritance, and does not remove foreign org_forward.
- Added test/test_runtime_compatibility.py, 12 tests.

## Reproduction before fixes

The new suite returned 3 FAIL and 2 ERROR out of the initial eight tests:
manual_cast left hooks after failed setup; safe-loader rejected its own .pt
because of .format_version; SDXL TI failed on metadata/checksum and did not
check matching vector counts. Classic TI round-trip and delayed FP8/fp16 cache
had already passed before fixes. FP8 runtime was not changed in this step.

## Repeat checks after fixes

- 21 test_runtime_stability + 12 test_runtime_recovery + 12
  test_runtime_compatibility: 45 CPU/production-AST tests PASS.
- Compatibility: classic TI and optimizer-state save/read, SDXL metadata
  round-trip, checksum of both encoders, mismatched counts, legacy Torch
  serialization, unknown ZIP member and harmless forbidden pickle global,
  early refusal of SDXL training, real float8 storage with delayed fp16-cache
  transfer, manual_cast setup failure/inheritance/nesting and LoRA+cast in both orders.
- A separate subprocess imported the complete current modules/safe.py with
  real Torch 2.10.0+cpu and already available NumPy 2.4.2. Globally wrapped
  torch.load successfully read a self-saved tensor .pt. Only WebUI config/
  reporting was mocked.
- 23 NumPy LoRA loader tests: PASS.
- 37 schema/parser/preset and 17 Node tests (51 DOM checks): PASS.
  Previous pytest warning about base_url remains.
- Installer/inspector: 25 PASS, 1 SKIP out of 26 (OS disallows symlink).
  Entire payload installed and restored byte-for-byte in synthetic host on D:.
- Ruff on modified runtime files, tests, and build/install scripts: PASS.

## Sources and package

Working project on D: was modified directly. Three runtime files and docs
were saved before editing:
<HISTORICAL_AUDIT_ROOT>/before-20261104-compat

v5 includes everything from v4, 80 verifiable assets and 58 install operations.
New complete replacement modules/safe.py is permitted by the installer only
for that particular path; unknown versions still require review and explicit
opt-in. processing.py retains four contextual fragments from v4. v4 and its
ZIP were unchanged. lyco_helpers.py unchanged. No libraries installed.

## Limitations

This is not completion of the entire plan or a guarantee of safety for
arbitrary pickle. Unknown/untrusted models must not be loaded with protections
disabled.

Torch was checked in one CPU environment (2.10.0, Python 3.12.10). Most tests
isolate production functions via AST; they do not start the server or an actual
checkpoint. SDXL TI matrices in tests are small synthetic ones, not user files.
FP8 was checked for CPU storage/merge/restore, not CUDA inference or benchmark.
Optimizer-state round-trip is not support for SDXL TI training.

Remaining: actual SD1.5/SDXL and CEB/Velvette, GPU generation, browser,
different library versions, real external attention extensions, complex
cache-hit/VAE/backend failures, and broader ownership of temporary loader
hooks. Porting ideas from other WebUIs is deferred until after these tests.
