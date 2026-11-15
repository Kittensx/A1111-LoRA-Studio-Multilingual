# Consolidated Re-Audit of UI, LoRA, VAE, and Embeddings

## Current status — 2026-11-06, v10

Confirmed V9-1–5, C1–4, and L1–2 findings have been fixed in the working
project and the separate v10 candidate. 104 CPU runtime / 54 parser+API+config
PASS; 13 subtests. The Options/callback lock ordering was also fixed during
the final review. Details: FIXES_V10_2026-11-06_EN.md and
RUNTIME_STABILITY_FIXES.md. First complete actual WAI/LoRA/TI GPU and Gradio
acceptance testing, then UX U1/history/blocks. The audit history below is
preserved: earlier FAIL and "not fixed" labels refer to v8/v9, not current
code. These local tests do not replace generation using the user's files.

Date: 2026-11-05, additional review following the v8 candidate release.
Project: `<A1111_SOURCE>`.
All relative source paths below refer to this project.

## Status after v9 fixes (2026-11-05)

**A1–A5 were fixed in working code and included in the separate v9 candidate.**
The material below is retained as history of the original findings. Its FAIL
results reflect the pre-fix state, not the current result. Full project
acceptance remains incomplete.

- A1: shared access dependency on all six LoRA API endpoints. Basic API
  credentials or a valid Gradio session; anonymous/forged cookies cannot write.
  Cross-origin browser writes are rejected. Six in-process API test methods PASS.
- A2/A3: bias backup tied to weight-backup initialization; original None stays
  None. Q/K/V bias is applied/restored for MHA and packed QKV, including
  partially absent bias. Float8 bias is converted to a supported floating
  dtype before summation. Repeat/change/stack/disable/rollback and fp16/FP8
  storage are covered by tests.
- A4/A5: actual TI consumer widths, finiteness/shape and overflow on float32
  conversion are checked before device transfer. Standalone and bundled-TI
  imports are included; registration rechecks model compatibility.
- 81 runtime PASS, 37 parser/schema/preset PASS, 17 Node PASS (51 UI assertions).
  Independent runtime/VAE reproducer: 6 PASS; API reproducer: 1 PASS.
  The prebuilt stub was removed from the TI harness: actual imports with
  model wrappers and explicit early rejection without registration are tested.
  Safety assertions were not weakened.
- From v9: 81 runtime PASS, 6 API PASS, 23 NumPy loading PASS,
  installer/inspector 25 PASS + 1 OS-symlink SKIP; 87 assets / 60 operations.
  Package check: 0 changes / 0 UNKNOWN. The old NumPy harness gained the new
  expected_shapes argument and an assertion that it is passed; this checks
  routing, not TI mathematics itself.

Runtime changes: networks.py, scripts/lora_script.py, new lora_studio_access.py,
modules/textual_inversion/textual_inversion.py. VAE, UI code, default weights,
and lyco_helpers.py were not changed. Backup:
<HISTORICAL_AUDIT_ROOT>/before-20261105-a1-a5.
v8/ZIP were preserved. v9 installation requires a full WebUI restart.

Next work: Gradio/GPU integration (including API-only, UI auth, mixed auth,
and proxy), UX U1/compact window/blocks/history, failure handling and concurrent
saving of shared config. A1 is not a complete security audit of A1111.
Real Gradio was not installed: the session contract was inspected in the
official pinned 3.41.2 source; HTTP tests used FastAPI with mocked
app.auth/app.tokens.
Source: https://github.com/gradio-app/gradio/blob/gradio%403.41.2/gradio/routes.py

## Overall conclusion and boundaries

**The plan needs additions. v8 is a testable candidate, not a finally accepted
release.** The previous 72 runtime tests pass, but new tests revealed five
groups of runtime/API problems: Studio authorization, two separate bias-handling
defects, insufficient SDXL TI dimension validation, and missed NaN checks.
UI usability and focus-return problems were also confirmed. Whether each defect
was inherited or introduced by our patches was not established in this pass.

This was a selective in-depth review, not proof that all other files/extensions
are defect-free. Runtime, JavaScript, and the v8 package were not changed.
Reproducers and documentation were added. Nothing was downloaded or installed.
`lyco_helpers.py`, source extensions, and user models were not modified.

## Confirmed findings

| ID | Priority | Observation | Status |
|---|---|---|---|
| A1 | P1 when reachable over a network | Studio library PUT bypasses `--api-auth` | Reproduced with in-process HTTP |
| A2 | P1 for adapters with output bias | Second application to an originally biasless Linear crashes | Reproduced on CPU using production Full adapter |
| A3 | P1 for projected bias | Q/K/V output bias silently discarded in MHA and packed QKV | Two CPU reproductions |
| A4 | P2 | SDXL TI is registered because summed widths match, then fails during conditioning | Registration and consumer error reproduced |
| A5 | P2 | SDXL TI containing NaN is accepted by loader | Reproduced; no early refusal |
| U1 | P2 UX | Focus goes to BODY after Cancel; quick actions require scrolling | Browser verified |

### A1. Studio route authorization

`extensions-builtin/Lora/scripts/lora_script.py:140–160`: GET/PUT are registered
directly on `app`. `modules/api/api.py:274` adds `Depends(self.auth)` only to
routes created via `Api.add_api_route`. `api_only` creates a plain FastAPI app;
setup_middleware adds GZip/CORS, not a shared Basic-auth guard.

The test used a real FastAPI TestClient without a network listener: a protected
control route returned 401; production Studio PUT without credentials returned
**200** and invoked the mocked save once. Real config was not written.
This is not just cosmetic: a network client could modify the library despite
an expectation of API protection. Exposure differs for localhost-only setups;
actual internet exposure of the user's installation was not assessed.

Resolution: shared access policy across all Studio endpoints (presets/coverage/
sidecar/refresh) compatible with API-only and Gradio-session modes. Do not
blindly add Basic auth to PUT alone: browser fetch from authenticated Gradio
must continue working without storing passwords in JS. Tests: anonymous
401/403, valid client read/write, bad credentials cannot write, unauthenticated
mode continues to work as expected. CORS is not authorization.

### A2. Absent bias is confused with absent backup

`extensions-builtin/Lora/networks.py:540–554`: None means both "no backup yet"
and "original layer had no bias." A Full adapter with `diff_b` creates bias
on its first merge. Applying it again tries to back up the already modified
bias and raises:
`no backup bias found and current bias are not unchanged`.

The reproducer uses actual `NetworkModuleFull`, real tensors, and production
merge/cache. This does not mean every ordinary LoRA contains output bias:
it concerns adapters producing `ex_bias`.

Resolution: explicitly distinguish initialized backup whose original was None
from an absent backup. Simply swallowing the exception is not enough.
Acceptance: first/repeated apply, strength change, two LoRAs, disable and
restore to **None**, rollback after failure; then Conv2d and MHA out-projection.
Check requires_grad/dtype/device.

### A3. Q/K/V bias not applied

`extensions-builtin/Lora/networks.py:601–603,633–635`: the second return value
from `calc_updown` is assigned to `_`. A production Full adapter returns
nonzero `diff_b`, but MHA.in_proj_bias / packed QKV.bias remain unchanged.
Reproduced on two paths with zero weight delta and unit bias delta.

Resolution: fully apply/restore Q/K/V bias in the correct packing, or explicitly
reject the unsupported format BEFORE changing the layer. Silent omission is
unacceptable. If explicit rejection is chosen, update the negative test for
that contract; current failing tests expect full application. Check cache
identity, original None, multiple adapters, rollback, and FP8. Fixing MHA
cannot mean only adding in_proj_bias: current backup/restore covers
out_proj.bias, so the whole lifecycle must change consistently.

### A4. SDXL Embedding: summed dimensions are insufficient

`modules/textual_inversion/textual_inversion.py:165,300,414`: the code checks
rank two, float type, and equal vector counts, but model compatibility is
reduced to a sum of widths. clip_l=1024 and clip_g=1024 pass expected=2048.
Then the actual `EmbeddingsWithFixes` with a 768-wide token embedding fails
when inserting clip_l=1024. The TI file has already been registered.

Resolution: obtain each encoder's requirements from the active model and
validate each part before registration/GPU transfer; diagnostics must name
file, encoder, expected width, and actual width. Do not hardcode 768/1280
for every future architecture. Tests: correct pair, wrong split with correct
sum, checkpoint switch, classic TI, missing part, wrong rank/vector count.

### A5. Early rejection of nonfinite TI weights

`Embedding.validate_sdxl` accepts float matrices containing NaN. The new
test expects diagnostic ValueError, but the loader returns Embedding.
No image generation with NaN was run: contaminated conditioning is a risk,
not a measured generation result in this audit. Inf was not separately reproduced.

Resolution: shared check of finiteness, shape, and nonempty vectors for
classic/SDXL before registration, with a clear error rather than silently
replacing NaN with zero. Cover NaN, ±Inf, empty/invalid matrices, and valid
large values. Do not conflate this check with allowing unsafe pickle.

## VAE: what was confirmed and what the tests do not promise

- Earlier tests of seven failure points, preflight, and success/no-op pass.
- A new check uses a real `.pt` in a temporary directory on D:, actual
  `load_state_dict` and production load_vae/_load_vae_dict/reload_vae_weights.
  A valid weight is written first; then a wrongly sized bias causes an
  exception. v8 removes the damaged object from active/cache, another cached
  model survives, loaded_vae_file is cleared, and the invalid VAE is not cached.
- This is **isolation of a damaged model**, not restoration of its original
  weights. The object remains partly modified; external references to it are
  not neutralized. An explicit checkpoint reload and clear UI message are needed.
- Important clarification of older tests: callback failure was injected into
  a mocked dispatcher. Actual `modules/script_callbacks.py:270–275` catches
  errors in individual extensions and only logs them. Thus v8 cannot be claimed
  to isolate the model automatically after every arbitrary extension callback
  failure.

Next VAE acceptance: A → external VAE → None/Automatic → B → cached A;
corrupt file/incorrect shape, error during Hires override, settings restoration
and next successful generation, fp16/bf16, and real low-VRAM. Also check
preflight VAE reading before removing hijack: currently some errors surface
after the modifying phase begins and conservatively disable the entire model.

## UI: follow-up review

The computer-use skill was used for a real-browser review of the local fixture.
**55 UI checks** passed again, including light/dark themes with/without tokens.
The narrow fixture confirms an overloaded popup: Apply/Cancel are below the
initial visible area. The fixture limits the app region, but does not test
all viewport sizes or mobile browsers.

New finding: after Cancel, `document.activeElement` is BODY; the original
prompt remains unchanged. Escape in code separately restores focus to prompt,
but the button does not. Add one closing mechanism that remembers/restores
the initiator and test Apply/Cancel/close icon/Escape, transition to Full Studio,
and tab switching.

Main shortcomings from the previous UX audit remain: block text inputs without
sliders are buried in Advanced, no variant history, overloaded non-draggable
popup. Trigger workflows, preview, keyboard actions, and curves also need
separate acceptance. Basic Studio attaches to `#<tab>_prompt > label > textarea`;
do not claim negative/third-party editors are supported merely because right-
click exists.

Limitations: mocked API, missing SourceSansPro CSS (fallback font), Gradio not
started. Unreadable white fields were not reproduced in the fixture. This cannot
be generalized to all user themes. Older extensions were not activated.

## Additional verification limits

- LoRA/FP8 cache, meta-loader, hook installation/removal order, and cleanup
  after exceptions are covered by existing CPU/AST regressions, not the full
  backend matrix.
- Staged TI saving and SHA256 optimizer binding are covered by existing tests;
  `.pt`+`.optim` is not an atomic pair, and concurrent writers/power loss are
  not guaranteed safe.
- Safe-unpickle was not disabled. Header inspector and installer are tested.
- Sidecar code bounds read size and adjacent directory, and presets have a
  schema/item limit. These are useful constraints but do not replace auth.
  File/symlink races and simultaneous library saves are not proven safe.
- Studio PUT calls shared Options.save: a direct open('w') of the entire config.
  Config write failure was not injected in this pass. Add disk-full/permissions/
  two-client checks and verify there is no false "saved" notification;
  an in-memory change is not equivalent to successful disk persistence.
  Do not fix only the frontend notification.
- Actual CUDA kernels, VRAM consumption/speed, SD1.5/SDXL images, style quality,
  TI training, external quantizers, and Python/Torch/Gradio versions have not
  been tested. SDXL TI training remains explicitly unsupported. New architecture
  is unnecessary for closing A1–A5 and must not conceal these defects.

## Results of this review pass

| Suite | Result |
|---|---|
| Existing CPU/AST runtime | 72 PASS |
| New `reaudit_v8_20261105.py` | 1 PASS (real partial VAE load), 5 substantive FAIL, 0 harness errors |
| New `reaudit_studio_api_20261105.py` | 1 FAIL: anonymous PUT=200, save invoked |
| Node | 17 PASS, including 51 UI assertions |
| Real browser fixture | 55 PASS; 51 scenarios overlap with Node, do not add as independent |
| Parser/schema/quick presets | 37 PASS |
| NumPy loading from v8 package | 23 PASS |
| v8 installer/header inspector | 25 PASS + 1 OS-symlink SKIP |
| v8 verify/check | 84 assets; 0 changes / 0 UNKNOWN for the project |

Both new reproducers have expected exit code 1 before the listed contracts are
fixed. They are stored separately from the working project/release package.
The first uses real Torch and a Full adapter, AST-extracted production functions,
and mocked WebUI services; the second uses real FastAPI without a network
listener or writes to the real config. This is not full A1111 startup.
The TI harness initially lacked a cache stub; the harness was fixed and the
final result above is free of that issue. NumPy loading was first mistakenly
run from a nonexistent source tests directory; rerun from the actual v8
package directory, 23 PASS. These harness errors are not WebUI bugs.

Reproduction commands (PowerShell, dependencies already installed):

```powershell
$env:TEMP='<HISTORICAL_AUDIT_ROOT>/tmp'; $env:TMP=$env:TEMP
& '<HISTORICAL_AUDIT_ROOT>/venv\Scripts\python.exe' -B '<HISTORICAL_AUDIT_SCRIPTS>/reaudit_v8_20261105.py' '<A1111_SOURCE>'
& '<PYTHON_3_12>' -B '<HISTORICAL_AUDIT_SCRIPTS>/reaudit_studio_api_20261105.py' '<A1111_SOURCE>'
```

## Revised roadmap and completion criteria

1. **API protection and persistence.** Close A1; test authentication modes,
   unauthorized writes, save failure, and concurrent library use. Do not
   regard v8 as safe for internet exposure solely because `--api-auth` exists.
2. **Adapter/Embedding correctness.** A2/A3: explicit backup state and full
   bias contract; A4/A5: per-encoder and finiteness validation before registration.
   Write a red test for each defect, then make it green; retain all older tests.
3. **VAE/checkpoint recovery and integration.** Run the matrix above in actual
   Gradio/GPU with LoRA + TI together, Hires/cancel/retry, and foreign hooks.
   Do not substitute callback stubs or high CPU-test counts for integration.
4. **Usable Studio.** Start with a compact window offering accessible actions
   and focus return; then quick blocks and history. Shared Quick/Full Studio
   components, one owner per draft, no default-weight changes. Then preview/
   triggers, interactive curves, and user-selected interface placement.
   Independent UI improvements can proceed without waiting for all backend
   research, but do not call the release fully accepted until integration step 3.
5. **Release acceptance.** Small new candidate after fixes, source↔payload hashes,
   installer/rollback against a known baseline, documented unsupported cases,
   environment versions, and user-scenario tests. Keep v8/ZIP unchanged.
6. **Later enhancements.** Runtime trace, XYZ/Effective Block experiments,
   architecture adapters, and other-WebUI ideas are separate tasks after the
   corresponding checks.

Stages 1–5 precede an accepted baseline release; stage 6 adds extensions
later. This is a list of work packages, not a promise of five short iterations.
Additional defects may broaden scope. The next concrete step at this point
is A1, followed by A2/A3.

## v8 immutability

ZIP SHA256: `9f97c5fb65ab019cb8dd55eb8466436e3a70746cbc4188dbf53bd19577baadc6`.
Matches released v8. No new candidate was created and no installation was
performed. Earlier audits remain as history; this document clarifies priorities.
