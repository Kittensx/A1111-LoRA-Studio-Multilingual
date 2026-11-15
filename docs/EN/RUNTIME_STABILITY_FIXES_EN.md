# Runtime stability fixes — v12 candidate, 2026-11-12

## v12: externally supplied audit corrections

Bind API access dependency during script import. Install REGION runtime instead
of shipping it only as a test reference; require its operation in the manifest.
VAE uses exact adjacent sidecars, visible duplicate paths and disk signatures.
Checkpoint forced reload bypasses RAM state cache; cached model identity includes
the file generation. Shared bounded header preflight rejects unrelated TI files
before tensor allocation; only selected tensor keys are loaded. TI watcher/loader
share cycle-safe sorted traversal; image cache uses full stat signature and closes
handles. Expected per-file errors have status/reason and deduplicated warnings.
SDXL creation initializes each encoder separately; old emphasis uses validated
vector count. Same-key Options callbacks converge to the latest request without
holding the config lock while waiting on the generation queue.

Unknown checkpoint dictionaries no longer silently assume SD1; explicit YAML
keeps priority. SD2 explicit v_pred avoids probing; fallback cleanup uses finally.
SD3/Flux metadata is not mislabeled SD1. Flux generation is not thereby supported.
Force disk-read actions added to settings; actual Gradio rendering is pending.
See EXTERNAL_AUDIT_RU.md in v12 for the complete issue/status matrix and limits.

CPU/AST tests and installed-host/rollback checks are not full WebUI/CUDA tests.
Independent dual token spans, dual TI training, per-step TE, mixed-LoRA batch,
ModelFamily/PatchManager/ResourceManager, VAE tiling and Studio history remain open.

## v11: cross-feature corrections and UI controls

Hires scalar overrides now replace base curves; TE uses the first curve value
instead of stale sampling state. Indexed SDXL OpenCLIP embedding keys route to
TE. Short classic masks zero omitted known blocks, matching original LBW to26;
unknown keys retain fallback 1. Classic MID transformer keys no longer trigger
flat-DiT elemental matching. Mixed start/stop units resolve before comparison;
negative bounds are rejected. CP factors honor positive integer dynamic rank.
Conditioning signature includes model_profile.

UI honors custom overrides of builtin presets, projects classic masks by block
ID without dropping nonzero values, adds per-block sliders and absolute-step
curve midpoints, rejects incomplete masks, and stacks based on container width.
Start/Stop are inclusive U-Net gates (0..1 fractions, >1 zero-based steps).
TE per-step recomputation and per-block curves are NOT implemented.

112 real-Torch CPU/AST runtime tests PASS; new matrix has 330 timestep/layer
scenarios across 1/2/4 occurrences, base/Hires, five key types and eleven steps.
59 parser/API/config tests (+13 subtests), 22 Node tests, 66 browser fixture
assertions PASS. Synthetic tensors and isolated browser services are used;
full Gradio, CUDA, actual user safetensors and sampler lifecycle remain pending.
No dependencies installed. lyco_helpers.py and modules/* unchanged from v10.
Detailed report: COMBINATIONS_AUDIT_RU.md in the separate v11 package.

## v10: follow-up defects V9-1–5, C1–4, L1–2

- Native SD3 consumers expose width on the Embedding itself, unlike TI wrappers.
- TI registry tracks its model/encoder signature and a revision. Model changes
  trigger revalidation; conditioning keys include TI revision and LoRA payload
  identity/settings. Base and Hires caches use the same invalidation mechanism.
- LoRA maps known EmbeddingsWithFixes to its child; canonical aliases share the
  forward binding. Actual Torch apply/repeat/change/TI insertion/disable tested.
- First upsampler accepts SD1 and SDXL topology; two distinct matches fail
  explicitly. Unsupported weightless modules report their type and tensor key.
- Preset writes validate a snapshot, atomically replace config, then publish
  memory. Options saves/sets share a process-local RLock. Cooperating concurrent
  transactions tested. Direct opts.data writes/external processes are not a
  cross-process transaction protocol; simultaneous whole-library edits remain
  last-writer-wins (no revision/ETag conflict UX yet).
- Low-VRAM temporary detachments restore in finally; partial hooks/methods roll
  back. Both first and second transfer failures tested, followed by retry.
- Original emphasis avoids undefined zero-mean normalization, preserves the
  promoted output dtype and keeps weighted values when normalization is unsafe.
- Malformed Origin is rejected with 403 instead of leaking an unhandled 500.

104 CPU runtime tests and 54 parser/API/config tests pass in the working tree.
Callbacks run outside the config lock: model/VAE callbacks wait for the generation
queue, whose running job may itself change options. A dedicated cross-thread
regression test checks this lock ordering.
Full GPU, real Gradio/proxy and the user's two safetensors remain untested.
This is not proof that all external callbacks or all checkpoint formats are safe.
UI redesign remains outside this runtime correction. lyco_helpers.py unchanged.
Earlier sections below describe historical candidates, not current acceptance.

## Follow-up audit A1–A5 (v9)

- All six LoRA/Studio routes now share lora_studio_access. Configured API Basic
  credentials or a valid authenticated Gradio session are required; anonymous,
  forged cookies and cross-origin browser writes are rejected. API-only and
  no-auth operation are tested with in-process FastAPI. The session integration
  follows the pinned Gradio 3.41.2 auth/tokens cookie contract; real Gradio and
  reverse-proxy deployments remain integration checks. No passwords are put in JS.
- Weight backup initialization now also captures an originally absent bias.
  Repeat application cannot mistake the adapter-created bias for original state.
  Q/K/V output bias is applied for MHA and packed QKV, including missing individual
  projection biases, with backup/restore of MHA in_proj_bias. Float8 bias is
  accumulated in a supported floating dtype before the final storage cast.
- TI import validates nonempty finite float matrices, including conversion
  overflow. Per-encoder widths are read from active TI consumer modules, not
  hardcoded SDXL constants or a sum-only check. Standalone and bundled embeddings
  are checked before device transfer; registration rechecks model compatibility.

81 CPU/AST runtime tests PASS (72 previous + 9 methods); 6 API test methods PASS;
37 parser/schema/preset tests PASS; 17 Node tests PASS (51 UI assertions).
Independent follow-up: 6 runtime/VAE checks and 1 API check PASS. The independent
TI harness was upgraded from a preconstructed-embedding stub to the actual import
function and real model-width wrappers; it now asserts rejection before registration.

No UI code/default multipliers/VAE implementation/lyco_helpers.py changed in v9.
Remaining: real GPU/Gradio, third-party callback recovery, atomic/concurrent full
config saves, UX U1 and the planned Studio redesign. Do not infer that every
attention backend, malformed file or external callback is now supported.

Official pinned session implementation inspected:
https://github.com/gradio-app/gradio/blob/gradio%403.41.2/gradio/routes.py
No dependencies were installed. Restart the WebUI process after installation;
live models with old backup state must not be hot-patched in place.

## Standalone VAE reload follow-up (v8)

The mutable phase of sd_vae.reload_vae_weights now quarantines an incompletely
configured model on failure: discard_failed_model removes only that object from
the active/cache registry; clear_loaded_vae resets the global VAE marker. The
original exception is re-raised. Resolution and no-op checks stay before mutation,
so a preflight resolution error does not discard the untouched working model.
This is fail-closed behavior, NOT automatic recovery of the previous VAE. Reload
the checkpoint explicitly after failure. Deploy sd_vae with the bundled sd_models.

test/test_runtime_vae_failures.py has 3 methods covering seven mutation failures,
preflight and successful/no-op normal and low-VRAM paths using production functions
with service doubles. The complete CPU/AST runtime suite is 72 PASS; CUDA/backend
hook recovery and real Gradio generation are still unverified.

Studio was separately exercised in a real browser against the static mock-API
fixture: 55 checks passed, including four field-contrast modes. This is not real
Gradio integration or UX acceptance. UX gaps and the next increments are recorded
in LORA_EVOLUTION_PLAN.md and package UX_AUDIT_RU.md. No UI code/defaults changed
in v8. The sections below preserve the previous runtime verification history.

## Compatibility follow-up (v5)

- The safe loader accepts only the two additional exact Torch ZIP metadata
  records .format_version and .storage_alignment. Restricted pickle checks
  and caller weights_only behavior are unchanged; no unsafe fallback was added.
- SDXL TI loads clip_l/clip_g separately from metadata, validates vector counts,
  saves a reloadable two-encoder format and hashes both encoders for optimizer
  state checks. Legacy single-encoder format/checksum is unchanged.
- SDXL TI training remains unsupported by the single-encoder trainer. It now
  reports that limitation before changing job state or preparing a dataset.
- manual_cast setup is inside the cleanup boundary; failed installation rolls
  back earlier hooks and restores inherited forward attributes accurately.

test/test_runtime_compatibility.py exercises actual .pt files, restricted reads,
legacy serialization, real CPU float8 storage + delayed fp16 cache assignment,
and LoRA/manual_cast in both installation orders. It does not run FP8 CUDA
kernels, full SDXL TI conditioning/training, or arbitrary external extensions.

## Follow-up to the independent v3 review

The v3 audit found remaining recovery and projection-cache defects. v4:

- Preflights checkpoint reads/config selection before cache eviction/unhijack;
  direct load also parses/repairs config before destroying the previous model.
  A preflight failure must not unregister an untouched working model.
- Rollback uses the checkpoint belonging to the actual reused object. Preparation
  or finalization failure discards an incompletely configured object. Successful
  rollback restores the selected U-Net too. A fresh model built into a free cache
  slot is not loaded again from an already consumed meta-loader dictionary.
- Attempts both processing unhooks independently and clears base/Hires conditioning
  even after an unhook failure. A failed unhook remains attached for diagnosis.
- Rolls back earlier LoRA hook installations on a late constructor failure;
  removal preflights the whole hook set to avoid a partial uninstall on conflicts.
  Install this with its matching modules/patches.py (new undo_many API).
- Includes the actual projection keys/multipliers in merged-cache identity, so
  changing only Q/K/V/out elemental weights refreshes MHA and packed QKV weights.

test/test_runtime_recovery.py adds real-factor projection and production-control-flow
regressions. These are CPU/AST tests, not a full checkpoint or browser run.
The global-hook/dependency/FP8/TI integration matrix remains unfinished. This is
a test candidate, not a declaration that the entire stabilization plan is done.

This release corrects the v2 integration regression between the global LoRA
Embedding load hook and A1111's meta loader. The earlier statement that v2
needed no further changes was incorrect. Tensor-factor tests did not exercise
that integration.

## Implemented

- Meta loader: wrap an explicit Embedding load override, preserve inherited
  class attributes on context exit, handle root-level parameter names and meta
  buffers, reject checkpoint tensors that already lack data, clean consumed
  entries after loader exceptions, and return load_state_dict's result.
- Generation cleanup: read the already loaded model directly, attempt close and
  all option restoration even after a cleanup error, preserve the original error.
- Failed checkpoint initialization: discard a published incomplete candidate
  and the old trashed model; lazy reads do not automatically retry. A failed
  weight rollback discards the broken model without replacing the original
  exception with the rollback exception. Explicit reload can retry.
  Models whose storage is moved to meta are also removed from the loaded cache.
- Patch registry: reject out-of-order undo and external replacement before
  changing anything; correctly remove temporary subclass overrides. This is a
  strict LIFO registry, not arbitrary removal of middle wrappers.
- Manual cast: restore original per-parameter/buffer dtypes after failure,
  preserve integer/bool inputs, handle keyword inputs, remove only owned hooks.
- LoRA: payload identity participates in merge-cache keys; a model-loaded event
  invalidates bound adapters; cached loads also verify model tokens. Only
  targeted layers create backups (including packed projections). Partial
  failures restore the affected layer and abort generation, never mark success.
  Regular layers stage combined deltas; MHA/QKV restore backups on errors.
- FP16 cache: multiple regular LoRAs accumulate on the same working weight,
  instead of restarting from the FP16 base for each adapter.
- Unmatched tensor keys are logged as warnings and summarized in generation
  comments instead of appearing only at debug level.
- TI: recursive file signatures detect nested changes/deletions; scans record
  their starting snapshot; preview saves once and skips missing images; step
  accounting uses the completed step; saving does not mutate the live embedding.
  Multi-term string_to_param exports have an explicit error. Multi-vector
  single-term embeddings remain supported.

## Verification and limits

`python -B test/test_runtime_stability.py` uses real PyTorch CPU and selected
production AST definitions, with stand-ins only for unavailable WebUI services.
It covers real Embedding/Linear meta loads with the production LoraPatches class,
merged/functional Embedding, hot reload, failed merges, MHA/QKV, cache binding,
checkpoint recovery control flow, cast/patch errors and TI files/snapshots.

This is not full WebUI startup or CUDA generation. Real SDXL/SD1.5 checkpoints,
CEB/Velvette files, extension combinations, training and VRAM have not been run.
Successful unit tests do not establish their end-to-end correctness.

Embedding DoRA/CP and ambiguous layouts remain unsupported. Mixed full-matrix
and projected QKV adapters are rejected explicitly. Failed-merge rollback is
per layer with generation aborted, not a transaction spanning every model layer.
File reload detection still uses mtime_ns/size; replacement deliberately
preserving both requires a reload/restart. Recursive TI scans add filesystem IO.
The hook registry does not control third-party direct monkey-patches.

## Follow-up roadmap

1. On the real installation: same SDXL checkpoint with meta optimization on/off,
   with/without LoRA; CEB/Velvette; checkpoint A→B→A; failure then explicit retry.
2. Validate external hooks and dependency versions. Do not upgrade all libraries
   together or disable safe deserialization to bypass compatibility problems.
3. Inspect `D:/Users/Asus/Music/ВЕбы` for model lifecycle and patcher ideas after
   these correctness fixes; port small tested contracts, not entire backends.
4. Longer term: common hook ownership diagnostics, tensor cache separated from
   model-bound adapters, richer coverage UI, and GPU/browser CI.

Existing Studio UI and preset formats are preserved. `lyco_helpers.py` is unchanged.

## Follow-up verification through v6 (2026-11-04)

v4 fixed checkpoint preflight/reuse rollback, independent processing cleanup,
LoRA hook installation rollback/batch ownership checks, and per-projection merge
cache identity. v5 fixed safe-reader Torch ZIP metadata compatibility, SDXL TI
metadata/save/checksum validation and manual_cast partial setup rollback. SDXL TI
training remains explicitly unsupported by the single-encoder trainer.

v6 adds three failure-path corrections, reproduced before changing runtime:

- Cached checkpoint transfer/VAE failure unregisters the potentially damaged
  candidate; successful UI checkpoint/hash settings are written only after VAE
  reload completes. Other cached objects are not discarded. This is quarantine,
  not a claim that VAE mutation is reversible or the prior model was reactivated.
- Projected QkvLinear adapters now accumulate on the precise fp16 base when
  available, or the stored base otherwise, before one final storage cast. This
  avoids unsupported Float8/float in-place addition and repeated quantization.
  Later adapter failure still restores the original stored weights.
- Temporary loader contexts unwind partial __enter__, restore raw descriptors,
  remove inherited overrides, refuse reentry of the same active context and do
  not overwrite another owner's replacement. Independent hooks are restored on
  error; unresolved entries remain retryable. A primary body/setup error is
  preserved and restoration failure logged. Separate nested contexts work.

test/test_runtime_edgecases.py adds 11 CPU/AST tests (including eight setup-failure
positions under actual LoraPatches). All four runtime suites contain 56 tests.
This extends, but does not replace, the real GPU/checkpoint/browser acceptance
requirements above. No CUDA inference or perceptual-quality claim is made.

Known audit boundaries: arbitrary MHA/QKV projection output-bias formats, mixed
full/projected QKV, non-tensor/extra-state metadata in the destructive meta loader,
cached checkpoints under changed FP8 settings, low-VRAM external backends and
atomic TI/optimizer file replacement require separate acceptance work. The exact
transformers private-call signature remains version-dependent. Foreign hooks are
detected, not automatically spliced out of arbitrary wrapper chains.

## State IO follow-up: v7 (2026-11-05)

- Meta loading preserves OrderedDict version metadata and both tensor/non-tensor
  `_extra_state`. Materialized parameters use the converted source dtype, with
  explicit dtype exclusions and assign=True covered. This does not guarantee
  custom override migrations which require real tensor data before calling super.
- Embedding and optional optimizer are fully serialized to same-directory temp
  files before either destination is replaced. Each replacement is atomic; the
  two-file commit is NOT atomic as a pair. New optimizer sidecars include SHA256
  of the serialized embedding. New main files declare that binding, so a stale
  legacy sidecar cannot be accepted after an interrupted upgrade. Legacy main +
  legacy optimizer retains checksum-only compatibility. This protects training
  resume consistency, not authenticity or safe-pickle security.
- Failures during serialization preserve both previous files; a failed first
  replace leaves them unchanged. A failed second replace leaves a valid new
  embedding and a rejected stale optimizer. Temp cleanup failures are logged
  without replacing the original error. Power-loss durability on every filesystem
  and concurrent writers to the same filename are not guaranteed.
- FP8 reload decisions inspect actual parameters of the requested cached model,
  not a global mode belonging to the previously active model. Cache hits rebind
  devices.fp8; changed storage policy preflights and reloads precise checkpoint
  data. A quantized target is upcast before weight loading even when the previous
  model's global flag was false. External non-Torch quantizers remain out of scope.

test/test_runtime_state_io.py adds 13 CPU/AST tests; the five runtime suites now
contain 69 passing tests. Older harnesses gained dependencies for the extracted
helpers; their existing expectations were not relaxed. Previous audit boundaries
for meta metadata, cached FP8 selection and TI replacement are partially closed
by these explicit cases, not blanket support for arbitrary state/backend formats.
