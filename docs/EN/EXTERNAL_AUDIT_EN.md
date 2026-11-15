# v11 Review Against Three Supplied Audits → v12 Candidate

Review completed: 2026-11-12. The source documents were read in full:

1. `КОНТЕКСТ_V10_ПОСЛЕ_АУДИТА_2026-11-08.md`.
2. `АУДИТ_V10_НОВАЯ_ПЕРЕПРОВЕРКА_ЯДРА_2026-11-08.md`.
3. All three files from `A1111_V10_SDXL_CLIP_L_CLIP_G_АУДИТ.zip`: the Markdown
   audit, `verify_v10_sdxl_ti_issues.py`, and `inspect_sdxl_embedding_headers.py`.

Original attachments were not altered. Their recommendations were not treated
as proof: actual functions in the working project and v11 payload were checked,
and separate tests were added. The supplied reproducer hardcodes `/mnt/data/...`
for the second ZIP; rather than running it with incorrect paths, its scenarios
were incorporated into local regressions.

## Conclusion about v11

The user was right: a significant portion of the November 8 findings remained
in v11. v11 fixed LoRA/UI combinations but not the VAE/TI/packaging errors.
In the first independent set of 14 checks on saved pre-fix code, 11 FAIL,
1 ERROR, and 2 PASS (the last two checked new helper contracts). In particular,
wrong VAE selection, lost duplicates, stale caches, SDXL Create/Reload,
and dict.shape in old emphasis were reproduced. These are not GPU error statistics.

Backup: `D:\AI-tools\lora-audit\before-20261111-external-audit`.
v10/v11 archives and directories are not overwritten. Fixes are applied to
the working project and v12.

## Map of every finding from the documents

| Item | v11 verification | v12 resolution/limit |
|---|---|---|
| API-STARTUP, late import | Confirmed in callback | Import bound at script load time; test prevents late import and calls the real API callback |
| REGION missing from operations | Confirmed | Controlled operation added, required runtime paths and missing-operation test; tests run AFTER install |
| VAE-01 foo/foobar | Confirmed | Only exact adjacent `stem.vae.safetensors/.pt/.ckpt`; multiple adjacent matches → explicit warning, selection required |
| VAE-02 identical basenames | Confirmed | All files exposed under path-qualified names; stable ordering and warning, no arbitrary winner |
| VAE-03 stale cache/active file | Confirmed | realpath/dev/inode/size/mtime_ns/ctime_ns signature; check on cache hit and active reload; discard stale entries on refresh |
| CKPT-01 forced reload from RAM | Confirmed | Forced path passes force_disk_read; cache dictionary and constructed model check file generation |
| MODEL-01 unknown → SD1 | Confirmed | Unknown/some incomplete models rejected; SD1 needs explicit weight evidence; adjacent YAML retains priority. Complete ModelFamily registry NOT implemented |
| MODEL-02 SD2 probe | Expensive fallback confirmed in sources | Explicit v_pred bypasses probe; freeing in finally tested on load failure. Universal metadata-first prediction selector and GPU benchmark remain open |
| MODEL-03 header allocation | No explicit limit | Shared stdlib header reader checks 8 bytes, length up to 100 MB, actual file size, and JSON before further processing; DoS attack is not claimed proven |
| MODEL-04 global SDXL patches | Inherited design confirmed | Not declared an independent bug; migration to owner/patch registry and external hook/refiner tests remain open |
| TI symlinks/cycles | Watcher/loader diverged | Shared sorted traversal; visited realpaths not traversed again; cycle tested with OS-independent simulation |
| TI PNG mtime-only cache | Confirmed | Expanded signature, Image handle closed, explicit force refresh clears image cache; replacement with preserved mtime tested |
| TI readme/rescans | Confirmed | Watcher includes only supported formats and excludes preview. Full incremental index and OS watcher NOT implemented |
| TI duplicate names | Silent re-registration possible | Deterministic first file; later files marked DUPLICATE_NAME with both paths, no silent replacement. Asset-ID namespace remains future architecture |
| TI entire unrelated safetensors in RAM | Confirmed | Header preflight before tensor loading; only chosen TI tensor keys read; invalid checkpoint rejected before safe_open/get_tensor |
| Expected TI errors and tracebacks | Confirmed | file_errors stores status/reason per path; expected ValueError/OSError handled without traceback or repeated warning on unchanged state. Unexpected failures remain visible |
| Options callback ordering | Old callback could finish last | One callback owner per key; new changes coalesced, current value applied after old. Config lock not held while waiting on GPU queue; two-thread test |
| TE schedule / Hires | Partially fixed in v11 | TE uses first point, separate hrte replaces base; no dependence on residual sampler step. Full per-step TE not implemented; actual Hires lifecycle requires GPU |
| Batch with different LoRAs | Deliberate refusal | Still restricted: per-batch-row adapters need a separate implementation; not solved by silently mixing |
| LoRA metadata sd3/flux | Could be treated as SD1 | Explicit SdVersion.SD3/Flux, unknown metadata does not prove SD1, SD3 filter fixed. Flux label does NOT imply Flux generation support |
| HeaderTooLarge frauke_gesicht | User file unavailable | Cannot confirm HTML/LFS/corruption; bounded inspector available, bytes not inspected |
| Y00n unequal rows | Correct defensive rejection | Independent spans/tokenizers not implemented; no truncation/repetition/zero padding |
| Merger/ww/ehscaletrick single tensor on SDXL | Correct rejection with dual consumers | Not automatically converted into two matrices. Filenames/other people's directories do not establish the format of a particular copy |
| Create embedding SDXL → unusable file | Confirmed | Vectors obtained separately from identified encoders; saved as clip_l/clip_g; save→load tested with real Torch |
| Old emphasis + dict | Confirmed | Uses validated embedding.vectors; test preserves neighboring tokens. Not full coverage of every legacy chunking mode |
| SDXL TI training | Already explicit rejection | Not misrepresented as supported training; dual optimizer/training loop remains separate |
| Forge patcher/memory/VAE tiles | Architecture proposals, not ready patches | Deferred to next stage; no copying of someone else's sd_vae.py. Registry/resource budget/tiled fallback need separate implementation and GPU acceptance |
| Quality/FPS/VRAM and SD1/2/XL/SD3 matrix | No actual images/measurements | No improvement claimed; CUDA, WAI, refiner/inpaint, 1/4/8 LoRA, FP8, REGION/TONEG, external hooks remain open integration matrix |

## New user actions and compatibility changes

- Settings → Actions: Force reload checkpoint from disk; Force reload selected VAE
  from disk. Handlers run through the generation queue. The Gradio page was not
  run here; callback code and loader semantics were tested separately.
- VAE Automatic no longer looks for approximate names in arbitrary folders.
  Previous selection of an ambiguous basename must be replaced by a
  path-qualified entry.
- If None/checkpoint VAE is selected, there is no separate external VAE file;
  to reread the base from disk, use Force reload checkpoint rather than the
  external-VAE button.
- An unknown checkpoint without YAML now fails clearly instead of assuming SD1.
  Completeness of architecture fingerprints is not yet established.
- The file signature is not a cryptographic content hash. Filesystems with
  unchanged stat fields might conceal a modification; explicit forced reload
  is available for that case.
- Under concurrent Options.set calls, the second request may return accepted
  before a shared callback completes. On successful callback, the owner brings
  the same key up to date. This is NOT an interprocess transaction and does
  NOT guarantee callback ordering across different keys or foreign opts.data writes.

## Test installation, not just the files directory

`tools/post_install_smoke.py` creates a new copy under
D:\AI-tools\lora-audit\tmp from local
`Downloads/stable-diffusion-webui-dev.zip` and `source/modules.zip`.
This is the available, expanded local base with old REGION; it is not assumed
to be identical to the archive from another party's audit. After installation,
ONLY tests are copied, not modules from reference/verification-only.
Runtime, parser/API, and Node checks are run, then byte-for-byte rollback
of all affected original files is verified.

This local base has 16 UNKNOWN files. Override was used only in the fresh test
copy created by the script. The normal installer still rejects UNKNOWN on a
user installation; this test does not authorize blind replacement of any build.

Concrete results of the final run and the archive SHA256 are in the package's
VERIFICATION_EN.md and separate v12 receipt. Older reports inside the package
remain as history.

## Tool from the ZIP and its limitations

The supplied header inspector does not load tensor payloads, but its
SDXL_DUAL_OK result does not validate a positive number of rows, dtype/offsets,
or finiteness of tensor values. Matching shapes do not establish file validity.
v12 adds `tools/inspect_embedding_header.py`, using the same bounded preflight
as runtime and checking positive dimensions and floating-point dtype without
opening pickle; it explicitly reports HEADER_COMPATIBLE_ONLY. The safetensors
library checks offsets/content during loading; training compatibility with a
checkpoint cannot be established from the header alone.

## Next plan

1. Retain reproducible installed smoke and rollback as a mandatory gate.
2. Real WebUI/Gradio/CUDA run on user models and files: VAE hot swap/force,
   checkpoint A→B→A and file replacement, LoRA disable/re-enable, SDXL TI
   Create/Reload/old emphasis, Hires, cancellation, REGION/TONEG, batch limits.
3. After that, separate ModelFamily/asset-ID/index and patch-ownership facade;
   then memory budget and VAE tiling with failure tests. Do not substitute
   this architecture work for fixing actual defects.
4. Independent dual token counts and SDXL training are separate projects;
   prove adjacent tokens are not overwritten and conditioning remains aligned.
5. Studio history/undo, further UX, and A/B tests with identical seed/model hashes.

Result wording: the listed locally reproduced defects have been fixed; NOT
"all ideas have been implemented," NOT "all models are now compatible," and
NOT "the GPU plan is fully complete." No new dependencies were downloaded
or installed; the package itself was installed only into a fresh local test copy.
