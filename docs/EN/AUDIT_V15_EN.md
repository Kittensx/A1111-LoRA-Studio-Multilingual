# v15: Review of the Complete External v14 Audit — 2026-11-12–13

This is a candidate with limited fixes, **not an implementation of every audit idea**.
Executable code changed in seven files: `network.py`, `networks.py`,
`modules/cache.py`, `modules/hashes.py`, `modules/options.py`, `modules/api/api.py`,
`javascript/loraStudioView.js`. `lyco_helpers.py` was not changed.

## Materials read and claims checked

Both supplied bundles were extracted to separate directories on D:. The full
v14 (16 subsections), short v14, attached v12/v13, repro scripts, logs from
the first bundle, and the user's explanation were read. All 8 hashes from the
first list and 7 from the second were verified. The separately attached full
Markdown file matches the Markdown in the ZIP byte-for-byte: SHA256
`5296cc7fc7b3e5512574ccf17c2febdfbcdfda1fe948cd8109e4b30f48cb989b`.
The documents' recommendations were treated as testable proposals, not as
automatic permission to disable safeguards or add a new backend.

The supplied `v13_options_extra_repro.py` actually reproduced **both** races
on saved v14. `verify_v12_lora_routes.py` confirmed standard SDXL matching
and four missing routes on v14. Tests for the same five cases were added to
`test_runtime_known_routes.py`: negative expectations document unsupported
behavior; they do not claim it has been fixed.

The supplied metadata reproducer needs unavailable `safetensors` and could not
run locally. Instead of installing dependencies, tests were added using valid
small binary safetensors, a standard JSON/struct decoder, real Torch factors,
and extracted production functions. This is **not a test of safetensors mmap,
diskcache SQLite, or full WebUI startup**. Against old `cache.py`/`hashes.py`,
three new cases failed as expected: an old-format entry returned foreign
metadata; an identical title returned data for another path; a change during
reading was not rejected. Those cases pass on the corrected code.

## What was fixed

### One file-version identity method for related caches

LoRA payloads, metadata, and hashes now use
`modules.asset_files.file_signature`: resolved path, device, inode, size,
mtime_ns, and ctime_ns. The new on-disk cache field is `file_signature_v1`;
mtime-only entries are rejected and recalculated lazily. No bulk deletion of
the user's cache is performed.

On a same-size replacement preserving mtime, `NetworkOnDisk` is updated, the
catalog is rebuilt, old alias/hash lookups are removed, the architecture
version is updated, and new weights are loaded. The chain of actual functions
`list_available_networks → load_networks → reactivation` was tested with and
without explicit refresh, including Torch delta calculations and reuse of an
unchanged file. Checks before/after reading and before activation reject a
detected file swap. A real writer thread replaces a tempfile between payload
reading and result publication; that test uses a stdlib tensor-decoder fixture.
A file disappearing during refresh no longer causes a secondary
`None.filename` crash.

A declared `sshs_model_hash` is no longer accepted by the loader as a verified
weight hash. On first load, the actual hash is calculated, then cached.
This may increase initial load time, especially on a network drive. With
`--no-hashing`, a previously computed valid hash is allowed, but no new hash
is calculated; an old/declared hash is not substituted after a version change.
The Studio metadata panel now labels the field **Declared hash / unverified**.

The `hashes.py` cache is shared across WebUI: the change affects not only LoRA
but also other consumers. Normal SHA, addnet SHA, legacy partial hash,
identical titles for different paths, backward timestamps, old entries, and
file changes during reading were tested. This is not a complete re-audit of
all possible third-party extensions.

### Concurrent settings: explicit rejection instead of premature success

While a key's callback is running, another `Options.set` for the same key
raises a clear RuntimeError asking the caller to retry. This includes setting
the same value and using `run_callbacks=False`. Success is returned only after
the callback finishes. On callback failure, the value is rolled back; config
saving and `set_and_save` do not write intermediate state while a callback is
active. The global config_lock is **not held** during a callback or queue wait.

`POST /sdapi/v1/options` returns HTTP 409 for such conflicts or callback failure,
not a false success response. The actual API method was tested using a temporary
config: conflicts/errors are not saved, and retry after completion saves correctly.

**Contract change:** concurrent callers must handle the rejection and retry,
rather than counting on automatic request coalescing. This is not a transaction
covering a whole settings batch: other keys changed successfully earlier may
stay changed if a subsequent key fails. Direct writes to `opts.data`, attribute
assignments bypassing `set`, callbacks for different keys, and arbitrary
extension side effects have no shared transactional guarantee. Rolling back
a value cannot itself undo every GPU effect already performed.

### Accurate Studio labels

Labels distinguish `TE at text encoding (first point)` from
`U-Net per sampling step`. Coverage is labeled
`File-key coverage (not runtime application)`; the existing warning that
checkpoint matching was not checked remains. Stable field names used by
accessibility/automation are preserved. Full runtime coverage was not added.
This is a UI clarification, not a new generation algorithm.

## Status map of all 16 subsections of the full v14 audit

| Item | Status after v15 / next step |
|---|---|
| V.1 text_projection | Open. Requires metadata/shapes from an actual adapter and a parameter-level route with numerical backup/restore, dtype/device and pooled conditioning. No blind transpose was performed. |
| V.2 text_encoders.clip_g | Open. Minimal trees confirm NO MATCH; without the file, an incompatible family cannot be distinguished from a compatible alternate export. Model-aware resolver needed. |
| V.3 coverage | Label clarified. Separate real load/application report for CLIP-L/G, T5, UNet/DiT and projection still needed. |
| V.4 data-only race | Reproduced scenario fixed by explicitly rejecting a busy key; test reads callback value AFTER the competing attempt. |
| V.5 early ACK/config | Reproduced scenario, pending-state save, and API reply fixed. Real Gradio/queue/GPU still need acceptance tests. |
| V.6 unequal dual rows | Defensive refusal retained. Independent spans/encoder alignment require separate development. |
| V.7 single TI in SDXL | Defensive refusal retained. Missing vectors are not manufactured. Library diagnostics by shape could be improved. |
| V.8 duplicates / partial encoder | Registration-path review confirms the first valid file is not necessarily the best; there is no PARTIAL_ENCODER_USE warning yet. Report of both paths/encoders/hashes and version selection needed. |
| V.9 SDXL TI training | Not implemented. Explicit refusal in single-encoder trainer retained; Create/Reload is not represented as training. |
| V.10 per-step TE | Not implemented. UI explicitly differentiates TE and UNet semantics; future re-encoding requires conditioning cache and GPU benchmarks. |
| V.11 mixed-LoRA batch | Refusal retained. Until per-sample/grouped execution exists, use batch size=1 for differing LoRA sets. |
| V.12 ModelPatcher/VRAM/tiled VAE | Separate project, not a fix for these races. Forge modules and global hooks were not imported. |
| V.13 SD3/Flux | A block profile is not backend support. No new universal support is claimed. |
| V.14 GPU/Gradio/WAI | Not performed; CPU/DOM shim does not replace images, VRAM/FPS, ControlNet, OOM/cancel recovery. |
| V.15 cache consistency | Described local scenarios and migration fixed; FUSE/Drive/stat-invariant substitution remain outside guarantees. |
| V.16 status separation | Applied here: bugs, safeguards, new features, and acceptance tests are not collapsed into "everything fixed." |

## Limits and continuation order

1. Installation checks completed: 142 runtime, 64 parser/API/config (+13
   subtests), 22 Node/62 UI assertions passed on the installed tree; 73/73
   SAME, rollback of 59 modified paths, and byte-level comparison of all 73
   target paths succeeded. This is not a full WebUI run; method and caveats
   are in VERIFICATION_EN.md.
2. Inspect headers of real `hermitmoth` and `α` assets with
   `tools/inspect_lora_header.py` and record the active checkpoint. Design the
   resolver and parameter-level adapter separately; prove numeric results
   and restore behavior before GPU testing.
3. Add LoRA runtime diagnostics and a partial-encoder TI warning without
   automatically corrupting/deleting files. Strengthen Options UI/API/queue tests.
4. Run a real SD1.5/SDXL/WAI/Pony matrix, Hires/refiner/inpaint, 1/4/8 LoRAs,
   A→B→A, VAE swap, FP8, REGION/TONEG, hooks, cancel/OOM.
5. Only afterward tackle independent dual spans, TI training, per-step TE,
   mixed batch, and ModelPatcher/memory manager/VAE tiling as separate stages.

A file signature is not cryptographic protection and does not block a foreign
writer. If a filesystem preserves all stat fields, invalidation is not
guaranteed. Real symlink assets, Drive/FUSE, and continuous in-place writes
have not been tested. Full WebUI generation and browser review of images/
infotext were not performed. Existing infotext code takes the active
descriptor's hash, but a small-LoRA test is not equivalent to end-to-end PNG
saving. Rendered cards/JS metadata are not updated by push events: the
already-open UI needs a normal refresh.

The source project on D: was updated; `/content/A1111` was not changed.
Older releases were preserved. Nothing was installed or downloaded; existing
Torch CPU on D: was used. Before installation on a working WebUI, stop Python,
run `verify` and then `check`; do not blindly bypass UNKNOWN checks.
