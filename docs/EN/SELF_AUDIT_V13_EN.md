# Independent Audit v12 → v13 — 2026-11-12

## What was rechecked

This is not a retelling of the previous report. Targeted diffs were rechecked
against the backup made before the external audit, along with the install
manifest/payload/working project, the v12 archive, loader calls, and concurrent
branches of `Options.set`.

It was verified that the v12 ZIP and its directory match entry-by-entry; SHA256
hashes for v8–v11 did not change. Before the new fix, `install.py check` on the
working project reported 0 changes and 0 unrecognized items. Thus the previous
package was not corrupted after sealing.

## Defect found and fixed

`Options.set(key, value, run_callbacks=False)` intentionally updates a value
without invoking its callback: this is used by operations that must not reload
the model/VAE. After v12, a rare concurrency scenario could arise:

1. The first thread starts a long-running callback for value A.
2. The second thread sets B with `run_callbacks=False`.
3. After callback A, the owner notices the newer version B and incorrectly invokes
   the callback again, now for B.

This violated the flag of the second call. In v13, callback intent is tracked
per version. The owner of the current callback finishes without implicitly
launching another callback if the latest change was data-only. A multithreaded
test with blocking/release was added; before the fix it would have failed
because `['first']` was expected, but `['first', 'data-only']` was returned.

The remaining limitation is intentional: a data-only change made during an
in-progress callback does not undo the effects of A that have already occurred.
It only guarantees that no new callback for B is secretly started. Arbitrary
side effects of A cannot safely be canceled without a transactional API for
each callback.

## No new confirmed runtime defects found

- VAE: exact sidecars, duplicate labels, and cache signatures agree with tests.
- Checkpoint: forced read reaches disk loading; a normal cache hit requires
  a matching file signature.
- TI: safetensors is validated before reading the tensor payload; dual
  rows/widths are not "fixed" by an unsafe conversion.
- LoRA: SD3/Flux metadata does not falsely become SD1; Flux support is not claimed.
- Installation: runtime assets must have an operation, and REGION is separately
  covered by a negative test.

This is a static/CPU audit. It does not establish real GPU quality, VRAM,
Gradio behavior, user models, or compatibility with every external extension.
Those limits remain unchanged.
