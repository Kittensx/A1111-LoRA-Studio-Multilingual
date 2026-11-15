# In-depth Independent Audit v13 → v14 — 2026-11-12

## Method

Not only previously modified functions were rechecked, but also their boundaries:
LoRA loading path → memory cache → reactivation, combinations of 1/2/4 LoRAs,
TE/U-Net/DYN/LBW/LBWE/curves/Hires, API authentication, and the package install →
runtime tests → rollback path. Previous passing tests were not considered
sufficient: two negative test scenarios for LoRA file identity were added.

## Confirmed defect

Before v14, the LoRA cache's `network_file_signature()` consisted only of mtime
and file size. A same-size file written under the same name while preserving its
mtime could leave previously loaded tensors in use. This is especially plausible
when synchronizing a model from another directory or using `os.replace`.

v14 retains the compatible mtime first element but adds ctime, dev, inode, and
canonical realpath. Loading also rereads the signature after `read_state_dict`:
if the file changes during reading, loading fails with a clear message and can
be retried after the write completes, rather than publishing a potentially
mixed payload.

New checks:

1. An actual same-size replacement with restored mtime changes the signature.
2. A simulated signature change between the beginning and end of reading stops loading.

## Test run results

- 133 Torch CPU runtime tests PASS.
- 62 parser/API/config tests PASS + 13 subtests PASS.
- 22 Node tests PASS, 61 UI assertions in the DOM shim.
- Ruff checks on modified LoRA runtime/tests PASS.

## Limits of the conclusions

The signature is not a cryptographic hash: an unusual filesystem may retain
all stat fields after a replacement, in which case an explicit reload is still
necessary. Actual Gradio, CUDA/WAI, user-provided LoRA/TI, VRAM/FPS, and external
extensions were not tested. Therefore v14 confirms the correctness of the tested
CPU scenarios, but does not present them as full GPU acceptance testing.
