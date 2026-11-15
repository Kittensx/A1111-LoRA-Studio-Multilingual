# Receipt v15 — 2026-11-13

Archive: `A1111-LoRA-Studio-Pack-v15.zip`

- Size: **537494 bytes**.
- SHA256: **b2cb3e6e6d1786249edf6e848d31d5f37e8efb2d43c7fe6776d3b40bbc0a3923**.
- ZIP CRC and matching all 146 entries against the package directory: PASS.
- 114 manifest assets, 73 install operations, 77 Python payload files.
- All replacement payloads match the working project on D: byte-for-byte.
- Processing text-patch was checked against the source base and for idempotence.
- SHA256 hashes of older ZIPs v8–v14 are unchanged; directory contents for v10–v14
  match their archives. `lyco_helpers.py` is unchanged.

Checks on an actually installed, disposable test base:

- 142 Torch CPU runtime tests PASS.
- 64 pytest parser/API/config tests PASS, 13 subtests PASS.
- 22 Node tests PASS, 62 UI assertions PASS in the DOM shim.
- 73/73 SAME after installation; rollback of 59 modified/added paths;
  all 73 target paths checked byte-for-byte or for absence.
- Installer/header tests: 26 PASS, 1 SKIP (symlink unavailable to the current user).
- Ruff checks on modified Python and Node syntax: PASS.

Test host: `<HISTORICAL_AUDIT_ROOT>/tmp/lora-v15-installed-kha0e4qa`.
The base was a local `stable-diffusion-webui-dev.zip` plus `modules.zip`, not a
live installation. The 16 UNKNOWN entries were allowed only in this new test copy.
Logs are in the package's `evidence/`; intermediate failed UI runs are described
in VERIFICATION_EN.md. The smoke script eventually reported PASS after all rollback
checks; when the conversation resumed, its completed exec session was no longer
available, so the result was verified against the saved complete log. Final
installer tests and the seal verifier finished with exit code 0 during the
current continuation.

Two reproduced Options race conditions, consistency of local LoRA asset caches,
and diagnostic signatures were fixed. Full audit and remaining plan:
`A1111-LoRA-Studio-Pack-v15/AUDIT_V15_EN.md`.

This is not GPU/Gradio acceptance testing and not a universal resolver. Real
hermitmoth/α, Drive/FUSE, new backend functions, runtime coverage, and the
partial-encoder TI warning remain unresolved. Dependencies were not installed;
existing Torch CPU on D: was used. `/content/A1111` was not modified.
