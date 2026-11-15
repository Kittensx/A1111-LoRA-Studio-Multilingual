# Release verification and checksum diagnostics

This repository installs full A1111 file replacements. The original LoRA Studio source and its license remain included. These checks are **offline package checks**, not a substitute for a real A1111 runtime/GPU test.

## Before distributing a release

Run from the repository root:

```powershell
python run.py verify
python run.py docs-audit
python run.py help-audit
python run.py test
```

A successful runtime verification checks all manifest-tracked assets using SHA-256 over their **exact bytes**. The documentation audit separately checks English/Russian file pairs and the six optional documents. A help audit checks topic references and fallback behavior.

After committing your changes, run:

```powershell
python run.py release-verify
```

This packages the **committed Git HEAD** into a temporary ZIP, extracts it, and verifies the distribution's runtime assets, optional documentation and contextual help. It requires Git and a clean working tree. It does not publish or edit the repository.

## Interpreting checksum failures

The installer now reports *all* mismatches in a single run:

- `missing`: an expected asset cannot be found
- `line endings changed`: converting CRLF/LF yields the exact expected SHA-256
- `content/bytes differ`: not explainable solely by CRLF/LF conversion

No mismatch is automatically repaired or accepted. Restore the **intended release bytes**, not an unrelated upstream copy. Files that were intentionally modified for the multilingual release must remain modified. Never regenerate every hash merely to make verification pass: that can disguise unexpected changes.

Git's `.gitattributes` protects checksummed payload bytes across platforms. Recheck the published Git archive after each release. Avoid editing payload files in editors that change line endings implicitly.

## GitHub Actions

`.github/workflows/verify.yml` runs the verification commands on Windows and Linux with Python 3.10/3.12 and Node.js 22. The workflow runs on pushes to `main`, pull requests and manual dispatch. A failing check blocks no local work; protect the `main` branch with required checks if you want GitHub to reject unverified merges.

The workflow does **not** install or run A1111, verify model inference, or publish a release automatically.
