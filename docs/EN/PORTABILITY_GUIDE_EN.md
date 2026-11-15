# Portable historical build instructions (English)

The current LoRA Studio package installs using `run.py` and `install.py`.
**Do not run `tools/build_v10.py` through `build_v15.py` to install the extension.**
These are historical **reconstruction scripts**; they rebuild old candidate
packages using source checkouts and snapshots that are not bundled here.

## Working with the current release

```powershell
python run.py verify
python run.py docs-audit
python run.py help-audit
python run.py test
python run.py check "<A1111_INSTALL_DIR>"
```

Replace `<A1111_INSTALL_DIR>` with your own installed WebUI directory.
No developer-machine directories are required for these commands. The
six optional technical documents remain optional; use
`--docs-language EN` or `--docs-language RUS` when installing if desired.

## Reproducing a historical build (advanced)

To rebuild a historical version you need a matching A1111 source tree, and
for some versions, a pre-change snapshot and the previous release directory.
**These inputs are not included in the public repository.** Historical builders
require an explicit independent `--output` directory with a copy of the earlier
package; they refuse to write into the checked-out current repository.

```powershell
python tools/build_v15.py --help
python tools/build_v15.py --source "<A1111_SOURCE>" --backup "<BEFORE_SNAPSHOT>" --base "<V14_PACKAGE>" --output "<SEPARATE_V14_COPY>"
```

`--backup` can also be spelled `--before`. The following environment variables
are equivalent to the named options: `LORA_STUDIO_SOURCE`,
`LORA_STUDIO_BACKUP`, `LORA_STUDIO_BASE`, and `LORA_STUDIO_OUTPUT`.
Build scripts v13 and v14 need `--source` and `--output` only, because they
modify a pre-staged copy of the previous version. v10-v12 and v15 also require
backup snapshots. Version v12 expects a historical `modules.zip` alongside
its A1111 source files. The earlier v3-v9 scripts have their own `--base`
options and are not covered by the newer reconstruction helper.

The scripts are historical artifacts and are **not verified end-to-end here**:
rebuilding them requires their exact period inputs and may differ from the
current multilingual package. They are not the release packaging workflow.

## Historical reports and evidence

Some older audit documents describe paths such as `<HISTORICAL_AUDIT_ROOT>`,
`<A1111_SOURCE>`, or `<HISTORICAL_EXTENSIONS_DIR>`. These are **redactions of
machine-specific paths**, not directories users must create. The recorded
results and dates have not been recalculated. Reproduction commands mentioning
`<HISTORICAL_AUDIT_SCRIPTS>` refer to scripts not provided in this repository.

The `evidence/` logs and `archive/` upstream ZIPs remain unmodified for
provenance. Their historical paths may be visible. Do not interpret these
logs as current install directions. A build's checksum is defined by the
published `manifest.json` and verified using `python run.py verify`.
