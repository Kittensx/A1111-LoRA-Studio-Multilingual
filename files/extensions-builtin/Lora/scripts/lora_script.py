import re

import gradio as gr
from fastapi import FastAPI, HTTPException

import network
import networks
import lora  # noqa:F401
import lora_patches
import lora_presets
import lora_coverage
import lora_sidecars
import lora_help_registry
from lora_studio_access import create_access_dependency
import extra_networks_lora
import ui_extra_networks_lora
from modules import script_callbacks, ui_extra_networks, extra_networks, shared


def unload():
    networks.originals.undo()


def before_ui():
    ui_extra_networks.register_page(ui_extra_networks_lora.ExtraNetworksPageLora())

    networks.extra_network_lora = extra_networks_lora.ExtraNetworkLora()
    extra_networks.register_extra_network(networks.extra_network_lora)
    extra_networks.register_extra_network_alias(networks.extra_network_lora, "lyco")
    # Compatibility with sd-webui-loractl tags.  The implementation is native;
    # it does not install the extension's obsolete Network property monkeypatch.
    extra_networks.register_extra_network_alias(networks.extra_network_lora, "loractl")


networks.originals = lora_patches.LoraPatches()

script_callbacks.on_model_loaded(networks.assign_network_names_to_compvis_modules)
script_callbacks.on_script_unloaded(unload)
script_callbacks.on_before_ui(before_ui)
script_callbacks.on_infotext_pasted(networks.infotext_pasted)


shared.options_templates.update(shared.options_section(('extra_networks', "Extra Networks"), {
    "sd_lora": shared.OptionInfo("None", "Add network to prompt", gr.Dropdown, lambda: {"choices": ["None", *networks.available_networks]}, refresh=networks.list_available_networks),
    "lora_preferred_name": shared.OptionInfo("Alias from file", "When adding to prompt, refer to Lora by", gr.Radio, {"choices": ["Alias from file", "Filename"]}),
    "lora_add_hashes_to_infotext": shared.OptionInfo(True, "Add Lora hashes to infotext"),
    "lora_bundled_ti_to_infotext": shared.OptionInfo(True, "Add Lora name as TI hashes for bundled Textual Inversion").info('"Add Textual Inversion hashes to infotext" needs to be enabled'),
    "lora_show_all": shared.OptionInfo(False, "Always show all networks on the Lora page").info("otherwise, those detected as for incompatible version of Stable Diffusion will be hidden"),
    "lora_hide_unknown_for_versions": shared.OptionInfo([], "Hide networks of unknown versions for model versions", gr.CheckboxGroup, {"choices": ["SD1", "SD2", "SDXL"]}),
    "lora_in_memory_limit": shared.OptionInfo(0, "Number of Lora networks to keep cached in memory", gr.Number, {"precision": 0}),
    "lora_not_found_warning_console": shared.OptionInfo(False, "Lora not found warning in console"),
    "lora_not_found_gradio_warning": shared.OptionInfo(False, "Lora not found warning popup in webui"),
    "lora_gpu_backup": shared.OptionInfo(False, "Keep LoRA weight backups on GPU (eliminates PCIe transfer when switching LoRAs, uses ~model size extra VRAM). Recommended for 12GB+ VRAM."),
    "lora_block_weight_presets": shared.OptionInfo("", "Additional LoRA block-weight presets (one NAME:weights line each)", gr.Textbox, {"lines": 12}).info("Built-ins: NONE, ALL, INS, IND, INALL, MIDD, OUTD, OUTS, OUTALL, ALL0.5, FLUXALL. Use in a prompt as lbw=NAME."),
    "lora_elemental_weight_presets": shared.OptionInfo("", "Additional LoRA elemental presets (groups separated by a blank line)", gr.Textbox, {"lines": 12}).info("Rule format: NAME:blocks:module-substrings:weight. Built-ins: ATTNDEEPON, ATTNDEEPOFF, PROJDEEPOFF. Use in a prompt as lbwe=NAME; matching rules override lbw for that module."),
    "lora_studio_open_by_default": shared.OptionInfo(False, "Open LoRA Studio below txt2img and img2img prompts by default"),
    "lora_studio_presets": shared.OptionInfo('{"schema_version":3,"presets":[]}', "LoRA Studio preset library (managed by Studio)", gr.Textbox, {"lines": 8}).info("Versioned JSON. Use Studio to create, apply and delete presets; this field is available for backup."),
}))


shared.options_templates.update(shared.options_section(('compatibility', "Compatibility"), {
    "lora_functional": shared.OptionInfo(False, "Lora/Networks: use functional method (no weight merge for supported layers; MultiheadAttention still merges). Good for frequent LoRA changes; disable for best batch-generation performance."),
}))


def create_lora_json(obj: network.NetworkOnDisk):
    return {
        "name": obj.name,
        "alias": obj.alias,
        "path": obj.filename,
        "metadata": obj.metadata,
    }


def api_networks(_: gr.Blocks, app: FastAPI):
    from fastapi import Depends

    access = [Depends(create_access_dependency(shared.cmd_opts))]

    def preset_library_response():
        try:
            presets = lora_presets.parse_preset_library(
                shared.opts.lora_studio_presets,
                shared.opts.lora_block_weight_presets,
                shared.opts.lora_elemental_weight_presets,
            )
        except ValueError as error:
            return {
                "schema_version": lora_presets.PRESET_SCHEMA_VERSION,
                "presets": [],
                "error": f"Saved LoRA Studio presets are invalid: {error}",
            }
        return {
            "schema_version": lora_presets.PRESET_SCHEMA_VERSION,
            "presets": [preset.to_contract() for preset in presets],
        }

    # Contextual documentation is read-only and uses the existing A1111 API auth.
    @app.get("/sdapi/v1/lora-studio/help", dependencies=access)
    async def get_lora_studio_help_index(locale: str = "en"):
        return lora_help_registry.index(locale)

    @app.get("/sdapi/v1/lora-studio/help/{topic}", dependencies=access)
    async def get_lora_studio_help_article(topic: str, locale: str = "en"):
        try:
            return lora_help_registry.article(topic, locale)
        except (lora_help_registry.HelpError, UnicodeError, OSError) as error:
            raise HTTPException(status_code=404, detail="Help article is unavailable") from error

    @app.get("/sdapi/v1/loras", dependencies=access)
    async def get_loras():
        return [create_lora_json(obj) for obj in networks.available_networks.values()]

    @app.post("/sdapi/v1/refresh-loras", dependencies=access)
    async def refresh_loras():
        return networks.list_available_networks()

    @app.get("/sdapi/v1/lora-studio/coverage", dependencies=access)
    async def get_lora_studio_coverage(name: str):
        network_on_disk = networks.get_network_on_disk(name)
        if network_on_disk is None:
            raise HTTPException(status_code=404, detail="LoRA was not found in the current Extra Networks index")
        try:
            report = lora_coverage.inspect_safetensors_file(network_on_disk.filename)
        except lora_coverage.CoverageUnavailable as error:
            return {
                "schema_version": lora_coverage.COVERAGE_SCHEMA_VERSION,
                "status": "unavailable",
                "network_name": network_on_disk.name,
                "reason": str(error),
            }
        return {
            **report,
            "status": "ok",
            "network_name": network_on_disk.name,
        }

    @app.get("/sdapi/v1/lora-studio/sidecar", dependencies=access)
    async def get_lora_studio_sidecar(name: str):
        network_on_disk = networks.get_network_on_disk(name)
        if network_on_disk is None:
            raise HTTPException(status_code=404, detail="LoRA was not found in the current Extra Networks index")
        try:
            report = lora_sidecars.inspect_lora_sidecars(network_on_disk.filename)
        except lora_sidecars.SidecarUnavailable as error:
            return {
                "schema_version": lora_sidecars.SIDECAR_SCHEMA_VERSION,
                "status": "unavailable",
                "network_name": network_on_disk.name,
                "reason": str(error),
            }
        return {
            **report,
            "status": "ok",
            "network_name": network_on_disk.name,
        }

    @app.get("/sdapi/v1/lora-studio/presets", dependencies=access)
    async def get_lora_studio_presets():
        return preset_library_response()

    @app.put("/sdapi/v1/lora-studio/presets", dependencies=access)
    async def put_lora_studio_presets(payload: dict):
        try:
            presets = lora_presets.parse_preset_library(
                payload,
                shared.opts.lora_block_weight_presets,
                shared.opts.lora_elemental_weight_presets,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        try:
            shared.opts.set_and_save("lora_studio_presets", lora_presets.serialize_preset_library(presets), shared.config_filename)
        except (AssertionError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise HTTPException(status_code=409, detail=f"Could not save LoRA Studio presets: {error}") from error

        return preset_library_response()


script_callbacks.on_app_started(api_networks)

re_lora = re.compile(r"<(lora|lyco|loractl):([^:]+):")


def infotext_pasted(infotext, d):
    hashes = d.get("Lora hashes")
    if not hashes:
        return

    hashes = [x.strip().split(':', 1) for x in hashes.split(",")]
    hashes = {x[0].strip().replace(",", ""): x[1].strip() for x in hashes}

    def network_replacement(m):
        tag_type, alias = m.group(1), m.group(2)
        shorthash = hashes.get(alias)
        if shorthash is None:
            return m.group(0)

        network_on_disk = networks.available_network_hash_lookup.get(shorthash)
        if network_on_disk is None:
            return m.group(0)

        return f'<{tag_type}:{network_on_disk.get_alias()}:'

    d["Prompt"] = re.sub(re_lora, network_replacement, d["Prompt"])


script_callbacks.on_infotext_pasted(infotext_pasted)

shared.opts.onchange("lora_in_memory_limit", networks.purge_networks_from_memory)
