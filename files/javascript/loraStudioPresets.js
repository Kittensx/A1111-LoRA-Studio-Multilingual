/* Versioned, browser-side companion to lora_presets.py.
 *
 * Persistence is handled by the native validated API. This module only
 * constructs and applies JSON-compatible preset snapshots to a local Studio
 * draft; it never writes a prompt itself.
 */
(function (root) {
    "use strict";

    const SCHEMA_VERSION = 3;
    const LEGACY_SCHEMA_VERSION = 1;
    const HIRES_SCHEMA_VERSION = 2;
    const MAX_PRESETS = 100;

    function clone(value) {
        return JSON.parse(JSON.stringify(value));
    }

    function makePresetId() {
        if (root.crypto && typeof root.crypto.randomUUID === "function") return root.crypto.randomUUID();
        return `studio-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
    }

    function requireTitle(title) {
        if (typeof title !== "string") throw new Error("Preset title must be text");
        const text = title.trim();
        if (!text) throw new Error("Preset title is required");
        if (text.length > 120) throw new Error("Preset title must be 120 characters or fewer");
        return text;
    }

    function requirePresetId(presetId) {
        if (typeof presetId !== "string") throw new Error("Preset id must be text");
        const text = presetId.trim();
        if (!text) throw new Error("Preset id is missing");
        if (text.length > 160) throw new Error("Preset id must be 160 characters or fewer");
        return text;
    }

    function createPreset(title, spec, presetId) {
        if (!spec || !spec.weights || !spec.active_range) throw new Error("LoRA draft is incomplete");
        return {
            schema_version: SCHEMA_VERSION,
            preset_id: presetId || makePresetId(),
            title: requireTitle(title),
            model_profile: String(spec.model_profile || "auto"),
            weights: {
                te: clone(spec.weights.te),
                unet: clone(spec.weights.unet),
            },
            dyn_dim: spec.dyn_dim === undefined ? null : spec.dyn_dim,
            active_range: clone(spec.active_range),
            block_weights: spec.block_weights ? clone(spec.block_weights) : null,
            elemental_weights: spec.elemental_weights ? clone(spec.elemental_weights) : null,
            // A newly-created snapshot always makes the Hires choice
            // explicit, including the intentional "inherit" (null) case.
            hires: {
                te: spec.hires && spec.hires.te ? clone(spec.hires.te) : null,
                unet: spec.hires && spec.hires.unet ? clone(spec.hires.unet) : null,
            },
            favorite: false,
        };
    }

    function normalizeHires(hires) {
        if (hires === null) return null;
        if (!hires || typeof hires !== "object" || Array.isArray(hires)) {
            throw new Error("Preset Hires overrides must be an object or null");
        }
        if (!Object.prototype.hasOwnProperty.call(hires, "te") || !Object.prototype.hasOwnProperty.call(hires, "unet")) {
            throw new Error("Preset Hires overrides need te and unet fields");
        }
        return {te: hires.te === null ? null : clone(hires.te), unet: hires.unet === null ? null : clone(hires.unet)};
    }

    function normalizeElementalWeights(value) {
        if (value === null) return null;
        if (!value || typeof value !== "object" || Array.isArray(value) || typeof value.raw !== "string" || !value.raw.trim()) {
            throw new Error("Preset elemental weights must contain a named preset");
        }
        return {raw: value.raw.trim()};
    }

    function validatePreset(preset) {
        if (!preset || typeof preset !== "object" || ![LEGACY_SCHEMA_VERSION, HIRES_SCHEMA_VERSION, SCHEMA_VERSION].includes(preset.schema_version)) {
            throw new Error("Unsupported LoRA Studio preset");
        }
        // The first persisted Studio snapshots predate the optional profile
        // selector. Treat a missing field exactly as the native parser does.
        const normalizedProfile = preset.model_profile === undefined || (typeof preset.model_profile === "string" && !preset.model_profile.trim())
            ? {...preset, model_profile: "auto"}
            : preset;
        requirePresetId(normalizedProfile.preset_id);
        requireTitle(normalizedProfile.title);
        if (typeof normalizedProfile.model_profile !== "string" || !normalizedProfile.model_profile.trim()) {
            throw new Error("Preset model profile must be text");
        }
        if (!normalizedProfile.weights || typeof normalizedProfile.weights !== "object") throw new Error("Preset weights are missing");
        const favorite = normalizedProfile.favorite === undefined ? false : normalizedProfile.favorite;
        if (typeof favorite !== "boolean") throw new Error("Preset favorite must be true or false");
        // v1 snapshots deliberately did not include Hires.  Preserve their
        // old application semantics by migrating them to `hires: null`.
        // In v2+ a null means "do not alter Hires"; an object (even with both
        // fields null) is an explicit snapshot and will be applied.
        const hires = normalizedProfile.schema_version === LEGACY_SCHEMA_VERSION
            ? null
            : (() => {
                if (!Object.prototype.hasOwnProperty.call(normalizedProfile, "hires")) {
                    throw new Error(`Preset schema v${normalizedProfile.schema_version} needs a hires field`);
                }
                return normalizeHires(normalizedProfile.hires);
            })();
        const elementalWeights = normalizedProfile.schema_version < SCHEMA_VERSION
            ? null
            : (() => {
                if (!Object.prototype.hasOwnProperty.call(normalizedProfile, "elemental_weights")) {
                    throw new Error("Preset schema v3 needs an elemental_weights field");
                }
                return normalizeElementalWeights(normalizedProfile.elemental_weights);
            })();
        return {...normalizedProfile, schema_version: SCHEMA_VERSION, hires, elemental_weights: elementalWeights, favorite};
    }

    function applyPreset(spec, preset) {
        const normalized = validatePreset(preset);
        const result = clone(spec);
        result.model_profile = String(normalized.model_profile || "auto");
        if (normalized.weights.te) result.weights.te = clone(normalized.weights.te);
        if (normalized.weights.unet) result.weights.unet = clone(normalized.weights.unet);
        if (Object.prototype.hasOwnProperty.call(normalized, "dyn_dim")) result.dyn_dim = normalized.dyn_dim;
        if (Object.prototype.hasOwnProperty.call(normalized, "active_range")) {
            result.active_range = normalized.active_range ? clone(normalized.active_range) : {start: null, stop: null};
        }
        if (Object.prototype.hasOwnProperty.call(normalized, "block_weights")) {
            result.block_weights = normalized.block_weights ? clone(normalized.block_weights) : null;
        }
        if (Object.prototype.hasOwnProperty.call(normalized, "elemental_weights")) {
            result.elemental_weights = normalized.elemental_weights ? clone(normalized.elemental_weights) : null;
        }
        if (normalized.hires !== null) {
            result.hires = {
                te: normalized.hires.te === null ? null : clone(normalized.hires.te),
                unet: normalized.hires.unet === null ? null : clone(normalized.hires.unet),
            };
        }
        return result;
    }

    function envelope(presets) {
        const items = Array.isArray(presets) ? presets : [];
        if (items.length > MAX_PRESETS) {
            throw new Error(`Preset library supports at most ${MAX_PRESETS} presets`);
        }
        const result = {
            schema_version: SCHEMA_VERSION,
            presets: items.map((preset) => clone(validatePreset(preset))),
        };
        const identifiers = result.presets.map((preset) => preset.preset_id.toLocaleLowerCase());
        if (new Set(identifiers).size !== identifiers.length) {
            throw new Error("Preset identifiers must be unique");
        }
        return result;
    }

    function parseEnvelope(value) {
        let parsed = value;
        if (typeof value === "string") {
            try {
                parsed = JSON.parse(value);
            } catch (_) {
                throw new Error("Preset backup is not valid JSON");
            }
        }
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
            throw new Error("Preset backup must be an object");
        }
        if (![LEGACY_SCHEMA_VERSION, HIRES_SCHEMA_VERSION, SCHEMA_VERSION].includes(parsed.schema_version)) {
            throw new Error("Preset backup has an unsupported schema version");
        }
        if (!Array.isArray(parsed.presets)) {
            throw new Error("Preset backup must contain a presets array");
        }
        return envelope(parsed.presets);
    }

    // Quick recipes are small, explicit transformations of the existing
    // contract. They do not create a second preset format or a runtime state.
    const QUICK_MODES = Object.freeze([
        {id: "clear-masks", title: "All blocks / no module overrides", description: "Clears block and elemental masks. Keeps TE/U-Net weights, curves, timing, Hires and rank."},
        {id: "te-half", title: "Text encoder at 50% of U-Net", description: "Sets TE to half the current U-Net value (or curve). Keeps U-Net, Hires and other settings. Reapplying does not halve TE again. TE is evaluated during text encoding, not each denoising step."},
        {id: "te-off", title: "Text encoder off", description: "Sets the base-pass TE weight to zero. Keeps U-Net and any explicit Hires override. Only affects LoRAs containing text-encoder weights."},
        {id: "te-match", title: "Text encoder matches U-Net", description: "Copies the current U-Net value or curve to TE. Keeps all other settings; explicit Hires overrides take precedence. TE is evaluated during text encoding, not each denoising step."},
        {id: "fade-in", title: "U-Net fade-in", description: "Replaces a constant U-Net weight with a ramp: zero initially, full current strength at 35% of sampling. Keeps TE, timing and Hires. Existing curves are preserved: edit them in Full Studio."},
    ]);

    const ELEMENTAL_NAMES = ["ATTNDEEPON", "ATTNDEEPOFF", "PROJDEEPOFF"];
    const ELEMENTAL_DESCRIPTIONS = {
        ATTNDEEPON: "Sets the LoRA block multiplier to 1 for matching attn modules in IN05–OUT05. Overrides an existing block mask there; usually no effect without a mask.",
        ATTNDEEPOFF: "Sets LoRA contribution to zero for matching attn modules in IN05–OUT05. Other modules keep their block weights. Intended for classic U-Nets.",
        PROJDEEPOFF: "Sets LoRA contribution to zero for matching proj modules in IN05–OUT05. Other modules keep their block weights. Intended for classic U-Nets.",
    };

    function namedDefinitions(text, elemental) {
        const result = new Map();
        const groups = elemental ? String(text || "").split(/\r?\n[ \t]*\r?\n/) : String(text || "").split(/\r?\n/);
        groups.forEach((line) => {
            const separator = line.indexOf(":");
            const name = line.slice(0, separator).trim();
            const raw = line.slice(separator + 1).trim();
            if (separator < 1 || !name || !raw || (!elemental && name.startsWith("#"))) return;
            result.set(name.toLowerCase(), {name, raw});
        });
        return result;
    }

    function listQuickPresets(options = {}) {
        const catalog = QUICK_MODES.map((mode) => ({...mode, id: `mode:${mode.id}`, kind: "mode", mode: mode.id, group: "Quick modes"}));
        const blocks = root.loraStudioBlocks;
        const customBlocks = namedDefinitions(options.blockPresets, false);
        const blockNames = new Map((blocks ? blocks.builtinPresetNames() : []).map((name) => [name.toLowerCase(), name]));
        customBlocks.forEach(({name}, key) => blockNames.set(key, name));
        blockNames.forEach((name, key) => {
            const custom = customBlocks.get(key);
            // Native custom definitions contain numbers, not aliases to
            // another preset. Do not resolve nested aliases in the browser.
            const customValues = custom && blocks && blocks.parseExactBlockWeights(custom.raw);
            const resolved = custom ? (customValues ? {values: customValues} : null) : blocks && blocks.resolveBlockWeights(name);
            const count = resolved ? resolved.values.length : null;
            const table = resolved ? resolved.values.join(", ") : "Defined in WebUI settings";
            catalog.push({
                id: `block:${key}`, kind: "block", name, count,
                group: "Block Weight presets", title: `${name}${custom ? " (custom)" : ""}`,
                description: `Changes only the block mask${count ? ` (${count} values)` : ""}; keeps TE/U-Net, timing and module overrides. ${table}`,
            });
        });
        const customElements = namedDefinitions(options.elementalPresets, true);
        const elementNames = new Map(ELEMENTAL_NAMES.map((name) => [name.toLowerCase(), name]));
        customElements.forEach(({name}, key) => elementNames.set(key, name));
        elementNames.forEach((name, key) => {
            const custom = customElements.get(key);
            catalog.push({
                id: `elemental:${key}`, kind: "elemental", name, classicOnly: !custom,
                group: "Module presets", title: `${name}${custom ? " (custom)" : ""}`,
                description: custom ? `Custom module override: ${custom.raw}. Keeps the block mask and all other settings; layout has not been verified.` : ELEMENTAL_DESCRIPTIONS[name],
            });
        });
        (options.savedPresets || []).forEach((preset) => {
            const normalized = validatePreset(preset);
            catalog.push({
                id: `saved:${normalized.preset_id}`, kind: "saved", preset: normalized,
                group: normalized.favorite ? "Favourites" : "Saved presets", title: normalized.title,
                description: "Restores the saved control snapshot, including weights, masks, timing and rank. Keeps this LoRA's name and unknown tag arguments; legacy presets preserve Hires when absent.",
            });
        });
        return catalog;
    }

    function quickPresetProblem(spec, choice) {
        if (!spec || !spec.weights) return "Select a LoRA first.";
        if (!choice) return "Choose a preset.";
        if (choice.kind === "block" && choice.count === null) return "This preset does not contain a supported finite block-weight table. Check its definition in WebUI settings.";
        const profile = String(spec.model_profile || "auto");
        const flat = ["flux-61", "dit-61"].includes(profile);
        if (choice.kind === "block" && choice.count !== null && profile !== "auto") {
            if ((flat && choice.count !== 61) || (profile.startsWith("classic-") && choice.count === 61)) {
                return `This ${choice.count}-value preset does not fit the selected ${profile} layout.`;
            }
        }
        if (choice.kind === "elemental" && choice.classicOnly && flat) return `This IN/OUT module preset targets classic U-Nets, not ${profile}.`;
        if (choice.kind === "mode" && choice.mode === "fade-in" && spec.weights.unet.kind !== "scalar") return "U-Net already has a curve. Edit it in Full Studio; this shortcut will not replace it.";
        return "";
    }

    function scaleWeight(weight, factor) {
        if (weight.kind === "curve") return {...clone(weight), points: weight.points.map((point) => ({...point, value: point.value * factor}))};
        return {kind: "scalar", value: weight.value * factor};
    }

    // One dispatcher owns recipe semantics. Both views use this function;
    // adding a mode does not require another prompt-writing event handler.
    function applyQuickPreset(spec, choice) {
        const problem = quickPresetProblem(spec, choice);
        if (problem) throw new Error(problem);
        if (choice.kind === "saved") return applyPreset(spec, choice.preset);
        const result = clone(spec);
        if (choice.kind === "block") result.block_weights = {raw: choice.name};
        else if (choice.kind === "elemental") result.elemental_weights = {raw: choice.name};
        else if (choice.kind === "mode") {
            switch (choice.mode) {
            case "clear-masks":
                result.block_weights = null;
                result.elemental_weights = null;
                break;
            case "te-half": result.weights.te = scaleWeight(result.weights.unet, 0.5); break;
            case "te-off": result.weights.te = {kind: "scalar", value: 0}; break;
            case "te-match": result.weights.te = clone(result.weights.unet); break;
            case "fade-in":
                result.weights.unet = {kind: "curve", points: [{position: 0, value: 0}, {position: 0.35, value: result.weights.unet.value}, {position: 1, value: result.weights.unet.value}]};
                break;
            default: throw new Error("Unknown quick preset mode");
            }
        } else throw new Error("Unknown quick preset type");
        return result;
    }

    const api = {SCHEMA_VERSION, MAX_PRESETS, createPreset, validatePreset, applyPreset, envelope, parseEnvelope, listQuickPresets, quickPresetProblem, applyQuickPreset};
    root.loraStudioPresets = api;
    if (typeof module !== "undefined" && module.exports) module.exports = api;
}(typeof window === "undefined" ? globalThis : window));
