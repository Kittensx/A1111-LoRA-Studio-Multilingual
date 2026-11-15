/* Profile-aware, browser-only helpers for the LoRA block-weight editor.
 *
 * The profile list deliberately mirrors lora_schema.MODEL_PROFILES and is
 * checked against it in the torch-free contract tests. A profile controls the
 * editor layout only; the native runtime still validates the exact lbw value
 * when the prompt is processed.
 */
(function (root) {
    "use strict";

    const PROFILES = Object.freeze([
        {id: "auto", label: "Automatic / preserve raw value", blockIds: []},
        {id: "classic-12", label: "Classic 12-block layout", blockIds: ["BASE", "IN04", "IN05", "IN07", "IN08", "M00", "OUT00", "OUT01", "OUT02", "OUT03", "OUT04", "OUT05"]},
        {id: "classic-17", label: "Classic 17-block layout", blockIds: ["BASE", "IN01", "IN02", "IN04", "IN05", "IN07", "IN08", "M00", "OUT03", "OUT04", "OUT05", "OUT06", "OUT07", "OUT08", "OUT09", "OUT10", "OUT11"]},
        {id: "classic-20", label: "Classic 20-block layout", blockIds: ["BASE", "IN00", "IN01", "IN02", "IN03", "IN04", "IN05", "IN06", "IN07", "IN08", "M00", "OUT00", "OUT01", "OUT02", "OUT03", "OUT04", "OUT05", "OUT06", "OUT07", "OUT08"]},
        {id: "classic-26", label: "Classic 26-block layout", blockIds: ["BASE", "IN00", "IN01", "IN02", "IN03", "IN04", "IN05", "IN06", "IN07", "IN08", "IN09", "IN10", "IN11", "M00", "OUT00", "OUT01", "OUT02", "OUT03", "OUT04", "OUT05", "OUT06", "OUT07", "OUT08", "OUT09", "OUT10", "OUT11"]},
        {id: "flux-61", label: "Flux 61-block layout", blockIds: ["CLIP", "T5", "IN", "D00", "D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08", "D09", "D10", "D11", "D12", "D13", "D14", "D15", "D16", "D17", "D18", "S00", "S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18", "S19", "S20", "S21", "S22", "S23", "S24", "S25", "S26", "S27", "S28", "S29", "S30", "S31", "S32", "S33", "S34", "S35", "S36", "S37", "OUT"]},
        {id: "dit-61", label: "Flat DiT 61-block layout", blockIds: ["CLIP", "T5", "IN", "B00", "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B13", "B14", "B15", "B16", "B17", "B18", "B19", "B20", "B21", "B22", "B23", "B24", "B25", "B26", "B27", "B28", "B29", "B30", "B31", "B32", "B33", "B34", "B35", "B36", "B37", "B38", "B39", "B40", "B41", "B42", "B43", "B44", "B45", "B46", "B47", "B48", "B49", "B50", "B51", "B52", "B53", "B54", "B55", "B56", "OUT"]},
    ]);

    const CLASSIC_17_PRESETS = Object.freeze({
        NONE: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ALL: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        INS: [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        IND: [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        INALL: [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        MIDD: [1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        OUTD: [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        OUTS: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        OUTALL: [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "ALL0.5": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    });

    const BUILTIN_PRESETS = Object.freeze({...CLASSIC_17_PRESETS, FLUXALL: Object.freeze(Array(61).fill(1))});

    function cloneProfile(profile) {
        return {id: profile.id, label: profile.label, blockIds: [...profile.blockIds]};
    }

    function listProfiles() {
        return PROFILES.map(cloneProfile);
    }

    function getProfile(profileId) {
        const key = String(profileId || "auto").trim().toLocaleLowerCase();
        const profile = PROFILES.find((item) => item.id === key);
        return cloneProfile(profile || PROFILES[0]);
    }

    function parseExactBlockWeights(raw) {
        const text = String(raw || "").trim();
        if (!text) return null;
        const parts = text.replace(/\n/g, ",").split(",").map((part) => part.trim());
        if (parts.some(part => !part)) return null;
        if (!parts.length || ![12, 17, 20, 26, 61].includes(parts.length)) return null;
        const values = parts.map(Number);
        return values.every(Number.isFinite) ? values : null;
    }

    function resolveBlockWeights(raw, customPresets = "") {
        const presetName = String(raw || "").trim().toLocaleUpperCase();
        // Settings override built-ins on the server. The visual table must use
        // the same definition, including refusing an invalid overriding entry.
        let custom;
        String(customPresets).split(/\r?\n/).forEach(line => {
            const split = line.indexOf(":");
            const name = line.slice(0, split).trim();
            const value = line.slice(split + 1).trim();
            if (split > 0 && !name.startsWith("#") && name.toLocaleUpperCase() === presetName && value) custom = value;
        });
        if (custom !== undefined) {
            const values = parseExactBlockWeights(custom);
            return values ? {kind: "custom", presetName, values} : null;
        }
        const exact = parseExactBlockWeights(raw);
        if (exact) return {kind: "exact", values: exact};
        const preset = BUILTIN_PRESETS[presetName];
        return preset ? {kind: "builtin", presetName, values: [...preset]} : null;
    }

    function profileForValues(values, requestedProfile) {
        const requested = getProfile(requestedProfile);
        if (requested.id !== "auto" && requested.blockIds.length === values.length) return requested;
        const candidates = PROFILES.filter((profile) => profile.blockIds.length === values.length);
        return cloneProfile(candidates[0] || PROFILES[0]);
    }

    function projectClassicValues(values, requestedProfile) {
        const source = profileForValues(values, "auto");
        const target = getProfile(requestedProfile);
        if (!source.id.startsWith("classic-") || !target.id.startsWith("classic-")) return null;
        // Do not discard nonzero blocks when projecting to a smaller layout.
        if (source.blockIds.some((id, i) => !target.blockIds.includes(id) && values[i] !== 0)) return null;
        return target.blockIds.map(id => {
            const index = source.blockIds.indexOf(id);
            return index < 0 ? 0 : values[index];
        });
    }

    function groupBlocks(profileId) {
        const profile = getProfile(profileId);
        const groups = new Map();
        const add = (name, blockId) => {
            if (!groups.has(name)) groups.set(name, []);
            groups.get(name).push(blockId);
        };
        profile.blockIds.forEach((blockId) => {
            if (["BASE", "CLIP", "T5"].includes(blockId)) add("Text / base", blockId);
            else if (blockId === "M00") add("Middle", blockId);
            else if (blockId === "IN" || blockId.startsWith("IN")) add("Input", blockId);
            else if (blockId.startsWith("D")) add("Double blocks", blockId);
            else if (blockId.startsWith("S")) add("Single blocks", blockId);
            else if (blockId === "OUT" || blockId.startsWith("OUT")) add("Output", blockId);
            else add("Other", blockId);
        });
        return [...groups.entries()].map(([label, blockIds]) => ({label, blockIds}));
    }

    function formatNumber(value) {
        const number = Number(value);
        if (!Number.isFinite(number)) throw new Error("Block weight must be finite");
        if (root.loraStudioModel && typeof root.loraStudioModel.formatNumber === "function") {
            return root.loraStudioModel.formatNumber(number);
        }
        return String(number);
    }

    function serializeExactBlockWeights(values) {
        if (!Array.isArray(values) || ![12, 17, 20, 26, 61].includes(values.length)) {
            throw new Error("Block weights need 12, 17, 20, 26, or 61 values");
        }
        return values.map(formatNumber).join(",");
    }

    const api = {
        listProfiles,
        getProfile,
        parseExactBlockWeights,
        resolveBlockWeights,
        profileForValues,
        projectClassicValues,
        groupBlocks,
        serializeExactBlockWeights,
        builtinPresetNames: () => Object.keys(BUILTIN_PRESETS),
    };
    root.loraStudioBlocks = api;
    if (typeof module !== "undefined" && module.exports) module.exports = api;
}(typeof window === "undefined" ? globalThis : window));
