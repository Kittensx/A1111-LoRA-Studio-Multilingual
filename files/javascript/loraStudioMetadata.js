/* Read-only metadata helpers for LoRA Studio.
 *
 * The native /sdapi/v1/loras endpoint already exposes the safetensors
 * metadata.  This module only turns common, non-authoritative fields into
 * safe display information and explicit trigger suggestions.  It never
 * writes a prompt or assumes a metadata value is trustworthy.
 */
(function (root) {
    "use strict";

    const MAX_TRIGGER_COUNT = 32;
    const MAX_TRIGGER_LENGTH = 160;
    const MAX_METADATA_TEXT_LENGTH = 4000;
    const TRIGGER_FIELDS = new Set([
        "ss_tag_frequency",
        "ssmd_trigger_words",
        "trigger_words",
        "trigger words",
        "activation text",
        "ssmd_activation_text",
        "modelspec.tags",
    ]);

    function asObject(value) {
        return value && typeof value === "object" && !Array.isArray(value) ? value : {};
    }

    function nonEmpty(value) {
        const text = typeof value === "string" ? value.trim() : "";
        if (!text) return null;
        return text.length > MAX_METADATA_TEXT_LENGTH ? `${text.slice(0, MAX_METADATA_TEXT_LENGTH)}…` : text;
    }

    function parseStructured(value) {
        if (typeof value !== "string") return value;
        const text = value.trim();
        if (!text || !/^[{[]/.test(text)) return value;
        try {
            return JSON.parse(text);
        } catch (_) {
            return value;
        }
    }

    function cleanTrigger(value) {
        const text = String(value || "").replace(/\s+/g, " ").trim();
        if (!text || text.length > MAX_TRIGGER_LENGTH || /^\d+$/.test(text)) return null;
        return text;
    }

    function addTrigger(scores, value, score) {
        const trigger = cleanTrigger(value);
        if (!trigger) return;
        const key = trigger.toLocaleLowerCase();
        const current = scores.get(key);
        if (!current || score > current.score) scores.set(key, {trigger, score});
    }

    function addDelimitedTriggers(scores, value, score) {
        String(value || "").split(/[\n,;]/).forEach((item) => addTrigger(scores, item, score));
    }

    function collectTriggers(scores, value, depth) {
        if (depth > 4 || value === null || value === undefined) return;
        const structured = parseStructured(value);
        if (structured !== value) {
            collectTriggers(scores, structured, depth + 1);
            return;
        }
        if (typeof value === "string") {
            addDelimitedTriggers(scores, value, 1);
            return;
        }
        if (Array.isArray(value)) {
            value.forEach((item) => collectTriggers(scores, item, depth + 1));
            return;
        }
        if (typeof value !== "object") return;

        Object.entries(value).forEach(([key, item]) => {
            if (typeof item === "number" && Number.isFinite(item)) {
                addTrigger(scores, key, item);
            } else if (typeof item === "string" && /^\s*\d+(?:\.\d+)?\s*$/.test(item)) {
                addTrigger(scores, key, Number(item));
            } else if (item === true) {
                addTrigger(scores, key, 1);
            } else {
                collectTriggers(scores, item, depth + 1);
            }
        });
    }

    function extractTriggerWords(metadata) {
        const scores = new Map();
        Object.entries(asObject(metadata)).forEach(([key, value]) => {
            if (TRIGGER_FIELDS.has(key.toLocaleLowerCase())) collectTriggers(scores, value, 0);
        });
        return [...scores.values()]
            .sort((left, right) => right.score - left.score || left.trigger.localeCompare(right.trigger))
            .slice(0, MAX_TRIGGER_COUNT)
            .map((item) => item.trigger);
    }

    function firstMetadata(metadata, keys) {
        for (const key of keys) {
            const value = nonEmpty(metadata[key]);
            if (value) return value;
        }
        return null;
    }

    function describeLora(item) {
        const source = asObject(item);
        const metadata = asObject(source.metadata);
        return {
            title: firstMetadata(metadata, ["ssmd_display_name", "modelspec.title", "ss_output_name"]) || nonEmpty(source.alias) || nonEmpty(source.name),
            description: firstMetadata(metadata, ["modelspec.description", "ssmd_description", "description"]),
            baseModel: firstMetadata(metadata, ["ss_base_model_version", "ss_base_model", "modelspec.base_model"]),
            hash: firstMetadata(metadata, ["sshs_model_hash", "ss_network_hash"]),
            triggers: extractTriggerWords(metadata),
        };
    }

    const api = {extractTriggerWords, describeLora};
    root.loraStudioMetadata = api;
    if (typeof module !== "undefined" && module.exports) module.exports = api;
}(typeof window === "undefined" ? globalThis : window));
