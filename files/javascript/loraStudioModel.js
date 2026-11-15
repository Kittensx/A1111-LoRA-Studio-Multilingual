/*
 * Browser-side half of the LoRA tag contract.  It deliberately owns prompt
 * grammar only; UI rendering stays in loraStudioView.js once Studio exists.
 * The tiny CommonJS export is for the torch-free contract test, while WebUI
 * uses window.loraStudioModel.
 */
(function (root) {
    "use strict";

    const SCHEMA_VERSION = 1;
    const TAG_TYPES = new Set(["lora", "lyco", "loractl"]);
    const KNOWN = new Set(["te", "unet", "dyn", "hr", "hrte", "hrunet", "start", "stop", "step", "lbw", "w", "lbwe", "profile"]);

    function finiteNumber(value, description) {
        const result = Number(value);
        if (!Number.isFinite(result)) {
            throw new Error(`${description} must be finite, got ${JSON.stringify(value)}`);
        }
        return result;
    }

    function formatNumber(value) {
        const number = Number(value);
        if (!Number.isFinite(number)) {
            throw new Error(`LoRA number must be finite, got ${JSON.stringify(value)}`);
        }
        if (Object.is(number, -0)) return "0";

        // Python's schema is the canonical serializer and uses ``.12g``.
        // ``String(number)`` differs at both precision and exponent boundaries
        // (for example 0.30000000000000004 and 1e-7), so emulate its compact
        // 12-significant-digit spelling here.
        const precision = number.toPrecision(12);
        const trimMantissa = (text) => text
            .replace(/(\.\d*?[1-9])0+$/, "$1")
            .replace(/\.0+$/, "");
        const scientific = (text) => {
            const [mantissa, rawExponent] = text.toLowerCase().split("e");
            const exponent = Number(rawExponent);
            return `${trimMantissa(mantissa)}e${exponent < 0 ? "-" : "+"}${String(Math.abs(exponent)).padStart(2, "0")}`;
        };

        if (precision.includes("e")) return scientific(precision);

        // JavaScript keeps 1e-5 in fixed notation for toPrecision(), while
        // ``g`` switches below 1e-4.  Check the rounded precision value so a
        // value that rounds up to exactly 0.0001 retains fixed notation.
        if (Math.abs(Number(precision)) > 0 && Math.abs(Number(precision)) < 1e-4) {
            return scientific(number.toExponential(11));
        }
        return trimMantissa(precision);
    }

    function parseWeight(raw, fallback = 1) {
        if (raw === undefined || raw === null || String(raw).trim() === "") {
            return {kind: "scalar", value: Number(fallback)};
        }

        const text = String(raw).trim();
        if (!text.includes("@") && !text.includes("~")) {
            return {kind: "scalar", value: finiteNumber(text, "LoRA weight")};
        }

        const points = new Map();
        text.split(/[,;]/).forEach((item) => {
            if (!item.trim()) return;
            const fields = item.trim().split(/[@~]/);
            if (fields.length !== 2 || !fields[0].trim() || !fields[1].trim()) {
                throw new Error(`invalid LoRA weight curve item ${JSON.stringify(item)}`);
            }
            const value = finiteNumber(fields[0].trim(), "curve weight");
            const position = finiteNumber(fields[1].trim(), "curve position");
            points.set(position, value);
        });

        if (!points.size) {
            throw new Error("LoRA weight curve has no points");
        }
        return {
            kind: "curve",
            points: Array.from(points, ([position, value]) => ({position, value})).sort((left, right) => left.position - right.position),
        };
    }

    function serializeWeight(weight) {
        if (weight.kind === "scalar") {
            return formatNumber(weight.value);
        }
        return weight.points.map((point) => `${formatNumber(point.value)}@${formatNumber(point.position)}`).join(",");
    }

    function initialWeight(weight, fallback) {
        if (!weight) return Number(fallback);
        return weight.kind === "curve" ? weight.points[0].value : weight.value;
    }

    function parseStepRange(named) {
        let start = named.start;
        let stop = named.stop;
        if (named.step !== undefined && String(named.step).trim()) {
            const number = '([+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?)';
            const match = String(named.step).match(new RegExp(`^\\s*${number}\\s*[-~,;]\\s*${number}\\s*$`));
            if (!match) {
                throw new Error("step must use start-stop");
            }
            const values = match.slice(1);
            if (start === undefined) start = values[0].trim();
            if (stop === undefined) stop = values[1].trim();
        }
        const parsedStart = start === undefined || String(start).trim() === "" ? null : finiteNumber(start, "LoRA start");
        const parsedStop = stop === undefined || String(stop).trim() === "" ? null : finiteNumber(stop, "LoRA stop");
        if ([parsedStart, parsedStop].some(value => value !== null && value < 0)) throw new Error("LoRA start and stop must be non-negative");
        if (parsedStart !== null && parsedStop !== null && (parsedStart <= 1) === (parsedStop <= 1) && parsedStart > parsedStop) {
            throw new Error("LoRA start must not be greater than stop");
        }
        return {start: parsedStart, stop: parsedStop};
    }

    // Legacy tags accept signed integer ranks; interactive editors accept
    // positive ranks or Off. Do not silently clamp a user's input to 1.
    function parseDynamicRank(raw, positiveOnly = false) {
        const text = raw === undefined || raw === null ? "" : String(raw).trim();
        if (!text) return null;
        const rank = Number(text);
        if (!/^[+-]?\d+$/.test(text) || !Number.isSafeInteger(rank) || (positiveOnly && rank <= 0)) {
            throw new Error(positiveOnly ? "Dynamic rank must be a positive safe integer, or blank for Off" : `LoRA dyn must be a safe integer, got ${JSON.stringify(raw)}`);
        }
        return rank;
    }

    function parseTag(tag) {
        const match = String(tag).trim().match(/^<([A-Za-z0-9_-]+):([^<>]*)>$/);
        if (!match) {
            throw new Error("expected a complete <lora:...> tag");
        }
        const tagType = match[1].toLowerCase();
        if (!TAG_TYPES.has(tagType)) {
            throw new Error(`unsupported LoRA tag type ${JSON.stringify(tagType)}`);
        }

        const items = match[2].split(":");
        const networkName = String(items.shift() || "").trim();
        if (!networkName) {
            throw new Error("LoRA tag needs a network name as its first argument");
        }

        const positional = [networkName];
        const named = {};
        const namedItems = [];
        items.forEach((item) => {
            const equals = item.indexOf("=");
            if (equals > 0 && item.slice(0, equals).trim()) {
                const name = item.slice(0, equals).trim();
                const value = item.slice(equals + 1);
                named[name] = value;
                namedItems.push({name, value});
            } else {
                positional.push(item);
            }
        });

        // Match the native parser exactly: if U-Net is omitted, it inherits
        // the *raw TE specification*, not merely TE's first scalar value.
        // This matters for curves: <lora:x:0.2@0,0.8@1> controls both
        // branches with the same curve.
        const teRaw = named.te !== undefined ? named.te : (positional[1] !== undefined ? positional[1] : 1);
        const te = parseWeight(teRaw);
        const unetRaw = named.unet !== undefined ? named.unet : (positional[2] !== undefined ? positional[2] : teRaw);
        const unet = parseWeight(unetRaw, initialWeight(te, 1));
        const dynRaw = named.dyn !== undefined ? named.dyn : positional[3];
        const dynDim = parseDynamicRank(dynRaw);

        const hr = named.hr;
        const hrteRaw = named.hrte !== undefined ? named.hrte : hr;
        const hrunetRaw = named.hrunet !== undefined ? named.hrunet : hr;
        const profile = String(named.profile === undefined ? "auto" : named.profile).trim().toLowerCase() || "auto";

        return {
            schema_version: SCHEMA_VERSION,
            tag_type: tagType,
            network_name: networkName,
            weights: {te, unet},
            hires: {
                te: hrteRaw === undefined ? null : parseWeight(hrteRaw, initialWeight(te, 1)),
                unet: hrunetRaw === undefined ? null : parseWeight(hrunetRaw, initialWeight(unet, 1)),
            },
            dyn_dim: dynDim,
            active_range: parseStepRange(named),
            block_weights: named.lbw !== undefined || named.w !== undefined ? {raw: String(named.lbw !== undefined ? named.lbw : named.w)} : null,
            elemental_weights: named.lbwe !== undefined ? {raw: String(named.lbwe)} : null,
            model_profile: profile,
            passthrough_named: namedItems.filter((item) => !KNOWN.has(item.name)),
            passthrough_positional: positional.slice(4),
        };
    }

    function serializeTag(spec, canonicalTagType) {
        const tagType = canonicalTagType ? "lora" : spec.tag_type;
        const hasFuturePositionals = (spec.passthrough_positional || []).length > 0;
        const items = hasFuturePositionals ? [
            spec.network_name,
            serializeWeight(spec.weights.te),
            serializeWeight(spec.weights.unet),
            spec.dyn_dim === null || spec.dyn_dim === undefined ? "" : String(spec.dyn_dim),
        ] : [
            spec.network_name,
            `te=${serializeWeight(spec.weights.te)}`,
            `unet=${serializeWeight(spec.weights.unet)}`,
        ];
        if (!hasFuturePositionals && spec.dyn_dim !== null && spec.dyn_dim !== undefined) items.push(`dyn=${spec.dyn_dim}`);
        if (spec.hires.te) items.push(`hrte=${serializeWeight(spec.hires.te)}`);
        if (spec.hires.unet) items.push(`hrunet=${serializeWeight(spec.hires.unet)}`);
        if (spec.active_range.start !== null) items.push(`start=${formatNumber(spec.active_range.start)}`);
        if (spec.active_range.stop !== null) items.push(`stop=${formatNumber(spec.active_range.stop)}`);
        if (spec.block_weights) items.push(`lbw=${spec.block_weights.raw}`);
        if (spec.elemental_weights) items.push(`lbwe=${spec.elemental_weights.raw}`);
        if (spec.model_profile !== "auto") items.push(`profile=${spec.model_profile}`);
        (spec.passthrough_positional || []).forEach((item) => items.push(item));
        (spec.passthrough_named || []).forEach((item) => items.push(`${item.name}=${item.value}`));
        return `<${tagType}:${items.join(":")}>`;
    }

    function findTags(prompt) {
        const text = String(prompt || "");
        const pattern = /<[A-Za-z0-9_-]+:[^<>]*>/g;
        const result = [];
        let match;
        while ((match = pattern.exec(text)) !== null) {
            try {
                result.push({
                    start: match.index,
                    end: match.index + match[0].length,
                    tag: match[0],
                    spec: parseTag(match[0]),
                });
            } catch (_) {
                // It is another Extra Network or an invalid/incomplete LoRA
                // tag.  Leave it untouched; Studio owns only valid LoRA tags.
            }
        }
        return result;
    }

    function replaceTagSpan(prompt, span, replacement) {
        const text = String(prompt || "");
        if (!span || !Number.isInteger(span.start) || !Number.isInteger(span.end) || span.start < 0 || span.end < span.start) {
            throw new Error("invalid LoRA tag span");
        }
        if (text.slice(span.start, span.end) !== span.tag) {
            throw new Error("LoRA tag changed while its editor was open");
        }
        return text.slice(0, span.start) + replacement + text.slice(span.end);
    }

    function createDefaultSpec(networkName, defaultWeight, tagType) {
        const weight = Number.isFinite(Number(defaultWeight)) ? defaultWeight : 1;
        return parseTag(`<${tagType || "lora"}:${networkName}:${formatNumber(weight)}>`);
    }

    const api = {
        parseTag,
        serializeTag,
        parseWeight,
        parseStepRange,
        parseDynamicRank,
        serializeWeight,
        formatNumber,
        findTags,
        replaceTagSpan,
        createDefaultSpec,
    };
    root.loraStudioModel = api;
    if (typeof module !== "undefined" && module.exports) module.exports = api;
}(typeof window === "undefined" ? globalThis : window));
