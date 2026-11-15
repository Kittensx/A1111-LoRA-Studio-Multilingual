/* LoRA Studio localization runtime. UI only; model names and prompt tags are never translated. */
(function (root) {
    "use strict";
    const DEFAULT = "en";
    const STORAGE = "loraStudio.language.v1";
    const supported = Object.freeze({"en": "English", "ru": "Русский", "fr": "Français", "es": "Español", "it": "Italiano", "ja": "日本語", "zh-CN": "简体中文", "pt-BR": "Português (Brasil)", "de": "Deutsch", "ko": "한국어"});
    const catalogs = Object.create(null);
    const aliases = Object.create(null);
    const missing = new Set();
    function canonical(raw) {
        const value = String(raw || "").trim().replace(/_/g, "-");
        if (Object.prototype.hasOwnProperty.call(supported, value)) return value;
        const base = value.split("-")[0].toLowerCase();
        if (base === "zh") return "zh-CN";
        if (base === "pt") return "pt-BR";
        return Object.prototype.hasOwnProperty.call(supported, base) ? base : DEFAULT;
    }
    function stored() { try { return root.localStorage && root.localStorage.getItem(STORAGE); } catch (_) { return null; } }
    let locale = canonical(stored() || (root.navigator && root.navigator.language) || DEFAULT);
    function register(lang, messages) {
        if (!Object.prototype.hasOwnProperty.call(supported, lang)) throw new Error("Unsupported LoRA Studio locale: " + lang);
        if (!messages || typeof messages !== "object" || Array.isArray(messages)) throw new TypeError("Locale catalog must be an object");
        const clean = Object.create(null);
        for (const [key, value] of Object.entries(messages)) {
            if (key === "__proto__" || key === "constructor" || key === "prototype") continue;
            if (typeof value === "string") clean[key] = value;
        }
        catalogs[lang] = clean;
        if (lang === DEFAULT) {
            for (const [key, value] of Object.entries(clean)) {
                if (!Object.prototype.hasOwnProperty.call(aliases, value)) aliases[value] = key;
            }
        }
    }
    function t(key, vars) {
        if (typeof key !== "string") return key;
        const id = Object.prototype.hasOwnProperty.call(aliases, key) ? aliases[key] : key;
        let result = (catalogs[locale] && catalogs[locale][id]) || (catalogs.en && catalogs.en[id]);
        if (result === undefined) {
            // Unknown keys and user-supplied prompt strings never undergo translation.
            result = key;
            if (key.startsWith("studio.")) missing.add(key);
        }
        if (vars && typeof vars === "object") result = result.replace(/\{([a-z][a-z0-9_]*)\}/gi,
          (all, name) => Object.prototype.hasOwnProperty.call(vars, name) ? String(vars[name]) : all);
        return result;
    }
    function setLocale(wanted) {
        const next = canonical(wanted);
        if (next === locale) return locale;
        locale = next;
        try { if (root.localStorage) root.localStorage.setItem(STORAGE, next); } catch (_) { /* storage is optional */ }
        if (typeof root.dispatchEvent === "function") {
            let event;
            if (typeof root.CustomEvent === "function") event = new root.CustomEvent("lora-studio-locale-change", {detail: {locale: next}});
            else if (typeof root.Event === "function") event = new root.Event("lora-studio-locale-change");
            else event = {type:"lora-studio-locale-change"};
            root.dispatchEvent(event);
        }
        return locale;
    }
    const api = Object.freeze({supported, canonical, register, t, setLocale, getLocale: () => locale,
                             missingKeys: () => Array.from(missing).sort()});
    root.loraStudioI18n = api;
    // Locale files may load before this script in A1111's JS discovery order.
    const pending = root.loraStudioPendingLocales || [];
    root.loraStudioPendingLocales = [];
    for (const [lang, values] of pending) register(lang, values);
}(typeof window !== "undefined" ? window : globalThis));
