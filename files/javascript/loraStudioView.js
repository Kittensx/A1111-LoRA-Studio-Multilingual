/* Native LoRA Studio shell for txt2img and img2img.
 *
 * This file contains DOM behavior only.  Prompt syntax, spans and atomic
 * replacement belong to loraStudioModel.js so the card editor and Studio use
 * the same contract.
 */
(function () {
    "use strict";


    function t(key, vars) {
        const api = window.loraStudioI18n;
        return api ? api.t(key, vars) : (vars ? key.replace(/\{([a-z][a-z0-9_]*)\}/gi, (item, name) => Object.prototype.hasOwnProperty.call(vars, name) ? String(vars[name]) : item) : key);
    }

    const studioStates = Object.create(null);
    const studioTabs = ["txt2img", "img2img"];
    const presetLibrary = {items: null, loading: false, error: ""};
    const RECENT_PRESET_STORAGE_KEY = "loraStudio.recentPresetIds.v1";
    const RECENT_PRESET_LIMIT = 8;

    function studioModel() {
        if (!window.loraStudioModel) throw new Error("LoRA Studio prompt model is not loaded");
        return window.loraStudioModel;
    }

    function studioMetadata() {
        return window.loraStudioMetadata || null;
    }

    function studioBlocks() {
        return window.loraStudioBlocks || null;
    }

    function studioPresets() {
        return window.loraStudioPresets || null;
    }

    function studioCurves() {
        return window.loraStudioCurves || null;
    }

    function studioState(tabname) {
        if (!studioStates[tabname]) {
            studioStates[tabname] = {
                selectedId: null,
                draft: null,
                entries: [],
                message: "",
                messageKind: "",
                libraryItems: null,
                libraryLoading: false,
                advancedOpen: false,
                curveOpen: false,
                selectedPresetId: "",
                quickPresetId: "",
                presetLibraryOpen: false,
                quickSaveOpen: false,
                presetTitle: "",
                presetTransferOpen: false,
                presetTransferText: null,
                coverage: null,
                coverageForNetwork: "",
                coverageLoading: false,
                coverageError: "",
                coverageErrorForNetwork: "",
                coverageOpen: false,
                sidecar: null,
                sidecarForNetwork: "",
                sidecarLoading: false,
                sidecarError: "",
                sidecarErrorForNetwork: "",
                sidecarOpen: false,
                quickTune: null,
            };
        }
        return studioStates[tabname];
    }

    function studioTextarea(tabname) {
        return gradioApp().querySelector(`#${tabname}_prompt > label > textarea`);
    }

    function studioRoot(tabname) {
        return gradioApp().querySelector(`#${tabname}_lora_studio`);
    }

    function quickTuneRoot(tabname) {
        return gradioApp().querySelector(`#${tabname}_lora_quick_tune`);
    }

    function studioMountTarget(tabname, promptContainer) {
        // `*_prompt_container` and `*_actions_column` are direct children of
        // `*_toprow`. Inserting Studio after the prompt container makes it a
        // third flex column and steals width from both prompts and Generate.
        // The whole top row is the stable, full-width insertion boundary.
        return gradioApp().querySelector(`#${tabname}_toprow`) || promptContainer;
    }

    function entryId(entry) {
        return `${entry.start}:${entry.end}:${entry.tag}`;
    }

    function clone(value) {
        return JSON.parse(JSON.stringify(value));
    }

    function makeElement(tag, className, text) {
        const element = document.createElement(tag);
        if (className) element.className = className;
        if (text !== undefined) element.textContent = text;
        return element;
    }

    function button(label, className, handler) {
        const element = makeElement("button", className, label);
        element.type = "button";
        element.addEventListener("click", handler);
        return element;
    }

    // Help article identifiers are stable and independent of translated labels.
    function helpIcon(topic) {
        const control = button("?", "lora-studio-help-icon", (event) => {
            event.preventDefault();
            event.stopPropagation();
            if (window.loraStudioHelp) window.loraStudioHelp.open(topic);
        });
        control.title = t("studio.help_registry.open_topic");
        control.setAttribute("aria-label", t("studio.help_registry.open_topic"));
        control.setAttribute("data-lora-help-context", topic);
        control.addEventListener("keydown", (event) => event.stopPropagation());
        return control;
    }

    function helpHeading(tag, label, topic) {
        const heading = makeElement(tag, "lora-studio-help-heading");
        heading.append(makeElement("span", "", label), helpIcon(topic));
        return heading;
    }

    function weightInitial(weight, fallback) {
        if (!weight) return fallback;
        return weight.kind === "curve" ? weight.points[0].value : weight.value;
    }

    function weightSummary(weight) {
        if (!weight) return "—";
        if (weight.kind === "curve") return `curve (${weight.points.length})`;
        return String(weight.value);
    }

    function setMessage(tabname, message, kind, vars) {
        const state = studioState(tabname);
        const id = typeof message === "string" && message.startsWith("studio.") ? message : null;
        state.messageKey = id;
        state.messageVars = vars || null;
        state.message = id ? t(id, vars) : (message || "");
        state.messageKind = kind || "";
        const root = studioRoot(tabname);
        const target = root && root.querySelector(".lora-studio-message");
        if (target) {
            target.textContent = state.message;
            target.dataset.kind = state.messageKind;
            target.hidden = !state.message;
        }
        const quick = quickTuneRoot(tabname);
        const status = quick && quick.querySelector(".lora-studio-quick-tune-status");
        if (status) {
            status.textContent = state.message;
            status.dataset.kind = state.messageKind;
        }
    }

    function getSelectedEntry(tabname) {
        const state = studioState(tabname);
        return state.entries.find((entry) => entryId(entry) === state.selectedId) || null;
    }

    function displayedSpec(tabname) {
        const state = studioState(tabname);
        const selected = getSelectedEntry(tabname);
        return state.draft ? state.draft.spec : (selected && selected.spec);
    }

    function defaultMultiplier() {
        const value = Number(opts.extra_networks_default_multiplier);
        return Number.isFinite(value) ? value : 1;
    }

    function applyInitialOpenState(tabname) {
        const root = studioRoot(tabname);
        if (!root || root.dataset.loraStudioDefaultApplied || !Object.keys(opts).length) return;
        root.open = Boolean(opts.lora_studio_open_by_default);
        root.dataset.loraStudioDefaultApplied = "true";
    }

    function draftFor(tabname, mode) {
        const state = studioState(tabname);
        const textarea = studioTextarea(tabname);
        const selected = getSelectedEntry(tabname);
        if (state.draft) return state.draft;
        if (!selected) throw new Error("Select a LoRA tag first");

        state.draft = {
            mode: mode || "replace",
            entryId: entryId(selected),
            span: {start: selected.start, end: selected.end, tag: selected.tag},
            spec: clone(selected.spec),
            originalSpec: clone(selected.spec),
            sourcePrompt: textarea.value,
        };
        return state.draft;
    }

    function createNewDraft(tabname, networkName, spec) {
        const state = studioState(tabname);
        const textarea = studioTextarea(tabname);
        state.draft = {
            mode: "add",
            entryId: null,
            span: null,
            spec: spec || studioModel().createDefaultSpec(networkName, defaultMultiplier()),
            originalSpec: null,
            sourcePrompt: textarea.value,
        };
        state.selectedId = "__new__";
    }

    function draftIsStale(tabname) {
        const state = studioState(tabname);
        const textarea = studioTextarea(tabname);
        return Boolean(state.draft && state.draft.mode !== "add" && textarea && textarea.value !== state.draft.sourcePrompt);
    }

    function updatePreview(tabname) {
        const root = studioRoot(tabname);
        const state = studioState(tabname);
        const preview = root && root.querySelector(".lora-studio-tag-preview code");
        if (preview && state.draft) {
            preview.textContent = studioModel().serializeTag(state.draft.spec);
        }
        refreshCombinationSummary(tabname);
    }

    function updateQuickTune(tabname) {
        const root = quickTuneRoot(tabname);
        const state = studioState(tabname);
        if (!root || !state.draft) return;

        const preview = root.querySelector(".lora-studio-quick-tune-preview code");
        if (preview) preview.textContent = studioModel().serializeTag(state.draft.spec);

        const stale = draftIsStale(tabname);
        const status = root.querySelector(".lora-studio-quick-tune-status");
        if (status) {
            status.textContent = stale
                ? "The prompt changed after this draft was opened. Cancel it and open the tag again."
                : "Draft only — the prompt will change only after Apply.";
            status.dataset.kind = stale ? "error" : "info";
        }
        const apply = root.querySelector(".lora-studio-quick-tune-apply");
        if (apply) apply.disabled = stale;
    }

    function syncDraftActions(tabname) {
        const root = studioRoot(tabname);
        const state = studioState(tabname);
        if (!root || !state.draft) return;
        const apply = root.querySelector(".lora-studio-apply");
        const discard = root.querySelector(".lora-studio-discard");
        if (apply) apply.disabled = draftIsStale(tabname);
        if (discard) discard.hidden = false;
        root.querySelectorAll(".lora-studio-selected-action").forEach((element) => {
            element.hidden = true;
        });
    }

    function mutateDraft(tabname, source, mutator) {
        if (source) source.setAttribute("data-lora-draft-input", "true");
        try {
            const hadDraft = Boolean(studioState(tabname).draft);
            const draft = draftFor(tabname);
            mutator(draft.spec);
            if (source) source.setCustomValidity("");
            setMessage(tabname, "studio.status.draft_changed_apply_writes_it_to", "info");
            // Keep the active input focused.  The first edit merely swaps the
            // available actions in place; rebuilding the editor here would
            // interrupt normal multi-character typing.
            if (!hadDraft) syncDraftActions(tabname);
            updatePreview(tabname);
            updateQuickTune(tabname);
            refreshQuickPresetAvailability(tabname);
            return true;
        } catch (error) {
            if (source) source.setCustomValidity(error.message);
            setMessage(tabname, error.message, "error");
            return false;
        }
    }

    function draftInputsValid(tabname) {
        const editor = quickTuneRoot(tabname) || studioRoot(tabname);
        const invalid = editor && editor.querySelector('[data-lora-draft-input="true"]:invalid');
        if (!invalid) return true;
        invalid.reportValidity();
        setMessage(tabname, "studio.status.complete_or_correct_the_highlighted_field", "error");
        return false;
    }

    function applyDraft(tabname) {
        const state = studioState(tabname);
        const textarea = studioTextarea(tabname);
        const draft = state.draft;
        if (!draft || !textarea) return false;
        if (draftIsStale(tabname)) {
            setMessage(tabname, "studio.status.the_prompt_changed_after_this_draft", "error");
            updateQuickTune(tabname);
            return false;
        }
        // Preset-library names/backups are separate from the tag draft.
        // Only invalid draft controls may block writing this tag.
        if (!draftInputsValid(tabname)) return false;

        try {
            const model = studioModel();
            let prompt = textarea.value;
            if (draft.mode === "remove") {
                prompt = model.replaceTagSpan(prompt, draft.span, "");
            } else if (draft.mode === "add") {
                const separator = prompt ? opts.extra_networks_add_text_separator : "";
                prompt += separator + model.serializeTag(draft.spec);
            } else {
                prompt = model.replaceTagSpan(prompt, draft.span, model.serializeTag(draft.spec));
            }
            textarea.value = prompt;
            state.draft = null;
            closeQuickTune(tabname);
            setMessage(tabname, "studio.status.prompt_updated", "success");
            updateInput(textarea);
            refreshStudio(tabname);
            return true;
        } catch (error) {
            setMessage(tabname, error.message, "error");
            updateQuickTune(tabname);
            return false;
        }
    }

    function discardDraft(tabname) {
        studioState(tabname).draft = null;
        closeQuickTune(tabname);
        setMessage(tabname, "studio.status.draft_discarded", "info");
        refreshStudio(tabname);
    }

    function libraryItemFor(tabname, networkName) {
        const state = studioState(tabname);
        if (!state.libraryItems || !networkName) return null;
        const name = String(networkName).toLocaleLowerCase();
        return state.libraryItems.find((item) => [item.name, item.alias]
            .filter(Boolean)
            .some((candidate) => String(candidate).toLocaleLowerCase() === name)) || null;
    }

    function insertTriggerAtCursor(tabname, trigger) {
        const state = studioState(tabname);
        const textarea = studioTextarea(tabname);
        if (!textarea) return;
        if (state.draft) {
            setMessage(tabname, "studio.status.apply_or_discard_the_lora_draft", "error");
            return;
        }

        const prompt = textarea.value;
        const start = Number.isInteger(textarea.selectionStart) ? textarea.selectionStart : prompt.length;
        const end = Number.isInteger(textarea.selectionEnd) ? textarea.selectionEnd : start;
        const overlapsTag = studioModel().findTags(prompt).some((entry) => start < entry.end && end > entry.start);
        if (overlapsTag) {
            setMessage(tabname, "studio.status.place_the_text_cursor_outside_a", "error");
            return;
        }

        const before = prompt.slice(0, start);
        const after = prompt.slice(end);
        const leading = before && !/\s$/.test(before) ? " " : "";
        const trailing = after && !/^\s/.test(after) ? " " : "";
        const inserted = `${leading}${trigger}${trailing}`;
        textarea.value = before + inserted + after;
        const cursor = before.length + inserted.length;
        textarea.focus();
        textarea.selectionStart = cursor;
        textarea.selectionEnd = cursor;
        updateInput(textarea);
        setMessage(tabname, `Inserted trigger suggestion: ${trigger}`, "success");
    }

    function appendMetadataField(body, label, value) {
        if (!value) return;
        const row = makeElement("div", "lora-studio-metadata-field");
        row.append(makeElement("span", "", label), makeElement("code", "", value));
        body.append(row);
    }

    function renderMetadataSlot(tabname, slot) {
        slot.replaceChildren();
        const metadata = studioMetadata();
        const spec = displayedSpec(tabname);
        const item = spec && libraryItemFor(tabname, spec.network_name);
        if (!metadata || !item) return;

        const info = metadata.describeLora(item);
        if (!info.title && !info.description && !info.baseModel && !info.hash && !info.triggers.length) return;
        const details = makeElement("details", "lora-studio-metadata");
        details.append(makeElement("summary", "", t("studio.ui.lora_information_and_trigger_suggestions")));
        const body = makeElement("div", "lora-studio-metadata-body");
        if (info.title) body.append(makeElement("h4", "", info.title));
        if (info.description) body.append(makeElement("p", "lora-studio-description", info.description));
        appendMetadataField(body, t("studio.ui.base_model"), info.baseModel);
        appendMetadataField(body, t("studio.ui.declared_hash_file_metadata_unverified"), info.hash);
        if (info.triggers.length) {
            const triggers = makeElement("div", "lora-studio-triggers");
            triggers.append(makeElement("span", "", t("studio.ui.suggestions_click_to_insert_at_the")));
            const chips = makeElement("div", "lora-studio-trigger-chips");
            info.triggers.forEach((trigger) => {
                chips.append(button(trigger, "lora-studio-trigger", () => insertTriggerAtCursor(tabname, trigger)));
            });
            triggers.append(chips);
            body.append(triggers);
        } else {
            body.append(makeElement("p", "lora-studio-preserved", t("studio.status.no_trigger_suggestions_were_found_in")));
        }
        details.append(body);
        slot.append(details);
    }

    function refreshMetadataSlot(tabname) {
        const root = studioRoot(tabname);
        const slot = root && root.querySelector(".lora-studio-metadata-slot");
        if (slot) renderMetadataSlot(tabname, slot);
    }

    function renderMetadata(tabname, body) {
        const slot = makeElement("div", "lora-studio-metadata-slot");
        body.append(slot);
        renderMetadataSlot(tabname, slot);
    }

    function sidecarNetworkItem(tabname, spec) {
        return spec && libraryItemFor(tabname, spec.network_name);
    }

    function refreshSidecarSlot(tabname, spec) {
        const root = studioRoot(tabname);
        const slot = root && root.querySelector(".lora-studio-sidecar-slot");
        if (slot) renderSidecarSlot(tabname, slot, spec);
    }

    function loadSidecar(tabname, spec) {
        const item = sidecarNetworkItem(tabname, spec);
        if (!item) {
            setMessage(tabname, "studio.status.sidecar_inspection_is_available_after_this", "error");
            return;
        }
        const state = studioState(tabname);
        if (state.sidecarLoading) return;
        state.sidecarLoading = true;
        state.sidecarError = "";
        state.sidecarErrorForNetwork = "";
        refreshSidecarSlot(tabname, spec);
        const url = new URL("sdapi/v1/lora-studio/sidecar", window.location.href);
        url.searchParams.set("name", item.name);
        fetch(url)
            .then((response) => response.ok ? response.json() : response.json().catch(() => ({})).then((payload) => Promise.reject(new Error(payload.detail || `Sidecar inspection failed (${response.status})`))))
            .then((payload) => {
                if (!payload || payload.status === "unavailable") throw new Error((payload && payload.reason) || "Sidecar inspection is unavailable.");
                state.sidecar = payload;
                state.sidecarForNetwork = item.name;
            })
            .catch((error) => {
                state.sidecar = null;
                state.sidecarForNetwork = "";
                state.sidecarError = error.message;
                state.sidecarErrorForNetwork = item.name;
            })
            .finally(() => {
                state.sidecarLoading = false;
                refreshSidecarSlot(tabname, displayedSpec(tabname));
            });
    }

    function renderSidecarSlot(tabname, slot, spec) {
        slot.replaceChildren();
        const state = studioState(tabname);
        const item = sidecarNetworkItem(tabname, spec);
        if (!item) {
            slot.append(makeElement("p", "lora-studio-preserved", t("studio.status.sidecar_inspection_is_available_after_this")));
            return;
        }
        if (state.sidecarLoading) {
            slot.append(makeElement("p", "lora-studio-preserved", t("studio.status.inspecting_trusted_sibling_sidecar_files")));
            return;
        }
        if (state.sidecarErrorForNetwork === item.name && state.sidecarError) {
            slot.append(makeElement("p", "lora-studio-warning", state.sidecarError));
        }
        const report = state.sidecarForNetwork === item.name ? state.sidecar : null;
        const inspect = button(report ? t("studio.ui.inspect_sidecars_again") : t("studio.ui.inspect_local_sidecars"), "", () => loadSidecar(tabname, spec));
        inspect.title = t("studio.help.reads_only_bounded_txt_description_txt");
        slot.append(inspect);
        if (!report) return;
        if (report.description) {
            slot.append(makeElement("p", "lora-studio-description", report.description));
            slot.append(makeElement("p", "lora-studio-preserved", t("studio.status.description_source_source", {source: report.description_source || t("studio.ui.unknown")})));
        } else {
            slot.append(makeElement("p", "lora-studio-preserved", t("studio.status.no_sidecar_description_was_found")));
        }
        if (typeof report.preferred_weight === "number" && Number.isFinite(report.preferred_weight)) {
            appendMetadataField(slot, t("studio.ui.sidecar_preferred_weight"), studioModel().formatNumber(report.preferred_weight));
        }
        if (report.activation_text) {
            const activation = makeElement("div", "lora-studio-triggers");
            activation.append(makeElement("span", "", t("studio.ui.sidecar_activation_text_explicit_insertion_only")));
            activation.append(button(t("studio.ui.insert_activation_text_at_cursor"), "lora-studio-trigger", () => insertTriggerAtCursor(tabname, report.activation_text)));
            slot.append(activation);
        }
        if (Array.isArray(report.warnings)) {
            report.warnings.forEach((warning) => slot.append(makeElement("p", "lora-studio-warning", String(warning))));
        }
    }

    function renderSidecarControls(tabname, body, spec) {
        const state = studioState(tabname);
        const details = makeElement("details", "lora-studio-metadata lora-studio-sidecar");
        details.open = state.sidecarOpen;
        details.addEventListener("toggle", () => { state.sidecarOpen = details.open; });
        details.append(makeElement("summary", "", t("studio.ui.local_sidecar_information_read_only")));
        details.querySelector("summary").append(helpIcon("studio.sidecars"));
        const slot = makeElement("div", "lora-studio-sidecar-slot");
        details.append(slot);
        body.append(details);
        renderSidecarSlot(tabname, slot, spec);
    }

    function updatePresetLibrary(payload) {
        const presets = studioPresets();
        if (!presets || !payload || payload.schema_version !== presets.SCHEMA_VERSION || !Array.isArray(payload.presets)) {
            throw new Error("Preset library response has an unsupported format");
        }
        presetLibrary.items = payload.presets.map((preset) => presets.validatePreset(preset));
        pruneRecentPresetIds(presetLibrary.items);
        presetLibrary.error = payload.error || "";
    }

    function loadRecentPresetIds() {
        try {
            const value = JSON.parse(window.localStorage.getItem(RECENT_PRESET_STORAGE_KEY) || "[]");
            if (!Array.isArray(value)) return [];
            const result = [];
            const seen = new Set();
            value.forEach((presetId) => {
                if (typeof presetId !== "string" || !presetId.trim() || seen.has(presetId)) return;
                seen.add(presetId);
                result.push(presetId);
            });
            return result.slice(0, RECENT_PRESET_LIMIT);
        } catch (_) {
            return [];
        }
    }

    function saveRecentPresetIds(presetIds) {
        try {
            window.localStorage.setItem(RECENT_PRESET_STORAGE_KEY, JSON.stringify(presetIds.slice(0, RECENT_PRESET_LIMIT)));
        } catch (_) {
            // Private mode and storage quotas must not block preset use.
        }
    }

    function pruneRecentPresetIds(items) {
        const available = new Set((items || []).map((item) => item.preset_id));
        const current = loadRecentPresetIds();
        const pruned = current.filter((presetId) => available.has(presetId));
        if (pruned.length !== current.length) saveRecentPresetIds(pruned);
        return pruned;
    }

    function recordRecentPreset(presetId) {
        const recent = loadRecentPresetIds().filter((item) => item !== presetId);
        recent.unshift(presetId);
        saveRecentPresetIds(recent);
    }

    function appendPresetOptions(select, label, items) {
        if (!items.length) return;
        const group = document.createElement("optgroup");
        group.label = label;
        items.forEach((preset) => {
            const option = makeElement("option", "", preset.favorite ? `★ ${preset.title}` : preset.title);
            option.value = preset.preset_id;
            group.append(option);
        });
        select.append(group);
    }

    function refreshPresetSlots() {
        studioTabs.forEach((tabname) => {
            const root = studioRoot(tabname);
            const slot = root && root.querySelector(".lora-studio-preset-slot");
            if (slot) renderPresetSlot(tabname, slot);
            [root, quickTuneRoot(tabname)].filter(Boolean).forEach((surface) => {
                const recipe = surface.querySelector(".lora-studio-recipe-slot");
                if (recipe) renderQuickPresetSlot(tabname, recipe);
            });
        });
    }

    function quickPresetCatalog() {
        return studioPresets().listQuickPresets({
            blockPresets: opts.lora_block_weight_presets,
            elementalPresets: opts.lora_elemental_weight_presets,
            savedPresets: presetLibrary.items || [],
        });
    }

    function refreshQuickPresetAvailability(tabname) {
        [studioRoot(tabname), quickTuneRoot(tabname)].filter(Boolean).forEach((surface) => {
            const slot = surface.querySelector(".lora-studio-recipe-slot");
            if (slot && slot.updateAvailability) slot.updateAvailability();
        });
    }

    // Translate built-in recipes at rendering; never translate a saved user's preset title.
    function presetDisplayTitle(choice) {
        return choice.kind === "mode" ? t(choice.title) : choice.title;
    }

    function presetDisplayDescription(choice) {
        return (choice.kind === "mode" || (choice.kind === "elemental" && choice.classicOnly))
            ? t(choice.description) : choice.description;
    }

    function renderQuickPresetSlot(tabname, slot) {
        slot.replaceChildren();
        const presets = studioPresets();
        if (!presets || !displayedSpec(tabname)) return;
        const state = studioState(tabname);
        const catalog = quickPresetCatalog();
        const row = makeElement("div", "lora-studio-recipe-row");
        const label = makeElement("label", "lora-studio-recipe-label");
        label.append(makeElement("span", "", t("studio.ui.preset")));
        label.append(helpIcon("studio.presets"));
        const select = document.createElement("select");
        const placeholder = makeElement("option", "", t("studio.status.choose_a_mode_or_saved_preset"));
        placeholder.value = "";
        select.append(placeholder);
        const groups = new Map();
        const options = [];
        catalog.forEach((choice) => {
            if (!groups.has(choice.group)) {
                const group = document.createElement("optgroup");
                group.label = t(choice.group);
                select.append(group);
                groups.set(choice.group, group);
            }
            const option = makeElement("option", "", presetDisplayTitle(choice));
            option.value = choice.id;
            groups.get(choice.group).append(option);
            options.push({choice, option});
        });
        select.value = state.quickPresetId;
        const description = makeElement("p", "lora-studio-recipe-description");
        description.setAttribute("aria-live", "polite");
        const current = makeElement("p", "lora-studio-recipe-current");
        const use = button(t("studio.ui.use_preset"), "lora-studio-recipe-use", () => {
            if (!draftInputsValid(tabname)) return;
            const choice = catalog.find((item) => item.id === select.value);
            if (!choice) return;
            // Compute first: a rejected recipe must not leave a partial draft.
            try {
                const next = presets.applyQuickPreset(displayedSpec(tabname), choice);
                if (!mutateDraft(tabname, use, (spec) => Object.assign(spec, next))) return;
                if (choice.kind === "saved") recordRecentPreset(choice.preset.preset_id);
                if (quickTuneRoot(tabname)) renderQuickTune(tabname);
                renderStudio(tabname);
                setMessage(tabname, "studio.status.title_loaded_into_the_draft_apply", "info", {title: presetDisplayTitle(choice)});
            } catch (error) {
                setMessage(tabname, error.message, "error");
            }
        });
        slot.updateAvailability = () => {
            const spec = displayedSpec(tabname);
            if (!spec) return;
            options.forEach(({choice, option}) => {
                const problem = presets.quickPresetProblem(spec, choice);
                option.disabled = Boolean(problem);
                option.title = problem || presetDisplayDescription(choice);
            });
            const selected = catalog.find((choice) => choice.id === select.value);
            const problem = presets.quickPresetProblem(spec, selected);
            use.disabled = Boolean(problem) || draftIsStale(tabname);
            description.textContent = selected ? `${presetDisplayDescription(selected)}${problem ? ` ${problem}` : ""}` : t("studio.status.choose_a_preset_to_see_what");
            const blockRaw = spec.block_weights ? spec.block_weights.raw : "all";
            const blockLabel = blockRaw.includes(",") ? `${blockRaw.split(",").length}-value table` : blockRaw;
            current.textContent = t("studio.status.current_masks_blocks_blocks_modules_modules", {blocks: blockLabel, modules: spec.elemental_weights ? spec.elemental_weights.raw : t("studio.ui.no_override"), layout: spec.model_profile || "auto", verification: !spec.model_profile || spec.model_profile === "auto" ? t("studio.ui.not_verified") : ""});
        };
        select.addEventListener("change", () => {
            state.quickPresetId = select.value;
            slot.updateAvailability();
        });
        label.append(select);
        row.append(label, use);
        slot.append(row, description, current);

        const save = makeElement("details", "lora-studio-quick-save");
        save.open = state.quickSaveOpen;
        save.addEventListener("toggle", () => { state.quickSaveOpen = save.open; });
        save.append(makeElement("summary", "", t("studio.ui.save_these_settings_as_a_preset")));
        const saveRow = makeElement("div", "lora-studio-recipe-row");
        const title = document.createElement("input");
        title.type = "text";
        title.maxLength = 120;
        title.placeholder = t("studio.ui.preset_name");
        title.setAttribute("aria-label", t("studio.ui.save_preset_name"));
        title.value = state.presetTitle;
        title.addEventListener("input", () => { state.presetTitle = title.value; title.setCustomValidity(""); });
        const saveButton = button(t("studio.ui.save_preset"), "", () => {
            if (!draftInputsValid(tabname)) return;
            try {
                const preset = presets.createPreset(title.value, displayedSpec(tabname));
                state.selectedPresetId = preset.preset_id;
                state.quickPresetId = `saved:${preset.preset_id}`;
                state.presetTitle = "";
                savePresetLibrary(tabname, [...(presetLibrary.items || []), preset]);
            } catch (error) {
                title.setCustomValidity(error.message);
                title.reportValidity();
            }
        });
        saveButton.disabled = presetLibrary.loading || presetLibrary.items === null || Boolean(presetLibrary.error);
        saveRow.append(title, saveButton);
        save.append(saveRow, makeElement("p", "lora-studio-preserved", t("studio.status.saves_a_reusable_snapshot_does_not")));
        slot.append(save);
        const manage = button(t("studio.status.manage_saved_presets"), "lora-studio-recipe-manage", () => {
            state.presetLibraryOpen = true;
            if (quickTuneRoot(tabname)) openFullStudioFromQuickTune(tabname);
            else renderStudio(tabname);
            const library = studioRoot(tabname).querySelector(".lora-studio-presets");
            if (library) library.scrollIntoView({block: "nearest"});
        });
        slot.append(manage);
        if (presetLibrary.loading) slot.append(makeElement("p", "lora-studio-preserved", t("studio.status.loading_saved_presets_built_in_modes")));
        if (presetLibrary.error) slot.append(makeElement("p", "lora-studio-warning", t("studio.status.saved_presets_unavailable_error_use_manage", {error: presetLibrary.error})));
        slot.updateAvailability();
    }

    function renderQuickPresetControls(tabname, body) {
        const slot = makeElement("section", "lora-studio-recipe-slot");
        body.append(slot);
        loadPresetLibrary(false);
        renderQuickPresetSlot(tabname, slot);
    }

    function loadPresetLibrary(force) {
        if (presetLibrary.loading || (!force && presetLibrary.items !== null)) return;
        presetLibrary.loading = true;
        presetLibrary.error = "";
        fetch(new URL("sdapi/v1/lora-studio/presets", window.location.href))
            .then((response) => response.ok ? response.json() : Promise.reject(new Error(`Preset library request failed (${response.status})`)))
            .then(updatePresetLibrary)
            .catch((error) => {
                presetLibrary.items = [];
                presetLibrary.error = error.message;
            })
            .finally(() => {
                presetLibrary.loading = false;
                refreshPresetSlots();
            });
    }

    function savePresetLibrary(tabname, items) {
        const presets = studioPresets();
        if (!presets || presetLibrary.loading) return;
        let payload;
        try {
            // Validate before showing a loading state. This keeps a damaged
            // local preset from making both Studio panels look permanently
            // busy; the server repeats validation before persisting.
            payload = presets.envelope(items);
        } catch (error) {
            presetLibrary.error = error.message;
            setMessage(tabname, t("studio.ui.preset_library_was_not_saved_error", {error: error.message}), "error");
            refreshPresetSlots();
            return;
        }
        presetLibrary.loading = true;
        presetLibrary.error = "";
        refreshPresetSlots();
        fetch(new URL("sdapi/v1/lora-studio/presets", window.location.href), {
            method: "PUT",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload),
        })
            .then(async (response) => {
                const payload = await response.json().catch(() => ({}));
                if (!response.ok) throw new Error(payload.detail || `Preset library save failed (${response.status})`);
                return payload;
            })
            .then((payload) => {
                updatePresetLibrary(payload);
                studioState(tabname).presetTransferText = null;
                setMessage(tabname, "studio.status.studio_preset_library_saved", "success");
            })
            .catch((error) => {
                presetLibrary.error = error.message;
                setMessage(tabname, t("studio.ui.preset_library_was_not_saved_error", {error: error.message}), "error");
            })
            .finally(() => {
                presetLibrary.loading = false;
                refreshPresetSlots();
            });
    }

    function selectedPreset(tabname) {
        const state = studioState(tabname);
        return (presetLibrary.items || []).find((preset) => preset.preset_id === state.selectedPresetId) || null;
    }

    function renderPresetTransfer(tabname, slot, presets) {
        const state = studioState(tabname);
        const details = makeElement("details", "lora-studio-preset-transfer");
        details.open = state.presetTransferOpen;
        details.addEventListener("toggle", () => { state.presetTransferOpen = details.open; });
        details.append(makeElement("summary", "", t("studio.ui.backup_or_restore_preset_library")));
        const body = makeElement("div", "lora-studio-preset-transfer-body");
        body.append(makeElement("p", "lora-studio-preserved", t("studio.help.copy_this_versioned_json_for_a")));
        const text = document.createElement("textarea");
        text.spellcheck = false;
        text.setAttribute("aria-label", t("studio.ui.lora_studio_preset_json_backup"));
        text.value = state.presetTransferText === null ? JSON.stringify(presets.envelope(presetLibrary.items || []), null, 2) : state.presetTransferText;
        text.rows = 8;
        text.placeholder = t("studio.ui.paste_a_studio_preset_backup_here");
        text.addEventListener("input", () => {
            state.presetTransferText = text.value;
            text.setCustomValidity("");
        });
        body.append(text);
        const controls = makeElement("div", "lora-studio-preset-transfer-actions");
        controls.append(button(t("studio.ui.restore_this_json"), "", () => {
            try {
                const imported = presets.parseEnvelope(text.value);
                if (!window.confirm(`Replace the saved Studio preset library with ${imported.presets.length} preset(s)?`)) return;
                state.selectedPresetId = "";
                state.presetTransferText = text.value;
                savePresetLibrary(tabname, imported.presets);
            } catch (error) {
                text.setCustomValidity(error.message);
                text.reportValidity();
                setMessage(tabname, t("studio.ui.preset_backup_is_invalid_error", {error: error.message}), "error");
            }
        }));
        controls.append(button(t("studio.ui.reset_text_to_saved_library"), "", () => {
            state.presetTransferText = null;
            renderPresetSlot(tabname, slot);
        }));
        body.append(controls);
        details.append(body);
        slot.append(details);
    }

    function renderPresetSlot(tabname, slot) {
        slot.replaceChildren();
        const presets = studioPresets();
        if (!presets) {
            slot.append(makeElement("p", "lora-studio-warning", t("studio.status.preset_controls_did_not_load")));
            return;
        }
        if (presetLibrary.loading) {
            slot.append(makeElement("p", "lora-studio-preserved", t("studio.status.loading_saved_studio_presets")));
            return;
        }
        if (presetLibrary.error) slot.append(makeElement("p", "lora-studio-warning", presetLibrary.error));

        const state = studioState(tabname);
        const controls = makeElement("div", "lora-studio-preset-controls");
        const select = document.createElement("select");
        const placeholder = makeElement("option", "", t("studio.ui.choose_a_saved_studio_preset"));
        placeholder.value = "";
        select.append(placeholder);
        const allItems = presetLibrary.items || [];
        const favorites = allItems.filter((preset) => preset.favorite);
        const favoritesById = new Set(favorites.map((preset) => preset.preset_id));
        const byId = new Map(allItems.map((preset) => [preset.preset_id, preset]));
        const recent = pruneRecentPresetIds(allItems)
            .map((presetId) => byId.get(presetId))
            .filter((preset) => preset && !favoritesById.has(preset.preset_id));
        const recentById = new Set(recent.map((preset) => preset.preset_id));
        appendPresetOptions(select, t("studio.ui.favourites"), favorites);
        appendPresetOptions(select, t("studio.ui.recent"), recent);
        appendPresetOptions(select, t("studio.ui.all_saved_presets"), allItems.filter((preset) => !favoritesById.has(preset.preset_id) && !recentById.has(preset.preset_id)));
        select.value = state.selectedPresetId;
        select.addEventListener("change", () => {
            state.selectedPresetId = select.value;
            refreshPresetSlots();
        });
        controls.append(select);

        const apply = button(t("studio.ui.apply_preset_to_draft"), "", () => {
            const preset = selectedPreset(tabname);
            if (!preset) {
                setMessage(tabname, "studio.status.choose_a_studio_preset_first", "error");
                return;
            }
            if (mutateDraft(tabname, apply, (draftSpec) => Object.assign(draftSpec, presets.applyPreset(draftSpec, preset)))) {
                recordRecentPreset(preset.preset_id);
                setMessage(tabname, `Preset applied locally: ${preset.title}. Apply writes it to the prompt.`, "info");
                renderStudio(tabname);
            }
        });
        apply.disabled = !selectedPreset(tabname);
        controls.append(apply);

        const remove = button(t("studio.ui.delete_preset"), "", () => {
            const preset = selectedPreset(tabname);
            if (!preset) return;
            if (!window.confirm(`Delete Studio preset “${preset.title}”?`)) return;
            state.selectedPresetId = "";
            savePresetLibrary(tabname, (presetLibrary.items || []).filter((item) => item.preset_id !== preset.preset_id));
        });
        remove.disabled = !selectedPreset(tabname);
        controls.append(remove);

        const toggleFavorite = button(selectedPreset(tabname)?.favorite ? t("studio.ui.remove_from_favourites") : t("studio.ui.add_to_favourites"), "", () => {
            const preset = selectedPreset(tabname);
            if (!preset) return;
            const updated = (presetLibrary.items || []).map((item) => item.preset_id === preset.preset_id
                ? {...item, favorite: !item.favorite}
                : item);
            savePresetLibrary(tabname, updated);
        });
        toggleFavorite.disabled = !selectedPreset(tabname);
        controls.append(toggleFavorite);
        slot.append(controls);

        const save = makeElement("div", "lora-studio-preset-save");
        const title = document.createElement("input");
        title.type = "text";
        title.maxLength = 120;
        title.placeholder = t("studio.ui.new_preset_title");
        title.value = state.presetTitle;
        title.addEventListener("input", () => {
            state.presetTitle = title.value;
            title.setCustomValidity("");
        });
        const saveButton = button(t("studio.ui.save_current_controls"), "", () => {
            if (!draftInputsValid(tabname)) return;
            const spec = displayedSpec(tabname);
            if (!spec) {
                setMessage(tabname, "studio.status.select_or_add_a_lora_before", "error");
                return;
            }
            try {
                const preset = presets.createPreset(title.value, spec);
                state.presetTitle = "";
                state.selectedPresetId = preset.preset_id;
                savePresetLibrary(tabname, [...(presetLibrary.items || []), preset]);
            } catch (error) {
                title.setCustomValidity(error.message);
                title.reportValidity();
            }
        });
        save.append(title, saveButton);
        slot.append(save);

        renderPresetTransfer(tabname, slot, presets);

        const reload = button(t("studio.ui.reload_preset_library"), "", () => loadPresetLibrary(true));
        reload.classList.add("lora-studio-preset-reload");
        slot.append(reload);
    }

    function renderPresetControls(tabname, body) {
        const details = makeElement("details", "lora-studio-presets");
        details.open = studioState(tabname).presetLibraryOpen;
        details.addEventListener("toggle", () => { studioState(tabname).presetLibraryOpen = details.open; });
        details.append(makeElement("summary", "", t("studio.ui.manage_saved_presets_import_export")));
        details.querySelector("summary").append(helpIcon("studio.presets"));
        const slot = makeElement("div", "lora-studio-preset-slot");
        details.append(slot);
        body.append(details);
        renderPresetSlot(tabname, slot);
        loadPresetLibrary(false);
    }

    function svgElement(name, attributes, text) {
        const element = document.createElementNS("http://www.w3.org/2000/svg", name);
        Object.entries(attributes || {}).forEach(([key, value]) => element.setAttribute(key, String(value)));
        if (text !== undefined) element.textContent = text;
        return element;
    }

    function renderCurvePlot(label, weight, curves) {
        const snapshot = curves.chart(weight);
        const width = 520;
        const height = 190;
        const padding = {left: 44, right: 14, top: 14, bottom: 30};
        const plotWidth = width - padding.left - padding.right;
        const plotHeight = height - padding.top - padding.bottom;
        const xAt = (position) => padding.left + ((position - snapshot.x.min) / (snapshot.x.max - snapshot.x.min)) * plotWidth;
        const yAt = (value) => padding.top + (1 - ((value - snapshot.y.min) / (snapshot.y.max - snapshot.y.min))) * plotHeight;
        const svg = svgElement("svg", {
            class: "lora-studio-curve-plot",
            viewBox: `0 0 ${width} ${height}`,
            role: "img",
            "aria-label": `${label} curve preview`,
        });
        svg.append(svgElement("title", {}, `${label}: position ${studioModel().formatNumber(snapshot.x.min)} to ${studioModel().formatNumber(snapshot.x.max)}, value ${studioModel().formatNumber(snapshot.y.min)} to ${studioModel().formatNumber(snapshot.y.max)}`));
        svg.append(svgElement("path", {
            class: "lora-studio-curve-axis",
            d: `M ${padding.left} ${padding.top} V ${height - padding.bottom} H ${width - padding.right}`,
        }));
        if (snapshot.y.min <= 0 && snapshot.y.max >= 0) {
            svg.append(svgElement("path", {
                class: "lora-studio-curve-zero",
                d: `M ${padding.left} ${yAt(0)} H ${width - padding.right}`,
            }));
        }
        const path = snapshot.samples.map((sample, index) => `${index ? "L" : "M"} ${xAt(sample.position).toFixed(2)} ${yAt(sample.value).toFixed(2)}`).join(" ");
        svg.append(svgElement("path", {class: "lora-studio-curve-line", d: path}));
        snapshot.points.forEach((point) => {
            svg.append(svgElement("circle", {
                class: "lora-studio-curve-point",
                cx: xAt(point.position),
                cy: yAt(point.value),
                r: 3.5,
            }));
        });
        svg.append(svgElement("text", {class: "lora-studio-curve-label", x: padding.left, y: height - 8}, studioModel().formatNumber(snapshot.x.min)));
        svg.append(svgElement("text", {class: "lora-studio-curve-label", x: width - padding.right, y: height - 8, "text-anchor": "end"}, studioModel().formatNumber(snapshot.x.max)));
        svg.append(svgElement("text", {class: "lora-studio-curve-label", x: padding.left - 5, y: padding.top + 9, "text-anchor": "end"}, studioModel().formatNumber(snapshot.y.max)));
        svg.append(svgElement("text", {class: "lora-studio-curve-label", x: padding.left - 5, y: height - padding.bottom, "text-anchor": "end"}, studioModel().formatNumber(snapshot.y.min)));
        return svg;
    }

    function curveWeightFromPoints(points) {
        if (!points.length) throw new Error("Curve needs at least one point");
        const raw = points.map((point) => `${studioModel().formatNumber(point.value)}@${studioModel().formatNumber(point.position)}`).join(",");
        return studioModel().parseWeight(raw);
    }

    function updateCurveWeight(tabname, source, key, points) {
        return mutateDraft(tabname, source, (draftSpec) => {
            draftSpec.weights[key] = curveWeightFromPoints(points);
        });
    }

    function renderCurveWeightEditor(tabname, body, label, key, weight, curves) {
        const panel = makeElement("section", "lora-studio-curve-editor");
        const heading = key === "te" ? t("studio.ui.te_at_text_encoding_first_point") : key === "unet" ? t("studio.ui.u_net_per_sampling_step") : label;
        panel.append(makeElement("h5", "", heading));
        try {
            panel.append(renderCurvePlot(label, weight, curves));
            const points = curves.pointsForWeight(weight);
            if (weight.kind !== "curve") {
                panel.append(makeElement("p", "lora-studio-preserved", t("studio.help.this_weight_is_scalar_convert_it")));
                panel.append(button(t("studio.ui.convert_to_flat_curve"), "", () => {
                    if (!draftInputsValid(tabname)) return;
                    if (updateCurveWeight(tabname, null, key, points)) renderStudio(tabname);
                }));
                body.append(panel);
                return;
            }

            const rows = makeElement("div", "lora-studio-curve-points");
            const rowLabels = makeElement("div", "lora-studio-curve-point-row");
            rowLabels.append(makeElement("span", "", "#"), makeElement("span", "", t("studio.ui.position")), makeElement("span", "", t("studio.ui.weight")), makeElement("span", "", ""));
            rows.append(rowLabels);
            const inputs = [];
            const applyRows = (source) => {
                try {
                    if (inputs.some(({position, value}) => [position, value].some((input) => !input.value.trim() || !Number.isFinite(Number(input.value))))) return;
                    const next = inputs.map(({position, value}) => ({
                        position: Number(position.value.trim()),
                        value: Number(value.value.trim()),
                    }));
                    if (!next.every((point) => Number.isFinite(point.position) && Number.isFinite(point.value))) {
                        throw new Error("Curve positions and values must be finite numbers");
                    }
                    if (updateCurveWeight(tabname, source, key, next)) {
                        points.splice(0, points.length, ...next);
                        syncWeightInputs(tabname, key, source, false);
                        const plot = panel.querySelector(".lora-studio-curve-plot");
                        if (plot) plot.replaceWith(renderCurvePlot(label, displayedSpec(tabname).weights[key], curves));
                        updateTimelineControls();
                    }
                } catch (error) {
                    source.setCustomValidity(error.message);
                    source.reportValidity();
                    setMessage(tabname, error.message, "error");
                }
            };
            points.forEach((point, index) => {
                const row = makeElement("div", "lora-studio-curve-point-row");
                row.append(makeElement("span", "lora-studio-curve-point-index", `#${index + 1}`));
                const position = document.createElement("input");
                position.type = "text";
                position.inputMode = "decimal";
                position.value = studioModel().formatNumber(point.position);
                position.setAttribute("aria-label", `${label} point ${index + 1} position`);
                const value = document.createElement("input");
                value.type = "text";
                value.inputMode = "decimal";
                value.value = studioModel().formatNumber(point.value);
                value.setAttribute("aria-label", `${label} point ${index + 1} value`);
                [position, value].forEach((input) => {
                    input.required = true;
                    input.setAttribute("data-lora-draft-input", "true");
                    input.addEventListener("input", () => {
                        const text = input.value.trim();
                        input.setCustomValidity(text && Number.isFinite(Number(text)) ? "" : "Enter a finite curve value; blank is not zero");
                        if (!input.validationMessage) applyRows(input);
                    });
                });
                inputs.push({position, value});
                row.append(position, value);
                const remove = button(t("studio.ui.remove"), "", () => {
                    if (!draftInputsValid(tabname)) return;
                    if (points.length <= 1) return;
                    const next = points.filter((_, pointIndex) => pointIndex !== index);
                    if (updateCurveWeight(tabname, remove, key, next)) renderStudio(tabname);
                });
                remove.disabled = points.length <= 1;
                row.append(remove);
                rows.append(row);
            });
            panel.append(rows);
            const add = button(t("studio.ui.add_midpoint"), "", () => {
                if (!draftInputsValid(tabname)) return;
                try {
                    const next = curves.insertMidpoint(points);
                    if (updateCurveWeight(tabname, add, key, next)) renderStudio(tabname);
                } catch (error) {
                    add.setCustomValidity(error.message);
                    add.reportValidity();
                    setMessage(tabname, error.message, "error");
                }
            });
            const updateTimelineControls = () => {
                add.disabled = false;
                add.title = curves.isNormalizedTimeline(points) ? t("studio.status.insert_an_interpolated_point_in_the") : t("studio.status.insert_an_interpolated_point_between_absolute");
            };
            updateTimelineControls();
            panel.append(add);
        } catch (error) {
            panel.append(makeElement("p", "lora-studio-warning", `Curve preview is unavailable: ${error.message}`));
        }
        body.append(panel);
    }

    function renderCurveControls(tabname, body) {
        const curves = studioCurves();
        const spec = displayedSpec(tabname);
        if (!spec) return;
        if (!curves) {
            body.append(makeElement("p", "lora-studio-warning", t("studio.status.curve_editor_did_not_load_the")));
            return;
        }
        const state = studioState(tabname);
        const details = makeElement("details", "lora-studio-curves");
        details.open = state.curveOpen;
        details.addEventListener("toggle", () => { state.curveOpen = details.open; });
        details.append(makeElement("summary", "", t("studio.ui.curve_preview_and_point_editor")));
        details.querySelector("summary").append(helpIcon("studio.curves"));
        const content = makeElement("div", "lora-studio-curves-body");
        content.append(makeElement("p", "lora-studio-preserved", t("studio.help.curves_are_piecewise_linear_if_every")));
        renderCurveWeightEditor(tabname, content, "Text encoder", "te", spec.weights.te, curves);
        renderCurveWeightEditor(tabname, content, "U-Net", "unet", spec.weights.unet, curves);
        details.append(content);
        body.append(details);
    }

    function refreshCurveSlot(tabname) {
        const root = studioRoot(tabname);
        const slot = root && root.querySelector(".lora-studio-curve-slot");
        if (!slot) return;
        // Editing TE must not erase an unfinished U-Net curve point (or
        // vice versa) while refreshing the plot.
        if (slot.querySelector('[data-lora-draft-input="true"]:invalid')) return;
        slot.replaceChildren();
        renderCurveControls(tabname, slot);
    }

    function refreshCombinationSummary(tabname) {
        const spec = displayedSpec(tabname);
        if (!spec) return;
        [studioRoot(tabname), quickTuneRoot(tabname)].filter(Boolean).forEach((root) => {
            const summary = root.querySelector(".lora-studio-combination");
            if (!summary) return;
            const model = studioModel();
            const hires = ["te", "unet"].map((key) => spec.hires[key] ? model.serializeWeight(spec.hires[key]) : "inherit");
            summary.textContent = `TE ${model.serializeWeight(spec.weights.te)} · U-Net ${model.serializeWeight(spec.weights.unet)} · DYN ${spec.dyn_dim === null ? t("studio.ui.off_full_rank") : spec.dyn_dim}. ` +
                `Hires TE / U-Net: ${hires.join(" / ")}. ` +
                "TE and U-Net set strength independently; DYN limits rank in supported LoRA layers, not strength or sampling steps.";
            if (spec.passthrough_named.some((item) => ["te", "unet", "dyn"].includes(item.name.toLowerCase()))) {
                summary.textContent += " Parameter names are case-sensitive: use te=, unet=, dyn=. Uppercase names are preserved but do not set native weights.";
            }
        });
    }

    function addCombinationSummary(body) {
        body.append(makeElement("p", "lora-studio-combination lora-studio-preserved"));
    }

    function syncWeightInputs(tabname, key, source, refreshCurve = true) {
        const weight = displayedSpec(tabname).weights[key];
        [studioRoot(tabname), quickTuneRoot(tabname)].filter(Boolean).forEach((root) => {
            const control = root.querySelector(`[data-lora-weight-control="${key}"]`);
            if (control && (control.dataset.weightKind !== weight.kind || weight.kind === "curve")) {
                // Replace only this quick control, never the exact field being
                // typed into. A blur must not remove the next clicked button.
                const holder = document.createElement("div");
                addWeightControl(tabname, holder, key === "te" ? "Text encoder" : "U-Net", key);
                control.replaceWith(holder.firstElementChild);
            }
            root.querySelectorAll(`[data-lora-weight="${key}"]`).forEach((input) => {
                if (input === source) return;
                if (input.type === "text") {
                    input.value = studioModel().serializeWeight(weight);
                    input.setCustomValidity("");
                } else {
                    input.disabled = weight.kind === "curve";
                    if (weight.kind === "curve") return;
                    input.setCustomValidity("");
                    if (input.type === "range") {
                        input.min = String(Math.min(-2, weight.value));
                        input.max = String(Math.max(2, weight.value));
                    }
                    input.value = studioModel().formatNumber(weight.value);
                }
            });
        });
        if (refreshCurve) refreshCurveSlot(tabname);
    }

    function addWeightControl(tabname, body, label, key) {
        const state = studioState(tabname);
        const selected = getSelectedEntry(tabname);
        const spec = state.draft ? state.draft.spec : (selected && selected.spec);
        if (!spec) return;

        const control = makeElement("div", "lora-studio-weight-control");
        control.setAttribute("data-lora-weight-control", key);
        control.dataset.weightKind = spec.weights[key].kind;
        const heading = makeElement("div", "lora-studio-control-heading");
        heading.append(makeElement("span", "", label));
        control.append(heading);

        const weight = spec.weights[key];
        if (weight.kind === "curve") {
            const curve = makeElement("div", "lora-studio-weight-curve");
            curve.append(makeElement("span", "lora-studio-weight-curve-status", t("studio.ui.curve_count_points", {count: weight.points.length})));
            curve.append(makeElement("code", "", studioModel().serializeWeight(weight)));
            const edit = button(t("studio.ui.edit_curve"), "lora-studio-edit-curve", () => {
                if (!draftInputsValid(tabname)) return;
                state.advancedOpen = true;
                state.curveOpen = true;
                const root = studioRoot(tabname);
                if (root) root.open = true;
                closeQuickTune(tabname);
                renderStudio(tabname);
            });
            edit.title = t("studio.status.open_the_precise_curve_editor_the");
            curve.append(edit);
            control.append(curve);
            body.append(control);
            return;
        }

        // Most LoRAs use scalar weights. The range input gives the fast,
        // tactile workflow of the old helper while the number field remains
        // exact. Its range expands when an exact value sits outside it, so a
        // slider never hides or silently clamps a valid LoRA weight.
        const inputs = makeElement("div", "lora-studio-weight-inputs");
        const range = document.createElement("input");
        range.type = "range";
        range.min = String(Math.min(-2, Number(weight.value)));
        range.max = String(Math.max(2, Number(weight.value)));
        range.step = "0.01";
        range.title = t("studio.status.quick_weight_adjustment_its_range_expands");
        range.setAttribute("aria-label", `${label} quick weight`);
        range.setAttribute("data-lora-weight", key);
        const number = document.createElement("input");
        number.type = "number";
        number.required = true;
        number.setAttribute("data-lora-draft-input", "true");
        number.setAttribute("data-lora-weight", key);
        number.step = "any";
        number.inputMode = "decimal";
        number.spellcheck = false;
        number.setAttribute("aria-label", `${label} exact weight`);

        const setRangeValue = (value) => {
            const numeric = Number(value);
            if (!Number.isFinite(numeric)) return;
            if (numeric < Number(range.min)) range.min = String(numeric);
            if (numeric > Number(range.max)) range.max = String(numeric);
            range.value = String(numeric);
        };
        const initial = weight.value;
        number.value = studioModel().formatNumber(initial);
        setRangeValue(initial);

        number.addEventListener("input", () => {
            // An empty number is a normal intermediate while replacing it
            // with Ctrl+A.  Do not manufacture a draft until it is complete.
            if (!number.value.trim()) {
                number.setCustomValidity("");
                return;
            }
            if (mutateDraft(tabname, number, (draftSpec) => {
                draftSpec.weights[key] = studioModel().parseWeight(number.value, weightInitial(draftSpec.weights[key], 1));
            })) {
                setRangeValue(number.value);
                syncWeightInputs(tabname, key, number);
            }
        });
        range.addEventListener("input", () => {
            const value = studioModel().formatNumber(Number(range.value));
            if (mutateDraft(tabname, range, (draftSpec) => {
                draftSpec.weights[key] = studioModel().parseWeight(value, weightInitial(draftSpec.weights[key], 1));
            })) {
                number.value = value;
                syncWeightInputs(tabname, key, range);
            }
        });
        inputs.append(range, number);
        control.append(inputs);
        body.append(control);
    }

    function addDynamicRankControl(tabname, body, spec) {
        const control = makeElement("div", "lora-studio-dyn-control");
        const heading = makeElement("div", "lora-studio-control-heading");
        heading.append(makeElement("span", "", t("studio.ui.dynamic_rank")));
        const hint = makeElement("span", "lora-studio-control-hint", t("studio.ui.optional"));
        hint.title = t("studio.help.off_uses_the_full_rank_a");
        heading.append(hint);
        control.append(heading);

        const inputs = makeElement("div", "lora-studio-dyn-inputs");
        const value = document.createElement("input");
        value.type = "number";
        value.min = "1";
        value.step = "1";
        value.inputMode = "numeric";
        value.placeholder = t("studio.ui.off_custom_rank");
        value.setAttribute("aria-label", t("studio.ui.dynamic_rank_custom_value"));
        value.setAttribute("data-lora-draft-input", "true");
        value.value = spec.dyn_dim === null ? "" : String(spec.dyn_dim);

        const choices = [null, 4, 8, 16, 32];
        const choiceButtons = [];
        const updateChoiceState = (next) => {
            choiceButtons.forEach(({button: choice, rank}) => {
                const active = next === rank;
                choice.classList.toggle("selected", active);
                choice.setAttribute("aria-pressed", String(active));
            });
        };
        const setRank = (rank, source) => {
            if (mutateDraft(tabname, source, (draftSpec) => { draftSpec.dyn_dim = rank; })) {
                value.value = rank === null ? "" : String(rank);
                value.setCustomValidity("");
                updateChoiceState(rank);
            }
        };
        choices.forEach((rank) => {
            const choice = button(rank === null ? t("studio.ui.off") : String(rank), "lora-studio-dyn-choice", () => setRank(rank, choice));
            choice.title = rank === null ? t("studio.ui.do_not_set_a_dynamic_rank") : t("studio.ui.set_dynamic_rank_to_rank", {rank});
            choiceButtons.push({button: choice, rank});
            inputs.append(choice);
        });
        value.addEventListener("input", () => {
            try {
                // Browsers expose an incomplete number (e.g. "-") as an
                // empty value with badInput=true; it must not become Off.
                if (value.validity && value.validity.badInput) throw new Error(t("studio.ui.complete_the_dynamic_rank_or_clear"));
                setRank(studioModel().parseDynamicRank(value.value, true), value);
            } catch (error) {
                value.setCustomValidity(error.message);
                setMessage(tabname, error.message, "error");
            }
        });
        updateChoiceState(spec.dyn_dim);
        inputs.append(value);
        control.append(inputs);
        body.append(control);
    }

    function addExactWeightControl(tabname, body, label, key) {
        const state = studioState(tabname);
        const selected = getSelectedEntry(tabname);
        const spec = state.draft ? state.draft.spec : (selected && selected.spec);
        if (!spec) return;

        const labelElement = makeElement("label", "lora-studio-field lora-studio-exact-weight");
        labelElement.append(makeElement("span", "", t("studio.ui.exact_label", {label: label === "text encoder" ? t("studio.ui.text_encoder") : label})));
        const input = document.createElement("input");
        input.type = "text";
        input.required = true;
        input.setAttribute("data-lora-draft-input", "true");
        input.setAttribute("data-lora-weight", key);
        input.spellcheck = false;
        input.value = studioModel().serializeWeight(spec.weights[key]);
        input.placeholder = t("studio.ui.0_7_or_0_0_1");
        input.addEventListener("input", () => {
            // An empty field is a normal intermediate while replacing a value
            // with Ctrl+A. Leave it editable until a complete scalar or curve
            // is entered again.
            if (!input.value.trim()) {
                input.setCustomValidity("");
                return;
            }
            if (mutateDraft(tabname, input, (draftSpec) => {
                const fallback = weightInitial(draftSpec.weights[key], 1);
                draftSpec.weights[key] = studioModel().parseWeight(input.value, fallback);
            })) syncWeightInputs(tabname, key, input);
        });
        // syncWeightInputs updates the matching quick control in place; no
        // whole-editor rebuild on blur that could swallow the next click.
        labelElement.append(input);
        body.append(labelElement);
    }

    function addOptionalWeightControl(tabname, body, label, key) {
        const state = studioState(tabname);
        const selected = getSelectedEntry(tabname);
        const spec = state.draft ? state.draft.spec : (selected && selected.spec);
        if (!spec) return;

        const labelElement = makeElement("label", "lora-studio-field");
        labelElement.append(makeElement("span", "", label));
        const input = document.createElement("input");
        input.type = "text";
        input.spellcheck = false;
        input.value = spec.hires[key] ? studioModel().serializeWeight(spec.hires[key]) : "";
        input.placeholder = t("studio.ui.inherit_blank_0_7_or_0");
        input.addEventListener("input", () => {
            mutateDraft(tabname, input, (draftSpec) => {
                const raw = input.value.trim();
                if (!raw) {
                    draftSpec.hires[key] = null;
                    return;
                }
                const fallback = weightInitial(draftSpec.hires[key] || draftSpec.weights[key], 1);
                draftSpec.hires[key] = studioModel().parseWeight(raw, fallback);
            });
        });
        labelElement.append(input);
        body.append(labelElement);
    }

    function parseOptionalStep(raw, label) {
        const text = raw.trim();
        if (!text) return null;
        const value = Number(text);
        if (!Number.isFinite(value)) throw new Error(`${label} must be a finite number`);
        return value;
    }

    function addStepControl(tabname, body, label, key) {
        const state = studioState(tabname);
        const selected = getSelectedEntry(tabname);
        const spec = state.draft ? state.draft.spec : (selected && selected.spec);
        if (!spec) return;

        const labelElement = makeElement("label", "lora-studio-field");
        labelElement.append(makeElement("span", "", label));
        const input = document.createElement("input");
        input.type = "text";
        input.inputMode = "decimal";
        input.spellcheck = false;
        input.value = spec.active_range[key] === null ? "" : String(spec.active_range[key]);
        input.placeholder = t("studio.ui.off_fraction_or_sampler_step");
        input.addEventListener("input", () => {
            mutateDraft(tabname, input, (draftSpec) => {
                const value = parseOptionalStep(input.value, label);
                const start = key === "start" ? value : draftSpec.active_range.start;
                const stop = key === "stop" ? value : draftSpec.active_range.stop;
                studioModel().parseStepRange({start: start === null ? "" : start, stop: stop === null ? "" : stop});
                draftSpec.active_range[key] = value;
            });
        });
        labelElement.append(input);
        body.append(labelElement);
    }

    function nativeBlockPresetNames(blocks) {
        const names = new Map();
        blocks.builtinPresetNames().forEach((name) => names.set(name.toLocaleLowerCase(), name));
        String(opts.lora_block_weight_presets || "").split(/\r?\n/).forEach((line) => {
            const separator = line.indexOf(":");
            const name = separator >= 0 ? line.slice(0, separator).trim() : "";
            if (name) names.set(name.toLocaleLowerCase(), name);
        });
        return [...names.values()].sort((left, right) => left.localeCompare(right));
    }

    function nativeElementalPresetNames() {
        const names = new Map([
            ["attndeepon", "ATTNDEEPON"],
            ["attndeepoff", "ATTNDEEPOFF"],
            ["projdeepoff", "PROJDEEPOFF"],
        ]);
        // Elemental settings use a blank line between named groups.  Reading
        // just the group header here is deliberately tolerant: final
        // validation remains native when the prompt is applied.
        String(opts.lora_elemental_weight_presets || "").split(/\r?\n[ \t]*\r?\n/).forEach((group) => {
            const separator = group.indexOf(":");
            const name = separator >= 0 ? group.slice(0, separator).trim() : "";
            if (name) names.set(name.toLocaleLowerCase(), name);
        });
        return [...names.values()].sort((left, right) => left.localeCompare(right));
    }

    function coverageNetworkItem(tabname, spec) {
        return spec && libraryItemFor(tabname, spec.network_name);
    }

    function refreshCoverageSlot(tabname, blocks, spec) {
        const root = studioRoot(tabname);
        const slot = root && root.querySelector(".lora-studio-coverage-slot");
        if (!slot) return;
        renderCoverageSlot(tabname, slot, blocks, spec);
    }

    function loadCoverage(tabname, blocks, spec) {
        const item = coverageNetworkItem(tabname, spec);
        if (!item) {
            setMessage(tabname, "studio.status.this_lora_is_not_present_in", "error");
            return;
        }
        const state = studioState(tabname);
        if (state.coverageLoading) return;
        state.coverageLoading = true;
        state.coverageError = "";
        state.coverageErrorForNetwork = "";
        refreshCoverageSlot(tabname, blocks, spec);
        const url = new URL("sdapi/v1/lora-studio/coverage", window.location.href);
        url.searchParams.set("name", item.name);
        fetch(url)
            .then(async (response) => {
                const payload = await response.json().catch(() => ({}));
                if (!response.ok) throw new Error(payload.detail || `File-key inspection failed (${response.status})`);
                if (!payload || payload.schema_version !== 1 || !["ok", "unavailable"].includes(payload.status)) {
                    throw new Error("File-key inspection returned an unsupported response");
                }
                state.coverage = payload;
                state.coverageForNetwork = item.name;
            })
            .catch((error) => {
                state.coverage = null;
                state.coverageForNetwork = "";
                state.coverageError = error.message;
                state.coverageErrorForNetwork = item.name;
            })
            .finally(() => {
                state.coverageLoading = false;
                refreshCoverageSlot(tabname, blocks, displayedSpec(tabname));
            });
    }

    function renderCoverageSlot(tabname, slot, blocks, spec) {
        slot.replaceChildren();
        const state = studioState(tabname);
        const item = coverageNetworkItem(tabname, spec);
        if (!item) {
            slot.append(makeElement("p", "lora-studio-preserved", t("studio.status.file_key_coverage_is_available_after")));
            return;
        }
        if (state.coverageLoading) {
            slot.append(makeElement("p", "lora-studio-preserved", t("studio.status.inspecting_the_safetensors_header_no_tensor")));
            return;
        }
        if (state.coverageErrorForNetwork === item.name && state.coverageError) {
            slot.append(makeElement("p", "lora-studio-warning", state.coverageError));
        }
        const report = state.coverageForNetwork === item.name ? state.coverage : null;
        const inspect = button(report ? t("studio.ui.inspect_file_keys_again") : t("studio.ui.inspect_safetensors_file_keys"), "", () => loadCoverage(tabname, blocks, spec));
        inspect.title = t("studio.help.reads_only_the_selected_safetensors_json");
        slot.append(inspect);
        if (!report) return;
        if (report.status === "unavailable") {
            slot.append(makeElement("p", "lora-studio-warning", report.reason || "This file cannot be inspected safely."));
            return;
        }
        slot.append(makeElement("p", "lora-studio-preserved", report.notice || "This is file-key coverage only; it does not prove active-checkpoint compatibility."));
        const suggestion = report.suggestion || {};
        if (suggestion.profile_id) {
            const suggested = blocks.getProfile(suggestion.profile_id);
            slot.append(makeElement("p", "lora-studio-coverage-summary", `Suggested layout: ${suggested.label} (${suggestion.confidence || "unknown"} confidence). ${suggestion.reason || ""}`));
        } else if (suggestion.confidence === "ambiguous") {
            slot.append(makeElement("p", "lora-studio-warning", `No layout was selected automatically: ${suggestion.reason || "the file-key evidence is ambiguous"}`));
        } else {
            slot.append(makeElement("p", "lora-studio-warning", suggestion.reason || "No recognised layout pattern was found."));
        }

        const profile = blocks.getProfile(spec.model_profile);
        if (profile.id === "auto") {
            slot.append(makeElement("p", "lora-studio-preserved", t("studio.help.choose_a_block_layout_profile_above")));
            return;
        }
        const coverage = Array.isArray(report.profiles) && report.profiles.find((entry) => entry.profile_id === profile.id);
        if (!coverage) {
            slot.append(makeElement("p", "lora-studio-warning", t("studio.status.the_inspection_has_no_coverage_record")));
            return;
        }
        slot.append(makeElement("p", "lora-studio-coverage-summary", `${profile.label}: ${coverage.matched_block_ids.length}/${coverage.total_block_count} layout blocks and ${coverage.matched_module_count} file module key(s) recognised.`));
        if (coverage.matched_block_ids.length) {
            slot.append(makeElement("p", "lora-studio-preserved", `Matched blocks: ${coverage.matched_block_ids.join(", ")}.`));
        }
        if (coverage.profile_mismatch_module_count || coverage.unclassified_module_count) {
            slot.append(makeElement("p", "lora-studio-preserved", `${coverage.profile_mismatch_module_count || 0} key(s) map to another layout; ${coverage.unclassified_module_count || 0} key(s) have no recognised block pattern.`));
        }
    }

    function renderCoverageControls(tabname, body, blocks, spec) {
        const state = studioState(tabname);
        const details = makeElement("details", "lora-studio-coverage");
        details.open = state.coverageOpen;
        details.addEventListener("toggle", () => { state.coverageOpen = details.open; });
        details.append(makeElement("summary", "", t("studio.ui.file_key_coverage_not_runtime_application")));
        details.querySelector("summary").append(helpIcon("studio.metadata"));
        const slot = makeElement("div", "lora-studio-coverage-slot");
        details.append(slot);
        body.append(details);
        renderCoverageSlot(tabname, slot, blocks, spec);
    }

    function addProfileControl(tabname, body, blocks, spec) {
        const labelElement = makeElement("label", "lora-studio-field");
        labelElement.append(makeElement("span", "", t("studio.ui.block_layout_profile")));
        const select = document.createElement("select");
        const profiles = blocks.listProfiles();
        const current = String(spec.model_profile || "auto");
        if (!profiles.some((profile) => profile.id === current)) {
            const unknown = makeElement("option", "", `Preserve unknown profile: ${current}`);
            unknown.value = current;
            select.append(unknown);
        }
        profiles.forEach((profile) => {
            const option = makeElement("option", "", profile.label);
            option.value = profile.id;
            select.append(option);
        });
        select.value = current;
        select.addEventListener("change", () => {
            if (mutateDraft(tabname, select, (draftSpec) => { draftSpec.model_profile = select.value; })) {
                renderStudio(tabname);
            }
        });
        labelElement.append(select);
        body.append(labelElement);
    }

    function blockInputValues(grid, blocks) {
        const values = [];
        grid.querySelectorAll("[data-lora-block-index]").forEach((input) => {
            const raw = input.value.trim();
            const value = Number(raw);
            if (!raw || !Number.isFinite(value)) throw new Error(`Block ${input.dataset.loraBlockId} needs a finite value; blank is not zero`);
            values[Number(input.dataset.loraBlockIndex)] = value;
        });
        return values.map((value) => {
            if (!Number.isFinite(value)) throw new Error("Block editor has an incomplete value");
            return value;
        });
    }

    function writeExactBlockValues(tabname, blocks, values, source) {
        const changed = mutateDraft(tabname, source, (draftSpec) => {
            draftSpec.block_weights = {raw: blocks.serializeExactBlockWeights(values)};
        });
        if (changed) {
            const root = studioRoot(tabname);
            const raw = root.querySelector(".lora-studio-block-weights input");
            if (raw) raw.value = displayedSpec(tabname).block_weights.raw;
            const select = root.querySelector(".lora-studio-block-preset");
            if (select) select.value = "";
        }
        return changed;
    }

    function renderExactBlockEditor(tabname, body, blocks, profile, initialValues, sourceKind) {
        const editor = makeElement("div", "lora-studio-block-editor");
        if (sourceKind === "builtin") {
            editor.append(makeElement("p", "lora-studio-preserved", t("studio.help.this_native_preset_is_shown_for")));
        } else if (sourceKind === "empty") {
            editor.append(makeElement("p", "lora-studio-preserved", t("studio.help.the_table_is_still_a_preview")));
        }

        const grid = makeElement("div", "lora-studio-block-groups");
        const values = [...initialValues];
        const syncBlockSlider = (input) => {
            const slider = input.parentElement && input.parentElement.querySelector('input[type="range"]');
            const value = Number(input.value);
            if (!slider || !input.value.trim() || !Number.isFinite(value)) return;
            slider.min = String(Math.min(-1, value));
            slider.max = String(Math.max(2, value));
            slider.value = String(value);
        };
        const updateGridValues = (indices = null) => {
            grid.querySelectorAll("[data-lora-block-index]").forEach((input) => {
                if (indices && !indices.includes(Number(input.dataset.loraBlockIndex))) return;
                input.value = studioModel().formatNumber(values[Number(input.dataset.loraBlockIndex)]);
                input.setCustomValidity("");
                syncBlockSlider(input);
            });
        };
        const updateOne = (source) => {
            try {
                grid.querySelectorAll("[data-lora-block-index]").forEach((input) => {
                    const raw = input.value.trim();
                    input.setCustomValidity(raw && Number.isFinite(Number(raw)) ? "" : "Enter a finite block value; blank is not zero");
                });
                const next = blockInputValues(grid, blocks);
                values.splice(0, values.length, ...next);
                grid.querySelectorAll("[data-lora-block-index]").forEach(syncBlockSlider);
                writeExactBlockValues(tabname, blocks, values, source);
            } catch (error) {
                setMessage(tabname, error.message, "error");
            }
        };

        const bulk = makeElement("div", "lora-studio-block-bulk");
        bulk.append(makeElement("span", "", t("studio.ui.set_all_blocks")));
        const bulkValue = document.createElement("input");
        bulkValue.type = "text";
        bulkValue.inputMode = "decimal";
        bulkValue.placeholder = t("studio.ui.custom_value");
        bulkValue.setAttribute("aria-label", t("studio.ui.value_for_all_lora_blocks"));
        bulkValue.addEventListener("input", () => bulkValue.setCustomValidity(""));
        const setAll = (value) => {
            if (!Number.isFinite(value)) {
                bulkValue.setCustomValidity("Block value must be a finite number");
                bulkValue.reportValidity();
                return;
            }
            bulkValue.setCustomValidity("");
            bulkValue.value = studioModel().formatNumber(value);
            values.fill(value);
            updateGridValues();
            writeExactBlockValues(tabname, blocks, values, null);
        };
        const zero = button("0", "", () => setAll(0));
        zero.title = t("studio.ui.set_every_visible_block_to_0");
        const one = button("1", "", () => setAll(1));
        one.title = t("studio.ui.set_every_visible_block_to_1");
        const custom = button(t("studio.ui.set"), "", () => setAll(bulkValue.value.trim() ? Number(bulkValue.value.trim()) : NaN));
        bulk.append(zero, one, bulkValue, custom);
        editor.append(bulk);

        blocks.groupBlocks(profile.id).forEach((group) => {
            const section = makeElement("section", "lora-studio-block-group");
            const header = makeElement("div", "lora-studio-block-group-header");
            header.append(makeElement("h5", "", group.label));
            const groupValue = document.createElement("input");
            groupValue.type = "text";
            groupValue.inputMode = "decimal";
            groupValue.placeholder = t("studio.ui.set_group");
            groupValue.setAttribute("aria-label", `Value for ${group.label} blocks`);
            groupValue.addEventListener("input", () => groupValue.setCustomValidity(""));
            const setGroup = button(t("studio.ui.set"), "", () => {
                const value = groupValue.value.trim() ? Number(groupValue.value.trim()) : NaN;
                if (!Number.isFinite(value)) {
                    groupValue.setCustomValidity("Group value must be a finite number");
                    groupValue.reportValidity();
                    return;
                }
                groupValue.setCustomValidity("");
                group.blockIds.forEach((blockId) => {
                    values[profile.blockIds.indexOf(blockId)] = value;
                });
                updateGridValues(group.blockIds.map((blockId) => profile.blockIds.indexOf(blockId)));
                writeExactBlockValues(tabname, blocks, values, null);
            });
            header.append(groupValue, setGroup);
            section.append(header);
            const cells = makeElement("div", "lora-studio-block-cells");
            group.blockIds.forEach((blockId) => {
                const index = profile.blockIds.indexOf(blockId);
                const label = makeElement("label", "lora-studio-block-cell");
                label.append(makeElement("span", "", blockId));
                const input = document.createElement("input");
                input.type = "text";
                input.inputMode = "decimal";
                input.spellcheck = false;
                input.required = true;
                input.setAttribute("data-lora-draft-input", "true");
                input.dataset.loraBlockId = blockId;
                input.dataset.loraBlockIndex = String(index);
                input.setAttribute("aria-label", `${blockId} exact block weight`);
                input.value = studioModel().formatNumber(values[index]);
                input.addEventListener("input", () => updateOne(input));
                label.append(input);
                const slider = document.createElement("input");
                slider.type = "range";
                slider.step = "0.01";
                slider.setAttribute("aria-label", `${blockId} block weight slider`);
                slider.title = t("studio.status.adjust_this_block_only_exact_values");
                slider.addEventListener("input", () => {
                    input.value = studioModel().formatNumber(Number(slider.value));
                    updateOne(input);
                });
                label.append(slider);
                syncBlockSlider(input);
                cells.append(label);
            });
            section.append(cells);
            grid.append(section);
        });
        editor.append(grid);
        body.append(editor);
    }

    function renderVisualBlockEditor(tabname, body, blocks, spec) {
        const raw = spec.block_weights ? spec.block_weights.raw : "";
        const resolved = blocks.resolveBlockWeights(raw, opts.lora_block_weight_presets || "");
        const requested = blocks.getProfile(spec.model_profile);
        const profile = requested.id !== "auto" ? requested : (resolved ? blocks.profileForValues(resolved.values, "auto") : requested);
        if (raw && !resolved) {
            body.append(makeElement("p", "lora-studio-warning", t("studio.help.this_is_a_custom_named_preset")));
            return;
        }
        if (profile.id === "auto") {
            body.append(makeElement("p", "lora-studio-preserved", t("studio.help.select_a_block_layout_profile_to")));
            return;
        }
        const projected = resolved && resolved.values.length !== profile.blockIds.length ? blocks.projectClassicValues(resolved.values, profile.id) : null;
        if (resolved && resolved.values.length !== profile.blockIds.length && !projected) {
            body.append(makeElement("p", "lora-studio-warning", `The current block value has ${resolved.values.length} entries, but ${profile.label} needs ${profile.blockIds.length}. It is preserved unchanged.`));
            return;
        }
        if (projected) body.append(makeElement("p", "lora-studio-preserved", t("studio.help.classic_mask_mapped_by_block_name")));
        const values = projected || (resolved ? resolved.values : Array(profile.blockIds.length).fill(1));
        renderExactBlockEditor(tabname, body, blocks, profile, values, resolved ? resolved.kind : "empty");
    }

    function addBlockWeightControl(tabname, body) {
        const state = studioState(tabname);
        const selected = getSelectedEntry(tabname);
        const spec = state.draft ? state.draft.spec : (selected && selected.spec);
        if (!spec) return;

        const blocks = studioBlocks();
        if (blocks) {
            addProfileControl(tabname, body, blocks, spec);
            renderCoverageControls(tabname, body, blocks, spec);
        }
        else body.append(makeElement("p", "lora-studio-warning", t("studio.status.block_editor_did_not_load_the")));
        const labelElement = makeElement("label", "lora-studio-field lora-studio-block-weights");
        labelElement.append(makeElement("span", "", t("studio.ui.raw_block_weights")));
        const input = document.createElement("input");
        input.type = "text";
        input.spellcheck = false;
        input.value = spec.block_weights ? spec.block_weights.raw : "";
        input.placeholder = t("studio.ui.inherit_blank_outs_or_exact_comma");
        input.addEventListener("input", () => {
            if (mutateDraft(tabname, input, (draftSpec) => {
                const raw = input.value.trim();
                if (raw && blocks && !blocks.resolveBlockWeights(raw, opts.lora_block_weight_presets || "")) {
                    throw new Error("Choose a defined block preset or enter a complete finite 12/17/20/26/61-value table.");
                }
                draftSpec.block_weights = raw ? {raw} : null;
            }) && blocks) refreshBlockViews();
        });
        labelElement.append(input);
        body.append(labelElement);

        if (!blocks) return;
        const visual = makeElement("div", "lora-studio-block-visual");
        let presetSelect = null;
        const refreshBlockViews = () => {
            const next = displayedSpec(tabname);
            if (presetSelect) {
                const raw = next.block_weights ? next.block_weights.raw : "";
                presetSelect.value = nativeBlockPresetNames(blocks).find((name) => name.toLowerCase() === raw.toLowerCase()) || "";
            }
            visual.replaceChildren();
            renderVisualBlockEditor(tabname, visual, blocks, next);
        };
        const presets = nativeBlockPresetNames(blocks);
        if (presets.length) {
            const presetLabel = makeElement("label", "lora-studio-field");
            presetLabel.append(makeElement("span", "", t("studio.ui.native_block_preset")));
            const preset = document.createElement("select");
            preset.className = "lora-studio-block-preset";
            presetSelect = preset;
            const placeholder = makeElement("option", "", t("studio.ui.choose_a_preset_creates_a_local"));
            placeholder.value = "";
            preset.append(placeholder);
            presets.forEach((name) => {
                const option = makeElement("option", "", name);
                option.value = name;
                preset.append(option);
            });
            const currentRaw = spec.block_weights ? spec.block_weights.raw : "";
            const currentName = presets.find((name) => name.toLocaleLowerCase() === String(currentRaw).toLocaleLowerCase());
            preset.value = currentName || "";
            preset.addEventListener("change", () => {
                if (!preset.value) return;
                if (mutateDraft(tabname, preset, (draftSpec) => { draftSpec.block_weights = {raw: preset.value}; })) {
                    input.value = preset.value;
                    refreshBlockViews();
                }
            });
            presetLabel.append(preset);
            body.append(presetLabel);
        }
        body.append(visual);
        refreshBlockViews();
    }

    function addElementalWeightControl(tabname, body) {
        const spec = displayedSpec(tabname);
        if (!spec) return;

        const control = makeElement("section", "lora-studio-elemental-weights");
        control.append(makeElement("p", "lora-studio-preserved", t("studio.help.a_named_elemental_preset_can_override")));
        const labelElement = makeElement("label", "lora-studio-field");
        labelElement.append(makeElement("span", "", t("studio.ui.elemental_module_preset")));
        const input = document.createElement("input");
        input.type = "text";
        input.spellcheck = false;
        input.placeholder = t("studio.ui.inherit_blank_or_attndeepoff");
        input.value = spec.elemental_weights ? spec.elemental_weights.raw : "";
        let presetSelect = null;
        const listId = `${tabname}_lora_studio_elemental_presets`;
        input.setAttribute("list", listId);
        const datalist = document.createElement("datalist");
        datalist.id = listId;
        nativeElementalPresetNames().forEach((name) => {
            const option = document.createElement("option");
            option.value = name;
            datalist.append(option);
        });
        input.addEventListener("input", () => {
            if (mutateDraft(tabname, input, (draftSpec) => {
                const raw = input.value.trim();
                if (raw && !nativeElementalPresetNames().some(name => name.toLowerCase() === raw.toLowerCase())) {
                    throw new Error("Choose a defined elemental preset; add custom rules in WebUI settings first.");
                }
                draftSpec.elemental_weights = raw ? {raw} : null;
            }) && presetSelect) {
                presetSelect.value = nativeElementalPresetNames().find((name) => name.toLowerCase() === input.value.trim().toLowerCase()) || "";
            }
        });
        labelElement.append(input, datalist);
        control.append(labelElement);

        const presets = nativeElementalPresetNames();
        if (presets.length) {
            const selectLabel = makeElement("label", "lora-studio-field");
            selectLabel.append(makeElement("span", "", t("studio.ui.choose_elemental_preset")));
            const select = document.createElement("select");
            presetSelect = select;
            const placeholder = makeElement("option", "", t("studio.ui.choose_a_preset_creates_a_local"));
            placeholder.value = "";
            select.append(placeholder);
            presets.forEach((name) => {
                const option = makeElement("option", "", name);
                option.value = name;
                select.append(option);
            });
            const current = spec.elemental_weights ? spec.elemental_weights.raw : "";
            const currentName = presets.find((name) => name.toLocaleLowerCase() === String(current).toLocaleLowerCase());
            select.value = currentName || "";
            select.addEventListener("change", () => {
                if (!select.value) return;
                if (mutateDraft(tabname, select, (draftSpec) => { draftSpec.elemental_weights = {raw: select.value}; })) {
                    input.value = select.value;
                }
            });
            selectLabel.append(select);
            control.append(selectLabel);
        }
        body.append(control);
    }

    function renderAdvancedControls(tabname, body) {
        const state = studioState(tabname);
        const advanced = makeElement("details", "lora-studio-advanced");
        advanced.open = state.advancedOpen;
        advanced.addEventListener("toggle", () => { state.advancedOpen = advanced.open; });
        advanced.append(makeElement("summary", "", t("studio.ui.advanced_controls")));
        advanced.querySelector("summary").append(helpIcon("studio.quick"));
        const content = makeElement("div", "lora-studio-advanced-body");
        content.append(makeElement("p", "lora-studio-preserved", t("studio.status.blank_hires_and_timing_values_mean")));
        renderSidecarControls(tabname, content, displayedSpec(tabname));
        content.append(helpHeading("h4", t("studio.ui.exact_weights_and_curves"), "studio.quick"));
        content.append(makeElement("p", "lora-studio-preserved", t("studio.help.use_these_fields_for_precise_values")));
        addExactWeightControl(tabname, content, "text encoder", "te");
        addExactWeightControl(tabname, content, "U-Net", "unet");
        const curveSlot = makeElement("div", "lora-studio-curve-slot");
        content.append(curveSlot);
        renderCurveControls(tabname, curveSlot);
        content.append(helpHeading("h4", t("studio.ui.hires_fix_overrides"), "studio.hires"));
        addOptionalWeightControl(tabname, content, "Hires text encoder", "te");
        addOptionalWeightControl(tabname, content, "Hires U-Net", "unet");
        content.append(helpHeading("h4", t("studio.ui.activation_timing"), "studio.timing"));
        content.append(makeElement("p", "lora-studio-preserved", t("studio.help.start_stop_gate_this_lora_s")));
        addStepControl(tabname, content, "Start", "start");
        addStepControl(tabname, content, "Stop", "stop");
        content.append(helpHeading("h4", t("studio.ui.block_weights"), "studio.blocks"));
        addBlockWeightControl(tabname, content);
        content.append(helpHeading("h4", t("studio.ui.elemental_module_overrides"), "studio.elemental"));
        addElementalWeightControl(tabname, content);
        content.append(makeElement("p", "lora-studio-preserved", t("studio.help.native_parsing_validates_every_named_preset")));
        advanced.append(content);
        body.append(advanced);
    }

    function renderEditor(tabname, body) {
        const state = studioState(tabname);
        const selected = getSelectedEntry(tabname);
        const draft = state.draft;
        const spec = draft ? draft.spec : (selected && selected.spec);
        if (!spec) {
            body.append(makeElement("p", "lora-studio-empty", t("studio.status.add_a_lora_from_its_card")));
            return;
        }

        body.append(helpHeading("h3", draft && draft.mode === "add" ? t("studio.ui.new_lora_name", {name: spec.network_name}) : t("studio.ui.selected_lora_name", {name: spec.network_name}), "studio.add"));
        renderMetadata(tabname, body);
        if (draft && draft.mode === "remove") {
            body.append(makeElement("p", "lora-studio-warning", t("studio.status.this_draft_will_remove_only_the")));
        } else {
            renderQuickPresetControls(tabname, body);
            const quick = makeElement("section", "lora-studio-quick-controls");
            quick.append(helpHeading("h4", t("studio.ui.quick_controls"), "studio.quick"));
            addWeightControl(tabname, quick, "Text encoder", "te");
            addWeightControl(tabname, quick, "U-Net", "unet");
            addDynamicRankControl(tabname, quick, spec);
            body.append(quick);
            addCombinationSummary(body);

            renderAdvancedControls(tabname, body);
            renderPresetControls(tabname, body);
        }

        const preview = makeElement("div", "lora-studio-tag-preview");
        preview.append(makeElement("span", "", t("studio.ui.tag_preview")));
        preview.append(makeElement("code", "", studioModel().serializeTag(spec)));
        body.append(preview);

        const actions = makeElement("div", "lora-studio-actions");
        const apply = button(draft && draft.mode === "remove" ? t("studio.ui.apply_removal") : t("studio.ui.apply_to_prompt"), "lora-studio-apply", () => applyDraft(tabname));
        apply.disabled = !draft || draftIsStale(tabname);
        actions.append(apply);
        const discard = button(t("studio.ui.discard_draft"), "lora-studio-discard", () => discardDraft(tabname));
        discard.hidden = !draft;
        actions.append(discard);
        if (selected) {
            const duplicate = button(t("studio.ui.duplicate"), "lora-studio-selected-action", () => {
                createNewDraft(tabname, selected.spec.network_name, clone(selected.spec));
                setMessage(tabname, "studio.status.duplicate_is_a_local_draft_until", "info");
                renderStudio(tabname);
            });
            duplicate.hidden = Boolean(draft);
            actions.append(duplicate);
            const remove = button(t("studio.ui.remove"), "lora-studio-selected-action", () => {
                draftFor(tabname, "remove");
                setMessage(tabname, "studio.status.removal_is_a_local_draft_until", "info");
                renderStudio(tabname);
            });
            remove.hidden = Boolean(draft);
            actions.append(remove);
        }
        body.append(actions);
    }

    function renderList(tabname, body) {
        const state = studioState(tabname);
        const list = makeElement("div", "lora-studio-list");
        list.append(helpHeading("h3", t("studio.ui.active_loras_count", {count: state.entries.length}), "studio.list"));
        if (!state.entries.length) {
            list.append(makeElement("p", "lora-studio-empty", t("studio.status.no_lora_tags_in_this_prompt")));
        }
        state.entries.forEach((entry, index) => {
            const id = entryId(entry);
            const item = button("", "lora-studio-list-item", () => {
                if (state.draft) {
                    setMessage(tabname, "studio.status.apply_or_discard_the_current_draft_067b489", "error");
                    return;
                }
                state.selectedId = id;
                state.draft = null;
                renderStudio(tabname);
            });
            const name = makeElement("span", "lora-studio-list-item-name", `#${index + 1} ${entry.spec.network_name}`);
            name.title = entry.spec.network_name;
            const values = makeElement("span", "lora-studio-list-item-values", `TE ${weightSummary(entry.spec.weights.te)} · U-Net ${weightSummary(entry.spec.weights.unet)}`);
            item.append(name, values);
            item.classList.toggle("selected", state.selectedId === id);
            item.classList.toggle("draft", Boolean(state.draft && state.draft.entryId === id));
            list.append(item);
        });
        body.append(list);
    }

    function populateLibrary(tabname, datalist) {
        const state = studioState(tabname);
        const fill = (items) => {
            datalist.replaceChildren();
            items.forEach((item) => {
                const option = document.createElement("option");
                option.value = item.name;
                option.label = item.alias && item.alias !== item.name ? `${item.alias} — ${item.path || item.name}` : (item.path || item.name);
                datalist.append(option);
            });
        };
        if (state.libraryItems) {
            fill(state.libraryItems);
            return;
        }
        if (state.libraryLoading) return;
        state.libraryLoading = true;
        fetch(new URL("sdapi/v1/loras", window.location.href))
            .then((response) => response.ok ? response.json() : Promise.reject(new Error(`LoRA library request failed (${response.status})`)))
            .then((items) => {
                if (!Array.isArray(items)) throw new Error("LoRA library response is not a list");
                state.libraryItems = items;
                const current = studioRoot(tabname) && studioRoot(tabname).querySelector(`#${tabname}_lora_studio_library`);
                if (current) populateLibrary(tabname, current);
                refreshMetadataSlot(tabname);
            })
            .catch((error) => setMessage(tabname, t("studio.ui.lora_suggestions_unavailable_error", {error: error.message}), "info"))
            .finally(() => { state.libraryLoading = false; });
    }

    function renderAddControl(tabname, body) {
        const add = makeElement("div", "lora-studio-add lora-studio-toolbar");
        const label = makeElement("label", "lora-studio-add-label");
        label.append(makeElement("span", "", t("studio.ui.add_lora")));
        const input = document.createElement("input");
        input.type = "search";
        input.placeholder = t("studio.ui.search_installed_loras_or_enter_a");
        input.autocomplete = "off";
        const listId = `${tabname}_lora_studio_library`;
        input.setAttribute("list", listId);
        const datalist = document.createElement("datalist");
        datalist.id = listId;
        const addButton = button(t("studio.ui.add_as_draft"), "lora-studio-add-button", () => {
            if (studioState(tabname).draft) {
                setMessage(tabname, "studio.status.apply_or_discard_the_current_draft", "error");
                return;
            }
            const name = input.value.trim();
            if (!name) {
                input.setCustomValidity("Enter a LoRA name first");
                input.reportValidity();
                return;
            }
            input.setCustomValidity("");
            createNewDraft(tabname, name);
            setMessage(tabname, "studio.status.new_lora_is_a_local_draft", "info");
            renderStudio(tabname);
        });
        input.addEventListener("keydown", (event) => {
            if (event.key === "Enter") {
                event.preventDefault();
                addButton.click();
            }
        });
        label.append(input);
        add.append(label, helpIcon("studio.add"), addButton, datalist);
        body.append(add);
        populateLibrary(tabname, datalist);
    }

    function renderStudio(tabname) {
        const root = studioRoot(tabname);
        const state = studioState(tabname);
        if (!root) return;
        const body = root.querySelector(".lora-studio-body");
        const count = root.querySelector(".lora-studio-count");
        if (!body || !count) return;
        body.inert = Boolean(quickTuneRoot(tabname));
        count.textContent = state.entries.length ? `(${state.entries.length})` : "";
        body.replaceChildren();

        const message = makeElement("p", "lora-studio-message");
        message.hidden = !state.message;
        message.dataset.kind = state.messageKind;
        message.textContent = state.message;
        body.append(message);
        renderAddControl(tabname, body);
        const workspace = makeElement("div", "lora-studio-workspace");
        renderList(tabname, workspace);
        const editor = makeElement("div", "lora-studio-editor");
        renderEditor(tabname, editor);
        workspace.append(editor);
        body.append(workspace);
        refreshCombinationSummary(tabname);
    }

    function refreshStudio(tabname) {
        const textarea = studioTextarea(tabname);
        if (!textarea || !studioRoot(tabname)) return;
        const state = studioState(tabname);
        state.entries = studioModel().findTags(textarea.value);
        if (!state.draft && !state.entries.some((entry) => entryId(entry) === state.selectedId)) {
            state.selectedId = state.entries.length ? entryId(state.entries[0]) : null;
        }
        renderStudio(tabname);
    }

    function closeQuickTune(tabname) {
        const root = quickTuneRoot(tabname);
        if (root) root.remove();
        studioState(tabname).quickTune = null;
        const studio = studioRoot(tabname);
        const body = studio && studio.querySelector(".lora-studio-body");
        if (body) body.inert = false;
    }

    function openFullStudioFromQuickTune(tabname) {
        if (!draftInputsValid(tabname)) return;
        const root = studioRoot(tabname);
        if (root) root.open = true;
        closeQuickTune(tabname);
        renderStudio(tabname);
        setMessage(tabname, "studio.status.opened_the_full_studio_your_current", "info");
    }

    function positionQuickTune(root, clientX, clientY) {
        const margin = 12;
        const x = Number.isFinite(clientX) ? clientX : margin;
        const y = Number.isFinite(clientY) ? clientY : margin;
        root.style.left = `${Math.max(margin, x)}px`;
        root.style.top = `${Math.max(margin, y)}px`;
        const bounds = root.getBoundingClientRect();
        root.style.left = `${Math.max(margin, Math.min(x, window.innerWidth - bounds.width - margin))}px`;
        root.style.top = `${Math.max(margin, Math.min(y, window.innerHeight - bounds.height - margin))}px`;
        // Opening Save or a longer preset explanation may grow the panel.
        // Keep the bottom inside the viewport and scroll its content.
        root.style.maxHeight = `${Math.max(0, window.innerHeight - Number.parseFloat(root.style.top) - margin)}px`;
    }

    function openQuickTuneForTagAtCaret(tabname, event) {
        const textarea = studioTextarea(tabname);
        if (!textarea || textarea.selectionStart !== textarea.selectionEnd) return false;
        const state = studioState(tabname);
        if (state.draft) {
            // Do not silently discard a pending edit just because the user
            // right-clicked another tag.
            setMessage(tabname, "studio.status.apply_or_discard_the_current_lora", "error");
            return false;
        }
        const caret = textarea.selectionStart;
        const entry = studioModel().findTags(textarea.value).find((candidate) => candidate.start <= caret && caret < candidate.end);
        if (!entry) return false;

        event.preventDefault();
        state.entries = studioModel().findTags(textarea.value);
        state.selectedId = entryId(entry);
        const draft = draftFor(tabname, "replace");
        state.quickTune = {clientX: event.clientX, clientY: event.clientY};
        renderQuickTune(tabname);
        const studio = studioRoot(tabname);
        if (studio && studio.open) renderStudio(tabname);
        setMessage(tabname, "studio.status.opened_quick_controls_for_name_changes", "info", {name: draft.spec.network_name});
        return true;
    }

    function renderQuickTune(tabname) {
        const state = studioState(tabname);
        const draft = state.draft;
        if (!draft) return;
        const position = state.quickTune || {clientX: 12, clientY: 12};
        const previous = quickTuneRoot(tabname);
        if (previous) previous.remove();

        const root = makeElement("section", "lora-studio-quick-tune");
        root.id = `${tabname}_lora_quick_tune`;
        root.tabIndex = -1;
        root.setAttribute("role", "dialog");
        root.setAttribute("aria-label", t("studio.ui.quick_lora_editor_for_name", {name: draft.spec.network_name}));

        const header = makeElement("div", "lora-studio-quick-tune-header");
        header.append(makeElement("h3", "", t("studio.ui.quick_tune_name", {name: draft.spec.network_name})));
        const close = button("×", "lora-studio-quick-tune-close", () => discardDraft(tabname));
        close.setAttribute("aria-label", t("studio.ui.cancel_quick_edit"));
        header.append(close);
        root.append(header);
        root.append(makeElement("p", "lora-studio-quick-tune-status"));
        renderQuickPresetControls(tabname, root);

        const controls = makeElement("div", "lora-studio-quick-tune-controls");
        addWeightControl(tabname, controls, "Text encoder", "te");
        addWeightControl(tabname, controls, "U-Net", "unet");
        addDynamicRankControl(tabname, controls, draft.spec);
        root.append(controls);
        addCombinationSummary(root);

        const preview = makeElement("div", "lora-studio-quick-tune-preview");
        preview.append(makeElement("span", "", t("studio.ui.preview")));
        preview.append(makeElement("code", "", studioModel().serializeTag(draft.spec)));
        root.append(preview);

        const actions = makeElement("div", "lora-studio-quick-tune-actions");
        actions.append(button(t("studio.ui.full_studio"), "lora-studio-quick-tune-full", () => openFullStudioFromQuickTune(tabname)));
        actions.append(button(t("studio.ui.apply"), "lora-studio-quick-tune-apply", () => applyDraft(tabname)));
        actions.append(button(t("studio.ui.cancel"), "lora-studio-quick-tune-cancel", () => discardDraft(tabname)));
        root.append(actions);
        root.addEventListener("keydown", (keyEvent) => {
            if (keyEvent.key !== "Escape") return;
            keyEvent.preventDefault();
            discardDraft(tabname);
            studioTextarea(tabname).focus();
        });

        // Keep the floating editor inside its tab: switching to img2img
        // hides txt2img's panel along with the tab, without discarding edits.
        const studio = studioRoot(tabname);
        (studio.parentElement || gradioApp()).append(root);
        studio.querySelector(".lora-studio-body").inert = true;
        state.quickTune = position;
        positionQuickTune(root, position.clientX, position.clientY);
        updateQuickTune(tabname);
        refreshCombinationSummary(tabname);
        const firstControl = root.querySelector('input[type="number"]');
        if (firstControl) firstControl.focus();
    }

    function mountStudio(tabname) {
        const textarea = studioTextarea(tabname);
        const promptContainer = gradioApp().querySelector(`#${tabname}_prompt_container`);
        if (!textarea || !promptContainer) return;
        const mountTarget = studioMountTarget(tabname, promptContainer);
        let root = studioRoot(tabname);
        let mounted = false;
        if (!root) {
            mounted = true;
            root = document.createElement("details");
            root.id = `${tabname}_lora_studio`;
            root.lang = window.loraStudioI18n ? window.loraStudioI18n.getLocale() : "en";
            root.className = "lora-studio";
            const summary = makeElement("summary", "", "");
            summary.append(makeElement("span", "lora-studio-summary-title", t("studio.ui.lora_studio")));
            summary.append(makeElement("span", "lora-studio-count"));
            summary.append(helpIcon("studio.root"));
            const i18n = window.loraStudioI18n;
            if (i18n) {
                const selector = document.createElement("select");
                selector.className = "lora-studio-language-picker";
                selector.setAttribute("aria-label", t("studio.ui.lora_studio_language"));
                selector.title = t("studio.ui.lora_studio_language");
                for (const [code, label] of Object.entries(i18n.supported)) {
                    const option = document.createElement("option");
                    option.value = code;
                    option.textContent = label;
                    selector.append(option);
                }
                selector.value = i18n.getLocale();
                selector.addEventListener("click", (event) => event.stopPropagation());
                selector.addEventListener("keydown", (event) => event.stopPropagation());
                selector.addEventListener("change", () => i18n.setLocale(selector.value));
                summary.append(selector, helpIcon("studio.language"));
            }
            root.append(summary, makeElement("div", "lora-studio-body"));
            root.addEventListener("toggle", () => {
                if (root.open) refreshStudio(tabname);
            });
            textarea.addEventListener("input", () => {
                if (quickTuneRoot(tabname)) updateQuickTune(tabname);
                refreshQuickPresetAvailability(tabname);
                if (!root.open) return;
                if (studioState(tabname).draft) {
                    setMessage(tabname, "studio.status.prompt_changed_while_this_draft_was", "error");
                    renderStudio(tabname);
                } else {
                    refreshStudio(tabname);
                }
            });
            // The old Weight Helper's most useful interaction was opening an
            // editor directly from a tag in the prompt.  Keep that shortcut,
            // but route it through Studio's non-destructive draft workflow:
            // right-click a LoRA tag at the text caret; right-click elsewhere
            // retains the browser's normal context menu.
            textarea.addEventListener("contextmenu", (event) => openQuickTuneForTagAtCaret(tabname, event));
        }
        // Move an instance made by an earlier Studio version as well as a
        // newly mounted one. `insertAdjacentElement` moves rather than clones.
        if (root.previousElementSibling !== mountTarget) {
            mountTarget.insertAdjacentElement("afterend", root);
        }
        applyInitialOpenState(tabname);
        if (mounted) refreshStudio(tabname);
    }

    window.loraStudioOpen = function (tabname, networkName, start, initialSpec) {
        if (!studioTabs.includes(tabname)) return;
        mountStudio(tabname);
        const root = studioRoot(tabname);
        const state = studioState(tabname);
        if (state.draft) {
            closeQuickTune(tabname);
            root.open = true;
            refreshStudio(tabname);
            setMessage(tabname, "studio.status.your_existing_draft_is_still_open", "info");
            return;
        }
        state.entries = studioModel().findTags(studioTextarea(tabname).value);
        const match = state.entries.find((entry) => entry.start === start && entry.spec.network_name === networkName) || state.entries.find((entry) => entry.spec.network_name === networkName);
        if (match) {
            state.selectedId = entryId(match);
            state.draft = null;
        } else if (initialSpec) {
            createNewDraft(tabname, networkName, clone(initialSpec));
            setMessage(tabname, "studio.status.new_lora_is_a_local_draft", "info");
        }
        root.open = true;
        refreshStudio(tabname);
    };

    function mountAllStudios() {
        if (!window.loraStudioModel) return;
        studioTabs.forEach(mountStudio);
    }

    function initializeStudios() {
        mountAllStudios();
        // ui.js, which declares the global opts object, is loaded after this
        // file. Register only once the UI lifecycle has begun, otherwise the
        // helper would dereference opts before its declaration exists.
        onOptionsAvailable(() => studioTabs.forEach((tabname) => {
            mountStudio(tabname);
            applyInitialOpenState(tabname);
        }));
    }

    if (window.addEventListener) window.addEventListener("lora-studio-locale-change", () => {
        studioTabs.forEach((tabname) => {
            const root = studioRoot(tabname);
            if (!root) return;
            const label = root.querySelector(".lora-studio-summary-title");
            if (label) label.textContent = t("studio.ui.lora_studio");
            const picker = root.querySelector(".lora-studio-language-picker");
            if (picker) {
                picker.value = window.loraStudioI18n.getLocale();
                picker.title = t("studio.ui.lora_studio_language");
                picker.setAttribute("aria-label", t("studio.ui.lora_studio_language"));
            }
            root.lang = window.loraStudioI18n.getLocale();
            const quick = quickTuneRoot(tabname);
            if (quick) quick.lang = root.lang;
            if (root.open) refreshStudio(tabname);
            if (quick) renderQuickTune(tabname);
            const state = studioState(tabname);
            if (state.messageKey) setMessage(tabname, state.messageKey, state.messageKind, state.messageVars);
        });
    });
    onUiLoaded(initializeStudios);
    onAfterUiUpdate(mountAllStudios);
}());
