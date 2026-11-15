function toggleCss(key, css, enable) {
    var style = document.getElementById(key);
    if (enable && !style) {
        style = document.createElement('style');
        style.id = key;
        style.type = 'text/css';
        document.head.appendChild(style);
    }
    if (style && !enable) {
        document.head.removeChild(style);
    }
    if (style) {
        style.innerHTML == '';
        style.appendChild(document.createTextNode(css));
    }
}

function setupExtraNetworksForTab(tabname) {
    function registerPrompt(tabname, id) {
        var textarea = gradioApp().querySelector("#" + id + " > label > textarea");

        if (!activePromptTextarea[tabname]) {
            activePromptTextarea[tabname] = textarea;
        }

        textarea.addEventListener("focus", function() {
            activePromptTextarea[tabname] = textarea;
        });
    }

    var tabnav = gradioApp().querySelector('#' + tabname + '_extra_tabs > div.tab-nav');
    var controlsDiv = document.createElement('DIV');
    controlsDiv.classList.add('extra-networks-controls-div');
    tabnav.appendChild(controlsDiv);
    tabnav.insertBefore(controlsDiv, null);

    var this_tab = gradioApp().querySelector('#' + tabname + '_extra_tabs');
    this_tab.querySelectorAll(":scope > [id^='" + tabname + "_']").forEach(function(elem) {
        // tabname_full = {tabname}_{extra_networks_tabname}
        var tabname_full = elem.id;
        var search = gradioApp().querySelector("#" + tabname_full + "_extra_search");
        var sort_dir = gradioApp().querySelector("#" + tabname_full + "_extra_sort_dir");
        var refresh = gradioApp().querySelector("#" + tabname_full + "_extra_refresh");
        var currentSort = '';

        // If any of the buttons above don't exist, we want to skip this iteration of the loop.
        if (!search || !sort_dir || !refresh) {
            return; // `return` is equivalent of `continue` but for forEach loops.
        }

        var applyFilter = function(force) {
            var searchTerm = search.value.toLowerCase();
            gradioApp().querySelectorAll('#' + tabname + '_extra_tabs div.card').forEach(function(elem) {
                var searchOnly = elem.querySelector('.search_only');
                var text = Array.prototype.map.call(elem.querySelectorAll('.search_terms, .description'), function(t) {
                    return t.textContent.toLowerCase();
                }).join(" ");

                var visible = text.indexOf(searchTerm) != -1;
                if (searchOnly && searchTerm.length < 4) {
                    visible = false;
                }
                if (visible) {
                    elem.classList.remove("hidden");
                } else {
                    elem.classList.add("hidden");
                }
            });

            applySort(force);
        };

        var applySort = function(force) {
            var cards = gradioApp().querySelectorAll('#' + tabname_full + ' div.card');
            var parent = gradioApp().querySelector('#' + tabname_full + "_cards");
            var reverse = sort_dir.dataset.sortdir == "Descending";
            var activeSearchElem = gradioApp().querySelector('#' + tabname_full + "_controls .extra-network-control--sort.extra-network-control--enabled");
            var sortKey = activeSearchElem ? activeSearchElem.dataset.sortkey : "default";
            var sortKeyDataField = "sort" + sortKey.charAt(0).toUpperCase() + sortKey.slice(1);
            var sortKeyStore = sortKey + "-" + sort_dir.dataset.sortdir + "-" + cards.length;

            if (sortKeyStore == currentSort && !force) {
                return;
            }
            currentSort = sortKeyStore;

            var sortedCards = Array.from(cards);
            sortedCards.sort(function(cardA, cardB) {
                var a = cardA.dataset[sortKeyDataField];
                var b = cardB.dataset[sortKeyDataField];
                if (!isNaN(a) && !isNaN(b)) {
                    return parseInt(a) - parseInt(b);
                }

                return (a < b ? -1 : (a > b ? 1 : 0));
            });

            if (reverse) {
                sortedCards.reverse();
            }

            parent.innerHTML = '';

            var frag = document.createDocumentFragment();
            sortedCards.forEach(function(card) {
                frag.appendChild(card);
            });
            parent.appendChild(frag);
        };

        search.addEventListener("input", function() {
            applyFilter();
        });
        applySort();
        applyFilter();
        extraNetworksApplySort[tabname_full] = applySort;
        extraNetworksApplyFilter[tabname_full] = applyFilter;

        var controls = gradioApp().querySelector("#" + tabname_full + "_controls");
        controlsDiv.insertBefore(controls, null);

        if (elem.style.display != "none") {
            extraNetworksShowControlsForPage(tabname, tabname_full);
        }
    });

    registerPrompt(tabname, tabname + "_prompt");
    registerPrompt(tabname, tabname + "_neg_prompt");
}

function extraNetworksMovePromptToTab(tabname, id, showPrompt, showNegativePrompt) {
    if (!gradioApp().querySelector('.toprow-compact-tools')) return; // only applicable for compact prompt layout

    var promptContainer = gradioApp().getElementById(tabname + '_prompt_container');
    var prompt = gradioApp().getElementById(tabname + '_prompt_row');
    var negPrompt = gradioApp().getElementById(tabname + '_neg_prompt_row');
    var elem = id ? gradioApp().getElementById(id) : null;

    if (showNegativePrompt && elem) {
        elem.insertBefore(negPrompt, elem.firstChild);
    } else {
        promptContainer.insertBefore(negPrompt, promptContainer.firstChild);
    }

    if (showPrompt && elem) {
        elem.insertBefore(prompt, elem.firstChild);
    } else {
        promptContainer.insertBefore(prompt, promptContainer.firstChild);
    }

    if (elem) {
        elem.classList.toggle('extra-page-prompts-active', showNegativePrompt || showPrompt);
    }
}


function extraNetworksShowControlsForPage(tabname, tabname_full) {
    gradioApp().querySelectorAll('#' + tabname + '_extra_tabs .extra-networks-controls-div > div').forEach(function(elem) {
        var targetId = tabname_full + "_controls";
        elem.style.display = elem.id == targetId ? "" : "none";
    });
}


function extraNetworksUnrelatedTabSelected(tabname) { // called from python when user selects an unrelated tab (generate)
    extraNetworksMovePromptToTab(tabname, '', false, false);

    extraNetworksShowControlsForPage(tabname, null);
}

function extraNetworksTabSelected(tabname, id, showPrompt, showNegativePrompt, tabname_full) { // called from python when user selects an extra networks tab
    extraNetworksMovePromptToTab(tabname, id, showPrompt, showNegativePrompt);

    extraNetworksShowControlsForPage(tabname, tabname_full);
}

function applyExtraNetworkFilter(tabname_full) {
    var doFilter = function() {
        var applyFunction = extraNetworksApplyFilter[tabname_full];

        if (applyFunction) {
            applyFunction(true);
        }
    };
    setTimeout(doFilter, 1);
}

function applyExtraNetworkSort(tabname_full) {
    var doSort = function() {
        extraNetworksApplySort[tabname_full](true);
    };
    setTimeout(doSort, 1);
}

var extraNetworksApplyFilter = {};
var extraNetworksApplySort = {};
var activePromptTextarea = {};

function setupExtraNetworks() {
    setupExtraNetworksForTab('txt2img');
    setupExtraNetworksForTab('img2img');
}

// The arguments after an extra-network name may be positional strengths or
// named values such as ``te=`` / ``unet=`` / ``dyn=``. Keep the first two
// components for identity, but treat everything until ``>`` as its settings.
var re_extranet = /^<([^:^>]+:[^:>]+):[^>]*>(.*)$/s;
var re_extranet_g = /<([^:^>]+:[^:>]+):[^>]*>/g;

var re_extranet_neg = /\(([^:^>]+:[\d.]+)\)/;
var re_extranet_g_neg = /\(([^:^>]+:[\d.]+)\)/g;
function tryToRemoveExtraNetworkFromPrompt(textarea, text, isNeg) {
    var m = text.match(isNeg ? re_extranet_neg : re_extranet);
    var replaced = false;
    var newTextareaText;
    var extraTextBeforeNet = opts.extra_networks_add_text_separator;
    if (m) {
        var extraTextAfterNet = m[2];
        var partToSearch = m[1];
        var foundAtPosition = -1;
        newTextareaText = textarea.value.replaceAll(isNeg ? re_extranet_g_neg : re_extranet_g, function(found, net, pos) {
            m = found.match(isNeg ? re_extranet_neg : re_extranet);
            if (m[1] == partToSearch) {
                replaced = true;
                foundAtPosition = pos;
                return "";
            }
            return found;
        });
        if (foundAtPosition >= 0) {
            if (extraTextAfterNet && newTextareaText.substr(foundAtPosition, extraTextAfterNet.length) == extraTextAfterNet) {
                newTextareaText = newTextareaText.substr(0, foundAtPosition) + newTextareaText.substr(foundAtPosition + extraTextAfterNet.length);
            }
            if (newTextareaText.substr(foundAtPosition - extraTextBeforeNet.length, extraTextBeforeNet.length) == extraTextBeforeNet) {
                newTextareaText = newTextareaText.substr(0, foundAtPosition - extraTextBeforeNet.length) + newTextareaText.substr(foundAtPosition);
            }
        }
    } else {
        newTextareaText = textarea.value.replaceAll(new RegExp(`((?:${extraTextBeforeNet})?${text})`, "g"), "");
        replaced = (newTextareaText != textarea.value);
    }

    if (replaced) {
        textarea.value = newTextareaText;
        return true;
    }

    return false;
}

function updatePromptArea(text, textArea, isNeg) {
    if (!tryToRemoveExtraNetworkFromPrompt(textArea, text, isNeg)) {
        textArea.value = textArea.value + opts.extra_networks_add_text_separator + text;
    }

    updateInput(textArea);
}

function loraFiniteNumber(value, fallback) {
    var parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
}

function loraFormatNumber(value) {
    return String(Math.round(loraFiniteNumber(value, 0) * 1000) / 1000);
}

function loraSetRangeValue(range, value) {
    var numeric = Number(value);
    if (!Number.isFinite(numeric)) return;
    if (numeric < Number(range.min)) range.min = String(numeric);
    if (numeric > Number(range.max)) range.max = String(numeric);
    range.value = String(numeric);
}

function loraModel() {
    if (!window.loraStudioModel) {
        throw new Error("LoRA Studio prompt model is not loaded");
    }
    return window.loraStudioModel;
}

function loraWeightInitial(weight, fallback) {
    if (!weight) {
        return fallback;
    }
    return weight.kind === "curve" ? weight.points[0].value : weight.value;
}

function loraWeightInput(weight) {
    return weight && weight.kind === "curve" ? loraModel().serializeWeight(weight) : "";
}

function loraReadSettings(spec) {
    var defaultWeight = loraFiniteNumber(opts.extra_networks_default_multiplier, 1);
    var result = {
        te: defaultWeight, unet: defaultWeight, teCurve: "", unetCurve: "",
        hrt: "", hru: "", start: "", stop: "", lbw: "", lbwe: "", dyn: "", spec: spec, span: null
    };

    result.teCurve = loraWeightInput(spec.weights.te);
    result.unetCurve = loraWeightInput(spec.weights.unet);
    result.te = loraWeightInitial(spec.weights.te, defaultWeight);
    result.unet = loraWeightInitial(spec.weights.unet, result.te);
    result.dyn = spec.dyn_dim === null ? "" : String(spec.dyn_dim);
    result.hrt = spec.hires.te ? loraModel().serializeWeight(spec.hires.te) : "";
    result.hru = spec.hires.unet ? loraModel().serializeWeight(spec.hires.unet) : "";
    result.start = spec.active_range.start === null ? "" : String(spec.active_range.start);
    result.stop = spec.active_range.stop === null ? "" : String(spec.active_range.stop);
    result.lbw = spec.block_weights ? spec.block_weights.raw : "";
    result.lbwe = spec.elemental_weights ? spec.elemental_weights.raw : "";

    return result;
}

function loraOptionalFinite(value, description) {
    var text = String(value || "").trim();
    if (!text) {
        return null;
    }
    var parsed = Number(text);
    if (!Number.isFinite(parsed)) {
        throw new Error(description + " must be a finite number");
    }
    return parsed;
}

function loraSpecFromSettings(settings) {
    var model = loraModel();
    var spec = JSON.parse(JSON.stringify(settings.spec));
    var teRaw = String(settings.teCurve || "").trim() || String(settings.te);
    var unetRaw = String(settings.unetCurve || "").trim() || String(settings.unet);
    if (!teRaw.trim() || !unetRaw.trim()) {
        throw new Error("Complete both TE and U-Net weights; an empty field is not zero");
    }
    spec.weights.te = model.parseWeight(teRaw, settings.te);
    spec.weights.unet = model.parseWeight(unetRaw, settings.unet);

    spec.dyn_dim = model.parseDynamicRank(settings.dyn, true);

    var hrt = String(settings.hrt || "").trim();
    var hru = String(settings.hru || "").trim();
    spec.hires.te = hrt ? model.parseWeight(hrt, loraWeightInitial(spec.weights.te, settings.te)) : null;
    spec.hires.unet = hru ? model.parseWeight(hru, loraWeightInitial(spec.weights.unet, settings.unet)) : null;
    spec.active_range.start = loraOptionalFinite(settings.start, "LoRA start");
    spec.active_range.stop = loraOptionalFinite(settings.stop, "LoRA stop");
    if (spec.active_range.start !== null && spec.active_range.stop !== null && spec.active_range.start > spec.active_range.stop) {
        throw new Error("LoRA start must not be greater than stop");
    }
    var lbw = String(settings.lbw || "").trim();
    spec.block_weights = lbw ? {raw: lbw} : null;
    var lbwe = String(settings.lbwe || "").trim();
    spec.elemental_weights = lbwe ? {raw: lbwe} : null;
    return spec;
}

function loraSetSettings(textarea, settings, source) {
    var popupRoot = source && source.closest(".lora-tuning-popup");
    var status = popupRoot && popupRoot.querySelector(".lora-tuning-status");
    try {
        var model = loraModel();
        var spec = loraSpecFromSettings(settings);
        var tag = model.serializeTag(spec);
        if (settings.span) {
            textarea.value = model.replaceTagSpan(textarea.value, settings.span, tag);
            settings.span = {start: settings.span.start, end: settings.span.start + tag.length, tag: tag, spec: spec};
        } else {
            var separator = opts.extra_networks_add_text_separator;
            var start = textarea.value.length + separator.length;
            textarea.value += separator + tag;
            settings.span = {start: start, end: start + tag.length, tag: tag, spec: spec};
        }
        settings.spec = spec;
        if (source) {
            source.setCustomValidity("");
        }
        updateInput(textarea);
        if (status) status.textContent = loraStudioText("studio.status.prompt_updated_te_u_net_strengths");
        return true;
    } catch (error) {
        if (source) {
            source.setCustomValidity(error.message);
        }
        if (status) status.textContent = error.message + ". The prompt has not changed.";
        console.warn("LoRA tag was not updated:", error);
        return false;
    }
}

function loraStudioText(text) {
    return window.loraStudioI18n ? window.loraStudioI18n.t(text) : text;
}

function loraShowTuning(event) {
    event.preventDefault();
    event.stopPropagation();

    var card = event.target.closest(".card, .tree-list-content");
    var page = card && card.closest(".extra-page");
    if (!card || !page || !page.id.endsWith("_lora")) {
        return;
    }

    var tabname = page.id.slice(0, -"_lora".length);
    var textarea = gradioApp().querySelector("#" + tabname + "_prompt > label > textarea");
    var name = card.dataset.promptName;
    if (!textarea || !name) {
        return;
    }

    var model;
    try {
        model = loraModel();
    } catch (error) {
        console.warn("LoRA tuning popup is unavailable:", error);
        return;
    }
    var selected = model.findTags(textarea.value).find(function(entry) {
        return entry.spec.network_name === name;
    });
    var settings = loraReadSettings(selected ? selected.spec : model.createDefaultSpec(name, opts.extra_networks_default_multiplier));
    settings.span = selected || null;
    var root = document.createElement("div");
    root.classList.add("lora-tuning-popup");

    var title = document.createElement("h2");
    title.textContent = "LoRA: " + name;
    root.appendChild(title);

    var openStudio = document.createElement("button");
    openStudio.type = "button";
    openStudio.classList.add("lora-tuning-open-studio");
    openStudio.textContent = loraStudioText("studio.ui.open_in_studio");
    openStudio.addEventListener("click", function() {
        if (typeof window.loraStudioOpen !== "function") {
            console.warn("LoRA Studio is not loaded yet");
            return;
        }
        window.loraStudioOpen(tabname, name, settings.span && settings.span.start, settings.spec);
        closePopup();
    });
    root.appendChild(openStudio);
    var status = document.createElement("p");
    status.className = "lora-tuning-status";
    status.setAttribute("aria-live", "polite");
    status.textContent = loraStudioText("studio.help.changes_here_update_the_prompt_immediately");
    root.appendChild(status);

    var strengthControls = {};

    function addStrengthControl(labelText, key) {
        var label = document.createElement("label");
        var caption = document.createElement("span");
        caption.textContent = labelText;
        var number = document.createElement("input");
        number.type = "number";
        number.step = "any";
        number.required = true;
        number.value = loraFormatNumber(settings[key]);
        var slider = document.createElement("input");
        slider.type = "range";
        slider.min = "-2";
        slider.max = "2";
        slider.step = "0.01";
        loraSetRangeValue(slider, number.value);
        var apply = function(source) {
            settings[key + "Curve"] = "";
            settings[key] = number.value;
            if (loraSetSettings(textarea, settings, source)) number.setCustomValidity("");
        };
        number.addEventListener("input", function() {
            loraSetRangeValue(slider, number.value);
            apply(number);
        });
        slider.addEventListener("input", function() {
            number.value = slider.value;
            apply(slider);
        });
        label.append(caption, number, slider);
        root.appendChild(label);
        strengthControls[key] = {number: number, slider: slider};
        var disabled = Boolean(settings[key + "Curve"]);
        number.disabled = disabled;
        slider.disabled = disabled;
    }

    function addTextControl(labelText, key, placeholder, callback) {
        var label = document.createElement("label");
        label.classList.add("lora-tuning-text");
        var caption = document.createElement("span");
        caption.textContent = labelText;
        var input = document.createElement("input");
        input.type = "text";
        input.placeholder = placeholder || "";
        input.value = settings[key] || "";
        input.addEventListener("input", function() {
            settings[key] = input.value.trim();
            if (callback) {
                callback(input.value);
            }
            loraSetSettings(textarea, settings, input);
        });
        label.append(caption, input);
        root.appendChild(label);
        return input;
    }

    addStrengthControl("Text encoder strength", "te");
    addStrengthControl("UNet strength", "unet");
    var teCurve = addTextControl("Text encoder curve", "teCurve", "0@0, 1@1", function(value) {
        var control = strengthControls.te;
        control.number.disabled = Boolean(value.trim());
        control.slider.disabled = Boolean(value.trim());
    });
    var unetCurve = addTextControl("UNet curve", "unetCurve", "0@0, 1@1", function(value) {
        var control = strengthControls.unet;
        control.number.disabled = Boolean(value.trim());
        control.slider.disabled = Boolean(value.trim());
    });

    var dynLabel = document.createElement("label");
    var dynCaption = document.createElement("span");
    dynCaption.textContent = "Dynamic rank (empty = off)";
    var dyn = document.createElement("input");
    dyn.type = "number";
    dyn.min = "1";
    dyn.step = "1";
    dyn.placeholder = "Off / full rank";
    dyn.title = "Limits rank in supported LoRA layers, not strength. Some adapter types ignore DYN.";
    dyn.value = settings.dyn;
    dyn.addEventListener("input", function() {
        settings.dyn = dyn.validity.badInput ? "incomplete number" : dyn.value;
        loraSetSettings(textarea, settings, dyn);
    });
    dynLabel.append(dynCaption, dyn);
    root.appendChild(dynLabel);

    var hrt = addTextControl("Hires text encoder", "hrt", "Inherit TE; or weight / curve");
    var hru = addTextControl("Hires UNet", "hru", "Inherit U-Net; or weight / curve");
    var start = addTextControl("Start step", "start", "No limit; or 0.2 / 4");
    var stop = addTextControl("Stop step", "stop", "No limit; or 0.8 / 18");
    var lbw = addTextControl("Block weights / preset", "lbw", "All blocks; or OUTS / exact values");
    var lbwe = addTextControl("Elemental module preset", "lbwe", "No override; or ATTNDEEPOFF");

    var reset = document.createElement("button");
    reset.classList.add("lora-tuning-reset");
    reset.textContent = "Reset to default";
    reset.addEventListener("click", function() {
        var defaultWeight = loraFiniteNumber(opts.extra_networks_default_multiplier, 1);
        settings.te = defaultWeight;
        settings.unet = defaultWeight;
        settings.teCurve = "";
        settings.unetCurve = "";
        settings.dyn = "";
        settings.hrt = "";
        settings.hru = "";
        settings.start = "";
        settings.stop = "";
        settings.lbw = "";
        settings.lbwe = "";
        strengthControls.te.number.value = loraFormatNumber(settings.te);
        strengthControls.unet.number.value = loraFormatNumber(settings.unet);
        loraSetRangeValue(strengthControls.te.slider, settings.te);
        loraSetRangeValue(strengthControls.unet.slider, settings.unet);
        strengthControls.te.number.disabled = false;
        strengthControls.te.slider.disabled = false;
        strengthControls.unet.number.disabled = false;
        strengthControls.unet.slider.disabled = false;
        teCurve.value = "";
        unetCurve.value = "";
        dyn.value = "";
        hrt.value = "";
        hru.value = "";
        start.value = "";
        stop.value = "";
        lbw.value = "";
        lbwe.value = "";
        root.querySelectorAll("input").forEach(function(input) { input.setCustomValidity(""); });
        loraSetSettings(textarea, settings, reset);
    });
    root.appendChild(reset);
    popup(root);
}

function cardClicked(tabname, textToAdd, textToAddNegative, allowNegativePrompt) {
    if (textToAddNegative.length > 0) {
        updatePromptArea(textToAdd, gradioApp().querySelector("#" + tabname + "_prompt > label > textarea"));
        updatePromptArea(textToAddNegative, gradioApp().querySelector("#" + tabname + "_neg_prompt > label > textarea"), true);
    } else {
        var textarea = allowNegativePrompt ? activePromptTextarea[tabname] : gradioApp().querySelector("#" + tabname + "_prompt > label > textarea");
        updatePromptArea(textToAdd, textarea);
    }
}

function saveCardPreview(event, tabname, filename) {
    var textarea = gradioApp().querySelector("#" + tabname + '_preview_filename  > label > textarea');
    var button = gradioApp().getElementById(tabname + '_save_preview');

    textarea.value = filename;
    updateInput(textarea);

    button.click();

    event.stopPropagation();
    event.preventDefault();
}

function extraNetworksSearchButton(tabname, extra_networks_tabname, event) {
    var searchTextarea = gradioApp().querySelector("#" + tabname + "_" + extra_networks_tabname + "_extra_search");
    var button = event.target;
    var text = button.classList.contains("search-all") ? "" : button.textContent.trim();

    searchTextarea.value = text;
    updateInput(searchTextarea);
}

function extraNetworksTreeProcessFileClick(event, btn, tabname, extra_networks_tabname) {
    /**
     * Processes `onclick` events when user clicks on files in tree.
     *
     * @param event                     The generated event.
     * @param btn                       The clicked `tree-list-item` button.
     * @param tabname                   The name of the active tab in the sd webui. Ex: txt2img, img2img, etc.
     * @param extra_networks_tabname    The id of the active extraNetworks tab. Ex: lora, checkpoints, etc.
     */
    // NOTE: Currently unused.
    return;
}

function extraNetworksTreeProcessDirectoryClick(event, btn, tabname, extra_networks_tabname) {
    /**
     * Processes `onclick` events when user clicks on directories in tree.
     *
     * Here is how the tree reacts to clicks for various states:
     * unselected unopened directory: Directory is selected and expanded.
     * unselected opened directory: Directory is selected.
     * selected opened directory: Directory is collapsed and deselected.
     * chevron is clicked: Directory is expanded or collapsed. Selected state unchanged.
     *
     * @param event                     The generated event.
     * @param btn                       The clicked `tree-list-item` button.
     * @param tabname                   The name of the active tab in the sd webui. Ex: txt2img, img2img, etc.
     * @param extra_networks_tabname    The id of the active extraNetworks tab. Ex: lora, checkpoints, etc.
     */
    var ul = btn.nextElementSibling;
    // This is the actual target that the user clicked on within the target button.
    // We use this to detect if the chevron was clicked.
    var true_targ = event.target;

    function _expand_or_collapse(_ul, _btn) {
        // Expands <ul> if it is collapsed, collapses otherwise. Updates button attributes.
        if (_ul.hasAttribute("hidden")) {
            _ul.removeAttribute("hidden");
            _btn.dataset.expanded = "";
        } else {
            _ul.setAttribute("hidden", "");
            delete _btn.dataset.expanded;
        }
    }

    function _remove_selected_from_all() {
        // Removes the `selected` attribute from all buttons.
        var sels = document.querySelectorAll("div.tree-list-content");
        [...sels].forEach(el => {
            delete el.dataset.selected;
        });
    }

    function _select_button(_btn) {
        // Removes `data-selected` attribute from all buttons then adds to passed button.
        _remove_selected_from_all();
        _btn.dataset.selected = "";
    }

    function _update_search(_tabname, _extra_networks_tabname, _search_text) {
        // Update search input with select button's path.
        var search_input_elem = gradioApp().querySelector("#" + tabname + "_" + extra_networks_tabname + "_extra_search");
        search_input_elem.value = _search_text;
        updateInput(search_input_elem);
    }


    // If user clicks on the chevron, then we do not select the folder.
    if (true_targ.matches(".tree-list-item-action--leading, .tree-list-item-action-chevron")) {
        _expand_or_collapse(ul, btn);
    } else {
        // User clicked anywhere else on the button.
        if ("selected" in btn.dataset && !(ul.hasAttribute("hidden"))) {
            // If folder is select and open, collapse and deselect button.
            _expand_or_collapse(ul, btn);
            delete btn.dataset.selected;
            _update_search(tabname, extra_networks_tabname, "");
        } else if (!(!("selected" in btn.dataset) && !(ul.hasAttribute("hidden")))) {
            // If folder is open and not selected, then we don't collapse; just select.
            // NOTE: Double inversion sucks but it is the clearest way to show the branching here.
            _expand_or_collapse(ul, btn);
            _select_button(btn, tabname, extra_networks_tabname);
            _update_search(tabname, extra_networks_tabname, btn.dataset.path);
        } else {
            // All other cases, just select the button.
            _select_button(btn, tabname, extra_networks_tabname);
            _update_search(tabname, extra_networks_tabname, btn.dataset.path);
        }
    }
}

function extraNetworksTreeOnClick(event, tabname, extra_networks_tabname) {
    /**
     * Handles `onclick` events for buttons within an `extra-network-tree .tree-list--tree`.
     *
     * Determines whether the clicked button in the tree is for a file entry or a directory
     * then calls the appropriate function.
     *
     * @param event                     The generated event.
     * @param tabname                   The name of the active tab in the sd webui. Ex: txt2img, img2img, etc.
     * @param extra_networks_tabname    The id of the active extraNetworks tab. Ex: lora, checkpoints, etc.
     */
    var btn = event.currentTarget;
    var par = btn.parentElement;
    if (par.dataset.treeEntryType === "file") {
        extraNetworksTreeProcessFileClick(event, btn, tabname, extra_networks_tabname);
    } else {
        extraNetworksTreeProcessDirectoryClick(event, btn, tabname, extra_networks_tabname);
    }
}

function extraNetworksControlSortOnClick(event, tabname, extra_networks_tabname) {
    /** Handles `onclick` events for Sort Mode buttons. */

    var self = event.currentTarget;
    var parent = event.currentTarget.parentElement;

    parent.querySelectorAll('.extra-network-control--sort').forEach(function(x) {
        x.classList.remove('extra-network-control--enabled');
    });

    self.classList.add('extra-network-control--enabled');

    applyExtraNetworkSort(tabname + "_" + extra_networks_tabname);
}

function extraNetworksControlSortDirOnClick(event, tabname, extra_networks_tabname) {
    /**
     * Handles `onclick` events for the Sort Direction button.
     *
     * Modifies the data attributes of the Sort Direction button to cycle between
     * ascending and descending sort directions.
     *
     * @param event                     The generated event.
     * @param tabname                   The name of the active tab in the sd webui. Ex: txt2img, img2img, etc.
     * @param extra_networks_tabname    The id of the active extraNetworks tab. Ex: lora, checkpoints, etc.
     */
    if (event.currentTarget.dataset.sortdir == "Ascending") {
        event.currentTarget.dataset.sortdir = "Descending";
        event.currentTarget.setAttribute("title", "Sort descending");
    } else {
        event.currentTarget.dataset.sortdir = "Ascending";
        event.currentTarget.setAttribute("title", "Sort ascending");
    }
    applyExtraNetworkSort(tabname + "_" + extra_networks_tabname);
}

function extraNetworksControlTreeViewOnClick(event, tabname, extra_networks_tabname) {
    /**
     * Handles `onclick` events for the Tree View button.
     *
     * Toggles the tree view in the extra networks pane.
     *
     * @param event                     The generated event.
     * @param tabname                   The name of the active tab in the sd webui. Ex: txt2img, img2img, etc.
     * @param extra_networks_tabname    The id of the active extraNetworks tab. Ex: lora, checkpoints, etc.
     */
    var button = event.currentTarget;
    button.classList.toggle("extra-network-control--enabled");
    var show = !button.classList.contains("extra-network-control--enabled");

    var pane = gradioApp().getElementById(tabname + "_" + extra_networks_tabname + "_pane");
    pane.classList.toggle("extra-network-dirs-hidden", show);
}

function extraNetworksControlRefreshOnClick(event, tabname, extra_networks_tabname) {
    /**
     * Handles `onclick` events for the Refresh Page button.
     *
     * In order to actually call the python functions in `ui_extra_networks.py`
     * to refresh the page, we created an empty gradio button in that file with an
     * event handler that refreshes the page. So what this function here does
     * is it manually raises a `click` event on that button.
     *
     * @param event                     The generated event.
     * @param tabname                   The name of the active tab in the sd webui. Ex: txt2img, img2img, etc.
     * @param extra_networks_tabname    The id of the active extraNetworks tab. Ex: lora, checkpoints, etc.
     */
    var btn_refresh_internal = gradioApp().getElementById(tabname + "_" + extra_networks_tabname + "_extra_refresh_internal");
    btn_refresh_internal.dispatchEvent(new Event("click"));
}

var globalPopup = null;
var globalPopupInner = null;

function closePopup() {
    if (!globalPopup) return;
    globalPopup.style.display = "none";
}

function popup(contents) {
    if (!globalPopup) {
        globalPopup = document.createElement('div');
        globalPopup.classList.add('global-popup');

        var close = document.createElement('div');
        close.classList.add('global-popup-close');
        close.addEventListener("click", closePopup);
        close.title = "Close";
        globalPopup.appendChild(close);

        globalPopupInner = document.createElement('div');
        globalPopupInner.classList.add('global-popup-inner');
        globalPopup.appendChild(globalPopupInner);

        gradioApp().querySelector('.main').appendChild(globalPopup);
    }

    globalPopupInner.innerHTML = '';
    globalPopupInner.appendChild(contents);

    globalPopup.style.display = "flex";
}

var storedPopupIds = {};
function popupId(id) {
    if (!storedPopupIds[id]) {
        storedPopupIds[id] = gradioApp().getElementById(id);
    }

    popup(storedPopupIds[id]);
}

function extraNetworksFlattenMetadata(obj) {
    const result = {};

    // Convert any stringified JSON objects to actual objects
    for (const key of Object.keys(obj)) {
        if (typeof obj[key] === 'string') {
            try {
                const parsed = JSON.parse(obj[key]);
                if (parsed && typeof parsed === 'object') {
                    obj[key] = parsed;
                }
            } catch (error) {
                continue;
            }
        }
    }

    // Flatten the object
    for (const key of Object.keys(obj)) {
        if (typeof obj[key] === 'object' && obj[key] !== null) {
            const nested = extraNetworksFlattenMetadata(obj[key]);
            for (const nestedKey of Object.keys(nested)) {
                result[`${key}/${nestedKey}`] = nested[nestedKey];
            }
        } else {
            result[key] = obj[key];
        }
    }

    // Special case for handling modelspec keys
    for (const key of Object.keys(result)) {
        if (key.startsWith("modelspec.")) {
            result[key.replaceAll(".", "/")] = result[key];
            delete result[key];
        }
    }

    // Add empty keys to designate hierarchy
    for (const key of Object.keys(result)) {
        const parts = key.split("/");
        for (let i = 1; i < parts.length; i++) {
            const parent = parts.slice(0, i).join("/");
            if (!result[parent]) {
                result[parent] = "";
            }
        }
    }

    return result;
}

function extraNetworksShowMetadata(text) {
    try {
        let parsed = JSON.parse(text);
        if (parsed && typeof parsed === 'object') {
            parsed = extraNetworksFlattenMetadata(parsed);
            const table = createVisualizationTable(parsed, 0);
            popup(table);
            return;
        }
    } catch (error) {
        console.error(error);
    }

    var elem = document.createElement('pre');
    elem.classList.add('popup-metadata');
    elem.textContent = text;

    popup(elem);
    return;
}

function requestGet(url, data, handler, errorHandler) {
    var xhr = new XMLHttpRequest();
    var args = Object.keys(data).map(function(k) {
        return encodeURIComponent(k) + '=' + encodeURIComponent(data[k]);
    }).join('&');
    xhr.open("GET", url + "?" + args, true);

    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                try {
                    var js = JSON.parse(xhr.responseText);
                    handler(js);
                } catch (error) {
                    console.error(error);
                    errorHandler();
                }
            } else {
                errorHandler();
            }
        }
    };
    var js = JSON.stringify(data);
    xhr.send(js);
}

function extraNetworksCopyCardPath(event) {
    navigator.clipboard.writeText(event.target.getAttribute("data-clipboard-text"));
    event.stopPropagation();
}

function extraNetworksRequestMetadata(event, extraPage) {
    var showError = function() {
        extraNetworksShowMetadata("there was an error getting metadata");
    };

    var cardName = event.target.parentElement.parentElement.getAttribute("data-name");

    requestGet("./sd_extra_networks/metadata", {page: extraPage, item: cardName}, function(data) {
        if (data && data.metadata) {
            extraNetworksShowMetadata(data.metadata);
        } else {
            showError();
        }
    }, showError);

    event.stopPropagation();
}

var extraPageUserMetadataEditors = {};

function extraNetworksEditUserMetadata(event, tabname, extraPage) {
    var id = tabname + '_' + extraPage + '_edit_user_metadata';

    var editor = extraPageUserMetadataEditors[id];
    if (!editor) {
        editor = {};
        editor.page = gradioApp().getElementById(id);
        editor.nameTextarea = gradioApp().querySelector("#" + id + "_name" + ' textarea');
        editor.button = gradioApp().querySelector("#" + id + "_button");
        extraPageUserMetadataEditors[id] = editor;
    }

    var cardName = event.target.parentElement.parentElement.getAttribute("data-name");
    editor.nameTextarea.value = cardName;
    updateInput(editor.nameTextarea);

    editor.button.click();

    popup(editor.page);

    event.stopPropagation();
}

function extraNetworksRefreshSingleCard(page, tabname, name) {
    requestGet("./sd_extra_networks/get-single-card", {page: page, tabname: tabname, name: name}, function(data) {
        if (data && data.html) {
            var card = gradioApp().querySelector(`#${tabname}_${page.replace(" ", "_")}_cards > .card[data-name="${name}"]`);

            var newDiv = document.createElement('DIV');
            newDiv.innerHTML = data.html;
            var newCard = newDiv.firstElementChild;

            newCard.style.display = '';
            card.parentElement.insertBefore(newCard, card);
            card.parentElement.removeChild(card);
        }
    });
}

window.addEventListener("keydown", function(event) {
    if (event.key == "Escape") {
        closePopup();
    }
});

/**
 * Setup custom loading for this script.
 * We need to wait for all of our HTML to be generated in the extra networks tabs
 * before we can actually run the `setupExtraNetworks` function.
 * The `onUiLoaded` function actually runs before all of our extra network tabs are
 * finished generating. Thus we needed this new method.
 *
 */

var uiAfterScriptsCallbacks = [];
var uiAfterScriptsTimeout = null;
var executedAfterScripts = false;

function scheduleAfterScriptsCallbacks() {
    clearTimeout(uiAfterScriptsTimeout);
    uiAfterScriptsTimeout = setTimeout(function() {
        executeCallbacks(uiAfterScriptsCallbacks);
    }, 200);
}

onUiLoaded(function() {
    var mutationObserver = new MutationObserver(function(m) {
        let existingSearchfields = gradioApp().querySelectorAll("[id$='_extra_search']").length;
        let neededSearchfields = gradioApp().querySelectorAll("[id$='_extra_tabs'] > .tab-nav > button").length - 2;

        if (!executedAfterScripts && existingSearchfields >= neededSearchfields) {
            mutationObserver.disconnect();
            executedAfterScripts = true;
            scheduleAfterScriptsCallbacks();
        }
    });
    mutationObserver.observe(gradioApp(), {childList: true, subtree: true});
});

uiAfterScriptsCallbacks.push(setupExtraNetworks);
