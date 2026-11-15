/* Contextual LoRA Studio Help. Read-only viewer; Markdown is rendered as text/DOM,
 * never injected as HTML. Python registry scans help_documentation at request time.
 */
(function (root) {
    "use strict";
    const state = {overlay: null, dialog: null, list: null, article: null, input: null,
                   topics: [], topic: null, request: 0, previousFocus: null};
    function tr(key, vars) {
        const api = root.loraStudioI18n;
        return api ? api.t("studio.help_registry." + key, vars) : key;
    }
    function language() { return (root.loraStudioI18n && root.loraStudioI18n.getLocale()) || "en"; }
    function el(tag, cls, value) {
        const node = document.createElement(tag);
        if (cls) node.className = cls;
        if (value !== undefined) node.textContent = value;
        return node;
    }
    function action(label, callback, cls) {
        const node = el("button", cls || "", label);
        node.type = "button";
        node.addEventListener("click", callback);
        return node;
    }
    function safeId(value) { return typeof value === "string" && /^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$/.test(value); }
    function safeContext(value) { return typeof value === "string" && /^studio\.[a-z][a-z0-9_.-]*$/.test(value); }
    function close() {
        state.request += 1;
        if (state.overlay) state.overlay.remove();
        state.overlay = state.dialog = state.list = state.article = state.input = null;
        const focus = state.previousFocus;
        state.previousFocus = null;
        if (focus && typeof focus.focus === "function") focus.focus();
    }
    function showError(error) {
        if (!state.article) return;
        state.article.replaceChildren(el("p", "lora-help-error", tr("error", {error: error.message || String(error)})));
    }
    async function fetchJSON(path) {
        const url = new URL("sdapi/v1/lora-studio/help" + path, root.location.href);
        const response = await root.fetch(url, {credentials: "same-origin"});
        if (!response.ok) throw new Error("HTTP " + response.status);
        return response.json();
    }
    function appendInline(container, text) {
        // Only a small, explicitly recognized Markdown subset. Any HTML, links
        // with unsafe protocols, and user-supplied tags remain literal text.
        const pattern = /(\*\*([^*\n]+)\*\*|`([^`\n]+)`|\[([^\]\n]+)\]\((https?:\/\/[^)\s]+|help:[a-z0-9-]+)\))/g;
        let start = 0;
        for (const match of text.matchAll(pattern)) {
            const index = match.index;
            if (index > start) container.append(document.createTextNode(text.slice(start, index)));
            if (match[2]) container.append(el("strong", "", match[2]));
            else if (match[3]) container.append(el("code", "", match[3]));
            else if (match[5].startsWith("help:")) {
                const id = match[5].slice(5);
                container.append(action(match[4], () => showArticle(id), "lora-help-text-link"));
            } else {
                const a = el("a", "", match[4]);
                a.href = match[5];
                a.target = "_blank";
                a.rel = "noopener noreferrer";
                container.append(a);
            }
            start = index + match[0].length;
        }
        if (start < text.length) container.append(document.createTextNode(text.slice(start)));
    }
    function markdown(container, source) {
        let inCode = false;
        let codeLines = [];
        const flushCode = () => {
            if (codeLines.length) {
                const pre = el("pre", "lora-help-code");
                pre.append(el("code", "", codeLines.join("\n")));
                container.append(pre);
            }
            codeLines = [];
        };
        for (const line of String(source).split(/\r?\n/)) {
            if (/^\s*```/.test(line)) { if (inCode) flushCode(); inCode = !inCode; continue; }
            if (inCode) { codeLines.push(line); continue; }
            if (!line.trim()) continue;
            const heading = line.match(/^(#{1,4})\s+(.+)$/);
            const bullet = line.match(/^\s*[-*]\s+(.+)$/);
            if (heading) {
                const title = el("h" + Math.min(heading[1].length + 1, 5));
                appendInline(title, heading[2]);
                container.append(title);
            } else if (bullet) {
                const p = el("p", "lora-help-bullet");
                p.append(document.createTextNode("• "));
                appendInline(p, bullet[1]);
                container.append(p);
            } else {
                const p = el("p", "");
                appendInline(p, line);
                container.append(p);
            }
        }
        if (inCode) flushCode();
    }
    function showTopics() {
        state.topic = null;
        if (!state.list || !state.article) return;
        state.article.hidden = true;
        state.list.hidden = false;
        state.input.focus();
        filterTopics();
    }
    function filterTopics() {
        if (!state.list) return;
        const needle = (state.input && state.input.value || "").trim().toLocaleLowerCase();
        const matches = state.topics.filter(item =>
            [item.title, item.summary, ...(item.tags || [])].join(" ").toLocaleLowerCase().includes(needle));
        state.list.replaceChildren();
        if (!matches.length) state.list.append(el("p", "lora-help-empty", tr("empty")));
        matches.forEach(item => {
            const target = action(item.title, () => showArticle(item.id), "lora-help-topic");
            target.append(el("small", "", item.summary));
            state.list.append(target);
        });
    }
    async function showArticle(topic) {
        if (!safeId(topic) || !state.article || !state.list) return;
        state.topic = topic;
        state.list.hidden = true;
        state.article.hidden = false;
        state.article.replaceChildren(el("p", "", "…"));
        const request = ++state.request;
        try {
            const data = await fetchJSON("/" + encodeURIComponent(topic) + "?locale=" + encodeURIComponent(language()));
            if (request !== state.request || !state.article) return;
            if (data.id !== topic || typeof data.content !== "string") throw new Error("Invalid help article response");
            const page = el("section", "lora-help-page");
            page.append(action(tr("back"), showTopics, "lora-help-back"));
            if (data.fallback) page.append(el("p", "lora-help-fallback", tr("fallback")));
            markdown(page, data.content);
            const related = (data.related || []).filter(id => safeId(id) && id !== topic);
            if (related.length) {
                page.append(el("h3", "", tr("related")));
                const links = el("nav", "lora-help-related");
                related.forEach(id => {
                    const entry = state.topics.find(item => item.id === id);
                    links.append(action(entry ? entry.title : id, () => showArticle(id), "lora-help-related-link"));
                });
                page.append(links);
            }
            state.article.replaceChildren(page);
            state.article.scrollTop = 0;
        } catch (error) {
            if (request === state.request) showError(error);
        }
    }
    function createDialog() {
        state.previousFocus = document.activeElement;
        const overlay = el("div", "lora-help-overlay");
        const panel = el("section", "lora-help-dialog");
        panel.setAttribute("role", "dialog");
        panel.setAttribute("aria-modal", "true");
        panel.setAttribute("aria-label", tr("topics"));
        panel.lang = language();
        const header = el("header", "lora-help-header");
        header.append(el("h2", "", tr("topics")));
        header.append(action("↻", () => reload(state.topic), "lora-help-reload"));
        header.lastElementChild.title = tr("reload");
        header.lastElementChild.setAttribute("aria-label", tr("reload"));
        header.append(action("×", close, "lora-help-close"));
        header.lastElementChild.setAttribute("aria-label", tr("close"));
        const input = el("input", "lora-help-search");
        input.type = "search";
        input.placeholder = tr("search");
        input.setAttribute("aria-label", tr("search"));
        input.addEventListener("input", filterTopics);
        const list = el("div", "lora-help-index");
        const article = el("div", "lora-help-article");
        article.hidden = true;
        panel.append(header, input, list, article);
        overlay.append(panel);
        overlay.addEventListener("click", event => {if (event.target === overlay) close();});
        overlay.addEventListener("keydown", event => {
            if (event.key === "Escape") {event.preventDefault(); close(); return;}
            if (event.key !== "Tab" || !state.dialog) return;
            const items = Array.from(panel.querySelectorAll('button, input, a[href]')).filter(node => !node.disabled && !node.hidden && node.getClientRects().length);
            if (!items.length) return;
            if (event.shiftKey && document.activeElement === items[0]) {event.preventDefault();items[items.length - 1].focus();}
            else if (!event.shiftKey && document.activeElement === items[items.length - 1]) {event.preventDefault();items[0].focus();}
        });
        document.body.append(overlay);
        state.overlay = overlay;
        state.dialog = panel;
        state.input = input;
        state.list = list;
        state.article = article;
        input.focus();
    }
    async function reload(topic) {
        if (!state.list || !state.article) return;
        const request = ++state.request;
        const needle = state.input.value;
        state.list.hidden = false;
        state.article.hidden = true;
        state.list.replaceChildren(el("p", "", "…"));
        try {
            const data = await fetchJSON("?locale=" + encodeURIComponent(language()));
            if (request !== state.request || !state.list) return;
            if (!Array.isArray(data.topics)) throw new Error("Invalid help index response");
            state.topics = data.topics.filter(item => safeId(item.id));
            state.input.value = needle;
            filterTopics();
            if (topic) {
                const resolved = safeContext(topic)
                    ? state.topics.find(item => Array.isArray(item.contexts) && item.contexts.includes(topic))?.id
                    : topic;
                if (resolved && safeId(resolved)) await showArticle(resolved);
            }
        } catch (error) {
            if (request === state.request) state.list.replaceChildren(el("p", "lora-help-error", tr("error", {error: error.message || String(error)})));
        }
    }
    async function open(topic) {
        if (topic && !safeId(topic) && !safeContext(topic)) return;
        if (!state.overlay) createDialog();
        await reload(topic || null);
    }
    if (root.addEventListener) root.addEventListener("lora-studio-locale-change", () => {
        if (!state.overlay) return;
        const topic = state.topic;
        close();
        open(topic);
    });
    root.loraStudioHelp = Object.freeze({open, close, reload, safeId, safeContext, appendInline, markdown});
}(typeof window !== "undefined" ? window : globalThis));
