// Dependency-free event/state regression runner. This minimal DOM does NOT
// validate browser layout, CSS, accessibility or Gradio integration. Open
// lora_studio_fixture.html in a real browser for those separate checks.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class UIEvent {
    constructor(type, options = {}) { this.type = type; Object.assign(this, options); }
    preventDefault() { this.defaultPrevented = true; }
}

class Element {
    constructor(tag) {
        this.tagName = tag.toLowerCase();
        this.children = [];
        this.parentElement = null;
        this.attributes = {};
        this.dataset = {};
        this.style = {};
        this.listeners = {};
        this.className = '';
        this.value = '';
        this._text = '';
        this.hidden = false;
        this.disabled = false;
        this.inert = false;
        this.classList = {
            toggle: (name, force) => {
                const classes = new Set(this.className.split(/\s+/).filter(Boolean));
                const enabled = force === undefined ? !classes.has(name) : force;
                if (enabled) classes.add(name); else classes.delete(name);
                this.className = [...classes].join(' ');
                return enabled;
            },
            add: name => this.classList.toggle(name, true),
        };
    }
    get open() { return Boolean(this._open); }
    set open(value) {
        const changed = Boolean(value) !== this.open;
        this._open = Boolean(value);
        if (changed) setTimeout(() => this.dispatchEvent(new UIEvent('toggle')), 0);
    }
    get previousElementSibling() {
        if (!this.parentElement) return null;
        return this.parentElement.children[this.parentElement.children.indexOf(this) - 1] || null;
    }
    get firstElementChild() { return this.children[0] || null; }
    replaceWith(element) {
        const parent = this.parentElement;
        const index = parent.children.indexOf(this);
        element.remove();
        parent.children[index] = element;
        element.parentElement = parent;
        this.parentElement = null;
    }
    get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
    set textContent(value) { this.replaceChildren(); this._text = String(value); }
    append(...children) {
        for (let child of children) {
            if (typeof child === 'string') { const text = new Element('#text'); text._text = child; child = text; }
            child.remove();
            child.parentElement = this;
            this.children.push(child);
        }
    }
    replaceChildren(...children) {
        this.children.forEach(child => { child.parentElement = null; });
        this.children = [];
        this._text = '';
        this.append(...children);
    }
    remove() {
        if (this.parentElement) this.parentElement.children = this.parentElement.children.filter(child => child !== this);
        this.parentElement = null;
    }
    insertAdjacentElement(position, element) {
        assert.equal(position, 'afterend');
        const parent = this.parentElement;
        element.remove();
        element.parentElement = parent;
        parent.children.splice(parent.children.indexOf(this) + 1, 0, element);
    }
    setAttribute(key, value) {
        this.attributes[key] = String(value);
        if (key === 'class') this.className = String(value);
        if (key === 'id') this.id = String(value);
    }
    getAttribute(key) {
        if (key.startsWith('data-')) {
            const name = key.slice(5).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
            return this.attributes[key] ?? this.dataset[name] ?? null;
        }
        return this.attributes[key] ?? null;
    }
    addEventListener(type, callback) { (this.listeners[type] ||= []).push(callback); }
    dispatchEvent(event) {
        event.target ||= this;
        (this.listeners[event.type] || []).forEach(callback => callback(event));
        if (event.bubbles && this.parentElement) this.parentElement.dispatchEvent(event);
        return !event.defaultPrevented;
    }
    click() {
        if (this.disabled) return;
        if (this.onclick) return this.onclick(new UIEvent('click'));
        return this.dispatchEvent(new UIEvent('click', {bubbles:true}));
    }
    focus() {}
    scrollIntoView() {}
    setSelectionRange(start, end) { this.selectionStart = start; this.selectionEnd = end; }
    setCustomValidity(message) { this.validationMessage = message; }
    reportValidity() { return !this.invalid(); }
    invalid() { return Boolean(this.validationMessage || (this.required && !String(this.value).trim())); }
    getBoundingClientRect() { return {width:496, height:600}; }
    getClientRects() {
        for (let parent = this; parent; parent = parent.parentElement) if (parent.hidden) return [];
        return [this.getBoundingClientRect()];
    }
    matchesSimple(selector) {
        if (selector.endsWith(':invalid')) return this.matchesSimple(selector.slice(0, -8)) && this.invalid();
        const attribute = selector.match(/\[([^=\]]+)(?:="([^"]*)")?\]$/);
        if (attribute) {
            const value = this.getAttribute(attribute[1]) ?? this[attribute[1]];
            if (attribute[2] !== undefined ? String(value) !== attribute[2] : value === undefined) return false;
            selector = selector.slice(0, attribute.index);
        }
        const id = selector.match(/#([\w-]+)/);
        if (id && this.id !== id[1]) return false;
        for (const cls of selector.matchAll(/\.([\w-]+)/g)) if (!this.className.split(/\s+/).includes(cls[1])) return false;
        const tag = selector.match(/^[\w-]+/);
        return !tag || tag[0] === this.tagName;
    }
    matchesChain(selector) {
        const parts = selector.match(/(?:\[[^\]]*\]|[^\s>])+|>/g) || [];
        let current = this;
        if (!current.matchesSimple(parts.pop() || '*')) return false;
        while (parts.length) {
            const part = parts.pop();
            current = current.parentElement;
            if (part === '>') {
                if (!current || !current.matchesSimple(parts.pop())) return false;
            } else {
                while (current && !current.matchesSimple(part)) current = current.parentElement;
                if (!current) return false;
            }
        }
        return true;
    }
    querySelectorAll(selector) {
        const found = [];
        const visit = element => {
            for (const child of element.children) {
                if (selector.split(/,\s*/).some(group => child.matchesChain(group))) found.push(child);
                visit(child);
            }
        };
        visit(this);
        return found;
    }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}

test('real Studio handlers pass the isolated event/state scenarios', async () => {
    const document = new Element('document');
    document.createElement = tag => new Element(tag);
    document.createElementNS = (_, tag) => new Element(tag);
    document.getElementById = id => document.querySelector(`#${id}`);
    const add = (parent, tag, id, text) => {
        const element = new Element(tag);
        element.id = id;
        if (text) element.textContent = text;
        parent.append(element);
        return element;
    };
    document.body = add(document, 'body');
    for (const id of ['run-tests', 'quick', 'theme', 'switch-tab', 'narrow']) add(document.body, 'button', id);
    const app = add(document.body, 'main', 'app');
    for (const tab of ['txt2img', 'img2img']) {
        const section = add(app, 'section', `tab_${tab}`);
        section.hidden = tab === 'img2img';
        const top = add(section, 'div', `${tab}_toprow`);
        const container = add(top, 'div', `${tab}_prompt_container`);
        const prompt = add(container, 'div', `${tab}_prompt`);
        const label = add(prompt, 'label');
        const textarea = add(label, 'textarea');
        textarea.value = tab === 'txt2img' ? 'portrait <lora:same:0.5> light <lora:same:0.8:plugin=keep>' : 'image <lora:other:0.7>';
    }
    const results = add(document.body, 'pre', 'test-results');
    const context = vm.createContext({document, console, setTimeout, clearTimeout, URL, Event:UIEvent, MouseEvent:UIEvent, KeyboardEvent:UIEvent, location:{href:'http://fixture.invalid/test/lora_studio_fixture.html'}, innerWidth:1100, innerHeight:900, localStorage:{getItem:() => null, setItem:() => {}}});
    context.window = context;
    const html = fs.readFileSync(path.join(__dirname, 'lora_studio_fixture.html'), 'utf8');
    context.navigator = {language: 'en-US'};
    const localeListeners = {};
    context.addEventListener = (name, listener) => { (localeListeners[name] ||= []).push(listener); };
    context.dispatchEvent = (event) => { (localeListeners[event.type] || []).forEach(fn => fn(event)); };
    context.CustomEvent = class { constructor(type, options) {this.type = type; this.detail = options.detail;} };
    const setup = html.match(/<script>([\s\S]*?)<\/script>/)[1];
    vm.runInContext(setup, context);
    for (const name of ['I18n', 'LocaleEn', 'LocaleRu', 'LocaleFr', 'LocaleEs', 'LocaleIt', 'LocaleJa', 'LocaleZhCn', 'LocalePtBr', 'LocaleDe', 'LocaleKo', 'Model', 'Blocks', 'Curves', 'Presets', 'View']) {
        const filename = path.join(__dirname, `../javascript/loraStudio${name}.js`);
        vm.runInContext(fs.readFileSync(filename, 'utf8'), context, {filename});
    }
    vm.runInContext(fs.readFileSync(path.join(__dirname, 'lora_studio_fixture.js'), 'utf8'), context);
    await document.querySelector('#run-tests').click();
    assert.equal(results.dataset.result, 'passed', results.textContent);
    const promptBefore = document.querySelector('#txt2img_prompt > label > textarea').value;
    const picker = document.querySelector('.lora-studio-language-picker');
    assert.ok(picker, 'Studio should mount a locale selector');
    assert.equal(picker.value, 'en');
    context.window.loraStudioI18n.setLocale('fr');
    assert.equal(picker.value, 'fr');
    assert.ok(document.querySelector('#txt2img_lora_studio .lora-studio-add-label').textContent.includes('Ajouter une LoRA'));
    assert.equal(document.querySelector('#txt2img_prompt > label > textarea').value, promptBefore);
    context.window.loraStudioI18n.setLocale('ru');
    assert.equal(picker.value, 'ru');
    assert.ok(document.querySelector('#txt2img_lora_studio .lora-studio-add-label').textContent.includes('Добавить LoRA'));
    assert.equal(document.querySelector('#txt2img_prompt > label > textarea').value, promptBefore);
    console.log(results.textContent+'\nLanguage picker: French and Russian switched, prompt preserved.');
});
