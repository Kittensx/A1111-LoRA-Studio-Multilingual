/* No dependencies or browser required: test external language catalogs in isolation. */
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const dir = path.resolve(__dirname, '../javascript');
const packs = ['En','Ru','Fr','Es','It','Ja','ZhCn','PtBr','De','Ko'];
function load(preferred='en-US') {
    const saved = new Map();
    const events = [];
    const window = {
        navigator: {language: preferred},
        localStorage: {getItem: key => saved.get(key) || null, setItem: (key,value) => saved.set(key,value)},
        dispatchEvent: event => events.push(event.type),
        CustomEvent: class { constructor(type,options) { this.type=type; this.detail=options.detail; } },
    };
    const context = vm.createContext({window});
    for (const item of ['I18n', ...packs.map(name => 'Locale'+name)]) {
        vm.runInContext(fs.readFileSync(path.join(dir, `loraStudio${item}.js`), 'utf8'), context);
    }
    return {api: window.loraStudioI18n, saved, events};
}
test('ten locale packs register; English messages are a lossless fallback', () => {
    const {api} = load();
    assert.equal(Object.keys(api.supported).length, 10);
    assert.equal(api.t('Apply'), 'Apply');
    assert.equal(api.t('Curves are piecewise-linear. If every position is 0…1, positions are fractions; otherwise ALL positions are absolute zero-based steps. U-Net changes during sampling; TE uses only the first point during text encoding, not a per-step TE schedule. Explicit Hires weights replace the base weight or curve.'), 'Curves are piecewise-linear. If every position is 0…1, positions are fractions; otherwise ALL positions are absolute zero-based steps. U-Net changes during sampling; TE uses only the first point during text encoding, not a per-step TE schedule. Explicit Hires weights replace the base weight or curve.');
    assert.equal(api.t('<lora:portrait:te=0.8>'), '<lora:portrait:te=0.8>');
});
test('Russian, French, Spanish, Japanese, Chinese and Portuguese labels are available', () => {
    const {api} = load();
    const expected = [['ru','Применить'],['fr','Appliquer'],['es','Aplicar'],['it','Applica'],['ja','適用'],['zh-CN','应用'],['pt-BR','Aplicar'],['de','Anwenden'],['ko','적용']];
    for (const [locale,label] of expected) {
        assert.equal(api.setLocale(locale),locale);
        assert.equal(api.t('Apply'),label);
    }
});
test('Switching persists and interpolates only allowed named placeholders', () => {
    const {api,saved,events} = load('en-US');
    assert.equal(api.setLocale('es'), 'es');
    assert.equal(saved.get('loraStudio.language.v1'), 'es');
    assert.equal(api.t('Selected LoRA: {name}', {name:'my<lora> & {hack}'}), 'LoRA seleccionada: my<lora> & {hack}');
    assert.deepEqual(events, ['lora-studio-locale-change']);
});
test('Browser locale fallbacks and unsupported locale are safe', () => {
    assert.equal(load('pt-PT').api.getLocale(),'pt-BR');
    assert.equal(load('zh-Hans').api.getLocale(),'zh-CN');
    assert.equal(load('fr-CA').api.getLocale(),'fr');
    const {api} = load();
    assert.equal(api.setLocale('__unknown__'), 'en');
    assert.equal(api.t('a raw model name'), 'a raw model name');
});
test('Language dictionary never modifies source prompt tags or unknown user text', () => {
    const {api} = load();
    const rawPrompt = 'a portrait <lora:miku:te=0.9:unet=0@0,1@0.5:dyn=8> test';
    for (const locale of Object.keys(api.supported)) {
        api.setLocale(locale);
        assert.equal(api.t(rawPrompt), rawPrompt);
    }
});


test('all UI catalogs contain the same stable message IDs and retain placeholders', () => {
    const {api} = load();
    const en = api.t('studio.ui.apply');
    assert.equal(en, 'Apply');
    const strings = {};
    for (const name of packs) {
        const filename = path.join(dir, `loraStudioLocale${name}.js`);
        const match = fs.readFileSync(filename, 'utf8').match(/const messages = ([\s\S]*?);\s*if \(root\.loraStudioI18n\)/);
        assert.ok(match, name);
        const messages = JSON.parse(match[1]);
        strings[name] = messages;
        assert.ok(Object.keys(messages).length >= 205, name);
        assert.deepEqual(Object.keys(messages).sort(), Object.keys(strings.En).sort(), name);
        for (const [key,value] of Object.entries(messages)) {
            assert.match(key, /^studio\.(ui|status|help|preset|help_registry)\.[a-z0-9_]+$/);
            assert.ok(value.trim(), `${name}:${key}`);
            const tokens = text => [...text.matchAll(/\{([a-z][a-z0-9_]*)\}/gi)].map(match=>match[1]).sort();
            assert.deepEqual(tokens(value), tokens(strings.En[key]), `${name}:${key} placeholders`);
        }
    }
    for (const lang of Object.keys(api.supported)) {
        api.setLocale(lang);
        assert.notEqual(api.t('studio.ui.apply'), 'studio.ui.apply');
    }
});

test('locale packs register correctly even when loaded before the engine', () => {
    const window = {navigator:{language:'ja-JP'}, localStorage:{getItem:()=>null,setItem:()=>{}}};
    const context = vm.createContext({window});
    for (const name of ['LocaleJa','LocaleEn','I18n']) {
        vm.runInContext(fs.readFileSync(path.join(dir, `loraStudio${name}.js`), 'utf8'), context);
    }
    assert.equal(window.loraStudioI18n.getLocale(),'ja');
    assert.equal(window.loraStudioI18n.t('studio.ui.apply'),'適用');
    assert.equal(window.loraStudioI18n.missingKeys().length,0);
    assert.equal(window.loraStudioI18n.t('studio.ui.absent_text'),'studio.ui.absent_text');
    assert.equal(window.loraStudioI18n.missingKeys()[0],'studio.ui.absent_text');
});

test('locale lookup leaves model names and prompt values completely untouched', () => {
    const {api} = load();
    for (const lang of Object.keys(api.supported)) {
        api.setLocale(lang);
        for (const value of ['<lora:same:0.5>','0.5@5,1@9','ATTNDEEPOFF','my custom Studio preset']) {
            assert.equal(api.t(value),value);
        }
    }
});
