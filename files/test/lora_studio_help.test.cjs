// Fast isolated tests for help rendering / navigation; not a real-browser test.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../javascript/loraStudioHelp.js'), 'utf8');

class Node {
    constructor(tag, value='') {
        this.tagName = tag;
        this._text = value;
        this.children = [];
        this.listeners = {};
        this.attributes = {};
        this.hidden = false;
        this.value = '';
        this.lastElementChild = null;
    }
    append(...nodes) { nodes.forEach(n => { this.children.push(n); this.lastElementChild = n; }); }
    replaceChildren(...nodes) { this.children=[]; this._text=''; this.lastElementChild=null; this.append(...nodes); }
    get textContent() { return this._text + this.children.map(n => n.textContent).join(''); }
    set textContent(value) { this._text=String(value); this.children=[]; }
    setAttribute(k,v) { this.attributes[k]=v; }
    addEventListener(k,fn) { (this.listeners[k] ||= []).push(fn); }
    focus() { this.focused = true; }
    remove() { this.removed = true; }
    getClientRects() { return [1]; }
    querySelectorAll() { return []; }
    click() { for (const fn of this.listeners.click || []) fn({preventDefault(){}}); }
}

function context(locale='fr') {
    const document = {
        activeElement: new Node('button'),
        body: new Node('body'),
        createElement: tag => new Node(tag),
        createTextNode: value => new Node('#text',value),
    };
    const topics = [
        {id:'getting-started', title:'Getting started',summary:'Add a LoRA',tags:[],related:['weights-and-rank'],contexts:['studio.root']},
        {id:'weights-and-rank',title:'Weights and rank',summary:'Use sliders',tags:['TE'],related:['getting-started'],contexts:['studio.quick']}
    ];
    const calls=[];
    const window = {
        location: {href:'http://localhost:7860/'},
        loraStudioI18n:{getLocale:()=>locale, t:(key,vars)=>key+(vars ? ':'+vars.error : '')},
        addEventListener(){},
        fetch: async (url) => {
            calls.push(String(url));
            const detail = String(url).includes('/help/');
            const id = String(url).includes('/help/weights-and-rank') ? 'weights-and-rank' : 'getting-started';
            return {ok:true, json: async () => detail
                ? {id, content:'# Getting started\n\nUntrusted <img src=x onerror=alert(1)>\n\n**Safe text** and `code`.',
                    related:['weights-and-rank'], locale:'en', fallback:true}
                : {schema_version:1,topics}};
        },
    };
    vm.runInNewContext(source,{window,document,URL});
    return {api:window.loraStudioHelp,window,document,calls};
}
function walk(node,tag) {
    const own = node.tagName === tag ? [node] : [];
    return own.concat(...node.children.map(c => walk(c,tag)));
}

test('help topics validate IDs and use authenticated same-origin API', async () => {
    const {api,document,calls} = context();
    assert.equal(api.safeId('getting-started'),true);
    assert.equal(api.safeId('../../etc/passwd'),false);
    await api.open('getting-started');
    assert.equal(calls.length,2);
    assert.ok(calls.every(call => call.startsWith('http://localhost:7860/sdapi/v1/lora-studio/help')));
    assert.ok(calls[1].includes('locale=fr'));
    assert.equal(walk(document.body,'img').length,0);
    assert.ok(document.body.textContent.includes('<img src=x onerror=alert(1)>'));
    assert.equal(walk(document.body,'strong').length,1);
    assert.equal(walk(document.body,'code').length,1);
    const links = walk(document.body,'button').filter(b => b.textContent === 'Weights and rank');
    assert.ok(links.length>0);
    api.close();
});

test('search and related topics display safe text, without browser markup injection', async () => {
    const {api,document} = context();
    await api.open();
    assert.equal(walk(document.body,'button').filter(n => n.textContent.startsWith('Getting started')).length,1);
    const search=walk(document.body,'input')[0];
    search.value='unmatched';
    for (const event of search.listeners.input || []) event({});
    assert.ok(document.body.textContent.includes('studio.help_registry.empty'));
    api.close();
});


test('context registry resolves a stable button context to the matching article', async () => {
    const {api,calls} = context();
    assert.equal(api.safeContext('studio.quick'),true);
    assert.equal(api.safeContext('../../home'),false);
    await api.open('studio.quick');
    assert.equal(calls.length,2);
    assert.ok(calls[1].includes('/help/weights-and-rank?locale=fr'));
    api.close();
});
