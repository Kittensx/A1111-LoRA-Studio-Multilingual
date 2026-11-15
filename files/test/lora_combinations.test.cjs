const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const model = require('../javascript/loraStudioModel.js');

test('TE/U-Net/DYN named, positional and mixed syntax keep independent values', () => {
    for (const tag of [
        '<lora:x:te=0.3:unet=-0.8:dyn=16>',
        '<lora:x:dyn=16:unet=-0.8:te=0.3>',
        '<lora:x:0.3:-0.8:16>',
        '<lora:x:9:8:4:te=0.3:unet=-0.8:dyn=16>',
        '<lora:x:0.3:unet=-0.8:dyn=16>',
    ]) {
        const spec = model.parseTag(tag);
        assert.equal(spec.weights.te.value, 0.3, tag);
        assert.equal(spec.weights.unet.value, -0.8, tag);
        assert.equal(spec.dyn_dim, 16, tag);
        assert.deepEqual(model.parseTag(model.serializeTag(spec)), spec);
    }
});

test('empty tag weights inherit native defaults, never NaN / blank number controls', () => {
    for (const tag of ['<lora:x:>', '<lora:x:te=:unet=:dyn=>', '<lora:x:::16>', '<lora:x:te=:dyn=16:hr=>']) {
        const spec = model.parseTag(tag);
        assert.equal(spec.weights.te.value, 1, tag);
        assert.equal(spec.weights.unet.value, 1, tag);
        assert.doesNotThrow(() => model.serializeTag(spec));
    }
    const inherited = model.parseTag('<lora:x:te=0.3:unet=:dyn=16>');
    assert.equal(inherited.weights.unet.value, 0.3);
});

test('rank input rejects fractions, overflow and negative values instead of clamping', () => {
    assert.equal(model.parseDynamicRank('', true), null);
    assert.equal(model.parseDynamicRank('+16', true), 16);
    for (const value of ['0', '-1', '2.5', '1e2', 'Infinity', 'NaN', '9007199254740992']) {
        assert.throws(() => model.parseDynamicRank(value, true), /positive safe integer/, value);
    }
    // Legacy tags are not silently reinterpreted by an unrelated UI patch.
    assert.equal(model.parseTag('<lora:x:dyn=-1>').dyn_dim, -1);
});

test('uppercase names remain unknown arguments rather than covertly changing defaults', () => {
    const spec = model.parseTag('<lora:x:0.7:TE=0.2:UNET=0.3:DYN=16>');
    assert.equal(spec.weights.te.value, 0.7);
    assert.equal(spec.weights.unet.value, 0.7);
    assert.equal(spec.dyn_dim, null);
    assert.equal(spec.passthrough_named.length, 3);
    assert.deepEqual(model.parseTag(model.serializeTag(spec)), spec);
});

function cardContext() {
    const context = vm.createContext({
        opts: {extra_networks_default_multiplier: 1, extra_networks_add_text_separator: ' '},
        onUiLoaded: () => {}, addEventListener: () => {}, updateInput: () => {},
        console: {warn: () => {}}, loraStudioModel: model,
    });
    context.window = context;
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../javascript/extraNetworks.js'), 'utf8'), context);
    return context;
}

test('card settings retain combined curves, Hires, rank, masks and unknown arguments', () => {
    const card = cardContext();
    const spec = model.parseTag('<loractl:x:te=0.3:unet=0@0,0.8@1:dyn=16:hr=0.4:lbw=OUTS:lbwe=ATTNDEEPOFF:plugin=keep>');
    const settings = card.loraReadSettings(spec);
    settings.te = 0.2;
    const result = JSON.parse(JSON.stringify(card.loraSpecFromSettings(settings)));
    assert.deepEqual(result, {...spec, weights: {...spec.weights, te: {kind: 'scalar', value: 0.2}}});
    for (const rank of ['0', '-2', '2.5', '9007199254740992']) {
        assert.throws(() => card.loraSpecFromSettings({...settings, dyn: rank}), /positive safe integer/);
    }
    assert.throws(() => card.loraSpecFromSettings({...settings, te: ''}), /Complete both/);
});

test('card refuses invalid settings without writing, and edits only the selected occurrence', () => {
    const card = cardContext();
    const prompt = {value: '<lora:x:0.5> <lora:x:te=0.3:unet=0.8:dyn=16>'};
    const before = prompt.value;
    const span = model.findTags(before)[1];
    const settings = card.loraReadSettings(span.spec);
    settings.span = span;
    const source = {closest: () => null, setCustomValidity: () => {}};
    settings.dyn = '2.5';
    assert.equal(card.loraSetSettings(prompt, settings, source), false);
    assert.equal(prompt.value, before);
    settings.dyn = '8';
    assert.equal(card.loraSetSettings(prompt, settings, source), true);
    assert.equal(prompt.value, '<lora:x:0.5> <lora:x:te=0.3:unet=0.8:dyn=8>');
});
