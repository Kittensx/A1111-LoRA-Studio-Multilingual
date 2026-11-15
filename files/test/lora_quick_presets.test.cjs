// Run with: node --test test/lora_quick_presets.test.cjs (no dependencies).
const {test} = require('node:test');
const assert = require('node:assert/strict');
const model = require('../javascript/loraStudioModel.js');
require('../javascript/loraStudioBlocks.js');
const presets = require('../javascript/loraStudioPresets.js');

const choice = (id, options) => presets.listQuickPresets(options).find((item) => item.id === id);
const parse = model.parseTag;

test('catalog combines builtin, custom and saved presets without touching the source', () => {
    const saved = presets.createPreset('My look', parse('<lora:test:0.8>'), 'saved-1');
    saved.favorite = true;
    const options = {blockPresets: 'mine:1,0,1,0,1,0,1,0,1,0,1,0', elementalPresets: 'my rules:IN05:attn:0.5', savedPresets: [saved]};
    const before = JSON.stringify(options);
    const all = presets.listQuickPresets(options);
    for (const id of ['mode:te-half', 'block:outs', 'block:mine', 'elemental:attndeepoff', 'elemental:my rules', 'saved:saved-1']) {
        assert.ok(all.some((item) => item.id === id), id);
    }
    assert.equal(all.find((item) => item.id === 'saved:saved-1').group, 'Favourites');
    assert.equal(new Set(all.map((item) => item.id)).size, all.length);
    assert.equal(JSON.stringify(options), before);
});

test('half-TE mode is idempotent and preserves explicit Hires, unknown fields and U-Net curve', () => {
    const spec = parse('<lyco:test:te=0.9:unet=0.4@0,0.8@1:hrte=0.3:lbw=OUTS:plugin=yes>');
    const before = JSON.stringify(spec);
    const next = presets.applyQuickPreset(spec, choice('mode:te-half'));
    assert.deepEqual(next.weights.te.points, [{position: 0, value: 0.2}, {position: 1, value: 0.4}]);
    assert.deepEqual(next, presets.applyQuickPreset(next, choice('mode:te-half')));
    assert.equal(next.hires.te.value, 0.3);
    assert.deepEqual(next.weights.unet, spec.weights.unet);
    assert.deepEqual(next.passthrough_named, spec.passthrough_named);
    assert.equal(next.tag_type, 'lyco');
    assert.equal(JSON.stringify(spec), before);
    next.weights.unet.points[0].value = 99;
    assert.equal(spec.weights.unet.points[0].value, 0.4);
});

test('block preset replaces only block mask and respects module override', () => {
    const spec = parse('<lora:test:0.8:lbwe=ATTNDEEPOFF:dyn=16:start=0.2:hr=0.4>');
    const next = presets.applyQuickPreset(spec, choice('block:outs'));
    assert.deepEqual(next, {...spec, block_weights: {raw: 'OUTS'}});
});

test('elemental preset replaces only module override; clear masks preserves everything else', () => {
    const spec = parse('<lora:test:0.6:lbw=OUTS:dyn=8:stop=0.8:hr=0.3>');
    const next = presets.applyQuickPreset(spec, choice('elemental:attndeepoff'));
    assert.deepEqual(next, {...spec, elemental_weights: {raw: 'ATTNDEEPOFF'}});
    assert.deepEqual(presets.applyQuickPreset(next, choice('mode:clear-masks')), {...spec, block_weights: null, elemental_weights: null});
});

test('TE off/match modes keep other values and do not link object references', () => {
    const spec = parse('<lora:test:te=0.3:unet=0.7:hrte=0.4>');
    assert.equal(presets.applyQuickPreset(spec, choice('mode:te-off')).weights.te.value, 0);
    const matched = presets.applyQuickPreset(spec, choice('mode:te-match'));
    assert.equal(matched.weights.te.value, 0.7);
    matched.weights.te.value = 0.1;
    assert.equal(matched.weights.unet.value, 0.7);
    assert.equal(matched.hires.te.value, 0.4);
});

test('fade-in keeps strength, supports negative weights, rejects existing curve unchanged', () => {
    const spec = parse('<lora:test:te=0.4:unet=-0.8>');
    const next = presets.applyQuickPreset(spec, choice('mode:fade-in'));
    assert.deepEqual(next.weights.unet.points, [{position: 0, value: 0}, {position: 0.35, value: -0.8}, {position: 1, value: -0.8}]);
    const before = JSON.stringify(next);
    assert.throws(() => presets.applyQuickPreset(next, choice('mode:fade-in')), /already has a curve/);
    assert.equal(JSON.stringify(next), before);
    assert.equal(next.weights.te.value, 0.4);
});

test('explicit incompatible layouts are blocked, auto is not claimed compatible', () => {
    for (const profile of ['flux-61', 'dit-61']) {
        const spec = parse(`<lora:test:profile=${profile}>`);
        assert.throws(() => presets.applyQuickPreset(spec, choice('block:outs')), /does not fit/);
        assert.throws(() => presets.applyQuickPreset(spec, choice('elemental:attndeepoff')), /classic U-Nets/);
        assert.equal(presets.applyQuickPreset(spec, choice('block:fluxall')).block_weights.raw, 'FLUXALL');
    }
    assert.throws(() => presets.applyQuickPreset(parse('<lora:test:profile=classic-20>'), choice('block:fluxall')), /does not fit/);
    assert.equal(presets.quickPresetProblem(parse('<lora:test>'), choice('block:outs')), '');
});

test('custom overrides of builtin names keep their true table length and spelling', () => {
    const options = {blockPresets: `outs:${Array(61).fill(0.5).join(',')}`};
    const item = choice('block:outs', options);
    assert.equal(item.count, 61);
    const next = presets.applyQuickPreset(parse('<lora:test:profile=flux-61>'), item);
    assert.equal(next.block_weights.raw, 'outs');
    assert.throws(() => presets.applyQuickPreset(next, choice('block:alias', {blockPresets: 'alias:OUTS'})), /finite block-weight table/);
});

test('saved snapshot restores controls but preserves LoRA identity, alias and unknown arguments', () => {
    const saved = presets.createPreset('Saved', parse('<lora:first:te=0.2:unet=0.8:lbwe=ATTNDEEPOFF>'), 'a');
    const target = parse('<loractl:second:0.5:custom=keep>');
    const result = presets.applyQuickPreset(target, choice('saved:a', {savedPresets: [saved]}));
    assert.equal(result.network_name, 'second');
    assert.equal(result.tag_type, 'loractl');
    assert.deepEqual(result.passthrough_named, target.passthrough_named);
    assert.deepEqual(result.weights, saved.weights);
    assert.deepEqual(result.elemental_weights, saved.elemental_weights);
    assert.deepEqual(parse(model.serializeTag(result)), result);
});

test('recipe edit replaces exactly the selected occurrence in a repeated-name prompt', () => {
    const prompt = 'a <lora:same:0.5> b <lora:same:0.8:custom=keep> c';
    const tag = model.findTags(prompt)[1];
    const changed = presets.applyQuickPreset(tag.spec, choice('mode:te-half'));
    const result = model.replaceTagSpan(prompt, tag, model.serializeTag(changed));
    assert.equal(result, 'a <lora:same:0.5> b <lora:same:te=0.4:unet=0.8:custom=keep> c');
    assert.throws(() => model.replaceTagSpan('prefix ' + prompt, tag, model.serializeTag(changed)));
});
