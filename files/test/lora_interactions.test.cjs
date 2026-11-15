const {test} = require('node:test');
const assert = require('node:assert/strict');
const model = require('../javascript/loraStudioModel.js');
const blocks = require('../javascript/loraStudioBlocks.js');
const curves = require('../javascript/loraStudioCurves.js');
const presets = require('../javascript/loraStudioPresets.js');

test('absolute-step midpoint is editable without converting units', () => {
    const points = [{position:5, value:0.5}, {position:9, value:1}];
    assert.deepEqual(curves.insertMidpoint(points), [points[0], {position:7, value:0.75}, points[1]]);
});

test('custom definitions override builtins in the actual visual block table', () => {
    const custom = 'MIDD:' + Array(17).fill(0.25).join(',');
    assert.deepEqual(blocks.resolveBlockWeights('midd', custom).values, Array(17).fill(0.25));
    assert.equal(blocks.resolveBlockWeights('MIDD', 'MIDD:broken'), null);
});

test('classic mask expansion preserves block IDs, not index offsets', () => {
    const source = blocks.resolveBlockWeights('MIDD');
    const expanded = blocks.projectClassicValues(source.values, 'classic-20');
    const ids = blocks.getProfile('classic-20').blockIds;
    assert.equal(expanded[ids.indexOf('IN00')], 0);
    assert.equal(expanded[ids.indexOf('IN05')], 1);
    assert.equal(expanded[ids.indexOf('M00')], 1);
    assert.equal(expanded[ids.indexOf('OUT00')], 0);
    assert.equal(expanded.length, 20);
});

test('mixed timing units survive tag and saved-preset roundtrip', () => {
    const spec = model.parseTag('<lora:a:0.8:start=5:stop=0.8:lbw=MIDD>');
    const saved = presets.createPreset('timed', spec, 'test');
    assert.deepEqual(model.parseTag(model.serializeTag(presets.applyPreset(spec, saved))), spec);
    assert.throws(() => model.parseTag('<lora:a:1:step=-1-5>'), /non-negative/);
});

test('editing each of four occurrences never changes a neighbour or drops controls', () => {
    const prompt = '<lora:a:0.5:start=0.2:stop=0.8:lbw=MIDD> <lora:b:te=0.3:unet=0.5@5,1@9:hr=0.4> <lora:a:0.7:dyn=1:lbwe=ATTNDEEPOFF> <lora:d:-0.2:plugin=keep>';
    const original = model.findTags(prompt);
    original.forEach((entry, index) => {
        const draft = presets.applyQuickPreset(entry.spec, {kind:'mode', mode:'te-off'});
        const changed = model.findTags(model.replaceTagSpan(prompt, entry, model.serializeTag(draft)));
        changed.forEach((other, j) => {
            if (j !== index) assert.equal(other.tag, original[j].tag);
            else assert.deepEqual(other.spec, draft);
        });
    });
});
