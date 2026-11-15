/* Isolated DOM regression fixture; no dependencies or generation calls. */
loaded.forEach(callback => callback());
const field = tab => document.querySelector(`#${tab}_prompt textarea`);
const quickRoot = () => document.querySelector('#txt2img_lora_quick_tune');
const fullRoot = () => document.querySelector('#txt2img_lora_studio');
const tick = () => new Promise(resolve => setTimeout(resolve, 30));
const initial = field('txt2img').value;

function openSecond() {
    const prompt = field('txt2img');
    prompt.focus();
    prompt.setSelectionRange(prompt.value.lastIndexOf('<lora:') + 7, prompt.value.lastIndexOf('<lora:') + 7);
    prompt.dispatchEvent(new MouseEvent('contextmenu', {bubbles:true, cancelable:true, clientX:300, clientY:155}));
}

function selectRecipe(surface, id, apply = true) {
    const select = surface.querySelector('.lora-studio-recipe-label select');
    select.value = id;
    select.dispatchEvent(new Event('change', {bubbles:true}));
    if (apply) surface.querySelector('.lora-studio-recipe-use').click();
}

function typeValue(input, value) {
    input.value = value;
    input.dispatchEvent(new Event('input', {bubbles:true}));
}

document.querySelector('#quick').onclick = openSecond;
document.querySelector('#theme').onclick = () => {
    const light = document.body.classList.toggle('light');
    document.body.classList.toggle('dark', !light);
};
document.querySelector('#switch-tab').onclick = () => {
    document.querySelectorAll('#tab_txt2img, #tab_img2img').forEach(tab => { tab.hidden = !tab.hidden; });
};
document.querySelector('#narrow').onclick = () => {
    const app = gradioApp();
    app.style.maxWidth = app.style.maxWidth ? '' : '390px';
};

document.querySelector('#run-tests').onclick = async () => {
    const output = document.querySelector('#test-results');
    const results = [];
    const check = (condition, message) => {
        if (!condition) throw new Error(message);
        results.push(`PASS ${message}`);
        output.textContent = results.join('\n');
    };
    try {
        await tick();
        if (quickRoot()) quickRoot().querySelector('.lora-studio-quick-tune-cancel').click();
        const discard = fullRoot().querySelector('.lora-studio-discard');
        if (discard && !discard.hidden) discard.click();
        field('txt2img').value = initial;
        updateInput(field('txt2img'));
        openSecond();
        check(Boolean(quickRoot()), 'right click opens quick editor');
        check(fullRoot().querySelector('.lora-studio-body').inert, 'background editor cannot change the same draft');
        selectRecipe(quickRoot(), 'mode:te-half', false);
        check(field('txt2img').value === initial, 'selecting a recipe does not write the prompt');
        quickRoot().querySelector('.lora-studio-recipe-use').click();
        check(quickRoot().querySelector('input[aria-label="Text encoder exact weight"]').value === '0.4', 'preset refreshes the actual TE control');
        check(field('txt2img').value === initial, 'using preset edits only the draft');
        quickRoot().querySelector('.lora-studio-quick-tune-cancel').click();
        check(field('txt2img').value === initial && !quickRoot(), 'Cancel leaves original prompt byte-for-byte');
        check(!fullRoot().querySelector('.lora-studio-body').inert, 'Cancel releases full editor');

        openSecond();
        const emptyWeight = quickRoot().querySelector('input[aria-label="Text encoder exact weight"]');
        emptyWeight.value = '';
        emptyWeight.dispatchEvent(new Event('input', {bubbles:true}));
        quickRoot().querySelector('.lora-studio-quick-tune-apply').click();
        check(field('txt2img').value === initial && Boolean(quickRoot()), 'empty exact weight does not silently apply its previous value');
        quickRoot().dispatchEvent(new KeyboardEvent('keydown', {key:'Escape', bubbles:true}));
        check(!quickRoot() && field('txt2img').value === initial, 'Escape cancels without writing');

        openSecond();
        selectRecipe(quickRoot(), 'mode:te-half');
        quickRoot().querySelector('.lora-studio-quick-tune-apply').click();
        check(field('txt2img').value === 'portrait <lora:same:0.5> light <lora:same:te=0.4:unet=0.8:plugin=keep>', 'Apply changes only second duplicate LoRA and keeps plugin argument');

        openSecond();
        selectRecipe(quickRoot(), 'mode:te-off');
        field('txt2img').value += ' manually changed';
        updateInput(field('txt2img'));
        check(quickRoot().querySelector('.lora-studio-quick-tune-apply').disabled, 'stale prompt disables Apply');
        quickRoot().querySelector('.lora-studio-quick-tune-cancel').click();

        field('txt2img').value = initial;
        updateInput(field('txt2img'));
        openSecond();
        selectRecipe(quickRoot(), 'mode:fade-in');
        check(quickRoot().querySelector('.lora-studio-weight-curve'), 'curve mode renders a curve, not scalar sliders');
        quickRoot().querySelector('.lora-studio-quick-tune-full').click();
        check(!quickRoot() && !fullRoot().querySelector('.lora-studio-apply').disabled, 'Full Studio preserves editable draft');
        fullRoot().querySelectorAll('.lora-studio-list-item')[1].click();
        check(!fullRoot().querySelector('.lora-studio-discard').hidden, 'reselecting same LoRA does not discard draft');
        window.loraStudioOpen('txt2img', 'same', 9);
        check(fullRoot().querySelector('.lora-studio-tag-preview code').textContent.includes('0.8@0.35'), 'card handoff preserves existing curve draft');
        fullRoot().querySelector('.lora-studio-discard').click();
        fullRoot().querySelectorAll('.lora-studio-list-item')[0].click();
        selectRecipe(fullRoot(), 'mode:te-half');
        check(fullRoot().querySelector('input[aria-label="Text encoder exact weight"]').value === '0.25', 'full Studio picker edits the selected first LoRA');
        check(field('txt2img').value === initial, 'full Studio picker keeps changes in draft');
        fullRoot().querySelector('.lora-studio-discard').click();

        openSecond();
        typeValue(quickRoot().querySelector('input[aria-label="Text encoder exact weight"]'), '0.3');
        typeValue(quickRoot().querySelector('input[aria-label="U-Net exact weight"]'), '-0.75');
        typeValue(quickRoot().querySelector('input[aria-label="Dynamic rank custom value"]'), '16');
        check(quickRoot().querySelector('.lora-studio-quick-tune-preview code').textContent.includes('te=0.3:unet=-0.75:dyn=16'), 'TE + U-Net + DYN combine without replacing each other');
        check(quickRoot().querySelector('.lora-studio-combination').textContent.includes('DYN 16'), 'combination summary shows actual rank');
        typeValue(quickRoot().querySelector('input[aria-label="Dynamic rank custom value"]'), '2.5');
        quickRoot().querySelector('.lora-studio-quick-tune-apply').click();
        check(field('txt2img').value === initial && Boolean(quickRoot()), 'invalid rank blocks Apply without losing the other edits');
        quickRoot().querySelector('.lora-studio-quick-tune-full').click();
        check(Boolean(quickRoot()), 'invalid rank cannot silently disappear during Full Studio handoff');
        selectRecipe(quickRoot(), 'mode:te-off');
        check(quickRoot().querySelector('input[aria-label="Text encoder exact weight"]').value === '0.3', 'preset cannot silently discard an invalid field');
        [...quickRoot().querySelectorAll('.lora-studio-dyn-choice')].find(button => button.textContent === '8').click();
        check(!quickRoot().querySelector('input[aria-label="Dynamic rank custom value"]').validationMessage, 'rank shortcut clears previous validation error');
        quickRoot().querySelector('.lora-studio-quick-tune-apply').click();
        check(field('txt2img').value === 'portrait <lora:same:0.5> light <lora:same:te=0.3:unet=-0.75:dyn=8:plugin=keep>', 'all three settings are applied only to the selected occurrence');

        field('txt2img').value = 'portrait <lora:same:0.5> light <lora:same:te=0.3:unet=0@0,0.8@1:dyn=16:hr=0.4:lbw=OUTS:lbwe=ATTNDEEPOFF:plugin=keep>';
        updateInput(field('txt2img'));
        openSecond();
        check(Boolean(quickRoot().querySelector('.lora-studio-weight-curve')), 'curve + scalar + rank opens without empty scalar field');
        typeValue(quickRoot().querySelector('input[aria-label="Text encoder exact weight"]'), '0.2');
        quickRoot().querySelector('.lora-studio-quick-tune-apply').click();
        const combined = window.loraStudioModel.findTags(field('txt2img').value)[1].spec;
        check(combined.weights.unet.kind === 'curve' && combined.dyn_dim === 16 && combined.hires.unet.value === 0.4 && combined.elemental_weights.raw === 'ATTNDEEPOFF', 'editing TE keeps U-Net curve, DYN, Hires and module preset');

        fullRoot().querySelectorAll('.lora-studio-list-item')[0].click();
        typeValue(fullRoot().querySelector('input[aria-label="Text encoder exact weight"]'), '0.25');
        check(fullRoot().querySelector('.lora-studio-exact-weight input').value === '0.25', 'quick scalar edit synchronizes advanced exact field');
        typeValue(fullRoot().querySelector('.lora-studio-exact-weight input'), '0.35');
        check(fullRoot().querySelector('input[aria-label="Text encoder exact weight"]').value === '0.35', 'advanced exact edit synchronizes quick scalar field');
        typeValue(fullRoot().querySelector('.lora-studio-exact-weight input'), '');
        const beforeEmptyApply = field('txt2img').value;
        fullRoot().querySelector('.lora-studio-apply').click();
        check(field('txt2img').value === beforeEmptyApply && !fullRoot().querySelector('.lora-studio-discard').hidden, 'empty advanced exact field blocks Apply instead of using a hidden old value');
        const exactField = fullRoot().querySelector('.lora-studio-exact-weight input');
        typeValue(exactField, '0.2@0,0.6@1');
        exactField.dispatchEvent(new Event('change', {bubbles:true}));
        check(fullRoot().querySelector('.lora-studio-exact-weight input') === exactField && Boolean(fullRoot().querySelector('.lora-studio-weight-curve')), 'curve conversion refreshes quick view without replacing focused exact field');
        const curvePoint = fullRoot().querySelector('input[aria-label="Text encoder point 1 value"]');
        typeValue(curvePoint, '');
        curvePoint.dispatchEvent(new Event('change', {bubbles:true}));
        fullRoot().querySelector('.lora-studio-apply').click();
        check(field('txt2img').value === beforeEmptyApply, 'empty curve point is not converted to zero or applied');
        fullRoot().querySelector('.lora-studio-discard').click();
        field('txt2img').value = initial;
        updateInput(field('txt2img'));

        openSecond();
        selectRecipe(quickRoot(), 'elemental:attndeepoff');
        selectRecipe(quickRoot(), 'block:outs');
        check(quickRoot().querySelector('.lora-studio-recipe-current').textContent.includes('ATTNDEEPOFF'), 'block preset keeps existing module override visible');
        const saveName = quickRoot().querySelector('input[aria-label="Save preset name"]');
        saveName.value = 'Fixture preset';
        saveName.dispatchEvent(new Event('input', {bubbles:true}));
        quickRoot().querySelector('.lora-studio-quick-save button').click();
        await tick();
        check(library.presets.length > 0 && library.presets.at(-1).elemental_weights.raw === 'ATTNDEEPOFF', 'saving persists the combined control snapshot');
        check(field('txt2img').value === initial, 'saving preset does not apply draft');
        selectRecipe(quickRoot(), 'mode:clear-masks');
        selectRecipe(quickRoot(), `saved:${library.presets.at(-1).preset_id}`);
        check(quickRoot().querySelector('.lora-studio-quick-tune-preview code').textContent.includes('lbwe=ATTNDEEPOFF'), 'saved preset restores module override in quick editor');
        document.querySelector('#switch-tab').click();
        check(quickRoot().getClientRects().length === 0, 'txt2img popup hides when switching to img2img');
        check(field('img2img').value === 'image <lora:other:0.7>', 'img2img prompt remains independent');
        document.querySelector('#switch-tab').click();
        quickRoot().querySelector('.lora-studio-quick-tune-cancel').click();

        field('txt2img').value = '<lora:blocks:0.8:profile=classic-26>';
        updateInput(field('txt2img'));
        fullRoot().open = true;
        const blockCells = () => fullRoot().querySelectorAll('[data-lora-block-index]');
        check(blockCells().length === 26, 'all block weight fields are present');
        const firstBlock = blockCells()[0];
        typeValue(firstBlock, '0.35');
        check(fullRoot().querySelector('.lora-studio-block-weights input').value.startsWith('0.35,'), 'editing block cell synchronizes raw table while typing');
        firstBlock.dispatchEvent(new Event('change', {bubbles:true}));
        check(blockCells()[0] === firstBlock, 'leaving block cell keeps field and next click target');
        const blockDraft = fullRoot().querySelector('.lora-studio-tag-preview code').textContent;
        const beforeBlock = field('txt2img').value;
        typeValue(blockCells()[1], '');
        fullRoot().querySelector('.lora-studio-apply').click();
        check(field('txt2img').value === beforeBlock && fullRoot().querySelector('.lora-studio-tag-preview code').textContent === blockDraft, 'empty block cell is not converted to zero or applied');
        typeValue(firstBlock, '0.25');
        typeValue(blockCells()[1], '0.6');
        check(!firstBlock.validationMessage && !blockCells()[1].validationMessage, 'fixing incomplete block table clears only actual errors');
        const bulk = fullRoot().querySelector('.lora-studio-block-bulk');
        const beforeBulk = fullRoot().querySelector('.lora-studio-tag-preview code').textContent;
        typeValue(bulk.querySelector('input'), '');
        [...bulk.querySelectorAll('button')].find(button => button.textContent === 'Set').click();
        check(fullRoot().querySelector('.lora-studio-tag-preview code').textContent === beforeBulk, 'empty Set all command does not zero every block');
        const group = fullRoot().querySelector('.lora-studio-block-group-header');
        typeValue(group.querySelector('input'), '');
        group.querySelector('button').click();
        check(fullRoot().querySelector('.lora-studio-tag-preview code').textContent === beforeBulk, 'empty group command does not zero its blocks');

        const rawBlocks = fullRoot().querySelector('.lora-studio-block-weights input');
        typeValue(rawBlocks, 'OUTS');
        rawBlocks.dispatchEvent(new Event('change', {bubbles:true}));
        check(fullRoot().querySelector('.lora-studio-block-weights input') === rawBlocks, 'raw block text keeps its DOM field on blur');
        const moduleField = fullRoot().querySelector('.lora-studio-elemental-weights input');
        typeValue(moduleField, 'ATTNDEEPOFF');
        moduleField.dispatchEvent(new Event('change', {bubbles:true}));
        check(fullRoot().querySelector('.lora-studio-elemental-weights input') === moduleField, 'module preset text keeps its DOM field on blur');
        check(fullRoot().querySelector('.lora-studio-elemental-weights select').value === 'ATTNDEEPOFF', 'module preset text and selector agree');
        typeValue(fullRoot().querySelector('.lora-studio-exact-weight input'), '0.2@0,0.8@1');
        const point = fullRoot().querySelector('input[aria-label="Text encoder point 1 value"]');
        typeValue(point, '0.4');
        point.dispatchEvent(new Event('change', {bubbles:true}));
        check(fullRoot().querySelector('input[aria-label="Text encoder point 1 value"]') === point, 'curve point entry does not replace the focused field');
        check(fullRoot().querySelector('.lora-studio-exact-weight input').value === '0.4@0,0.8@1', 'curve points synchronize exact weights while typing');
        const backup = fullRoot().querySelector('.lora-studio-preset-transfer textarea');
        check(backup.value.includes('schema_version'), 'backup textarea starts with saved JSON rather than an empty panel');
        typeValue(backup, '');
        fullRoot().querySelector('.lora-studio-discard').click();
        check(fullRoot().querySelector('.lora-studio-preset-transfer textarea').value === '', 'intentionally cleared JSON stays empty after rerender');

        field('txt2img').value = '<lora:a:0.5:start=0.2:stop=0.8:lbw=MIDD> <lora:b:te=0.3:unet=0.5@5,1@9:hr=0.4:profile=classic-20:lbw=MIDD> <lora:a:0.7:dyn=1:lbwe=ATTNDEEPOFF> <lora:d:-0.2:plugin=keep>';
        updateInput(field('txt2img'));
        const fourBefore = window.loraStudioModel.findTags(field('txt2img').value);
        fullRoot().querySelectorAll('.lora-studio-list-item')[1].click();
        const rawCheck = fullRoot().querySelector('.lora-studio-block-weights input');
        typeValue(rawCheck, 'missing-preset');
        const invalidBefore = field('txt2img').value;
        fullRoot().querySelector('.lora-studio-apply').click();
        check(field('txt2img').value === invalidBefore && Boolean(rawCheck.validationMessage), 'unknown LBW preset blocks Apply instead of failing during generation');
        typeValue(rawCheck, 'MIDD');
        const elementalCheck = fullRoot().querySelector('.lora-studio-elemental-weights input');
        typeValue(elementalCheck, 'missing-module-preset');
        fullRoot().querySelector('.lora-studio-apply').click();
        check(field('txt2img').value === invalidBefore && Boolean(elementalCheck.validationMessage), 'unknown LBWE preset blocks Apply without changing other controls');
        typeValue(elementalCheck, '');
        check(fullRoot().querySelectorAll('[data-lora-block-index]').length === 20, 'MIDD maps by name into explicit classic-20 layout without an unusable empty panel');
        const blockSlider = fullRoot().querySelector('input[aria-label="M00 block weight slider"]');
        typeValue(blockSlider, '0.35');
        check(fullRoot().querySelector('input[aria-label="M00 exact block weight"]').value === '0.35', 'middle block slider synchronizes exact field');
        typeValue(fullRoot().querySelector('input[aria-label="M00 exact block weight"]'), '3.5');
        check(blockSlider.value === '3.5' && Number(blockSlider.max) >= 3.5, 'exact block value extends slider range without clamping');
        const curvePanels = [...fullRoot().querySelectorAll('.lora-studio-curve-editor')];
        const curvePanel = curvePanels.find(panel => panel.querySelector('h5').textContent === 'U-Net per sampling step');
        check(Boolean(curvePanel) && curvePanels.some(panel => panel.querySelector('h5').textContent === 'TE at text encoding (first point)'), 'curve headings distinguish TE encoding from U-Net sampling');
        const midpoint = [...curvePanel.querySelectorAll('button')].find(button => button.textContent === 'Add midpoint');
        check(!midpoint.disabled, 'absolute-step curve allows adding a point visually');
        midpoint.click();
        check(fullRoot().querySelector('input[aria-label="U-Net point 2 position"]').value === '7', 'absolute midpoint keeps steps 5 / 7 / 9');
        fullRoot().querySelector('.lora-studio-apply').click();
        const fourAfter = window.loraStudioModel.findTags(field('txt2img').value);
        check(fourAfter.length === 4 && [0, 2, 3].every(index => fourAfter[index].tag === fourBefore[index].tag), 'block plus curve edits affect only selected LoRA out of four');
        check(fourAfter[1].spec.hires.unet.value === 0.4 && fourAfter[1].spec.weights.te.value === 0.3, 'editing absolute curve and LBW preserves Hires and independent TE');

        opts.lora_block_weight_presets = 'MIDD:' + Array(17).fill(0.25).join(',');
        field('txt2img').value = '<lora:custom:0.8:lbw=MIDD>';
        updateInput(field('txt2img'));
        check([...fullRoot().querySelectorAll('[data-lora-block-index]')].every(input => input.value === '0.25'), 'custom preset overriding MIDD is displayed using its actual runtime values');
        opts.lora_block_weight_presets = '';

        if (typeof window.getComputedStyle === 'function') {
            const app = gradioApp();
            const savedWidth = app.style.maxWidth;
            app.style.maxWidth = '390px';
            await tick();
            check(fullRoot().querySelector('.lora-studio-editor').getBoundingClientRect().width > 300, 'narrow Studio uses its own width, not browser viewport, to stack the editor');
            app.style.maxWidth = savedWidth;
            const luminance = colour => {
                const channels = colour.match(/[\d.]+/g).slice(0, 3).map(value => {
                    const channel = Number(value) / 255;
                    return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
                });
                return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
            };
            const savedClass = document.body.className;
            for (const mode of ['dark', 'light', 'dark fixture-no-tokens', 'light fixture-no-tokens']) {
                document.body.className = mode;
                const controls = [...fullRoot().querySelectorAll('input, textarea, select')].filter(input => !['range', 'checkbox', 'radio'].includes(input.type));
                const readable = controls.every(input => {
                    const style = window.getComputedStyle(input);
                    const foreground = luminance(style.color);
                    const background = luminance(style.backgroundColor);
                    return (Math.max(foreground, background) + 0.05) / (Math.min(foreground, background) + 0.05) >= 4.5;
                });
                check(readable, `browser field contrast survives forms reset: ${mode}`);
            }
            document.body.className = savedClass;
        } else {
            output.textContent += '\nBrowser colour/layout checks not run in the simulated DOM.';
        }
        output.textContent += `\n${results.length} UI checks passed.`;
        output.dataset.result = 'passed';
    } catch (error) {
        output.textContent += `\nFAIL ${error.stack}`;
        output.dataset.result = 'failed';
    }
};
