"""Cross-language checks for combined tags and saved presets, without Torch."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'extensions-builtin/Lora'))
import lora_schema  # noqa: E402
import lora_presets  # noqa: E402


@pytest.mark.parametrize('tag', [
    '<lora:a:te=0.3:unet=0.5@5,1@9:start=5:stop=0.8:lbw=MIDD:dyn=1:hr=0.4>',
    '<lora:a:0.8:step=0.2-0.8:lbw=NONE:lbwe=ATTNDEEPON:profile=classic-20>',
    '<loractl:a:te=0.2@0,0.6@1:unet=-0.5:hrunet=0@0,0.4@1:lbwe=PROJDEEPOFF>',
    '<lyco:a:0.4:dyn=16:plugin=keep:lbw=OUTS:start=0:stop=1>',
])
def test_browser_native_preset_tag_contract(tag):
    program = """
    const m = require('./javascript/loraStudioModel.js');
    const p = require('./javascript/loraStudioPresets.js');
    const spec = m.parseTag(JSON.parse(process.argv[1]));
    console.log(JSON.stringify({spec, tag:m.serializeTag(spec), preset:p.createPreset('matrix', spec, 'matrix')}));
    """
    result = json.loads(subprocess.check_output(['node', '-e', program, json.dumps(tag)], cwd=ROOT, text=True))
    native = lora_schema.parse_lora_tag(tag)
    assert result['spec'] == native.to_contract()
    assert lora_schema.parse_lora_tag(result['tag']).to_contract() == native.to_contract()
    # Exercise the actual preset validation used by the API, including mixed units.
    normalized = lora_presets.parse_preset_library({'schema_version': 3, 'presets': [result['preset']]})
    assert normalized[0].active_range.to_contract() == result['spec']['active_range']


def test_empty_lbw_entry_is_not_silently_removed():
    with pytest.raises(ValueError, match='empty entry'):
        lora_schema.parse_lora_tag('<lora:a:1:lbw=' + ','.join(['1'] * 17) + ',>')
