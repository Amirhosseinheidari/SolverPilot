"""Current signatures and patch-line compatibility preserve the full API."""
import inspect
import json
from pathlib import Path

import solverpilot


def test_complete_current_public_api_snapshot():
    root=Path(__file__).resolve().parents[1]
    payload=json.loads((root/'PUBLIC-API-V0_4_2.json').read_text(encoding='utf-8'))
    assert payload['package_version']==solverpilot.__version__=='0.4.2'
    assert [row['name'] for row in payload['symbols']]==solverpilot.__all__
    for row in payload['symbols']:
        obj=getattr(solverpilot,row['name'])
        try: signature=str(inspect.signature(obj))
        except (ValueError,TypeError): signature=None
        assert row=={'name':row['name'],'module':getattr(obj,'__module__',None),
                     'qualname':getattr(obj,'__qualname__',None),'signature':signature}


def test_patch_release_preserves_full_0_4_1_signatures():
    root=Path(__file__).resolve().parents[1]
    old=json.loads((root/'PUBLIC-API-V0_4_1.json').read_text(encoding='utf-8'))
    current=json.loads((root/'PUBLIC-API-V0_4_2.json').read_text(encoding='utf-8'))
    assert old['package_version']=='0.4.1'
    assert old['symbols']==current['symbols']
