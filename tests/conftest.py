from __future__ import annotations
import json
from pathlib import Path
import pytest

_CONFIG=json.loads(Path(__file__).with_name('test-tiers.json').read_text(encoding='utf-8'))
_NATIVE_MODULES={Path(p).name for p in _CONFIG['native_files']}
_EVIDENCE_NODES={(Path(n.split('::',1)[0]).name,n.split('::',1)[1]) for n in _CONFIG['evidence_nodes']}

def pytest_collection_modifyitems(items:list[pytest.Item])->None:
    for item in items:
        basename=Path(str(item.fspath)).name
        if basename in _NATIVE_MODULES: item.add_marker(pytest.mark.native)
        if (basename,item.name) in _EVIDENCE_NODES: item.add_marker(pytest.mark.evidence)
