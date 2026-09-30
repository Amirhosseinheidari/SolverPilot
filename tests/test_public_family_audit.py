import importlib
from pathlib import Path
import pytest


@pytest.mark.parametrize('train,test', [('greenbea','greenbeb'),('pilot4','pilotnov'),
                                     ('pilot.ja','pilot.we'),('standata','standgub')])
def test_alphabetic_variants_cannot_leak_across_public_split(monkeypatch,train,test):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'benchmarks'))
    audit=importlib.import_module('public_lp_families').audit_split
    def case(name,digest):return dict(source='netlib',name=name,group='netlib:'+name,data_hash=digest)
    result=audit([case(train,'a'*64)],[case(test,'b'*64)])
    assert not result['passed'] and result['overlapping_families']
    assert not audit([], [case(test,'b'*64)],prior_names=[train])['passed']
    assert audit([case(train,'a'*64)],[case('unrelated','b'*64)])['passed']
