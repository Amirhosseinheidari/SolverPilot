from __future__ import annotations
import importlib.util
from pathlib import Path
import solverpilot as om
from solverpilot.cp.ortools_backend import _translate_to_cp_sat, VERIFIED_ORTOOLS_VERSION


class _Expr:
    def __init__(self, text): self.text=str(text)
    def __add__(self,o): return _Expr(f'({self.text}+{getattr(o,"text",o)})')
    __radd__=__add__
    def __mul__(self,o): return _Expr(f'({self.text}*{o})')
    __rmul__=__mul__
    def __eq__(self,o): return ('eq',self.text,getattr(o,'text',o))
    def __ge__(self,o): return ('ge',self.text,getattr(o,'text',o))
    def __le__(self,o): return ('le',self.text,getattr(o,'text',o))

class _Domain:
    @staticmethod
    def from_values(values): return ('domain',tuple(values))

class _FakeModel:
    def __init__(self): self.calls=[]
    def new_bool_var(self,name): self.calls.append(('bool',name)); return _Expr(name)
    def new_int_var_from_domain(self,dom,name): self.calls.append(('int',name,dom)); return _Expr(name)
    def add(self,x): self.calls.append(('add',x)); return x
    def new_interval_var(self,s,size,e,name): self.calls.append(('interval',name,size)); return ('iv',name)
    def add_all_different(self,x): self.calls.append(('alldiff',len(x)))
    def add_exactly_one(self,x): self.calls.append(('exactlyone',len(x)))
    def add_allowed_assignments(self,x,t): self.calls.append(('table',len(x),tuple(t)))
    def add_element(self,i,v,t): self.calls.append(('element',tuple(v)))
    def add_circuit(self,a): self.calls.append(('circuit',len(a)))
    def add_no_overlap(self,i): self.calls.append(('nooverlap',len(i)))
    def add_cumulative(self,i,d,c): self.calls.append(('cumulative',len(i),tuple(d),c))
    def minimize(self,e): self.calls.append(('minimize',e.text))
    def maximize(self,e): self.calls.append(('maximize',e.text))

class _FakeCP:
    CpModel=_FakeModel
    Domain=_Domain


def test_static_translation_uses_true_boolvars_and_all_p9_families():
    m=om.CPModel('static-contract')
    b=[m.bool_var(f'b{i}') for i in range(3)]
    x=m.int_var(0,2,'x'); y=m.int_var(0,2,'y')
    m.add_exactly_one(b); m.add_all_different([x,y])
    m.add_allowed_assignments([x,y],[(0,1),(1,0)])
    m.add_element(x,[2,1,0],y)
    m.add_circuit([(0,0,b[0]),(1,1,b[1]),(0,1,b[2]),(1,0,b[2])])
    s1=m.int_var(0,3,'s1'); s2=m.int_var(0,3,'s2')
    i1=m.interval_var(s1,1,'i1'); i2=m.interval_var(s2,1,'i2')
    m.add_no_overlap([i1,i2]); m.add_cumulative([i1,i2],[1,1],1); m.minimize(x+y)
    fm,_,_=_translate_to_cp_sat(m.compile(),_FakeCP)
    kinds=[c[0] for c in fm.calls]
    assert kinds.count('bool')==3
    for expected in ('exactlyone','alldiff','table','element','circuit','nooverlap','cumulative','minimize'):
        assert expected in kinds


def test_verified_binding_lock_is_exact():
    assert VERIFIED_ORTOOLS_VERSION=='9.15.6755'


def test_track_p_merge_provenance_is_frozen_and_hash_locked():
    root = Path(__file__).resolve().parents[1]
    import json
    manifest = json.loads((root / "TRACK-P-MERGE-PROVENANCE.json").read_text())
    assert manifest["source_bundle"]["sha256"] == "6cc282b9428450ab7bbc39cd6c768de2da2fc092f8d7f79cbbf319f657df24a2"
    assert manifest["source_bundle"]["internal_checksum_rows"] == 816
    assert manifest["source_bundle"]["internal_checksum_mismatches"] == 0
    assert manifest["ortools_qualification"]["version"] == "9.15.6755"
    assert manifest["ortools_qualification"]["wheel_sha256"] == "ebd5aea00374e3aad7a78de59058aca5e871a26a3c385cd0860ef1d685d03c9a"
    assert manifest["integration"]["version"] == "0.0.40rc2"


def test_cp_canonical_json_roundtrip_preserves_hashes():
    from solverpilot.cp.serialization import problem_from_canonical_dict
    m=om.CPModel('roundtrip')
    x=m.int_var(0,3,'x'); y=m.int_var(0,3,'y'); b=m.bool_var('b')
    m.add(x+y>=2); m.add_exactly_one([b]); m.minimize(2*x+y)
    p=m.compile(); q=problem_from_canonical_dict(p.canonical_dict())
    assert q.canonical_dict()==p.canonical_dict()
    assert q.structural_hash==p.structural_hash
    assert q.data_hash==p.data_hash


def test_ortools_backend_uses_isolated_worker_and_parent_revalidation_contract():
    import inspect
    import solverpilot.cp.ortools_backend as ob
    solve_text=inspect.getsource(ob.ORToolsCPSATBackend.solve)
    command=ob._isolated_worker_command()
    assert '-I' in command
    assert 'solverpilot.cp.ortools_worker' in command[command.index('-c')+1]
    assert 'validate_cp_solution(problem' in solve_text
    assert 'worker_request_bound' in solve_text
    assert 'worker_binding_verified' in solve_text
    assert 'worker_proof_consistent' in solve_text
