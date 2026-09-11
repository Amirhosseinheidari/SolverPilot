import json
from importlib.resources import files
import pytest
import solverpilot as om
from solverpilot.cp.ir import (
    CPAllDifferentIR, CPExactlyOneIR, CPTableIR, CPElementIR,
    CPCircuitIR, CPNoOverlapIR, CPCumulativeIR,
)


def test_domains_bool_int_and_hash_stability():
    m=om.CPModel('d'); x=m.int_var_from_values([3,1,3,2],'x'); b=m.bool_var('b'); p=m.compile()
    assert x.domain==(1,2,3) and b.domain==(0,1)
    assert p.data_hash==m.compile().data_hash and p.structural_hash==m.compile().structural_hash


def test_domain_data_hash_changes_but_structural_hash_does_not():
    a=om.CPModel(); a.int_var(0,2); pa=a.compile()
    b=om.CPModel(); b.int_var(4,6); pb=b.compile()
    assert pa.structural_hash==pb.structural_hash
    assert pa.data_hash!=pb.data_hash


def test_linear_expression_and_minimize():
    m=om.CPModel(); x=m.int_var(0,4,'x'); y=m.int_var(0,4,'y'); m.add(x+y>=5); m.minimize(2*x+y)
    r=m.solve(); assert r.optimality_proven and r.validation.valid
    # domains cap y at 4, hence x=1,y=4 => objective 6
    assert r.objective==6 and r.assignment[x.var_id]==1 and r.assignment[y.var_id]==4


def test_maximize():
    m=om.CPModel(); x=m.int_var(0,3); y=m.int_var(0,3); m.add(x+y<=4); m.maximize(2*x+y)
    r=m.solve(); assert r.objective==7 and r.optimality_proven


def test_all_different():
    m=om.CPModel(); xs=[m.int_var(1,3) for _ in range(3)]; m.add_all_different(xs); m.add(xs[0]==1); m.minimize(xs[2]); p=m.compile()
    assert isinstance(p.constraints[0],CPAllDifferentIR)
    r=m.solve(); assert r.objective==2 and len({r.assignment[x.var_id] for x in xs})==3


def test_exactly_one():
    m=om.CPModel(); bs=[m.bool_var() for _ in range(4)]; m.add_exactly_one(bs); m.maximize(sum((i+1)*b for i,b in enumerate(bs)))
    p=m.compile(); assert any(isinstance(c,CPExactlyOneIR) for c in p.constraints)
    r=m.solve(); assert r.objective==4 and sum(r.assignment[b.var_id] for b in bs)==1


def test_table_allowed_assignments():
    m=om.CPModel(); x=m.int_var(0,2); y=m.int_var(0,2); m.add_allowed_assignments([x,y],[(0,2),(2,0)]); m.minimize(x+y)
    assert isinstance(m.compile().constraints[0],CPTableIR)
    r=m.solve(); assert r.validation.valid and (r.assignment[x.var_id],r.assignment[y.var_id]) in {(0,2),(2,0)}


def test_table_arity_error():
    m=om.CPModel(); x=m.int_var(0,1); y=m.int_var(0,1)
    with pytest.raises(ValueError,match='arity'): m.add_allowed_assignments([x,y],[(1,)])


def test_element():
    m=om.CPModel(); i=m.int_var(0,2); t=m.int_var(0,10); m.add_element(i,[4,7,9],t); m.add(i==1); m.minimize(t)
    assert isinstance(m.compile().constraints[0],CPElementIR)
    r=m.solve(); assert r.objective==7 and r.assignment[t.var_id]==7


def test_element_bad_index_is_infeasible():
    m=om.CPModel(); i=m.int_var(3,4); t=m.int_var(0,10); m.add_element(i,[4,7,9],t)
    r=m.solve(); assert r.status=='infeasible' and r.optimality_proven


def test_interval_derived_end_and_validation():
    m=om.CPModel(); s=m.int_var(0,3); iv=m.interval_var(s,2,'task'); m.add(s==1); r=m.solve()
    assert r.validation.valid and r.assignment[iv.end.var_id]==3


def test_no_overlap_touching_allowed():
    m=om.CPModel(); s1=m.int_var(0,2); s2=m.int_var(0,2); a=m.interval_var(s1,2); b=m.interval_var(s2,2); m.add_no_overlap([a,b]); m.minimize(s1+s2)
    assert isinstance(m.compile().constraints[0],CPNoOverlapIR)
    r=m.solve(); assert r.validation.valid and sorted([r.assignment[s1.var_id],r.assignment[s2.var_id]])==[0,2]


def test_cumulative_capacity():
    m=om.CPModel(); starts=[m.int_var(0,2) for _ in range(3)]; iv=[m.interval_var(s,2) for s in starts]; m.add_cumulative(iv,[2,2,1],3); m.minimize(sum(starts))
    assert isinstance(m.compile().constraints[0],CPCumulativeIR)
    r=m.solve(); assert r.validation.valid
    # jobs with demand 2 cannot overlap each other
    assert abs(r.assignment[starts[0].var_id]-r.assignment[starts[1].var_id])>=2


def test_cumulative_negative_demand_rejected():
    m=om.CPModel(); s=m.int_var(0,1); i=m.interval_var(s,1)
    with pytest.raises(ValueError,match='nonnegative'): m.add_cumulative([i],[-1],2)


def test_circuit_three_node_cycle():
    m=om.CPModel(); arcs=[]; lits={}
    for i,j in [(0,1),(1,2),(2,0),(0,0),(1,1),(2,2)]:
        lits[i,j]=m.bool_var(f'a{i}{j}'); arcs.append((i,j,lits[i,j]))
    m.add_circuit(arcs)
    for e in [(0,1),(1,2),(2,0)]: m.add(lits[e]==1)
    for e in [(0,0),(1,1),(2,2)]: m.add(lits[e]==0)
    assert isinstance(m.compile().constraints[0],CPCircuitIR)
    r=m.solve(); assert r.validation.valid


def test_circuit_empty_tour_via_self_loops_is_valid_cp_sat_semantics():
    m=om.CPModel(); loops=[]
    for i in range(3):
        b=m.bool_var(); loops.append((i,i,b)); m.add(b==1)
    m.add_circuit(loops); r=m.solve(); assert r.validation.valid


def test_circuit_skipped_node_requires_selected_self_loop():
    m=om.CPModel(); a01=m.bool_var(); a10=m.bool_var(); l2=m.bool_var(); m.add_circuit([(0,1,a01),(1,0,a10),(2,2,l2)]); m.add(a01==1); m.add(a10==1); m.add(l2==1)
    assert m.solve().validation.valid


def test_circuit_duplicate_self_loop_rejected():
    m=om.CPModel(); a=m.bool_var(); b=m.bool_var()
    with pytest.raises(ValueError,match='multiple self-loops'): m.add_circuit([(0,0,a),(0,0,b)])


def test_reference_state_limit_fail_closed():
    m=om.CPModel(); [m.int_var(0,9) for _ in range(5)]
    r=m.solve(om.ReferenceCPBackend(max_states=10)); assert r.status=='unknown_state_limit' and not r.optimality_proven


def test_reference_infeasibility_proof_by_full_enumeration():
    m=om.CPModel(); x=m.int_var(0,1); m.add(x>=2); r=m.solve(); assert r.status=='infeasible' and r.optimality_proven


def test_validator_rejects_wrong_objective():
    m=om.CPModel(); x=m.int_var(0,1); m.minimize(x); p=m.compile(); rep=om.validate_cp_solution(p,{x.var_id:0},objective_reported=1)
    assert not rep.valid and rep.objective_consistent is False


def test_validator_rejects_fractional_not_possible_by_int_cast_semantics():
    m=om.CPModel(); x=m.int_var(0,2); p=m.compile(); rep=om.validate_cp_solution(p,{x.var_id:7}); assert not rep.valid


def test_reference_conformance_passes():
    r=om.reference_cp_conformance(); assert r.passed and r.available and all(c.passed for c in r.checks)


def test_ortools_adapter_availability_contract(monkeypatch):
    b=om.ORToolsCPSATBackend()
    m=om.CPModel(); x=m.bool_var(); m.maximize(x)
    if b.is_available():
        r=b.solve(m.compile())
        assert r.validation.valid and r.optimality_proven and r.objective==1
    else:
        with pytest.raises(RuntimeError,match='not installed|not the verified version'):
            b.solve(m.compile())

    # Fail-closed behavior must remain testable even on machines where the
    # exact verified OR-Tools binding is installed.
    import solverpilot.cp.ortools_backend as ob
    monkeypatch.setattr(ob.importlib.util, 'find_spec', lambda name: None if name=='ortools' else __import__('importlib').util.find_spec(name))
    bb=om.ORToolsCPSATBackend()
    with pytest.raises(RuntimeError,match='not installed'):
        bb.solve(m.compile())


def test_ortools_conformance_matches_runtime_availability():
    b=om.ORToolsCPSATBackend()
    r=om.ortools_cp_sat_conformance()
    assert r.verified_version=='9.15.6755'
    assert r.available is b.is_available()
    if b.is_available():
        assert r.passed and all(c.passed for c in r.checks)
    else:
        assert not r.passed


def test_schema_packaged_source_tree():
    schema=json.loads(files('solverpilot.cp').joinpath('cp-ir-v1.schema.json').read_text()); assert schema['$id']=='solverpilot.cp-ir.v1'


def test_cross_model_expression_rejected():
    a=om.CPModel(); b=om.CPModel(); x=a.int_var(0,1); y=b.int_var(0,1)
    with pytest.raises(ValueError,match='cross-model'): _=x+y


def test_jobshop_small_known_optimum():
    # Two jobs: J0 M0(2)->M1(1); J1 M1(2)->M0(1). Optimal makespan=3.
    m=om.CPModel();
    s00=m.int_var(0,3,'s00'); s01=m.int_var(0,3,'s01'); s10=m.int_var(0,3,'s10'); s11=m.int_var(0,3,'s11')
    i00=m.interval_var(s00,2,'j0m0'); i01=m.interval_var(s01,1,'j0m1'); i10=m.interval_var(s10,2,'j1m1'); i11=m.interval_var(s11,1,'j1m0')
    m.add(s01>=i00.end); m.add(s11>=i10.end); m.add_no_overlap([i00,i11]); m.add_no_overlap([i10,i01])
    makespan=m.int_var(0,6,'makespan'); m.add(makespan>=i01.end); m.add(makespan>=i11.end); m.minimize(makespan)
    r=m.solve(om.ReferenceCPBackend(max_states=2_000_000)); assert r.optimality_proven and r.validation.valid and r.objective==3


def test_cumulative_event_point_semantics_half_open():
    m=om.CPModel(); s1=m.int_var_from_values([0]); s2=m.int_var_from_values([2]); i1=m.interval_var(s1,2); i2=m.interval_var(s2,2); m.add_cumulative([i1,i2],[3,3],3)
    assert m.solve().validation.valid  # touching at t=2 does not overlap


def test_metadata_and_version():
    assert om.__version__=='0.1'; p=om.CPModel('x').compile(); assert p.metadata['compiled_by']=='solverpilot-p9-cp-core'
