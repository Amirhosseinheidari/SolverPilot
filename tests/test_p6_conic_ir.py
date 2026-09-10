from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

import solverpilot as om
from solverpilot.conic import ConeAffineBlock, ConeKind, ConicProblem, validate_conic_solution


def _base_problem(blocks):
    return ConicProblem.from_data(
        P=np.zeros((2,2)), q=np.array([1.0, 0.0]), A=np.zeros((0,2)),
        variable_lower=np.array([-5.0,-5.0]), variable_upper=np.array([5.0,5.0]),
        constraint_lower=np.zeros(0), constraint_upper=np.zeros(0), cones=blocks,
    )


def test_cone_block_is_immutable_and_hashes_data_separately():
    F=sparse.csr_matrix([[1.,0.],[0.,1.]])
    a=ConeAffineBlock(ConeKind.SECOND_ORDER,F,np.array([1.,2.]),(2,))
    b=ConeAffineBlock(ConeKind.SECOND_ORDER,F,np.array([1.,3.]),(2,))
    assert a.structural_hash == b.structural_hash
    assert a.data_hash != b.data_hash
    assert not a.g.flags.writeable
    with pytest.raises(ValueError):
        a.g[0]=9


def test_soc_validation_accepts_and_rejects():
    block=ConeAffineBlock(ConeKind.SECOND_ORDER,sparse.eye(2),np.zeros(2),(2,))
    p=_base_problem([block])
    assert validate_conic_solution(p,np.array([2.,1.])).valid
    bad=validate_conic_solution(p,np.array([0.5,1.]))
    assert not bad.valid and bad.max_cone_violation > 0


def test_rotated_soc_validation():
    F=sparse.eye(3,format='csr')[:,:2]
    # y=[x0,x1,1]
    block=ConeAffineBlock(ConeKind.ROTATED_SECOND_ORDER,F,np.array([0.,0.,1.]),(3,))
    p=_base_problem([block])
    assert validate_conic_solution(p,np.array([1.,1.])).valid
    assert not validate_conic_solution(p,np.array([0.1,0.1])).valid


def test_psd_validation():
    # [[x0,0],[0,x1]]
    F=sparse.csr_matrix([[1,0],[0,0],[0,0],[0,1]],dtype=float)
    block=ConeAffineBlock(ConeKind.POSITIVE_SEMIDEFINITE,F,np.zeros(4),(2,2))
    p=_base_problem([block])
    assert validate_conic_solution(p,np.array([1.,2.])).valid
    assert not validate_conic_solution(p,np.array([-1.,2.])).valid


def test_conic_problem_rejects_nonconvex_quadratic():
    block=ConeAffineBlock(ConeKind.SECOND_ORDER,sparse.csr_matrix([[1.,0.],[0.,1.]]),np.zeros(2),(2,))
    with pytest.raises(ValueError, match='not convex'):
        ConicProblem.from_data(P=np.diag([-1.,1.]),q=np.zeros(2),A=np.zeros((0,2)),variable_lower=-np.ones(2),variable_upper=np.ones(2),constraint_lower=np.zeros(0),constraint_upper=np.zeros(0),cones=[block])


def test_requirements_v2_detect_cones_and_verified_bridge():
    m=om.Model(); x=m.variable(1); t=m.variable(lower=0); m.soc(t,x); m.minimize(t)
    p=m.compile(use_cache=False).execution_ir
    req=om.requirements_v2_for(p)
    assert om.CapabilityKey.PROBLEM_CONIC in req.required
    assert om.CapabilityKey.CONSTRAINT_SOC in req.required
    backend=om.CasadiSuperSCSBackend()
    manifest=backend.capability_manifest_v2
    ok0,_=om.compatible_v2(manifest, req)
    ok1,checks=om.compatible_v2(manifest, req, allow_safe_emulation=True)
    assert not ok0
    assert ok1
    assert all(c.usable for c in checks)
