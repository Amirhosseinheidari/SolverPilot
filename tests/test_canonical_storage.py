import pickle
import json
import numpy as np
import pytest
from scipy import sparse

import solverpilot as sp
from solverpilot.conic.ir import ConicProblem, ConeAffineBlock, ConeKind


def lp():
    return sp.LinearProblem.from_data(A=[[1.,2.]], c=[1.,2.], variable_lower=[0.,0.],
        variable_upper=[2.,3.], constraint_lower=[0.], constraint_upper=[5.])


def assert_array_frozen(a):
    before = np.array(a, copy=True)
    for edit in [lambda: a.setflags(write=True), lambda: setattr(a, 'dtype', np.uint8),
                 lambda: setattr(a, 'shape', (a.size,1)), lambda: a.resize((a.size,1))]:
        with pytest.raises((ValueError, AttributeError)):
            edit()
    np.testing.assert_array_equal(a, before)
    b = a.copy()
    b.setflags(write=True)
    if b.size:
        b.flat[0] = 0
    np.testing.assert_array_equal(a, before)


def assert_matrix_frozen(a):
    for buf in (a.data, a.indices, a.indptr):
        assert_array_frozen(buf)
    for edit in [lambda: setattr(a, 'data', np.zeros(a.nnz)), lambda: setattr(a, 'shape', (2,1)),
                 lambda: a.set_shape((2,1)), lambda: a.__dict__.__setitem__('data', np.zeros(a.nnz)),
                 lambda: a.__dict__.__setitem__('_canonical_frozen', False)]:
        with pytest.raises((ValueError, AttributeError, TypeError)):
            edit()
    b = a.copy()
    b.data[:] = 0
    if a.nnz:
        assert np.any(a.data != 0)


@pytest.mark.parametrize('roundtrip', [False, True])
def test_lp_and_qp_buffers_preserve_identity_and_curvature(roundtrip):
    p = lp()
    q = sp.QuadraticProblem(p, sparse.eye(2), sp.ConvexityStatus.CONFIRMED)
    identity = p.data_hash, q.data_hash
    if roundtrip:
        p, q = pickle.loads(pickle.dumps((p,q)))
    for a in (p.c,p.variable_lower,p.variable_upper,p.constraint_lower,p.constraint_upper,p.domains):
        assert_array_frozen(a)
    assert_matrix_frozen(p.A)
    assert_matrix_frozen(q.P)
    assert (p.data_hash,q.data_hash) == identity
    assert q.convexity_status is sp.ConvexityStatus.CONFIRMED


def test_sparse_operations_on_owned_storage_remain_compatible():
    p = lp()
    a = p.A
    np.testing.assert_array_equal(a @ [1.,2.], [5.])
    for b in (a.copy(),a.astype(bool),a.T.tocsc(),a+a, a*.5,abs(a),a[:,[1,0]],sparse.vstack([a,a])):
        assert np.isfinite(b.data).all()
        restored = pickle.loads(pickle.dumps(b))
        assert restored.dtype == b.dtype
        restored.data[:] = 0
    assert sp.solve(p,backend='scipy-highs-ds').validation.valid


def test_conic_storage_and_pickle_preserve_identity():
    block = ConeAffineBlock(ConeKind.POWER, np.zeros((3,1)), [1.,1.,0.], (3,), metadata={'alpha':np.array(.5)})
    p = ConicProblem.from_data(P=[[1.]], q=[0.], A=[[1.]], variable_lower=[0.],
        variable_upper=[1.], constraint_lower=[0.], constraint_upper=[1.], cones=[block])
    restored = pickle.loads(pickle.dumps(p))
    assert restored.data_hash == p.data_hash and type(restored.cones[0].metadata['alpha']) is float
    assert_array_frozen(restored.q)
    assert_array_frozen(restored.cones[0].g)
    assert_matrix_frozen(restored.P)
    assert_matrix_frozen(restored.cones[0].F)
    nested = ConeAffineBlock(ConeKind.SECOND_ORDER, [[0.],[0.]], [1.,0.], (2,), metadata={'custom':{'label':'a'}})
    assert json.loads(json.dumps(nested.to_canonical_dict()))['metadata']['custom']['label'] == 'a'
    assert pickle.loads(pickle.dumps(nested)).data_hash == nested.data_hash
    offset = np.array(0.)
    from dataclasses import replace
    direct = replace(p,objective_offset=offset)
    identity = direct.data_hash
    offset[...] = 100.
    assert direct.objective_value([0.]) == 0. and direct.data_hash == identity


def test_nlp_parameter_snapshots_cannot_change_during_a_solve():
    m = sp.Model()
    x = m.variable(lower=0,upper=2)
    parameter = m.parameter(value=1.)
    m.minimize(sp.exp(x)*parameter)
    p = m.compile().execution_ir
    identity = p.data_hash
    for a in (p.variable_lower,p.variable_upper,*p.parameter_values.values()):
        assert_array_frozen(a)
    restored = pickle.loads(pickle.dumps(p))
    assert restored.data_hash == identity
    parameter.value = 2.
    assert p.data_hash == identity and m.compile().execution_ir.data_hash != identity
