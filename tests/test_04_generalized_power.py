import json
import numpy as np
import pytest
from scipy import sparse
from solverpilot.model import Model, GeneralizedPowerCone
from solverpilot.conic import ConeAffineBlock, ConeKind
from solverpilot.conic.validation import _cone_check


@pytest.mark.parametrize(
    "weights,tail",
    [
        ((0.2, 0.3), 1),
        ((0.0, 1.0), 1),
        ((float("nan"), 1.0), 1),
        ((True, 0.5), 1),
        ((0.5, 0.5), 0),
        ((0.5, 0.5), True),
    ],
)
def test_invalid_definition(weights, tail):
    with pytest.raises(ValueError):
        GeneralizedPowerCone(weights, tail)


def test_hash_and_immutable_weights():
    weights = [0.2, 0.3, 0.5]
    block = ConeAffineBlock(
        ConeKind.GENERALIZED_POWER,
        sparse.eye(5),
        np.zeros(5),
        (5,),
        metadata={"weights": weights, "tail_dimension": 2},
    )
    weights[0] = 0.8
    assert block.metadata["weights"] == (0.2, 0.3, 0.5)
    changed = ConeAffineBlock(
        ConeKind.GENERALIZED_POWER,
        sparse.eye(5),
        np.zeros(5),
        (5,),
        metadata={"weights": (0.3, 0.2, 0.5), "tail_dimension": 2},
    )
    assert changed.structural_hash != block.structural_hash
    assert json.loads(json.dumps(block.to_canonical_dict()))["kind"] == "generalized_power"


def test_boundary_and_extreme_scaling():
    def check(x):
        return _cone_check(
            ConeKind.GENERALIZED_POWER,
            np.array(x),
            atol=1e-10,
            rtol=1e-10,
            source_id=None,
            weights=(0.5, 0.5),
        ).valid

    assert check([0.0, 1.0, 0.0])
    assert not check([0.0, 1.0, 0.01])
    assert check([1e300, 1e300, 1e300])
    assert not check([-1e-5, 1e300, 0.0])
    assert not check([1.0, 1.0, 2.0])


@pytest.mark.parametrize("fixed", [[1.0, 4.0, 9.0], [4.0, 4.0, 4.0], [0.01, 100.0, 1.0]])
def test_native_model_transport_and_original_validation(fixed):
    pytest.importorskip("clarabel")
    from solverpilot.conic import ClarabelBackend

    m = Model()
    x = m.variable(5, lower=fixed + [-10.0, -10.0], upper=fixed + [10.0, 10.0])
    m.add_in_set(x, GeneralizedPowerCone((0.2, 0.3, 0.5), 2))
    m.minimize(-x[3])
    compiled = m.compile()
    result = compiled.solve(backend=ClarabelBackend(tolerance=1e-10))
    assert result.backend_status in ("Solved", "AlmostSolved")
    assert result.validation.valid
    assert result.objective_reported == pytest.approx(
        -np.exp(np.dot([0.2, 0.3, 0.5], np.log(fixed))), abs=1e-7
    )
    assert compiled.execution_ir.cones[0].kind is ConeKind.GENERALIZED_POWER
    import jsonschema
    from importlib.resources import files

    schema = json.loads(files("solverpilot.conic").joinpath("conic-ir-v1.schema.json").read_text())
    jsonschema.validate(compiled.execution_ir.to_canonical_dict(), schema)
