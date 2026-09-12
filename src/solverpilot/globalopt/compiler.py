import hashlib
import numpy as np
from solverpilot.model.compiler import CompiledModel, CompilationReport, _variable_layout
from solverpilot.model.model import IndicatorConstraint
from solverpilot.model.sets import LessThan, GreaterThan, EqualTo, Interval
from solverpilot.model.errors import CompileError
from solverpilot.nlp.ir import NLPProblem, NLPConstraintBlock
from .problem import FactorableProblem, GlobalIndicator
from .expressions import freeze_node
from .intervals import validate_domains, bound_node


def compile_global_model(model, *, use_cache=True):
    """Compile an explicit bounded global path without relaxing the convex/OA APIs."""
    if model.objective is None:
        raise CompileError("model has no objective")
    offsets, n, lo, hi, domains, source_variables = _variable_layout(model)
    layout = {key: (offsets[key][0], data.shape) for key, data in model._variables.items()}
    values = {key: data.value.copy() for key, data in model._parameters.items()}
    blocks = []
    indicators = []
    source_constraints = {}
    for c in model.constraints:
        if not isinstance(c.set, (LessThan, GreaterThan, EqualTo, Interval)):
            raise CompileError(
                "global factorable compilation accepts algebraic scalar sets, not cones"
            )
        node = freeze_node(c.function._node)
        size = c.function.size

        def vector(value):
            return (
                np.broadcast_to(np.asarray(value, dtype=float), c.function.shape).reshape(-1).copy()
            )

        if isinstance(c.set, LessThan):
            lower = np.full(size, -np.inf)
            upper = vector(c.set.upper)
        elif isinstance(c.set, GreaterThan):
            lower = vector(c.set.lower)
            upper = np.full(size, np.inf)
        elif isinstance(c.set, EqualTo):
            lower = vector(c.set.value)
            upper = lower.copy()
        else:
            lower = vector(c.set.lower)
            upper = vector(c.set.upper)
        block = NLPConstraintBlock(c.entity_id.value, node, lower, upper, c.function.shape)
        if isinstance(c, IndicatorConstraint):
            indicators.append(
                GlobalIndicator(int(offsets[c.indicator.id.value][0]), c.active_value, block)
            )
        else:
            blocks.append(block)
        source_constraints[c.entity_id.value] = {
            "kind": "indicator" if isinstance(c, IndicatorConstraint) else "algebraic",
            "shape": list(c.function.shape),
        }
    relaxation = NLPProblem(
        n,
        lo,
        hi,
        layout,
        values,
        freeze_node(model.objective.expression._node),
        tuple(blocks),
        model.objective.sense.value,
        model.name,
    )
    problem = FactorableProblem(
        relaxation, tuple(domains), tuple(indicators), {"semantic_hash": model.semantic_hash}
    )
    for node in (
        relaxation.objective_node,
        *(c.node for c in blocks),
        *(i.constraint.node for i in indicators),
    ):
        validate_domains(problem, node)
    for indicator in indicators:
        if bound_node(problem, indicator.constraint.node) is None:
            raise CompileError(
                "indicator body requires a finite exact polynomial interval enclosure"
            )
    report = CompilationReport(
        "global-factorable-v1",
        0.0,
        0.0,
        0.0,
        len(model._variables),
        n,
        len(model.constraints),
        problem.n_constraints,
        cache_status="global_rebuild",
    )
    return CompiledModel(
        problem,
        model.semantic_hash,
        model.data_hash,
        hashlib.sha256(("global-v1" + model.data_hash).encode()).hexdigest(),
        {key: {"patchable": False, "plan_sensitive": True} for key in values},
        {"variables": source_variables, "constraints": source_constraints},
        (),
        "scip-global-factorable-v1",
        {"mode": "global_rebuild"},
        tuple(values),
        (),
        report,
        {
            "primal": "identity_by_source_map",
            "semantic_variable_count": n,
            "optimality": "backend_reported_numerical_global",
            "original_space_validation": "validate_global_solution",
        },
        schema_version="solverpilot.compiled-model.global.v1",
    )
