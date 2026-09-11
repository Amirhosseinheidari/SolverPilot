"""Convex atoms with checked composition and explicit conic graph lowering."""

from dataclasses import replace
import builtins
import hashlib
import numpy as np
from scipy import sparse
from scipy.special import logsumexp
from .expression import Expression, ExprNode
from .errors import DomainError, CompileError
from .types import Curvature, SignDomain


def _atom(kind, expression, payload=None):
    if not isinstance(expression, Expression) or expression.polynomial_degree not in (0, 1):
        raise DomainError("convex atom arguments must be affine SolverPilot expressions")
    if kind != "abs" and expression.ndim > 1:
        raise DomainError("this atom requires a scalar or vector; flatten matrices explicitly")
    shape = expression.shape if kind in ("abs", "huber") else ()
    return Expression(
        expression._model,
        ExprNode(
            "atom_" + kind,
            shape,
            args=(expression._node,),
            payload=payload,
            variable_dependencies=expression.variable_dependencies,
            parameter_dependencies=expression.parameter_dependencies,
            sign=SignDomain.UNKNOWN if kind == "log_sum_exp" else SignDomain.NONNEGATIVE,
            degree=None,
            curvature=Curvature.UNKNOWN,
        ),
    )


def abs(expression):
    return _atom("abs", expression)


def norm(expression, order=2):
    if order not in (1, 2, np.inf):
        raise ValueError("norm order must be 1, 2, or infinity")
    return _atom("norm", expression, float(order))


def huber(expression, delta=1.0):
    """x² inside [-delta, delta], 2*delta*|x|-delta² outside."""
    if isinstance(delta, bool) or not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive")
    return _atom("huber", expression, float(delta))


def log_sum_exp(expression):
    if not expression.size:
        raise ValueError("log_sum_exp requires at least one entry")
    return _atom("log_sum_exp", expression)


def quad_form(expression, matrix):
    """Return x.T @ matrix @ x for a symmetric positive-semidefinite matrix."""
    value = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix, dtype=float)
    if value.shape != (expression.size, expression.size) or not np.isfinite(value).all():
        raise ValueError("quadratic matrix must be finite with matching square shape")
    if not np.array_equal(value, value.T) or np.linalg.eigvalsh(value).min(initial=0.0) < 0.0:
        raise DomainError("quad_form requires a symmetric positive-semidefinite matrix")
    value = value.copy()
    value.flags.writeable = False
    return _atom("quad_form", expression, value)


def has_atoms(node):
    return node.kind.startswith("atom_") or any(has_atoms(a) for a in node.args)


def evaluate(model, node, x):
    """Evaluate original expressions independently of auxiliary epigraph values."""
    if node.kind == "variable":
        start = 0
        for key, value in model._variables.items():
            size = int(np.prod(value.shape)) if value.shape else 1
            if key == node.payload:
                return np.asarray(x[start : start + size]).reshape(value.shape)
            start += size
        raise KeyError(node.payload)
    if node.kind == "parameter":
        return model._parameters[node.payload].value
    if node.kind == "constant":
        return node.payload
    args = [evaluate(model, a, x) for a in node.args]
    kind = node.kind
    if kind == "add":
        return args[0] + args[1]
    if kind == "mul":
        return args[0] * args[1]
    if kind == "neg":
        return -args[0]
    if kind == "matmul":
        return args[0] @ args[1]
    if kind == "index":
        return args[0][node.payload]
    if kind == "transpose":
        return args[0].T
    if kind == "sum":
        return np.sum(args[0], axis=node.payload)
    if kind == "concat":
        return np.concatenate([np.asarray(a).reshape(-1) for a in args])
    if kind == "atom_abs":
        return np.abs(args[0])
    if kind == "atom_norm":
        return np.linalg.norm(np.asarray(args[0]).reshape(-1), ord=node.payload)
    if kind == "atom_huber":
        a = np.abs(args[0])
        d = node.payload
        return np.where(a <= d, a * a, 2 * d * a - d * d)
    if kind == "atom_log_sum_exp":
        return logsumexp(args[0])
    if kind == "atom_quad_form":
        a = np.asarray(args[0]).reshape(-1)
        return a @ node.payload @ a
    raise CompileError(f"unsupported atom composition: {kind}")


def _curvature(model, node):
    # 0 affine, +1 convex, -1 concave. Unsupported compositions fail closed.
    if node.degree is not None and node.degree <= 1:
        return 0
    if node.kind.startswith("atom_"):
        return 1
    if node.kind == "neg":
        return -_curvature(model, node.args[0])
    if node.kind in ("sum", "index", "transpose"):
        return _curvature(model, node.args[0])
    if node.kind == "add":
        left, right = [_curvature(model, a) for a in node.args]
        if left == 0 or right == 0 or left == right:
            return left or right
    if node.kind in ("mul", "matmul"):
        for i, a in enumerate(node.args):
            if a.degree == 0:
                values = evaluate(model, a, [])
                values = values.toarray() if sparse.issparse(values) else np.asarray(values)
                curvature = _curvature(model, node.args[1 - i])
                if np.all(values >= 0):
                    return curvature
                if np.all(values <= 0):
                    return -curvature
        if node.kind == "mul" and node.args[0] is node.args[1] and node.args[0].degree in (0, 1):
            return 1
    raise CompileError(
        "unsupported convex composition; use affine arguments and nonnegative weights"
    )


def compile_atoms(model, *, use_cache=True, bridge_policy=None, capabilities=None):
    from .model import Model, IndicatorConstraint
    from .sets import LessThan, GreaterThan, ExponentialCone
    from solverpilot.problem import ObjectiveSense, VariableDomain

    if model.objective is None:
        raise CompileError("model has no objective")
    if any(v.domain is not VariableDomain.CONTINUOUS for v in model.variables):
        raise CompileError("convex atoms currently require continuous variables")
    objective_curvature = _curvature(model, model.objective.expression._node)
    wanted = 1 if model.objective.sense is ObjectiveSense.MINIMIZE else -1
    if objective_curvature not in (0, wanted):
        raise CompileError("objective uses a convex atom in the wrong direction")
    for c in model.constraints:
        curvature = _curvature(model, c.function._node)
        if isinstance(c, IndicatorConstraint):
            raise CompileError("convex atoms cannot be mixed with indicator bridges")
        if curvature and not (
            (isinstance(c.set, LessThan) and curvature == 1)
            or (isinstance(c.set, GreaterThan) and curvature == -1)
        ):
            raise CompileError("non-affine atom constraints must have a convex feasible direction")
    key = "atoms:" + model.data_hash
    if use_cache and key in model._compiler_cache:
        return model._compiler_cache[key]
    lowered = Model(model.name)
    values, original_variables = {}, {}
    for v in model.variables:
        values[v.id.value] = lowered.variable(v.shape, lower=v.lower, upper=v.upper, name=v.name)
        original_variables[v.id.value] = values[v.id.value]
    for p in model.parameters:
        values[p.id.value] = lowered.parameter(p.shape, value=p.value, sign=p.sign, name=p.name)
    cache = {}

    def flatten(expr):
        return (
            lowered._concat(*[expr[index] for index in np.ndindex(expr.shape)])
            if expr.shape
            else lowered._concat(expr)
        )

    def lower(node):
        if id(node) in cache:
            return cache[id(node)]
        if node.kind in ("variable", "parameter"):
            return values[node.payload]
        if node.kind == "constant":
            return lowered.constant(node.payload)
        args = [lower(a) for a in node.args]
        if node.kind.startswith("atom_"):
            kind, a = node.kind[5:], args[0]
            if kind == "abs":
                out = lowered.variable(a.shape, lower=0.0, name="abs_epigraph")
                lowered.add(out >= a)
                lowered.add(out >= -a)
            elif kind == "norm" and node.payload in (1, np.inf):
                out = lowered.variable(
                    a.shape if node.payload == 1 else (), lower=0.0, name="norm_epigraph"
                )
                lowered.add(out >= a)
                lowered.add(out >= -a)
                if node.payload == 1:
                    out = out.sum()
            elif kind == "norm":
                out = lowered.variable(lower=0.0, name="norm_epigraph")
                lowered.soc(out, flatten(a))
            elif kind == "huber":
                d = node.payload
                v = lowered.variable(a.shape, lower=-d, upper=d, name="huber_quadratic")
                u = lowered.variable(a.shape, lower=0.0, name="huber_absolute")
                w = lowered.variable(a.shape, lower=0.0, name="huber_square")
                lowered.add(u >= a - v)
                lowered.add(u >= v - a)
                for index in np.ndindex(a.shape):
                    lowered.rotated_soc(w[index], 0.5, lowered._concat(v[index]))
                out = w + 2 * d * u
            elif kind == "log_sum_exp":
                out = lowered.variable(name="log_sum_exp_epigraph")
                u = lowered.variable(a.size, lower=0.0, name="exponential_slack")
                aa = flatten(a)
                for i in range(a.size):
                    lowered.add_in_set(lowered._concat(aa[i] - out, 1.0, u[i]), ExponentialCone())
                lowered.add(u.sum() <= 1.0)
            elif kind == "quad_form":
                eigenvalues, vectors = np.linalg.eigh(node.payload)
                factor = np.sqrt(np.maximum(eigenvalues, 0.0))[:, None] * vectors.T
                out = lowered.variable(lower=0.0, name="quadratic_epigraph")
                lowered.rotated_soc(out, 0.5, lowered.constant(factor) @ flatten(a))
            else:
                raise CompileError(kind)
        else:
            if node.kind == "add":
                out = args[0] + args[1]
            elif node.kind == "mul":
                out = args[0] * args[1]
            elif node.kind == "matmul":
                out = args[0] @ args[1]
            elif node.kind == "neg":
                out = -args[0]
            elif node.kind == "index":
                out = args[0][node.payload]
            elif node.kind == "transpose":
                out = args[0].T
            elif node.kind == "sum":
                out = args[0].sum(axis=node.payload)
            elif node.kind == "concat":
                out = lowered._concat(*args)
            else:
                raise CompileError(f"unsupported atom composition: {node.kind}")
        cache[id(node)] = out
        return out

    constraints = {}
    for c in model.constraints:
        constraints[c.entity_id.value] = lowered.add_in_set(
            lower(c.function._node), c.set, name=c.name
        )
    objective = lower(model.objective.expression._node)
    (lowered.minimize if wanted == 1 else lowered.maximize)(objective)
    compiled = lowered.compile(
        use_cache=use_cache, bridge_policy=bridge_policy, capabilities=capabilities
    )
    n = builtins.sum(v.size for v in model.variables)
    source = {
        "variables": {
            key: compiled.source_map["variables"][v.id.value]
            for key, v in original_variables.items()
        },
        "constraints": {
            key: compiled.source_map["constraints"].get(c.entity_id.value, {})
            for key, c in constraints.items()
        },
        "objective": {model.objective.entity_id.value: {"execution_objective": True}},
    }
    result = replace(
        compiled,
        semantic_hash=model.semantic_hash,
        data_hash=model.data_hash,
        compilation_hash=hashlib.sha256(
            (model.semantic_hash + "convex-atoms-v1").encode()
        ).hexdigest(),
        source_map=source,
        parameter_map={
            p.id.value: {"lowered_parameter": values[p.id.value].id.value} for p in model.parameters
        },
        reconstruction_contract={
            **compiled.reconstruction_contract,
            "semantic_variable_count": n,
            "convex_atoms": True,
        },
        compilation_report=replace(
            compiled.compilation_report,
            n_semantic_variables=n,
            n_semantic_constraints=len(model.constraints),
        ),
    )
    if use_cache:
        # Keep bounded numeric snapshots without affecting the core compiler cache.
        atom_keys = [k for k in model._compiler_cache if k.startswith("atoms:")]
        if len(atom_keys) >= 8:
            model._compiler_cache.pop(atom_keys[0])
        model._compiler_cache[key] = result
    return result
