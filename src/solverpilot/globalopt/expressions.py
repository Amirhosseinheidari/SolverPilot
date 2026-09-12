"""Explicit nonsmooth global atoms and original-coordinate evaluation."""

import builtins
from dataclasses import replace
import numpy as np
from scipy import sparse
from solverpilot.model.expression import Expression, ExprNode
from solverpilot.model.errors import DomainError
from solverpilot._immutability import readonly_array


def _atom(kind, expression, other=None):
    if not isinstance(expression, Expression):
        raise TypeError("a SolverPilot expression is required")
    args = (expression,) if other is None else (expression, expression._coerce(other))
    if kind == "global_abs":
        shape = expression.shape
    elif len(args) == 2:
        shape = tuple(np.broadcast_shapes(args[0].shape, args[1].shape))
    else:
        if expression.size == 0:
            raise ValueError("min/max require at least one element")
        shape = ()
    node = ExprNode(
        kind,
        shape,
        args=tuple(a._node for a in args),
        variable_dependencies=frozenset().union(*(a.variable_dependencies for a in args)),
        parameter_dependencies=frozenset().union(*(a.parameter_dependencies for a in args)),
    )
    return Expression(expression._model, node)


def absolute(expression):
    return _atom("global_abs", expression)


def maximum(expression, other=None):
    return _atom("global_max", expression, other)


def minimum(expression, other=None):
    return _atom("global_min", expression, other)


ALLOWED = {
    "constant",
    "parameter",
    "variable",
    "index",
    "transpose",
    "sum",
    "neg",
    "add",
    "mul",
    "div",
    "matmul",
    "pow",
    "concat",
    "sin",
    "cos",
    "exp",
    "log",
    "sqrt",
    "tanh",
    "atom_abs",
    "global_abs",
    "global_max",
    "global_min",
}


def freeze_node(node):
    if node.kind not in ALLOWED:
        raise DomainError(f"global solver does not support expression {node.kind!r}")
    payload = node.payload
    if node.kind == "constant":
        payload = readonly_array(
            payload.toarray() if sparse.issparse(payload) else payload, dtype=float
        )
        if not np.isfinite(payload).all():
            raise DomainError("nonfinite expression constant")
    if node.kind == "pow" and (type(payload) is not int or not 0 <= payload <= 64):
        raise DomainError("global polynomial exponent must be an integer in [0, 64]")
    return replace(node, payload=payload, args=tuple(freeze_node(a) for a in node.args))


def evaluate_node(problem, node, x):
    p = problem.relaxation
    kind = node.kind
    if kind == "constant":
        return np.asarray(node.payload, dtype=float)
    if kind == "parameter":
        return p.parameter_values[node.payload]
    if kind == "variable":
        off, shape = p.variable_layout[node.payload]
        return np.asarray(x[off : off + (int(np.prod(shape)) if shape else 1)]).reshape(shape)
    args = [evaluate_node(problem, a, x) for a in node.args]
    with np.errstate(all="ignore"):
        if kind == "index":
            value = args[0][node.payload]
        elif kind == "transpose":
            value = args[0].T
        elif kind == "sum":
            value = np.sum(args[0], axis=node.payload)
        elif kind == "neg":
            value = -args[0]
        elif kind == "add":
            value = args[0] + args[1]
        elif kind == "mul":
            value = args[0] * args[1]
        elif kind == "div":
            value = args[0] / args[1]
        elif kind == "matmul":
            value = args[0] @ args[1]
        elif kind == "pow":
            value = args[0] ** node.payload
        elif kind == "concat":
            value = np.concatenate([np.asarray(a).reshape(-1) for a in args])
        elif kind in ("atom_abs", "global_abs"):
            value = np.abs(args[0])
        elif kind == "global_max":
            value = np.max(args[0]) if len(args) == 1 else np.maximum(*args)
        elif kind == "global_min":
            value = np.min(args[0]) if len(args) == 1 else np.minimum(*args)
        elif kind in ("sin", "cos", "exp", "log", "sqrt", "tanh"):
            value = getattr(np, kind)(args[0])
        else:
            raise DomainError(f"unsupported global expression {kind!r}")
    return np.asarray(value)


def to_scip(problem, node, x, scip_module, cache):
    key = id(node)
    if key in cache:
        return cache[key]
    p = problem.relaxation
    kind = node.kind
    if kind == "constant":
        value = np.asarray(node.payload, dtype=object)
    elif kind == "parameter":
        value = np.asarray(p.parameter_values[node.payload], dtype=object)
    elif kind == "variable":
        off, shape = p.variable_layout[node.payload]
        value = np.asarray(
            x[off : off + (int(np.prod(shape)) if shape else 1)], dtype=object
        ).reshape(shape)
    else:
        a = [to_scip(problem, c, x, scip_module, cache) for c in node.args]
        if kind == "index":
            value = a[0][node.payload]
        elif kind == "transpose":
            value = a[0].T
        elif kind == "sum":
            value = np.sum(a[0], axis=node.payload)
        elif kind == "neg":
            value = -a[0]
        elif kind == "add":
            value = a[0] + a[1]
        elif kind == "mul":
            value = a[0] * a[1]
        elif kind == "div":
            value = a[0] / a[1]
        elif kind == "matmul":
            value = a[0] @ a[1]
        elif kind == "pow":
            value = a[0] ** node.payload
        elif kind == "concat":
            value = np.concatenate([np.asarray(v).reshape(-1) for v in a])
        elif kind in ("atom_abs", "global_abs"):
            value = np.vectorize(builtins.abs, otypes=[object])(a[0])
        elif kind in ("global_max", "global_min"):
            sign = 1 if kind == "global_max" else -1
            combine = lambda left, right: (left + right + sign * builtins.abs(left - right)) / 2
            if len(a) == 2:
                value = np.vectorize(combine, otypes=[object])(*a)
            else:
                seq = a[0].reshape(-1)
                value = seq[0]
                for term in seq[1:]:
                    value = combine(value, term)
        elif kind == "tanh":
            fn = lambda v: (scip_module.exp(2 * v) - 1) / (scip_module.exp(2 * v) + 1)
            value = np.vectorize(fn, otypes=[object])(a[0])
        elif kind in ("sin", "cos", "exp", "log", "sqrt"):
            value = np.vectorize(getattr(scip_module, kind), otypes=[object])(a[0])
        else:
            raise DomainError(f"unsupported global expression {kind!r}")
    value = np.asarray(value, dtype=object)
    if value.shape != node.shape:
        raise DomainError(f"global expression shape mismatch at {kind}")
    cache[key] = value
    return value
