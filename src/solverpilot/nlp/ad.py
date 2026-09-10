from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from solverpilot.model.expression import ExprNode
from .ir import NLPProblem


def _reshape_row_major_ca(ca, vec, shape):
    if shape == ():
        return vec[0] if hasattr(vec, '__len__') else vec
    if len(shape) == 1:
        return ca.reshape(vec, shape[0], 1)
    if len(shape) == 2:
        r, c = shape
        return ca.reshape(vec, c, r).T
    raise ValueError("P7 supports at most 2-D expressions")


def _flatten_row_major_ca(ca, expr, shape):
    if shape == ():
        return ca.vertcat(expr)
    if len(shape) == 1:
        return ca.reshape(expr, shape[0], 1)
    if len(shape) == 2:
        return ca.reshape(expr.T, shape[0]*shape[1], 1)
    raise ValueError("P7 supports at most 2-D expressions")


def _eval_np(node: ExprNode, x: np.ndarray, problem: NLPProblem):
    kind = node.kind
    if kind == 'constant': return np.asarray(node.payload, dtype=float)
    if kind == 'parameter': return np.asarray(problem.parameter_values[str(node.payload)], dtype=float)
    if kind == 'variable':
        off, shape = problem.variable_layout[str(node.payload)]
        size = int(np.prod(shape)) if shape else 1
        arr = np.asarray(x[off:off+size], dtype=float)
        return arr.reshape(shape if shape else ())
    args = [_eval_np(a, x, problem) for a in node.args]
    if kind == 'index': return args[0][node.payload]
    if kind == 'transpose': return np.asarray(args[0]).T
    if kind == 'sum': return np.sum(args[0], axis=node.payload)
    if kind == 'neg': return -args[0]
    if kind == 'add': return args[0] + args[1]
    if kind == 'mul': return args[0] * args[1]
    if kind == 'div': return args[0] / args[1]
    if kind == 'matmul': return args[0] @ args[1]
    if kind == 'pow': return np.power(args[0], int(node.payload))
    if kind == 'sin': return np.sin(args[0])
    if kind == 'cos': return np.cos(args[0])
    if kind == 'exp': return np.exp(args[0])
    if kind == 'log': return np.log(args[0])
    if kind == 'sqrt': return np.sqrt(args[0])
    if kind == 'tanh': return np.tanh(args[0])
    raise ValueError(f'unsupported P7 expression kind {kind!r}')


def evaluate_node(problem: NLPProblem, node: ExprNode, x: np.ndarray):
    with np.errstate(all='ignore'):
        return _eval_np(node, np.asarray(x, dtype=float).reshape(-1), problem)


def _to_ca(ca, node: ExprNode, x, problem: NLPProblem, cache: dict[int, Any]):
    key = id(node)
    if key in cache: return cache[key]
    kind = node.kind
    if kind == 'constant':
        arr = np.asarray(node.payload, dtype=float)
        out = ca.DM(float(arr)) if arr.shape == () else ca.DM(arr if arr.ndim == 2 else arr.reshape(-1,1))
    elif kind == 'parameter':
        arr = np.asarray(problem.parameter_values[str(node.payload)], dtype=float)
        out = ca.DM(float(arr)) if arr.shape == () else ca.DM(arr if arr.ndim == 2 else arr.reshape(-1,1))
    elif kind == 'variable':
        off, shape = problem.variable_layout[str(node.payload)]
        size = int(np.prod(shape)) if shape else 1
        out = _reshape_row_major_ca(ca, x[off:off+size], shape)
    else:
        args = [_to_ca(ca, a, x, problem, cache) for a in node.args]
        if kind == 'index':
            out = args[0][node.payload]
        elif kind == 'transpose': out = args[0].T
        elif kind == 'sum':
            if node.payload is None: out = ca.sum1(ca.sum2(args[0]))
            elif int(node.payload) == 0: out = ca.sum1(args[0])
            elif int(node.payload) == 1: out = ca.sum2(args[0])
            else: raise ValueError('P7 sum supports axes 0/1')
        elif kind == 'neg': out = -args[0]
        elif kind == 'add': out = args[0] + args[1]
        elif kind == 'mul': out = args[0] * args[1]
        elif kind == 'div': out = args[0] / args[1]
        elif kind == 'matmul':
            a0, a1 = node.args
            left = args[0].T if len(a0.shape) == 1 and len(a1.shape) == 2 else args[0]
            out = ca.mtimes(left, args[1])
        elif kind == 'pow': out = ca.power(args[0], int(node.payload))
        elif kind == 'sin': out = ca.sin(args[0])
        elif kind == 'cos': out = ca.cos(args[0])
        elif kind == 'exp': out = ca.exp(args[0])
        elif kind == 'log': out = ca.log(args[0])
        elif kind == 'sqrt': out = ca.sqrt(args[0])
        elif kind == 'tanh': out = ca.tanh(args[0])
        else: raise ValueError(f'unsupported P7 expression kind {kind!r}')
    cache[key] = out
    return out


@dataclass
class NLPDerivativeEngine:
    problem: NLPProblem

    def __post_init__(self) -> None:
        import casadi as ca
        self.ca = ca
        self.x = ca.MX.sym('x', self.problem.n_variables)
        cache: dict[int, Any] = {}
        f_raw = _to_ca(ca, self.problem.objective_node, self.x, self.problem, cache)
        obj_size = int(np.prod(self.problem.objective_node.shape)) if self.problem.objective_node.shape else 1
        if obj_size != 1:
            raise ValueError('NLP objective must be scalar')
        self.f = ca.reshape(f_raw, 1, 1)[0]
        g_parts = []
        for block in self.problem.constraints:
            e = _to_ca(ca, block.node, self.x, self.problem, cache)
            g_parts.append(_flatten_row_major_ca(ca, e, block.shape))
        self.g = ca.vertcat(*g_parts) if g_parts else ca.MX.zeros(0,1)
        self.grad = ca.gradient(self.f, self.x)
        self.jac = ca.jacobian(self.g, self.x)
        self.lam = ca.MX.sym('lam', self.problem.n_constraints)
        lag = self.f + ca.dot(self.lam, self.g) if self.problem.n_constraints else self.f
        self.hess = ca.hessian(lag, self.x)[0]
        self._fun = ca.Function('solverpilot_p7_eval', [self.x], [self.f, self.g, self.grad, self.jac])
        self._hfun = ca.Function('solverpilot_p7_hess', [self.x, self.lam], [self.hess])

    def evaluate(self, x):
        out = self._fun(np.asarray(x,dtype=float).reshape(-1))
        return (float(out[0]), np.asarray(out[1],dtype=float).reshape(-1), np.asarray(out[2],dtype=float).reshape(-1), np.asarray(out[3],dtype=float))

    def objective(self, x) -> float:
        return self.evaluate(x)[0]
    def constraints(self, x) -> np.ndarray:
        return self.evaluate(x)[1]
    def gradient(self, x) -> np.ndarray:
        return self.evaluate(x)[2]
    def jacobian(self, x) -> np.ndarray:
        return self.evaluate(x)[3]
    def lagrangian_hessian(self, x, multipliers=None) -> np.ndarray:
        lam = np.zeros(self.problem.n_constraints) if multipliers is None else np.asarray(multipliers,dtype=float).reshape(-1)
        return np.asarray(self._hfun(np.asarray(x,dtype=float).reshape(-1), lam), dtype=float)
    def jacobian_sparsity(self) -> tuple[tuple[int,int], ...]:
        sp = self.jac.sparsity(); rows, cols = sp.get_triplet()[0:2]
        return tuple(zip(map(int, rows), map(int, cols)))
    def hessian_sparsity(self) -> tuple[tuple[int,int], ...]:
        sp = self.hess.sparsity(); rows, cols = sp.get_triplet()[0:2]
        return tuple(zip(map(int, rows), map(int, cols)))
    def jvp(self, x, v) -> np.ndarray:
        import casadi as ca
        v_sym = ca.MX.sym('v', self.problem.n_variables)
        fun = ca.Function('solverpilot_p7_jvp', [self.x, v_sym], [ca.jtimes(self.g, self.x, v_sym)])
        return np.asarray(fun(np.asarray(x,float), np.asarray(v,float)), dtype=float).reshape(-1)
    def vjp(self, x, w) -> np.ndarray:
        import casadi as ca
        w_sym = ca.MX.sym('w', self.problem.n_constraints)
        fun = ca.Function('solverpilot_p7_vjp', [self.x, w_sym], [ca.jtimes(self.g, self.x, w_sym, True)])
        return np.asarray(fun(np.asarray(x,float), np.asarray(w,float)), dtype=float).reshape(-1)
