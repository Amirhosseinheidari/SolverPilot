from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import hashlib
import json
import uuid
import math

import numpy as np

from solverpilot.problem import LinearProblem, ObjectiveSense, QuadraticProblem, VariableDomain

from .errors import DomainError, OwnershipError, ShapeError, SymbolicTruthValueError
from .expression import ExprNode, Expression, _shape_tuple, _sign_of_constant
from .sets import (ConstraintSet, EqualTo, GreaterThan, LessThan, ScalarSet, SecondOrderCone, RotatedSecondOrderCone, PositiveSemidefiniteCone, ExponentialCone, PowerCone)
from .types import Curvature, EntityId, SignDomain


def _readonly_array(value: Any, shape: tuple[int, ...], *, finite: bool, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    try:
        arr = np.broadcast_to(arr, shape if shape else ()).copy()
    except ValueError as exc:
        raise ShapeError(f"{name} cannot broadcast to shape {shape}") from exc
    if np.isnan(arr).any():
        raise DomainError(f"{name} must not contain NaN")
    if finite and np.isinf(arr).any():
        raise DomainError(f"{name} must be finite")
    arr.flags.writeable = False
    return arr


def _canonical_float(x: float) -> str:
    x = float(x)
    if math.isnan(x):
        raise DomainError("NaN cannot be canonically serialized")
    if x == math.inf:
        return "+inf"
    if x == -math.inf:
        return "-inf"
    return x.hex()


def _array_semantic_payload(arr: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    return {"shape": list(arr.shape), "values": [_canonical_float(x) for x in flat]}


@dataclass(frozen=True, slots=True)
class _VariableData:
    entity_id: EntityId
    shape: tuple[int, ...]
    lower: np.ndarray
    upper: np.ndarray
    domain: VariableDomain
    name: str | None
    origin_id: str | None = None


@dataclass(slots=True)
class _ParameterData:
    entity_id: EntityId
    shape: tuple[int, ...]
    value: np.ndarray
    sign: SignDomain
    name: str | None
    origin_id: str | None = None


@dataclass(frozen=True, slots=True)
class PendingConstraint:
    function: Expression
    relation: str

    def __bool__(self) -> bool:
        raise SymbolicTruthValueError("symbolic constraint cannot be converted to bool")


@dataclass(frozen=True, slots=True)
class Constraint:
    entity_id: EntityId
    function: Expression
    set: ConstraintSet
    name: str | None = None


@dataclass(frozen=True, slots=True)
class IndicatorConstraint:
    entity_id: EntityId
    indicator: "Variable"
    active_value: int
    function: Expression
    set: ConstraintSet
    name: str | None = None


@dataclass(frozen=True, slots=True)
class Objective:
    entity_id: EntityId
    expression: Expression
    sense: ObjectiveSense
    name: str | None = None


class Variable(Expression):
    __slots__ = ("_entity_id",)

    def __init__(self, model: "Model", data: _VariableData) -> None:
        sign = model._variable_sign(data)
        node = ExprNode(
            "variable", data.shape, payload=data.entity_id.value,
            variable_dependencies=frozenset({data.entity_id.value}),
            sign=sign, degree=1, curvature=Curvature.AFFINE,
        )
        super().__init__(model, node)
        self._entity_id = data.entity_id

    @property
    def id(self) -> EntityId:
        return self._entity_id

    @property
    def name(self) -> str | None:
        return self._model._variables[self.id.value].name

    @property
    def lower(self) -> np.ndarray:
        return self._model._variables[self.id.value].lower

    @property
    def upper(self) -> np.ndarray:
        return self._model._variables[self.id.value].upper

    @property
    def domain(self) -> VariableDomain:
        return self._model._variables[self.id.value].domain


class Parameter(Expression):
    __slots__ = ("_entity_id",)

    def __init__(self, model: "Model", data: _ParameterData) -> None:
        node = ExprNode(
            "parameter", data.shape, payload=data.entity_id.value,
            parameter_dependencies=frozenset({data.entity_id.value}),
            sign=data.sign, degree=0, curvature=Curvature.CONSTANT,
        )
        super().__init__(model, node)
        self._entity_id = data.entity_id

    @property
    def id(self) -> EntityId:
        return self._entity_id

    @property
    def name(self) -> str | None:
        return self._model._parameters[self.id.value].name

    @property
    def value(self) -> np.ndarray:
        return self._model._parameters[self.id.value].value.copy()

    @value.setter
    def value(self, new_value: Any) -> None:
        self._model._set_parameter_value(self.id.value, new_value)


class Model:
    def __init__(self, name: str | None = None) -> None:
        self.name = name
        self._namespace = uuid.uuid4().hex
        self._next_ids = {"v": 0, "p": 0, "c": 0, "o": 0}
        self._variables: dict[str, _VariableData] = {}
        self._parameters: dict[str, _ParameterData] = {}
        self._constraints: list[Constraint | IndicatorConstraint] = []
        self._objective: Objective | None = None
        # P2 compiler/cache state. Expression nodes remain immutable; only the
        # model container tracks semantic/data revisions.
        self._semantic_revision = 0
        self._data_revision = 0
        self._parameter_versions: dict[str, int] = {}
        self._semantic_hash_cache: tuple[int, str] | None = None
        self._data_hash_cache: tuple[int, int, str] | None = None
        self._compiler_cache: dict[str, Any] = {}

    def _new_id(self, prefix: str) -> EntityId:
        self._next_ids[prefix] += 1
        return EntityId(f"{prefix}{self._next_ids[prefix]:06d}", self._namespace)

    @property
    def variables(self) -> tuple[Variable, ...]:
        return tuple(Variable(self, d) for d in self._variables.values())

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        return tuple(Parameter(self, d) for d in self._parameters.values())

    @property
    def constraints(self) -> tuple[Constraint | IndicatorConstraint, ...]:
        return tuple(self._constraints)

    @property
    def objective(self) -> Objective | None:
        return self._objective

    def variable(
        self,
        shape: int | Iterable[int] | tuple[int, ...] | None = None,
        *,
        lower: Any = -np.inf,
        upper: Any = np.inf,
        domain: VariableDomain | str = VariableDomain.CONTINUOUS,
        name: str | None = None,
    ) -> Variable:
        shape_t = _shape_tuple(shape)
        dom = VariableDomain(domain)
        lo = _readonly_array(lower, shape_t, finite=False, name="variable lower bound")
        hi = _readonly_array(upper, shape_t, finite=False, name="variable upper bound")
        if np.any(lo > hi):
            raise DomainError("variable lower bound exceeds upper bound")
        if dom is VariableDomain.BINARY and (np.any(lo < 0) or np.any(hi > 1)):
            raise DomainError("binary variable bounds must lie inside [0, 1]")
        entity_id = self._new_id("v")
        data = _VariableData(entity_id, shape_t, lo, hi, dom, name)
        self._variables[entity_id.value] = data
        self._mark_semantic_change()
        return Variable(self, data)

    def binary(self, shape: int | Iterable[int] | tuple[int, ...] | None = None, *, name: str | None = None) -> Variable:
        return self.variable(shape, lower=0.0, upper=1.0, domain=VariableDomain.BINARY, name=name)

    def integer(
        self,
        shape: int | Iterable[int] | tuple[int, ...] | None = None,
        *,
        lower: Any = -np.inf,
        upper: Any = np.inf,
        name: str | None = None,
    ) -> Variable:
        return self.variable(shape, lower=lower, upper=upper, domain=VariableDomain.INTEGER, name=name)

    def parameter(
        self,
        shape: int | Iterable[int] | tuple[int, ...] | None = None,
        *,
        value: Any | None = None,
        sign: SignDomain | str = SignDomain.UNKNOWN,
        name: str | None = None,
    ) -> Parameter:
        shape_t = _shape_tuple(shape)
        sign_v = SignDomain(sign)
        if value is None:
            value = np.zeros(shape_t if shape_t else (), dtype=np.float64)
        arr = _readonly_array(value, shape_t, finite=True, name="parameter value")
        self._validate_parameter_sign(arr, sign_v)
        entity_id = self._new_id("p")
        data = _ParameterData(entity_id, shape_t, arr, sign_v, name)
        self._parameters[entity_id.value] = data
        self._parameter_versions[entity_id.value] = 0
        self._mark_semantic_change()
        return Parameter(self, data)

    def constant(self, value: Any) -> Expression:
        arr = np.asarray(value, dtype=np.float64)
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise DomainError("expression constants must be finite")
        arr = np.array(arr, dtype=np.float64, copy=True)
        arr.flags.writeable = False
        node = ExprNode("constant", tuple(arr.shape), payload=arr, sign=_sign_of_constant(arr), degree=0, curvature=Curvature.CONSTANT)
        return Expression(self, node)

    def add_in_set(self, function: Any, set_: ConstraintSet, *, name: str | None = None) -> Constraint:
        expr = function if isinstance(function, Expression) else self.constant(function)
        if expr._model is not self:
            raise OwnershipError("constraint function belongs to another model")
        if not isinstance(set_, (LessThan, GreaterThan, EqualTo, SecondOrderCone, RotatedSecondOrderCone, PositiveSemidefiniteCone, ExponentialCone, PowerCone)):
            from .sets import Interval
            if not isinstance(set_, Interval):
                raise TypeError("unsupported constraint set")
        if isinstance(set_, (ExponentialCone, PowerCone)) and expr.shape != (3,):
            raise ShapeError("exponential and power cones require a vector of length three")
        if isinstance(set_, SecondOrderCone) and expr.shape != (set_.dimension,):
            raise ShapeError(f"SOC function must have shape ({set_.dimension},), got {expr.shape}")
        if isinstance(set_, RotatedSecondOrderCone) and expr.shape != (set_.dimension,):
            raise ShapeError(f"rotated SOC function must have shape ({set_.dimension},), got {expr.shape}")
        if isinstance(set_, PositiveSemidefiniteCone) and expr.shape != (set_.dimension, set_.dimension):
            raise ShapeError(f"PSD function must have shape ({set_.dimension}, {set_.dimension}), got {expr.shape}")
        constraint = Constraint(self._new_id("c"), expr, set_, name)
        self._constraints.append(constraint)
        self._mark_semantic_change()
        return constraint

    def add(self, relation: PendingConstraint, *, name: str | None = None) -> Constraint:
        if not isinstance(relation, PendingConstraint):
            raise TypeError("Model.add expects a symbolic relation such as x <= b")
        if relation.function._model is not self:
            raise OwnershipError("constraint belongs to another model")
        if relation.relation == "le":
            set_ = LessThan(0.0)
        elif relation.relation == "ge":
            set_ = GreaterThan(0.0)
        elif relation.relation == "eq":
            set_ = EqualTo(0.0)
        else:
            raise AssertionError(relation.relation)
        return self.add_in_set(relation.function, set_, name=name)

    def indicator(
        self,
        indicator: Variable,
        relation: PendingConstraint,
        *,
        active_value: int = 1,
        name: str | None = None,
    ) -> IndicatorConstraint:
        """Add a scalar binary indicator constraint without prematurely linearizing it.

        P4 deliberately preserves the semantic indicator until compilation chooses an exact
        lowering path. The current execution IR can transport only linear rows, so safe
        compilation uses fixed-premise simplification or a bound-certified Big-M bridge.
        """
        if not isinstance(indicator, Variable) or indicator._model is not self:
            raise OwnershipError("indicator variable must be a Variable owned by this model")
        if indicator.shape != ():
            raise ShapeError("P4 indicator variable must be scalar")
        if indicator.domain is not VariableDomain.BINARY:
            raise DomainError("P4 indicator variable must be binary")
        if active_value not in {0, 1}:
            raise DomainError("indicator active_value must be 0 or 1")
        if not isinstance(relation, PendingConstraint):
            raise TypeError("Model.indicator expects a symbolic relation such as x <= b")
        if relation.function._model is not self:
            raise OwnershipError("indicator body belongs to another model")
        if relation.function.shape != ():
            raise ShapeError("P4 indicator body must be scalar")
        if relation.relation == "le":
            set_ = LessThan(0.0)
        elif relation.relation == "ge":
            set_ = GreaterThan(0.0)
        elif relation.relation == "eq":
            set_ = EqualTo(0.0)
        else:
            raise AssertionError(relation.relation)
        constraint = IndicatorConstraint(self._new_id("c"), indicator, int(active_value), relation.function, set_, name)
        self._constraints.append(constraint)
        self._mark_semantic_change()
        return constraint

    def _concat(self, *values: Any) -> Expression:
        exprs: list[Expression] = []
        total = 0
        deps_v: frozenset[str] = frozenset()
        deps_p: frozenset[str] = frozenset()
        degree = 0
        for value in values:
            expr = value if isinstance(value, Expression) else self.constant(value)
            if expr._model is not self:
                raise OwnershipError("concatenated expression belongs to another model")
            if expr.ndim > 1:
                raise ShapeError("P6 cone packing supports scalar/vector expressions only")
            exprs.append(expr)
            total += expr.size
            deps_v = deps_v | expr.variable_dependencies
            deps_p = deps_p | expr.parameter_dependencies
            if expr.polynomial_degree is None:
                degree = None
            elif degree is not None:
                degree = max(degree, expr.polynomial_degree)
        curvature = Curvature.CONSTANT if degree == 0 else Curvature.AFFINE if degree is not None and degree <= 1 else Curvature.UNKNOWN
        return Expression(self, ExprNode("concat", (total,), args=tuple(e._node for e in exprs), variable_dependencies=deps_v, parameter_dependencies=deps_p, sign=SignDomain.UNKNOWN, degree=degree, curvature=curvature))

    def soc(self, t: Any, vector: Any, *, name: str | None = None) -> Constraint:
        """Add the exact semantic cone constraint ``||vector||_2 <= t``."""
        t_expr = t if isinstance(t, Expression) else self.constant(t)
        v_expr = vector if isinstance(vector, Expression) else self.constant(vector)
        if t_expr._model is not self or v_expr._model is not self:
            raise OwnershipError("SOC expressions must belong to this model")
        if t_expr.shape != ():
            raise ShapeError("SOC t must be scalar")
        if v_expr.ndim != 1:
            raise ShapeError("SOC vector must be one-dimensional")
        packed = self._concat(t_expr, v_expr)
        return self.add_in_set(packed, SecondOrderCone(packed.size), name=name)

    def rotated_soc(self, u: Any, v: Any, vector: Any, *, name: str | None = None) -> Constraint:
        """Add ``u>=0, v>=0, 2*u*v >= ||vector||_2^2`` as a semantic cone."""
        u_expr = u if isinstance(u, Expression) else self.constant(u)
        v_expr = v if isinstance(v, Expression) else self.constant(v)
        x_expr = vector if isinstance(vector, Expression) else self.constant(vector)
        if any(e._model is not self for e in (u_expr, v_expr, x_expr)):
            raise OwnershipError("rotated SOC expressions must belong to this model")
        if u_expr.shape != () or v_expr.shape != ():
            raise ShapeError("rotated SOC u and v must be scalars")
        if x_expr.ndim != 1:
            raise ShapeError("rotated SOC vector must be one-dimensional")
        packed = self._concat(u_expr, v_expr, x_expr)
        return self.add_in_set(packed, RotatedSecondOrderCone(packed.size), name=name)

    def psd(self, matrix: Any, *, name: str | None = None) -> Constraint:
        """Add affine symmetric matrix membership in the PSD cone."""
        expr = matrix if isinstance(matrix, Expression) else self.constant(matrix)
        if expr._model is not self:
            raise OwnershipError("PSD expression belongs to another model")
        if expr.ndim != 2 or expr.shape[0] != expr.shape[1]:
            raise ShapeError("PSD expression must be a square matrix")
        return self.add_in_set(expr, PositiveSemidefiniteCone(expr.shape[0]), name=name)

    def minimize(self, expression: Any, *, name: str | None = None) -> Objective:
        return self._set_objective(expression, ObjectiveSense.MINIMIZE, name=name)

    def maximize(self, expression: Any, *, name: str | None = None) -> Objective:
        return self._set_objective(expression, ObjectiveSense.MAXIMIZE, name=name)

    def _set_objective(self, expression: Any, sense: ObjectiveSense, *, name: str | None) -> Objective:
        expr = expression if isinstance(expression, Expression) else self.constant(expression)
        if expr._model is not self:
            raise OwnershipError("objective belongs to another model")
        if expr.shape != ():
            raise ShapeError(f"objective must be scalar, got shape {expr.shape}")
        objective = Objective(self._new_id("o"), expr, sense, name)
        self._objective = objective
        self._mark_semantic_change()
        return objective

    def compile(self, *, use_cache: bool = True, bridge_policy=None, capabilities=None):
        from .compiler import compile_model
        return compile_model(self, use_cache=use_cache, bridge_policy=bridge_policy, capabilities=capabilities)

    def clear_compile_cache(self) -> None:
        self._compiler_cache.clear()

    @property
    def compiler_cache_info(self):
        from .compiler import compiler_cache_info
        return compiler_cache_info(self)

    def solve(self, **kwargs):
        compiled = self.compile()
        return compiled.solve(**kwargs)

    @property
    def semantic_hash(self) -> str:
        cached = self._semantic_hash_cache
        if cached is not None and cached[0] == self._semantic_revision:
            return cached[1]
        payload = self._semantic_payload(include_parameter_values=False)
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        value = hashlib.sha256(raw).hexdigest()
        self._semantic_hash_cache = (self._semantic_revision, value)
        return value

    @property
    def data_hash(self) -> str:
        cached = self._data_hash_cache
        if cached is not None and cached[:2] == (self._semantic_revision, self._data_revision):
            return cached[2]
        payload = {
            "semantic_hash": self.semantic_hash,
            "parameters": {
                key: _array_semantic_payload(data.value)
                for key, data in sorted(self._parameters.items())
            },
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        value = hashlib.sha256(raw).hexdigest()
        self._data_hash_cache = (self._semantic_revision, self._data_revision, value)
        return value

    def _semantic_payload(self, *, include_parameter_values: bool) -> dict[str, Any]:
        variables = []
        for key, data in self._variables.items():
            variables.append({
                "id": key,
                "shape": list(data.shape),
                "lower": _array_semantic_payload(data.lower),
                "upper": _array_semantic_payload(data.upper),
                "domain": data.domain.value,
            })
        parameters = []
        for key, data in self._parameters.items():
            item = {"id": key, "shape": list(data.shape), "sign": data.sign.value}
            if include_parameter_values:
                item["value"] = _array_semantic_payload(data.value)
            parameters.append(item)
        has_cones = any(isinstance(c.set, (SecondOrderCone, RotatedSecondOrderCone, PositiveSemidefiniteCone, ExponentialCone, PowerCone)) for c in self._constraints)
        nonlinear_kinds = {"sin", "cos", "exp", "log", "sqrt", "tanh", "pow", "div"}
        def has_nonlinear(node):
            return node.kind in nonlinear_kinds or any(has_nonlinear(a) for a in node.args)
        has_nlp = (self._objective is not None and has_nonlinear(self._objective.expression._node)) or any(has_nonlinear(c.function._node) for c in self._constraints if hasattr(c, "function"))
        has_discrete = any(d.domain in {VariableDomain.INTEGER, VariableDomain.BINARY} for d in self._variables.values())
        has_nonlinear_degree = (self._objective is not None and (self._objective.expression.polynomial_degree is None or self._objective.expression.polynomial_degree > 1)) or any((c.function.polynomial_degree is None or c.function.polynomial_degree > 1) for c in self._constraints if hasattr(c, "function"))
        schema = "solverpilot.semantic-model.p8.v1" if (has_discrete and (has_nlp or has_nonlinear_degree)) else ("solverpilot.semantic-model.p7.v1" if has_nlp else ("solverpilot.semantic-model.p6.v1" if has_cones else "solverpilot.semantic-model.p1.v1"))
        return {
            "schema": schema,
            "variables": variables,
            "parameters": parameters,
            "constraints": [self._serialize_constraint(c) for c in self._constraints],
            "objective": None if self._objective is None else {
                "id": self._objective.entity_id.value,
                "sense": self._objective.sense.value,
                "expression": self._serialize_node(self._objective.expression._node, include_parameter_values=False),
            },
        }

    def _serialize_constraint(self, constraint: Constraint | IndicatorConstraint) -> dict[str, Any]:
        base = {
            "id": constraint.entity_id.value,
            "function": self._serialize_node(constraint.function._node, include_parameter_values=False),
            "set": self._serialize_set(constraint.set),
        }
        if isinstance(constraint, IndicatorConstraint):
            base.update({
                "kind": "IndicatorConstraint",
                "indicator": constraint.indicator.id.value,
                "active_value": constraint.active_value,
            })
        return base

    def _serialize_set(self, set_: ScalarSet) -> dict[str, Any]:
        if isinstance(set_, LessThan):
            return {"kind": "LessThan", "upper": _canonical_float(set_.upper)}
        if isinstance(set_, GreaterThan):
            return {"kind": "GreaterThan", "lower": _canonical_float(set_.lower)}
        if isinstance(set_, EqualTo):
            return {"kind": "EqualTo", "value": _canonical_float(set_.value)}
        from .sets import Interval
        if isinstance(set_, Interval):
            return {"kind": "Interval", "lower": _canonical_float(set_.lower), "upper": _canonical_float(set_.upper)}
        if isinstance(set_, ExponentialCone):
            return {"kind": "ExponentialCone"}
        if isinstance(set_, PowerCone):
            return {"kind": "PowerCone", "alpha": set_.alpha}
        if isinstance(set_, SecondOrderCone):
            return {"kind": "SecondOrderCone", "dimension": set_.dimension}
        if isinstance(set_, RotatedSecondOrderCone):
            return {"kind": "RotatedSecondOrderCone", "dimension": set_.dimension}
        if isinstance(set_, PositiveSemidefiniteCone):
            return {"kind": "PositiveSemidefiniteCone", "dimension": set_.dimension}
        raise TypeError(f"unsupported constraint set {type(set_).__name__}")

    def _serialize_node(self, node: ExprNode, *, include_parameter_values: bool) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": node.kind, "shape": list(node.shape)}
        if node.kind == "constant":
            out["value"] = _array_semantic_payload(node.payload)
        elif node.kind in {"variable", "parameter"}:
            out["entity"] = node.payload
            if node.kind == "parameter" and include_parameter_values:
                out["value"] = _array_semantic_payload(self._parameters[node.payload].value)
        elif node.kind == "index":
            key = node.payload if isinstance(node.payload, tuple) else (node.payload,)
            out["index"] = [
                {"int": int(x)} if isinstance(x, (int, np.integer)) else {"slice": [x.start, x.stop, x.step]}
                for x in key
            ]
        elif node.kind == "sum":
            out["axis"] = node.payload
        elif node.kind == "pow":
            out["exponent"] = int(node.payload)
        out["args"] = [self._serialize_node(arg, include_parameter_values=include_parameter_values) for arg in node.args]
        return out

    def _node_sort_key(self, node: ExprNode) -> str:
        return json.dumps(self._serialize_node(node, include_parameter_values=False), sort_keys=True, separators=(",", ":"))

    def _constant_value(self, node: ExprNode) -> np.ndarray:
        if node.kind != "constant":
            raise DomainError("expression is not a fixed constant")
        return np.asarray(node.payload, dtype=np.float64)

    def _variable_sign(self, data: _VariableData) -> SignDomain:
        if np.all(data.lower == 0) and np.all(data.upper == 0):
            return SignDomain.ZERO
        if np.all(data.lower >= 0):
            return SignDomain.NONNEGATIVE
        if np.all(data.upper <= 0):
            return SignDomain.NONPOSITIVE
        return SignDomain.UNKNOWN

    def _indexed_sign(self, node: ExprNode, key: Any) -> SignDomain:
        if node.kind == "constant":
            return _sign_of_constant(np.asarray(node.payload)[key])
        if node.kind == "variable":
            data = self._variables[node.payload]
            lo = np.asarray(data.lower)[key]
            hi = np.asarray(data.upper)[key]
            if np.all(lo == 0) and np.all(hi == 0):
                return SignDomain.ZERO
            if np.all(lo >= 0):
                return SignDomain.NONNEGATIVE
            if np.all(hi <= 0):
                return SignDomain.NONPOSITIVE
        if node.kind == "parameter":
            return self._parameters[node.payload].sign
        return SignDomain.UNKNOWN

    def _validate_parameter_sign(self, value: np.ndarray, sign: SignDomain) -> None:
        if sign is SignDomain.NONNEGATIVE and np.any(value < 0):
            raise DomainError("parameter declared nonnegative has negative values")
        if sign is SignDomain.NONPOSITIVE and np.any(value > 0):
            raise DomainError("parameter declared nonpositive has positive values")
        if sign is SignDomain.ZERO and np.any(value != 0):
            raise DomainError("parameter declared zero has nonzero values")

    def _set_parameter_value(self, entity: str, value: Any) -> None:
        data = self._parameters[entity]
        arr = _readonly_array(value, data.shape, finite=True, name="parameter value")
        self._validate_parameter_sign(arr, data.sign)
        if np.array_equal(arr, data.value):
            return
        data.value = arr
        self._parameter_versions[entity] = self._parameter_versions.get(entity, 0) + 1
        self._data_revision += 1
        self._data_hash_cache = None

    def _mark_semantic_change(self) -> None:
        self._semantic_revision += 1
        self._data_revision += 1
        self._semantic_hash_cache = None
        self._data_hash_cache = None
        # Semantic changes invalidate compiler templates. Parameter value updates
        # deliberately do not clear this cache.
        self._compiler_cache.clear()

    @classmethod
    def from_problem(cls, problem: LinearProblem | QuadraticProblem, *, name: str | None = None) -> "Model":
        """Import the current legacy LP/QP execution IR into the P1 semantic layer.

        This is a migration/debug path, not a replacement for the zero-overhead legacy direct path.
        It preserves variable ordering, bounds/domains, objective convention and row bounds.
        """
        from .sets import Interval
        linear = problem.linear if isinstance(problem, QuadraticProblem) else problem
        model = cls(linear.name if name is None else name)
        xs: list[Variable] = []
        for j in range(linear.n_variables):
            xs.append(model.variable(
                lower=float(linear.variable_lower[j]),
                upper=float(linear.variable_upper[j]),
                domain=VariableDomain(str(linear.domains[j])),
                name=f"x{j}",
            ))
        for i in range(linear.n_constraints):
            row = linear.A.getrow(i)
            expr: Expression = model.constant(0.0)
            for j, value in zip(row.indices, row.data):
                expr = expr + float(value) * xs[int(j)]
            lo = float(linear.constraint_lower[i]); hi = float(linear.constraint_upper[i])
            if lo == hi:
                model.add_in_set(expr, EqualTo(lo), name=f"row{i}")
            elif np.isneginf(lo):
                model.add_in_set(expr, LessThan(hi), name=f"row{i}")
            elif np.isposinf(hi):
                model.add_in_set(expr, GreaterThan(lo), name=f"row{i}")
            else:
                model.add_in_set(expr, Interval(lo, hi), name=f"row{i}")
        objective: Expression = model.constant(float(linear.objective_offset))
        for j, value in enumerate(linear.c):
            if value != 0.0:
                objective = objective + float(value) * xs[j]
        if isinstance(problem, QuadraticProblem):
            coo = problem.P.tocoo()
            for i, j, value in zip(coo.row, coo.col, coo.data):
                if value != 0.0:
                    objective = objective + 0.5 * float(value) * xs[int(i)] * xs[int(j)]
        if linear.objective_sense is ObjectiveSense.MINIMIZE:
            model.minimize(objective)
        else:
            model.maximize(objective)
        return model

    def clone(self) -> "Model":
        clone = Model(self.name)
        node_map: dict[str, Expression] = {}
        for key, data in self._variables.items():
            v = clone.variable(data.shape, lower=data.lower, upper=data.upper, domain=data.domain, name=data.name)
            node_map[key] = v
        for key, data in self._parameters.items():
            p = clone.parameter(data.shape, value=data.value, sign=data.sign, name=data.name)
            node_map[key] = p

        def rebuild(node: ExprNode) -> Expression:
            if node.kind == "variable" or node.kind == "parameter":
                return node_map[node.payload]
            if node.kind == "constant":
                return clone.constant(node.payload)
            args = [rebuild(x) for x in node.args]
            if node.kind == "add": return args[0] + args[1]
            if node.kind == "mul": return args[0] * args[1]
            if node.kind == "div": return args[0] / args[1]
            if node.kind == "pow": return args[0] ** node.payload
            if node.kind in {"exp", "log", "sqrt", "sin", "cos", "tanh"}:
                return getattr(args[0], node.kind)()
            if node.kind == "matmul": return args[0] @ args[1]
            if node.kind == "neg": return -args[0]
            if node.kind == "index": return args[0][node.payload]
            if node.kind == "transpose": return args[0].T
            if node.kind == "sum": return args[0].sum(axis=node.payload)
            if node.kind == "concat": return clone._concat(*args)
            raise DomainError(f"cannot clone unsupported expression node {node.kind!r}")

        for c in self._constraints:
            expr = rebuild(c.function._node)
            if isinstance(c.set, (SecondOrderCone, RotatedSecondOrderCone, PositiveSemidefiniteCone, ExponentialCone, PowerCone)):
                clone.add_in_set(expr, c.set, name=c.name)
                continue
            if isinstance(c.set, LessThan): relation = expr <= c.set.upper
            elif isinstance(c.set, GreaterThan): relation = expr >= c.set.lower
            elif isinstance(c.set, EqualTo): relation = expr == c.set.value
            else:
                if isinstance(c, IndicatorConstraint):
                    raise DomainError("P4 clone does not accept Interval indicator bodies")
                clone.add(expr >= c.set.lower, name=c.name)
                clone.add(expr <= c.set.upper, name=c.name)
                continue
            if isinstance(c, IndicatorConstraint):
                new_indicator = node_map[c.indicator.id.value]
                if not isinstance(new_indicator, Variable):
                    raise DomainError("cloned indicator did not resolve to a Variable")
                clone.indicator(new_indicator, relation, active_value=c.active_value, name=c.name)
            else:
                clone.add(relation, name=c.name)
        if self._objective is not None:
            expr = rebuild(self._objective.expression._node)
            if self._objective.sense is ObjectiveSense.MINIMIZE: clone.minimize(expr, name=self._objective.name)
            else: clone.maximize(expr, name=self._objective.name)
        return clone
