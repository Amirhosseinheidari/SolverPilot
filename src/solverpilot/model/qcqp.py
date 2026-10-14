"""Bounded exact quadratic-inequality graph bridges to rotated SOCs.

All constants are interpreted as their represented binary64 rationals. No
rounded eigendecomposition, square root, or tolerance establishes equivalence.
Unsupported or exhausted automatic bridges retain the existing NLP route.
"""
from dataclasses import dataclass, replace
from fractions import Fraction
from hashlib import sha256
import json
from math import lcm
from time import perf_counter

import numpy as np
from scipy import sparse

from solverpilot.problem import ObjectiveSense, VariableDomain
from .errors import CompileError
from .sets import GreaterThan, LessThan


class _Unsupported(Exception):
    pass


class _Limits:
    def __init__(self):
        self.work = 0

    def tick(self, amount=1):
        self.work += amount
        if self.work > 100000:
            raise _Unsupported("exact quadratic bridge work limit")

    def check(self, value):
        self.tick()
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > 2048:
            raise _Unsupported("exact quadratic bridge fraction limit")
        return value


@dataclass
class _Poly:
    coefficients: dict
    limits: _Limits

    @classmethod
    def constant(cls, value, limits):
        value = value if isinstance(value, Fraction) else Fraction.from_float(float(value))
        return cls({(): limits.check(value)} if value else {}, limits)

    @property
    def degree(self):
        return max(map(len, self.coefficients), default=0)

    def coerce(self, other):
        return other if isinstance(other, _Poly) else self.constant(other, self.limits)

    def __add__(self, other):
        other = self.coerce(other)
        values = dict(self.coefficients)
        for key, value in other.coefficients.items():
            values[key] = self.limits.check(values.get(key, Fraction(0)) + value)
            if not values[key]:
                del values[key]
        return _Poly(values, self.limits)

    __radd__ = __add__

    def __neg__(self):
        return _Poly({k: -v for k, v in self.coefficients.items()}, self.limits)

    def __mul__(self, other):
        other = self.coerce(other)
        if self.degree + other.degree > 2:
            raise _Unsupported("nonquadratic product")
        out = {}
        for left, a in self.coefficients.items():
            for right, b in other.coefficients.items():
                key = tuple(sorted(left + right))
                value = self.limits.check(a * b)
                out[key] = self.limits.check(out.get(key, Fraction(0)) + value)
                if not out[key]:
                    del out[key]
        return _Poly(out, self.limits)

    __rmul__ = __mul__


@dataclass
class _Squares:
    affine: _Poly
    terms: tuple = ()

    def coerce(self, other):
        return other if isinstance(other, _Squares) else _Squares(_Poly.constant(other, self.affine.limits))

    def __add__(self, other):
        other = self.coerce(other)
        self.affine.limits.tick(len(self.terms) + len(other.terms))
        return _Squares(self.affine + other.affine, self.terms + other.terms)

    __radd__ = __add__

    def __neg__(self):
        return _Squares(-self.affine, tuple((-w, a) for w, a in self.terms))

    def __mul__(self, other):
        other = self.coerce(other)
        for constant, value in [(self, other), (other, self)]:
            if not constant.terms and constant.affine.degree == 0:
                weight = constant.affine.coefficients.get((), Fraction(0))
                return _Squares(value.affine * weight,
                                tuple((value.affine.limits.check(weight * w), a) for w, a in value.terms))
        if not self.terms and not other.terms and self.affine.coefficients == other.affine.coefficients:
            return _Squares(_Poly.constant(0, self.affine.limits), ((Fraction(1), self.affine),))
        raise _Unsupported("not a graph-derived affine square")

    __rmul__ = __mul__


class _Evaluator:
    def __init__(self, model, offsets, limits, squares):
        self.model, self.offsets, self.limits, self.squares = model, offsets, limits, squares
        self.cache = {}

    def wrap(self, poly):
        return _Squares(poly) if self.squares else poly

    def evaluate(self, node):
        if id(node) in self.cache:
            return self.cache[id(node)]
        size = int(np.prod(node.shape)) if node.shape else 1
        if size > 4096:
            raise _Unsupported("exact quadratic bridge expression-size limit")
        self.limits.tick(size)
        if node.kind in {"constant", "parameter", "variable"}:
            out = np.empty(node.shape, dtype=object)
            if node.kind == "variable":
                start, stop = self.offsets[node.payload]
                for i, j in enumerate(range(start, stop)):
                    out.flat[i] = self.wrap(_Poly({(j,): Fraction(1)}, self.limits))
            else:
                raw = node.payload if node.kind == "constant" else self.model._parameters[node.payload].value
                raw = raw.toarray() if sparse.issparse(raw) else np.asarray(raw)
                for index in np.ndindex(node.shape):
                    out[index] = self.wrap(_Poly.constant(raw[index], self.limits))
        else:
            args = [self.evaluate(a) for a in node.args]
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
                out = np.asarray(np.sum(args[0], axis=node.payload), dtype=object).reshape(node.shape)
            elif node.kind == "concat":
                out = np.concatenate([a.reshape(-1) for a in args])
            else:
                raise _Unsupported("nonpolynomial quadratic bridge input")
        out = np.asarray(out, dtype=object).reshape(node.shape)
        expected = _Squares if self.squares else _Poly
        for index in np.ndindex(node.shape):
            if not isinstance(out[index], expected):
                out[index] = self.wrap(_Poly.constant(out[index], self.limits))
        self.cache[id(node)] = out
        return out


def _float_exact(value):
    try:
        result = float(value)
    except OverflowError as error:
        raise _Unsupported("bridge coefficient overflow") from error
    if not np.isfinite(result) or Fraction.from_float(result) != value:
        raise _Unsupported("bridge coefficient is not exactly representable")
    return result


def validate_qcqp_source(model, candidate, *, compiled, atol=1e-8):
    """Bounded rational scalar-row checks; other sets retain their validators.

    A numerical primal relation check, not an optimality proof. Direct square
    graphs avoid both expanded-float cancellation and dense polynomial growth.
    Every exhausted exact check rejects the candidate explicitly.
    """
    from solverpilot.bridges.validation import SemanticPrimalValidation, validate_semantic_primal
    from .compiler import _variable_layout
    from .sets import EqualTo, Interval

    if isinstance(atol, bool) or type(atol) not in (int, float) or not np.isfinite(atol) or atol < 0:
        raise ValueError('atol must be finite and non-negative')
    offsets, n, *_ = _variable_layout(model)
    try:
        raw = np.asarray(candidate, dtype=float).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        raw = np.array([])
    if raw.size not in {n, compiled.execution_ir.n_variables} or not np.isfinite(raw).all():
        return SemanticPrimalValidation(False, np.inf, ('invalid QCQP candidate',))
    primal = raw[:n]
    limits = _Limits()
    graph = _Evaluator(model, offsets, limits, True)
    polynomial = _Evaluator(model, offsets, limits, False)
    values = [Fraction.from_float(float(v)) for v in primal]

    def evaluate_poly(poly):
        total = Fraction(0)
        for key, coefficient in poly.coefficients.items():
            term = coefficient
            for i in key:
                term = limits.check(term * values[i])
            total = limits.check(total + term)
        return total

    def display(value):
        try:
            return float(value)
        except OverflowError:
            return np.inf

    def upper_violation(value, bound):
        if np.isposinf(bound):
            return Fraction(0)
        if not np.isfinite(bound):
            raise _Unsupported('nonfinite rejecting scalar bound')
        return max(Fraction(0), value - Fraction.from_float(float(bound)))

    def check(constraint, _primal, tolerance):
        label = f'constraint {constraint.entity_id.value}'
        try:
            try:
                rows = graph.evaluate(constraint.function._node)
                activities = []
                for row in rows.flat:
                    value = evaluate_poly(row.affine)
                    for weight, factor in row.terms:
                        activity = evaluate_poly(factor)
                        value = limits.check(value + weight * activity * activity)
                    activities.append(value)
            except _Unsupported:
                activities = [evaluate_poly(row) for row in polynomial.evaluate(constraint.function._node).flat]
            maximum, violations = Fraction(0), []
            threshold = Fraction.from_float(float(tolerance))
            for i, value in enumerate(activities):
                set_ = constraint.set
                if isinstance(set_, LessThan):
                    violation = upper_violation(value, set_.upper)
                elif isinstance(set_, GreaterThan):
                    violation = upper_violation(-value, -set_.lower)
                elif isinstance(set_, EqualTo):
                    if not np.isfinite(set_.value):
                        raise _Unsupported('nonfinite equality bound')
                    violation = abs(value - Fraction.from_float(float(set_.value)))
                elif isinstance(set_, Interval):
                    violation = max(upper_violation(-value, -set_.lower), upper_violation(value, set_.upper))
                else:
                    raise _Unsupported('unsupported scalar set')
                maximum = max(maximum, violation)
                if violation > threshold:
                    violations.append(f'{label}[{i}] rational violation={display(violation)}')
            return SemanticPrimalValidation(not violations, display(maximum), tuple(violations))
        except (_Unsupported, OverflowError, ValueError):
            return SemanticPrimalValidation(False, np.inf, (f'{label} exact source check unavailable or exhausted',))

    return validate_semantic_primal(model, primal, atol=atol, _scalar_validator=check)


def _ldl_squares(poly):
    """Exact small PSD decomposition, with linear/constant terms kept separate."""
    limits = poly.limits
    quadratic = {k: v for k, v in poly.coefficients.items() if len(k) == 2}
    active = sorted({i for key in quadratic for i in key})
    if len(active) > 32:
        raise _Unsupported("general exact LDL dimension limit")
    n = len(active)
    positions = {v: i for i, v in enumerate(active)}
    matrix = [[Fraction(0) for _ in active] for _ in active]
    for (i, j), value in quadratic.items():
        a, b = positions[i], positions[j]
        matrix[a][b] = value if i == j else value / 2
        matrix[b][a] = matrix[a][b]
    factors = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    diagonal = []
    for j in range(n):
        pivot = matrix[j][j]
        for k in range(j):
            pivot = limits.check(pivot - factors[j][k]**2 * diagonal[k])
        if pivot < 0:
            raise _Unsupported("quadratic inequality is not convex")
        diagonal.append(pivot)
        for i in range(j + 1, n):
            residual = matrix[i][j]
            for k in range(j):
                residual = limits.check(residual - factors[i][k] * factors[j][k] * diagonal[k])
            if not pivot:
                if residual:
                    raise _Unsupported("zero PSD pivot has a nonzero residual")
                factors[i][j] = Fraction(0)
            else:
                factors[i][j] = limits.check(residual / pivot)
    terms = []
    for j, weight in enumerate(diagonal):
        if weight:
            row = _Poly({(active[i],): factors[i][j] for i in range(j, n) if factors[i][j]}, limits)
            terms.append((weight, row))
    affine = _Poly({k: v for k, v in poly.coefficients.items() if len(k) < 2}, limits)
    return _Squares(affine, tuple(terms))


def _affine_expression(poly, lowered, variables):
    value = lowered.constant(_float_exact(poly.coefficients.get((), Fraction(0))))
    for key, coefficient in sorted(poly.coefficients.items()):
        if key:
            if len(key) != 1:
                raise _Unsupported("nonaffine emitted cone map")
            value = value + _float_exact(coefficient) * variables[key[0]]
    return value


def _scaled_term(weight, row):
    """Return exact RSOC head scales and affine tail without taking sqrt."""
    if weight <= 0:
        raise _Unsupported("nonpositive square weight")
    try:
        half = _float_exact(weight / 2)
        scaled = row * weight
        for value in scaled.coefficients.values():
            _float_exact(value)
        return Fraction(1), Fraction.from_float(half), scaled
    except _Unsupported:
        denominator = lcm(*(v.denominator for v in row.coefficients.values()))
        row.limits.check(Fraction(denominator))
        integer_row = row * denominator
        numerator, divisor = weight.numerator, weight.denominator
        head = row.limits.check(Fraction(divisor * denominator**2))
        half = row.limits.check(Fraction(numerator, 2))
        scaled = integer_row * numerator
        _float_exact(head)
        _float_exact(half)
        for value in scaled.coefficients.values():
            _float_exact(value)
        return head, half, scaled


def try_compile_qcqp(model, *, use_cache=True, bridge_policy=None, capabilities=None):
    """Return an exact supported bridge, otherwise ``None`` for existing routing."""
    from .compiler import _variable_layout
    from .model import IndicatorConstraint, Model
    if model.objective is None or model.objective.sense is not ObjectiveSense.MINIMIZE:
        return None
    if any(v.domain is not VariableDomain.CONTINUOUS for v in model.variables):
        return None
    candidates = [c for c in model.constraints if c.function.polynomial_degree == 2]
    if not candidates or any(isinstance(c, IndicatorConstraint) for c in model.constraints):
        return None
    if any(c.function.polynomial_degree is None or c.function.polynomial_degree > 2 for c in model.constraints):
        return None
    if any(not isinstance(c.set, (LessThan, GreaterThan)) for c in candidates):
        return None
    if bridge_policy is not None:
        from solverpilot.bridges.types import ExactnessClass
        if not bridge_policy.permits(ExactnessClass.EXACT_EQUIVALENT):
            return None
    key = "qcqp-exact-v1:" + model.data_hash
    if use_cache and key in model._compiler_cache:
        return model._compiler_cache[key]
    started = perf_counter()
    offsets, n, *_ = _variable_layout(model)
    limits = _Limits()
    graph = _Evaluator(model, offsets, limits, True)
    polynomial = _Evaluator(model, offsets, limits, False)
    decompositions = {}
    try:
        for c in candidates:
            sign = Fraction(1 if isinstance(c.set, LessThan) else -1)
            try:
                values = np.asarray(graph.evaluate(c.function._node) * sign, dtype=object).reshape(c.function.shape)
                if any(w < 0 for item in values.flat for w, _ in item.terms):
                    raise _Unsupported("graph has negative square weights")
            except _Unsupported:
                values = np.asarray([_ldl_squares(p * sign) for p in polynomial.evaluate(c.function._node).flat], dtype=object).reshape(c.function.shape)
            decompositions[c.entity_id.value] = values

        term_count = sum(len(item.terms) for values in decompositions.values() for item in values.flat)
        component_count = sum(values.size for values in decompositions.values())
        # Worst case: one three-coordinate RSOC per factor and one row per
        # scalar component. Count components, not vector constraint objects.
        if term_count > 4096 or 3 * term_count + component_count > 16384:
            raise _Unsupported("quadratic bridge output-size limit")

        lowered = Model(model.name)
        originals, values, scalar_variables = {}, {}, []
        for variable in model.variables:
            copy = lowered.variable(variable.shape, lower=variable.lower, upper=variable.upper, name=variable.name)
            originals[variable.id.value] = values[variable.id.value] = copy
            scalar_variables.extend(copy[index] for index in np.ndindex(copy.shape))
        for parameter in model.parameters:
            values[parameter.id.value] = lowered.parameter(parameter.shape, value=parameter.value, sign=parameter.sign, name=parameter.name)
        clone_cache = {}

        def clone(node):
            if id(node) in clone_cache:
                return clone_cache[id(node)]
            if node.kind in {"variable", "parameter"}:
                out = values[node.payload]
            elif node.kind == "constant":
                out = lowered.constant(node.payload)
            else:
                args = [clone(a) for a in node.args]
                if node.kind == "add": out = args[0] + args[1]
                elif node.kind == "mul": out = args[0] * args[1]
                elif node.kind == "matmul": out = args[0] @ args[1]
                elif node.kind == "neg": out = -args[0]
                elif node.kind == "index": out = args[0][node.payload]
                elif node.kind == "transpose": out = args[0].T
                elif node.kind == "sum": out = args[0].sum(axis=node.payload, keepdims=len(node.shape) == len(node.args[0].shape) and bool(node.shape))
                elif node.kind == "concat": out = lowered._concat(*args)
                else: raise _Unsupported("unsupported objective or affine graph")
            clone_cache[id(node)] = out
            return out

        generated, steps = {}, []
        for constraint in model.constraints:
            cid = constraint.entity_id.value
            if cid not in decompositions:
                generated[cid] = [lowered.add_in_set(clone(constraint.function._node), constraint.set, name=constraint.name)]
                continue
            generated[cid] = []
            bound = constraint.set.upper if isinstance(constraint.set, LessThan) else -constraint.set.lower
            proof_rows = []
            for component, item in enumerate(decompositions[cid].flat):
                rhs_poly = _Poly.constant(bound, limits) + (-item.affine)
                rhs = _affine_expression(rhs_poly, lowered, scalar_variables)
                groups = {}
                for weight, row in item.terms:
                    if weight:
                        groups.setdefault(weight, []).append(row)
                epigraphs = []
                for group_index, (weight, rows) in enumerate(groups.items()):
                    try:
                        coefficient = _float_exact(weight)
                        for row in rows:
                            for value in row.coefficients.values():
                                _float_exact(value)
                        unit_group = True
                    except _Unsupported:
                        unit_group = False
                    if unit_group:
                        # Put the represented weight in the linear epigraph sum,
                        # avoiding both sqrt(weight) and rounded weight*factor.
                        direct = len(groups) == 1 and weight == 1
                        epigraph = rhs if direct else lowered.variable(lower=0., name=f"qcqp[{cid},{component},{group_index}]")
                        if not direct:
                            epigraphs.append((coefficient, epigraph))
                        tail = lowered._concat(*[_affine_expression(row, lowered, scalar_variables) for row in rows])
                        generated[cid].append(lowered.rotated_soc(epigraph, .5, tail, name=constraint.name))
                        continue
                    scaled = [_scaled_term(weight, row) for row in rows]
                    # Direct grouping needs equal head scales. Integer-cleared LDL
                    # rows can have different scales, so retain their own epigraphs.
                    same = all(a[:2] == scaled[0][:2] for a in scaled)
                    batches = [scaled] if same else [[a] for a in scaled]
                    for batch_index, batch in enumerate(batches):
                        head, half, _ = batch[0]
                        direct = len(groups) == 1 and len(batches) == 1
                        epigraph = rhs if direct else lowered.variable(lower=0., name=f"qcqp[{cid},{component},{group_index},{batch_index}]")
                        if not direct:
                            epigraphs.append((1., epigraph))
                        tail = lowered._concat(*[_affine_expression(a, lowered, scalar_variables) for _, _, a in batch])
                        u = _affine_expression(rhs_poly * head, lowered, scalar_variables) if direct else _float_exact(head) * epigraph
                        cone = lowered.rotated_soc(u, _float_exact(half), tail, name=constraint.name)
                        generated[cid].append(cone)
                if epigraphs or not groups:
                    total = lowered.constant(0.)
                    for weight, epigraph in epigraphs:
                        total = total + weight * epigraph
                    generated[cid].append(lowered.add(total <= rhs, name=constraint.name))
                proof_rows.append({"component": component, "weights": [[str(w.numerator), str(w.denominator)] for w, _ in item.terms],
                                   "factors": [[[list(k), str(v.numerator), str(v.denominator)] for k, v in row.coefficients.items()] for _, row in item.terms]})
            from solverpilot.bridges.types import (TransformationStep, PreconditionCertificate,
                                                   ExactnessClass, MappingAvailability)
            certificate = PreconditionCertificate("exact_quadratic_decomposition", {
                "decomposition": proof_rows,
                "parameter_dependencies": sorted(constraint.function.parameter_dependencies),
                "arithmetic": "rational_graph_identity_and_exact_binary64_cone_coefficients",
            })
            steps.append(TransformationStep(
                step_id=f"qcqp:{cid}", bridge_id="qcqp_rsoc_exact", bridge_version="1",
                source_entity_ids=(cid,), generated_target_ids=tuple(c.entity_id.value for c in generated[cid]),
                exactness=ExactnessClass.EXACT_EQUIVALENT, preconditions=(certificate,),
                primal_mapping=MappingAvailability.AVAILABLE,
                notes=("Independent conic bounds concern the execution IR; semantic dual/certificate reconstruction is unavailable.",),
            ).to_dict())
        lowered.minimize(clone(model.objective.expression._node))
        compiled = lowered.compile(use_cache=False, bridge_policy=bridge_policy, capabilities=capabilities)
        metadata = {**compiled.execution_ir.metadata,
                    "qcqp_source_semantic_hash": model.semantic_hash,
                    "qcqp_source_data_hash": model.data_hash,
                    "lowered_semantic_hash": compiled.semantic_hash,
                    "qcqp_bridge_version": "exact-v1"}
        from solverpilot.problem import QuadraticProblem
        if isinstance(compiled.execution_ir, QuadraticProblem):
            execution_ir = replace(compiled.execution_ir, metadata=metadata,
                                   linear=replace(compiled.execution_ir.linear, metadata=metadata))
        else:
            execution_ir = replace(compiled.execution_ir, metadata=metadata)
        certificates = tuple(cert["sha256"] for step in steps for cert in step["preconditions"])
        layout = {"variables": execution_ir.n_variables,
                  "generated_counts": [len(items) for items in generated.values()],
                  "certificate_hashes": certificates, "policy": "exact-v1"}
        compilation_hash = sha256((model.semantic_hash + json.dumps(layout, sort_keys=True)).encode()).hexdigest()
        source = {
            "variables": {key: compiled.source_map["variables"][copy.id.value] for key, copy in originals.items()},
            "constraints": {key: {**({} if key in decompositions else compiled.source_map["constraints"][constraints[0].entity_id.value]),
                                    "semantic_kind": "exact_quadratic_inequality" if key in decompositions else "unchanged",
                                    "generated": [compiled.source_map["constraints"][c.entity_id.value] for c in constraints],
                                    "execution_rows": [r for c in constraints for r in compiled.source_map["constraints"][c.entity_id.value].get("execution_rows", [])],
                                    "cone_blocks": [compiled.source_map["constraints"][c.entity_id.value]["cone_block"] for c in constraints if "cone_block" in compiled.source_map["constraints"][c.entity_id.value]],
                                    "parameter_dependencies": sorted(next(c.function.parameter_dependencies for c in model.constraints if c.entity_id.value == key))}
                            for key, constraints in generated.items()},
            "objective": {model.objective.entity_id.value: {"execution_objective": True}},
        }
        result = replace(compiled, execution_ir=execution_ir, semantic_hash=model.semantic_hash, data_hash=model.data_hash,
                         compilation_hash=compilation_hash,
                         source_map=source, transformation_tape=tuple(steps),
                         precondition_certificate_hashes=certificates,
                         parameter_map={p.id.value: {"patchable": False, "parameter_compile_class": "qcqp_full_relower"} for p in model.parameters},
                         plan_sensitive_dependencies=tuple(p.id.value for p in model.parameters),
                         mutation_plan={"mode": "qcqp_full_relower", "incremental_parameter_patching": False},
                         reconstruction_contract={**compiled.reconstruction_contract, "semantic_variable_count": n,
                                                  "source_primal": "bounded_rational_qcqp_v1",
                                                  "dual": "unavailable_qcqp_v1", "certificate": "execution_ir_only_qcqp_v1"},
                         compilation_report=replace(compiled.compilation_report, total_s=perf_counter() - started,
                                                    n_semantic_variables=len(model.variables), n_semantic_constraints=len(model.constraints),
                                                    n_transformations=len(steps), cache_status="qcqp_full_relower"))
    except (_Unsupported, OverflowError, ValueError, CompileError):
        return None
    if use_cache:
        snapshots = [k for k in model._compiler_cache if k.startswith("qcqp-exact-v1:")]
        if len(snapshots) >= 8:
            model._compiler_cache.pop(snapshots[0])
        model._compiler_cache[key] = result
    return result
