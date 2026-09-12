from dataclasses import dataclass, field, replace
import numpy as np
from scipy import sparse
from solverpilot.problem import LinearProblem, ObjectiveSense, VariableDomain
from solverpilot.problem.linear import _readonly_csr
from solverpilot.problem.quadratic import _canonicalize_hessian
from solverpilot.problem.hashing import _hash_parts, hash_sparse_matrix
from solverpilot.nlp.ir import NLPProblem, NLPConstraintBlock
from solverpilot._immutability import deep_freeze, readonly_array


@dataclass(frozen=True, slots=True)
class GlobalQuadraticProblem:
    """Matrix-first 1/2 x' P x + q'x + offset, with optional integer domains.

    Convexity is not asserted. Both objective senses are supported.
    """

    linear: LinearProblem
    P: sparse.csr_matrix
    data_hash: str = field(init=False)
    structural_hash: str = field(init=False)

    def __post_init__(self):
        if not isinstance(self.linear, LinearProblem):
            raise TypeError("linear must be a LinearProblem")
        P = _canonicalize_hessian(_readonly_csr(self.P))
        if P.shape != (self.linear.n_variables, self.linear.n_variables):
            raise ValueError("quadratic matrix shape mismatch")
        object.__setattr__(self, "P", P)
        object.__setattr__(
            self,
            "data_hash",
            _hash_parts(
                [
                    b"global-quadratic-v1",
                    self.linear.data_hash.encode(),
                    hash_sparse_matrix(P).encode(),
                ]
            ),
        )
        object.__setattr__(
            self,
            "structural_hash",
            _hash_parts(
                [
                    b"global-quadratic-v1",
                    self.linear.structural_hash.encode(),
                    hash_sparse_matrix(P.astype(bool)).encode(),
                ]
            ),
        )

    @classmethod
    def from_data(
        cls,
        *,
        P,
        A,
        q,
        variable_lower,
        variable_upper,
        constraint_lower,
        constraint_upper,
        domains=None,
        objective_sense="minimize",
        objective_offset=0.0,
        name=None,
        metadata=None,
    ):
        linear = LinearProblem.from_data(
            A=A,
            c=q,
            variable_lower=variable_lower,
            variable_upper=variable_upper,
            constraint_lower=constraint_lower,
            constraint_upper=constraint_upper,
            domains=domains,
            objective_sense=objective_sense,
            objective_offset=objective_offset,
            name=name,
            metadata=metadata,
        )
        return cls(linear, sparse.csr_matrix(P))

    @property
    def n_variables(self):
        return self.linear.n_variables

    @property
    def n_constraints(self):
        return self.linear.n_constraints

    @property
    def objective_sense(self):
        return self.linear.objective_sense

    def objective_value(self, x):
        x = np.asarray(x, dtype=float)
        return float(0.5 * x @ (self.P @ x) + self.linear.c @ x + self.linear.objective_offset)


@dataclass(frozen=True, slots=True)
class GlobalIndicator:
    variable_index: int
    active_value: int
    constraint: NLPConstraintBlock


@dataclass(frozen=True, slots=True)
class FactorableProblem:
    """Bounded algebraic NLP/MINLP; arbitrary Python callbacks are not accepted."""

    relaxation: NLPProblem
    domains: tuple[str, ...]
    indicators: tuple[GlobalIndicator, ...] = ()
    metadata: dict = field(default_factory=dict)
    data_hash: str = field(init=False)

    def __post_init__(self):
        p = self.relaxation
        if not isinstance(p, NLPProblem):
            raise TypeError("relaxation must be NLPProblem")
        from .expressions import freeze_node, evaluate_node

        def freeze_block(c):
            if not isinstance(c, NLPConstraintBlock):
                raise TypeError("constraints must be NLPConstraintBlock")
            block = replace(c, node=freeze_node(c.node), shape=tuple(c.shape))
            object.__setattr__(block, "lower", readonly_array(block.lower, dtype=float))
            object.__setattr__(block, "upper", readonly_array(block.upper, dtype=float))
            return block

        indicators = tuple(self.indicators)
        if any(not isinstance(i, GlobalIndicator) for i in indicators):
            raise TypeError("indicators must be GlobalIndicator")
        indicators = tuple(replace(i, constraint=freeze_block(i.constraint)) for i in indicators)
        p = replace(
            p,
            objective_node=freeze_node(p.objective_node),
            constraints=tuple(freeze_block(c) for c in p.constraints),
        )
        object.__setattr__(p, "variable_lower", readonly_array(p.variable_lower, dtype=float))
        object.__setattr__(p, "variable_upper", readonly_array(p.variable_upper, dtype=float))
        object.__setattr__(p, "parameter_values", deep_freeze(p.parameter_values))
        object.__setattr__(p, "metadata", deep_freeze(p.metadata))
        object.__setattr__(self, "relaxation", p)
        object.__setattr__(self, "indicators", indicators)
        occupied = np.zeros(p.n_variables, dtype=bool)
        layout = {}
        for key, (offset, shape) in p.variable_layout.items():
            shape = tuple(shape)
            if (
                type(offset) is not int
                or offset < 0
                or any(type(d) is not int or d < 1 for d in shape)
            ):
                raise ValueError("invalid global variable layout")
            size = int(np.prod(shape)) if shape else 1
            if offset + size > p.n_variables or occupied[offset : offset + size].any():
                raise ValueError("overlapping or out-of-range global variable layout")
            occupied[offset : offset + size] = True
            layout[key] = (offset, shape)
        if not occupied.all():
            raise ValueError("global variable layout must cover all variables")
        object.__setattr__(p, "variable_layout", deep_freeze(layout))
        if any(not np.isfinite(v).all() for v in p.parameter_values.values()):
            raise ValueError("nonfinite global parameter")
        domains = tuple(VariableDomain(d).value for d in self.domains)
        if len(domains) != p.n_variables:
            raise ValueError("domains must match variable count")
        if not np.isfinite(p.variable_lower).all() or not np.isfinite(p.variable_upper).all():
            raise ValueError("factorable global problems require finite declared variable bounds")
        ObjectiveSense(p.objective_sense)
        if p.objective_node.shape != ():
            raise ValueError("global objective must be scalar")
        for i, d in enumerate(domains):
            if d == VariableDomain.BINARY.value and (
                p.variable_lower[i] < 0 or p.variable_upper[i] > 1
            ):
                raise ValueError("binary variable bounds must lie in [0, 1]")
        for c in (*p.constraints, *(i.constraint for i in self.indicators)):
            if np.isnan(c.lower).any() or np.isnan(c.upper).any() or np.any(c.lower > c.upper):
                raise ValueError("invalid nonlinear constraint bounds")
            if np.any(np.isposinf(c.lower)) or np.any(np.isneginf(c.upper)):
                raise ValueError("invalid nonlinear bound direction")
            if c.shape != c.node.shape or c.lower.size != (
                int(np.prod(c.node.shape)) if c.node.shape else 1
            ):
                raise ValueError("nonlinear constraint shape mismatch")
        for indicator in self.indicators:
            i = indicator.variable_index
            if (
                type(i) is not int
                or not 0 <= i < p.n_variables
                or domains[i] != VariableDomain.BINARY.value
            ):
                raise ValueError("indicator premise must index a binary variable")
            if type(indicator.active_value) is not int or indicator.active_value not in (0, 1):
                raise ValueError("indicator active_value must be zero or one")
        object.__setattr__(self, "domains", domains)
        object.__setattr__(self, "indicators", tuple(self.indicators))
        object.__setattr__(self, "metadata", deep_freeze(self.metadata))
        # Include indicator bodies in the canonical NLP hash, and premise semantics separately.
        all_rows = replace(
            p, constraints=tuple(p.constraints) + tuple(i.constraint for i in self.indicators)
        )
        payload = repr(
            (domains, tuple((i.variable_index, i.active_value) for i in self.indicators))
        )
        object.__setattr__(
            self,
            "data_hash",
            _hash_parts([b"factorable-global-v1", all_rows.data_hash.encode(), payload.encode()]),
        )
        from .intervals import validate_domains

        midpoint = 0.5 * p.variable_lower + 0.5 * p.variable_upper

        def check_node(node):
            for child in node.args:
                check_node(child)
            try:
                actual = evaluate_node(self, node, midpoint)
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError("invalid global expression reference") from exc
            if actual.shape != node.shape:
                raise ValueError("global expression shape mismatch")

        for node in (p.objective_node, *(c.node for c in all_rows.constraints)):
            check_node(node)
            validate_domains(self, node)

    @property
    def n_variables(self):
        return self.relaxation.n_variables

    @property
    def objective_sense(self):
        return ObjectiveSense(self.relaxation.objective_sense)

    @property
    def n_constraints(self):
        return self.relaxation.n_constraints + sum(i.constraint.lower.size for i in self.indicators)
