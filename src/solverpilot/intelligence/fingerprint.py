"""Versioned feature schema for SolverPilot's existing ProblemFingerprint."""
from __future__ import annotations

from solverpilot.inspect import ProblemFingerprint, inspect_problem
from solverpilot.problem import LinearProblem, QuadraticProblem

from .schema import FeatureDefinition, FeatureRecord, MissingnessPolicy, build_feature_record, build_feature_schema

_EXTRACTOR_ID = "solverpilot.inspect.problem_fingerprint"
_EXTRACTOR_VERSION = "1.0.0"


def _d(name: str, dtype: str, unit: str, source: tuple[str, ...], *, nullable: bool = False, allowed: tuple[str, ...] = (), lower: float | None = None, upper: float | None = None) -> FeatureDefinition:
    return FeatureDefinition(
        name=name,
        data_type=dtype,
        unit=unit,
        missingness=MissingnessPolicy.NULLABLE if nullable else MissingnessPolicy.FORBIDDEN,
        description=f"Pre-solve structural feature {name} derived from the canonical problem representation.",
        source_fields=source,
        lower_bound=lower,
        upper_bound=upper,
        allowed_values=allowed,
    )


def problem_fingerprint_feature_schema():
    defs = [
        _d("problem_class", "categorical", "category", ("problem.type",), allowed=("convex_qp", "lp", "milp")),
        _d("n_variables", "integer", "count", ("problem.variables",), lower=0),
        _d("n_constraints", "integer", "count", ("problem.constraints",), lower=0),
        _d("nnz_a", "integer", "count", ("problem.A",), lower=0),
        _d("density_a", "number", "ratio", ("problem.A",), lower=0, upper=1),
        _d("binary_fraction", "number", "ratio", ("problem.domains",), lower=0, upper=1),
        _d("integer_fraction", "number", "ratio", ("problem.domains",), lower=0, upper=1),
        _d("equality_fraction", "number", "ratio", ("problem.constraint_bounds",), lower=0, upper=1),
        _d("fully_bounded_variable_fraction", "number", "ratio", ("problem.variable_bounds",), lower=0, upper=1),
        _d("free_variable_fraction", "number", "ratio", ("problem.variable_bounds",), lower=0, upper=1),
        _d("objective_nnz_fraction", "number", "ratio", ("problem.objective_coefficients",), lower=0, upper=1),
        _d("coefficient_dynamic_range_log10", "number", "log10_ratio", ("problem.A",), nullable=True, lower=0),
        _d("objective_dynamic_range_log10", "number", "log10_ratio", ("problem.objective_coefficients",), nullable=True, lower=0),
        _d("rhs_dynamic_range_log10", "number", "log10_ratio", ("problem.constraint_bounds", "problem.variable_bounds"), nullable=True, lower=0),
        _d("singleton_rows", "integer", "count", ("problem.A",), lower=0),
        _d("singleton_cols", "integer", "count", ("problem.A",), lower=0),
        _d("positive_coefficient_fraction", "number", "ratio", ("problem.A",), lower=0, upper=1),
        _d("negative_coefficient_fraction", "number", "ratio", ("problem.A",), lower=0, upper=1),
        _d("quadratic_nnz", "integer", "count", ("problem.P",), nullable=True, lower=0),
        _d("quadratic_density", "number", "ratio", ("problem.P",), nullable=True, lower=0, upper=1),
        _d("convexity_status", "categorical", "category", ("problem.convexity_status",), nullable=True, allowed=("confirmed",)),
        _d("estimated_numeric_bytes", "integer", "bytes", ("problem.canonical_arrays",), lower=0),
    ]
    for prefix in ("row_nnz", "col_nnz"):
        for suffix in ("minimum", "q25", "median", "mean", "q75", "maximum"):
            defs.append(_d(f"{prefix}_{suffix}", "number", "count", ("problem.A",), lower=0))
    return build_feature_schema(domain="solverpilot_problem_fingerprint", version="1.0.0", definitions=defs)


PROBLEM_FINGERPRINT_FEATURE_SCHEMA = problem_fingerprint_feature_schema()


def _values(f: ProblemFingerprint) -> dict[str, object]:
    values: dict[str, object] = {
        "problem_class": f.problem_class,
        "n_variables": f.n_variables,
        "n_constraints": f.n_constraints,
        "nnz_a": f.nnz_a,
        "density_a": f.density_a,
        "binary_fraction": f.binary_fraction,
        "integer_fraction": f.integer_fraction,
        "equality_fraction": f.equality_fraction,
        "fully_bounded_variable_fraction": f.fully_bounded_variable_fraction,
        "free_variable_fraction": f.free_variable_fraction,
        "objective_nnz_fraction": f.objective_nnz_fraction,
        "coefficient_dynamic_range_log10": f.coefficient_dynamic_range_log10,
        "objective_dynamic_range_log10": f.objective_dynamic_range_log10,
        "rhs_dynamic_range_log10": f.rhs_dynamic_range_log10,
        "singleton_rows": f.singleton_rows,
        "singleton_cols": f.singleton_cols,
        "positive_coefficient_fraction": f.positive_coefficient_fraction,
        "negative_coefficient_fraction": f.negative_coefficient_fraction,
        "quadratic_nnz": f.quadratic_nnz,
        "quadratic_density": f.quadratic_density,
        "convexity_status": f.convexity_status,
        "estimated_numeric_bytes": f.estimated_numeric_bytes,
    }
    for prefix, stats in (("row_nnz", f.row_nnz), ("col_nnz", f.col_nnz)):
        for suffix in ("minimum", "q25", "median", "mean", "q75", "maximum"):
            values[f"{prefix}_{suffix}"] = getattr(stats, suffix)
    return values


def feature_record_from_problem(
    problem: LinearProblem | QuadraticProblem,
    *,
    instance_id: str | None = None,
) -> FeatureRecord:
    fingerprint = inspect_problem(problem)
    values = _values(fingerprint)
    reasons = {name: "not applicable to this problem class" for name, value in values.items() if value is None}
    return build_feature_record(
        schema=PROBLEM_FINGERPRINT_FEATURE_SCHEMA,
        instance_id=problem.data_hash if instance_id is None else instance_id,
        values=values,
        missingness_reasons=reasons,
        source_content_sha256=problem.data_hash,
        extractor_id=_EXTRACTOR_ID,
        extractor_version=_EXTRACTOR_VERSION,
    )
