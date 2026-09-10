from __future__ import annotations

from typing import Any, Mapping

from .ir import (
    CPAllDifferentIR,
    CPCircuitIR,
    CPCumulativeIR,
    CPElementIR,
    CPExactlyOneIR,
    CPIntVarIR,
    CPIntervalIR,
    CPLinearConstraintIR,
    CPLinearExprIR,
    CPNoOverlapIR,
    CPObjectiveSense,
    CPProblem,
    CPTableIR,
)


def _expr_from_dict(payload: Mapping[str, Any]) -> CPLinearExprIR:
    return CPLinearExprIR(
        terms=tuple((int(v), int(c)) for v, c in payload.get("terms", ())),
        constant=int(payload.get("constant", 0)),
    )


def problem_from_canonical_dict(payload: Mapping[str, Any]) -> CPProblem:
    """Reconstruct canonical CP IR from :meth:`CPProblem.canonical_dict` output.

    This deliberately accepts only the closed P9 schema families. It is used by
    the isolated CP-SAT worker so no pickle/arbitrary-code serialization crosses
    the process boundary.
    """
    variables = tuple(
        CPIntVarIR(
            var_id=int(v["id"]),
            name=str(v["name"]),
            domain=tuple(int(x) for x in v["domain"]),
            kind=str(v.get("kind", "int")),
            derived_from=None
            if v.get("derived_from") is None
            else (int(v["derived_from"][0]), int(v["derived_from"][1])),
        )
        for v in payload.get("variables", ())
    )
    intervals = tuple(
        CPIntervalIR(
            interval_id=int(i["id"]),
            name=str(i["name"]),
            start_var=int(i["start_var"]),
            size=int(i["size"]),
            end_var=int(i["end_var"]),
        )
        for i in payload.get("intervals", ())
    )

    constraints = []
    for c in payload.get("constraints", ()):
        kind = c["kind"]
        if kind == "linear":
            constraints.append(
                CPLinearConstraintIR(
                    _expr_from_dict(c["expr"]),
                    None if c.get("lower") is None else int(c["lower"]),
                    None if c.get("upper") is None else int(c["upper"]),
                )
            )
        elif kind == "all_different":
            constraints.append(CPAllDifferentIR(tuple(int(x) for x in c["var_ids"])))
        elif kind == "exactly_one":
            constraints.append(CPExactlyOneIR(tuple(int(x) for x in c["literal_ids"])))
        elif kind == "table":
            constraints.append(
                CPTableIR(
                    tuple(int(x) for x in c["var_ids"]),
                    tuple(tuple(int(x) for x in row) for row in c["allowed_tuples"]),
                )
            )
        elif kind == "element":
            constraints.append(
                CPElementIR(
                    int(c["index_var"]),
                    tuple(int(x) for x in c["values"]),
                    int(c["target_var"]),
                )
            )
        elif kind == "circuit":
            constraints.append(
                CPCircuitIR(tuple(tuple(int(x) for x in arc) for arc in c["arcs"]))
            )
        elif kind == "no_overlap":
            constraints.append(CPNoOverlapIR(tuple(int(x) for x in c["interval_ids"])))
        elif kind == "cumulative":
            constraints.append(
                CPCumulativeIR(
                    tuple(int(x) for x in c["interval_ids"]),
                    tuple(int(x) for x in c["demands"]),
                    int(c["capacity"]),
                )
            )
        else:
            raise ValueError(f"unsupported CP constraint kind in isolated payload: {kind!r}")

    objective_payload = payload.get("objective")
    objective = None if objective_payload is None else _expr_from_dict(objective_payload)
    return CPProblem(
        variables=variables,
        intervals=intervals,
        constraints=tuple(constraints),
        objective=objective,
        objective_sense=CPObjectiveSense(payload.get("objective_sense", "minimize")),
        name=payload.get("name"),
        metadata=dict(payload.get("metadata", {})),
        schema_version=str(payload.get("schema_version", "solverpilot.cp-ir.v1")),
    )
