def legacy_signature(obj):
    """Project documented additive optional parameters out of historical signatures."""
    import inspect
    additions = {
        "PhaseTimings": {"backend_update_s", "backend_total_s", "primal_validate_s", "certificate_s"},
        "SolveTrace": {"execution_id", "source_sha256", "replay_of"},
        "QuadraticProblem": {"_verify_convexity"},
        "ValidationTolerances": {"feasibility_rel"},
        "execute": {"tolerances", "proof_policy", "verification_workspace"}, "solve": {"tolerances", "certificate_recovery", "proof_policy", "verification_workspace"},
        "solve_production": {"tolerances", "certificate_recovery", "proof_policy", "verification_workspace"}, "execute_portfolio": {"tolerances"},
    }
    try:
        sig = inspect.signature(obj)
    except (ValueError, TypeError):
        return None
    allowed = additions.get(getattr(obj, '__name__', ''), set())
    for name in allowed:
        assert name in sig.parameters
        assert sig.parameters[name].default is not inspect.Parameter.empty
    return str(sig.replace(parameters=[p for n, p in sig.parameters.items() if n not in allowed]))


def published_04_signature(obj):
    """Preserve the published 0.4 signature while checking documented extensions."""
    import inspect
    try:
        sig = inspect.signature(obj)
    except (ValueError, TypeError):
        return None
    additions = {"execution_id", "source_sha256", "replay_of"} if getattr(obj, "__name__", "") == "SolveTrace" else set()
    if getattr(obj, "__name__", "") == "PhaseTimings":
        additions = {"primal_validate_s", "certificate_s"}
    if getattr(obj, "__name__", "") in {"solve", "solve_production", "execute"}:
        additions = {"proof_policy", "verification_workspace"}
    for name in additions:
        parameter = sig.parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is not inspect.Parameter.empty
    return str(sig.replace(parameters=[p for n, p in sig.parameters.items() if n not in additions]))
