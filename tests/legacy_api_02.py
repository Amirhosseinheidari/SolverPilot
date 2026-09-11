def legacy_signature(obj):
    """Project only documented 0.2 optional additions out of historical signatures."""
    import inspect
    additions = {
        "PhaseTimings": {"backend_update_s", "backend_total_s"},
        "QuadraticProblem": {"_verify_convexity"},
        "ValidationTolerances": {"feasibility_rel"},
        "execute": {"tolerances"}, "solve": {"tolerances"},
        "solve_production": {"tolerances"}, "execute_portfolio": {"tolerances"},
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
