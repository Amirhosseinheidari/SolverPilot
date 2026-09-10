class ModelingError(Exception):
    """Base error for the semantic modeling layer."""


class ShapeError(ModelingError, ValueError):
    pass


class DomainError(ModelingError, ValueError):
    pass


class OwnershipError(ModelingError, ValueError):
    pass


class SymbolicTruthValueError(ModelingError, TypeError):
    pass


class CompileError(ModelingError, ValueError):
    pass
