from .compiler import CompiledModel, CompilationReport, CompilerCacheInfo, compile_model, compiler_cache_info
from .errors import CompileError, DomainError, ModelingError, OwnershipError, ShapeError, SymbolicTruthValueError
from .expression import Expression
from .model import Constraint, IndicatorConstraint, Model, Objective, Parameter, PendingConstraint, Variable
from .sets import EqualTo, GreaterThan, Interval, LessThan, ExponentialCone, PowerCone, PositiveSemidefiniteCone, RotatedSecondOrderCone, SecondOrderCone
from .types import Curvature, EntityId, NumericType, SignDomain


def sum(expression, axis=None):
    return expression.sum(axis=axis)


__all__ = [
    "CompileError", "CompiledModel", "CompilationReport", "CompilerCacheInfo", "Constraint", "Curvature",
    "DomainError", "EntityId", "EqualTo", "Expression", "GreaterThan", "IndicatorConstraint", "Interval",
    "LessThan", "PositiveSemidefiniteCone", "RotatedSecondOrderCone", "SecondOrderCone", "Model", "ModelingError", "NumericType", "Objective", "OwnershipError",
    "Parameter", "PendingConstraint", "ShapeError", "SignDomain", "SymbolicTruthValueError",
    "Variable", "compile_model", "compiler_cache_info", "sum", "sin", "cos", "exp", "log", "sqrt", "tanh",
]


def sin(x): return x.sin()
def cos(x): return x.cos()
def exp(x): return x.exp()
def log(x): return x.log()
def sqrt(x): return x.sqrt()
def tanh(x): return x.tanh()

from .convenience import indexed_variables, soft_constraint, named_values, diagnose_model, SoftConstraint
__all__ += ["ExponentialCone", "PowerCone", "indexed_variables", "soft_constraint", "named_values", "diagnose_model", "SoftConstraint"]
