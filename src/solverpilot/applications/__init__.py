"""Optional application packs built on SolverPilot's canonical optimization core."""

__all__: list[str] = ['production_model', 'transportation_model', 'energy_dispatch_model', 'mpc_model']
from .templates import production_model, transportation_model, energy_dispatch_model, mpc_model
