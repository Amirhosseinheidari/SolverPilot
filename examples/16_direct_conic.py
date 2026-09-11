"""Direct PSD optimization; install solverpilot[clarabel] to run."""
import numpy as np
from solverpilot.model import Model
from solverpilot.conic import ClarabelBackend

backend = ClarabelBackend()
if not backend.is_available():
    print('Optional example: install solverpilot[clarabel].')
else:
    m = Model('small-psd')
    t = m.variable(lower=0, upper=10, name='spectral-shift')
    m.psd(t*m.constant(np.eye(2))+m.constant([[0., 1.], [1., 0.]]))
    m.minimize(t)
    result = m.compile().solve(backend=backend)
    assert result.validated and abs(result.objective_reported-1.) < 1e-6
    print(result.objective_reported)
