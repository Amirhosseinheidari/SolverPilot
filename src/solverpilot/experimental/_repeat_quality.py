"""Matched repeat success comparisons shared by LP research gates."""
import numpy as np


def paired_repeat_losses(selected, baseline):
    """Count baseline-successful repeats lost at the same index by the policy."""
    a, b = np.asarray(selected), np.asarray(baseline)
    if a.shape != b.shape or not a.size or not a.ndim:
        raise ValueError('paired repeat masks must have equal nonempty shapes')
    if any(not isinstance(v, (bool, np.bool_)) for mask in (a, b) for v in mask.flat):
        raise ValueError('paired repeat masks must contain boolean flags')
    return int(np.count_nonzero(b.astype(bool, copy=False) & ~a.astype(bool, copy=False)))
