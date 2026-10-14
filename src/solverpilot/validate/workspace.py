"""Bounded preparation reuse and cooperative numerical proof controls.

Only immutable matrix preparation is cached. Candidate validation and proof
acceptance are recomputed on every invocation. Limits are cooperative, not a
process deadline or an exact symbolic optimality claim.
"""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from threading import RLock
from time import perf_counter
import math
import hashlib
import numpy as np


@dataclass(frozen=True, slots=True)
class ProofPolicy:
    mode: str = 'standard'
    time_limit_s: float | None = None
    max_terms: int | None = None

    def __post_init__(self):
        if self.mode not in {'standard','budgeted','required'}: raise ValueError('unknown proof policy')
        if self.time_limit_s is not None and (isinstance(self.time_limit_s,bool) or not math.isfinite(self.time_limit_s) or self.time_limit_s<=0):
            raise ValueError('proof time limit must be positive and finite')
        if self.max_terms is not None and (type(self.max_terms) is not int or self.max_terms<=0):
            raise ValueError('proof max_terms must be a positive integer')
        if self.mode=='budgeted' and self.time_limit_s is None and self.max_terms is None:
            raise ValueError('budgeted proof policy requires a time or work limit')


class VerificationLimit(TimeoutError):
    pass


class ProofRequirementNotMetError(RuntimeError):
    def __init__(self,result):
        self.result=result
        super().__init__('required independent numerical proof was not established; result retained')


class VerificationWorkspace:
    """Owned LRU of matrix hashes and exact dyadics, bounded by coefficient count."""
    def __init__(self, *, max_entries=8, max_coefficient_terms=65536):
        if type(max_entries) is not int or max_entries<0 or type(max_coefficient_terms) is not int or max_coefficient_terms<0:
            raise ValueError('workspace bounds must be nonnegative integers')
        self.max_entries=max_entries; self.max_coefficient_terms=max_coefficient_terms
        self._entries=OrderedDict(); self._terms=0; self._lock=RLock()
        self.hits=0; self.misses=0

    def prepare(self,matrix):
        from ._certificate_arithmetic import _dyadic
        matrix=matrix.tocsr()
        if not self.max_entries or matrix.nnz>self.max_coefficient_terms or len(matrix.indptr)>self.max_coefficient_terms:
            self.misses+=1
            return None
        digest=hashlib.sha256()
        for values,dtype in ((matrix.shape,np.int64),(matrix.indptr,np.int64),(matrix.indices,np.int64),(matrix.data,np.float64)):
            digest.update(np.asarray(values,dtype=dtype).tobytes())
        key=('binary64-dyadic-csr-v2',digest.hexdigest())
        with self._lock:
            if key in self._entries:
                self.hits+=1; self._entries.move_to_end(key); return self._entries[key]
            self.misses+=1
            value=(tuple(map(int,matrix.indptr)),tuple(map(int,matrix.indices)),tuple(_dyadic(v) for v in matrix.data))
            size=matrix.nnz
            if self.max_entries and size<=self.max_coefficient_terms:
                while self._entries and (len(self._entries)>=self.max_entries or self._terms+size>self.max_coefficient_terms):
                    _,old=self._entries.popitem(last=False); self._terms-=len(old[2])
                self._entries[key]=value; self._terms+=size
            return value

    def info(self):
        with self._lock:
            return {'entries':len(self._entries),'coefficient_terms':self._terms,
                    'index_entries':sum(len(v[0])+len(v[1]) for v in self._entries.values()),
                    'hits':self.hits,'misses':self.misses}

    def clear(self):
        with self._lock: self._entries.clear(); self._terms=0


class _VerificationCall:
    def __init__(self,policy,workspace):
        self.policy=policy; self.workspace=workspace; self.started=perf_counter(); self.terms=0

    def tick(self,terms=0):
        self.terms+=terms
        if self.policy.max_terms is not None and self.terms>self.policy.max_terms:
            raise VerificationLimit('verification work limit exceeded')
        if self.policy.time_limit_s is not None and perf_counter()-self.started>=self.policy.time_limit_s:
            raise VerificationLimit('verification time limit exceeded')


_CURRENT=ContextVar('solverpilot_verification_call',default=None)


@contextmanager
def verification_scope(policy=None,workspace=None,*,existing=None):
    policy=policy or ProofPolicy()
    if not isinstance(policy,ProofPolicy): raise TypeError('proof_policy must be ProofPolicy')
    if workspace is not None and not isinstance(workspace,VerificationWorkspace): raise TypeError('workspace must be VerificationWorkspace')
    call=existing or _VerificationCall(policy,workspace); token=_CURRENT.set(call)
    try:
        call.tick(); yield call; call.tick()
    finally: _CURRENT.reset(token)


def controlled_verification(function):
    @wraps(function)
    def run(*args,proof_policy=None,workspace=None,**kwargs):
        if _CURRENT.get() is not None: return function(*args,**kwargs)
        try:
            with verification_scope(proof_policy,workspace): return function(*args,**kwargs)
        except VerificationLimit as exc:
            from .optimality import OptimalityCheck
            return OptimalityCheck(False,False,False,np.inf,np.inf,np.inf,str(exc))
    return run
