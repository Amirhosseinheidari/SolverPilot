"""Conservative PSD evidence about the supplied binary64 matrix.

Numerical eigensolver tolerances are not proofs of nonnegative curvature.
Sparse diagonal dominance and small rational LDL checks provide exact evidence;
an exhausted size/arithmetic budget returns unknown rather than accepting.
"""

from dataclasses import dataclass
from fractions import Fraction
import numpy as np
from scipy import sparse
from .hashing import hash_sparse_matrix


def _charge_verification(terms):
    from solverpilot.validate.workspace import _CURRENT
    call=_CURRENT.get()
    if call is not None: call.tick(terms)


@dataclass(frozen=True, slots=True)
class PSDEvidence:
    status: str
    method: str
    matrix_hash: str
    reason: str

    @property
    def certified(self) -> bool:
        return self.status == "positive_semidefinite"


def certify_psd(
    matrix, *, max_dense_dimension: int = 32, max_fraction_bits: int = 8192,
    max_component_work: int = 2_000_000,
) -> PSDEvidence:
    """Prove PSD, prove a negative direction, or return unknown.

    This checks the exact floating-point input, not an intended unrounded model.
    It handles singular PSD matrices. Size/bit limits bound the dense fallback.
    """
    if type(max_dense_dimension) is not int or max_dense_dimension < 0:
        raise ValueError("max_dense_dimension must be a nonnegative integer")
    if type(max_fraction_bits) is not int or max_fraction_bits < 64:
        raise ValueError("max_fraction_bits must be an integer >= 64")
    if type(max_component_work) is not int or max_component_work < 0:
        raise ValueError("max_component_work must be a nonnegative integer")
    a = sparse.csr_matrix(matrix, dtype=float, copy=True)
    a.sum_duplicates()
    a.eliminate_zeros()
    a.sort_indices()
    if a.shape[0] != a.shape[1] or not np.isfinite(a.data).all():
        raise ValueError("PSD evidence requires a finite square matrix")
    digest = hash_sparse_matrix(a)

    def result(status, method, reason):
        return PSDEvidence(status, method, digest, reason)

    if (a != a.T).nnz:
        return result("unknown", "symmetry", "matrix is not exactly symmetric")
    diagonal = a.diagonal()
    _charge_verification(a.nnz+a.shape[0])
    if np.any(diagonal < 0):
        return result("not_psd", "negative_diagonal", "coordinate direction has negative curvature")
    # A zero diagonal with a nonzero off-diagonal gives a negative 2x2 minor.
    for i in np.flatnonzero(diagonal == 0):
        if a.indptr[i + 1] > a.indptr[i]:
            return result("not_psd", "zero_diagonal_minor", "nonzero row at a zero diagonal")
    dominant = True
    for i in range(a.shape[0]):
        radius = sum(
            (
                abs(Fraction(float(a.data[k])))
                for k in range(a.indptr[i], a.indptr[i + 1])
                if a.indices[k] != i
            ),
            Fraction(0),
        )
        if Fraction(float(diagonal[i])) < radius:
            dominant = False
            break
    if dominant:
        return result(
            "positive_semidefinite",
            "exact_gershgorin",
            "all exact Gershgorin lower bounds are nonnegative",
        )
    if a.shape[0] > max_dense_dimension:
        from scipy.sparse.csgraph import connected_components
        count, labels = connected_components(a, directed=False)
        if count > 1:
            groups = np.split(np.argsort(labels,kind='stable'),np.cumsum(np.bincount(labels))[:-1])
            work = sum(len(g)**3 for g in groups)
            if work > max_component_work:
                return result("unknown", "component_work_limit", "sparse component LDL work budget exceeded")
            unknown = None
            for indices in groups:
                evidence = certify_psd(a[indices][:, indices], max_dense_dimension=max_dense_dimension,
                                       max_fraction_bits=max_fraction_bits, max_component_work=max_component_work)
                if evidence.status == "not_psd":
                    return result("not_psd", "exact_component_ldl", "a sparse connected component has negative curvature")
                if not evidence.certified: unknown = evidence
            if unknown is not None:
                return result("unknown", "exact_components", unknown.reason)
            return result("positive_semidefinite", "exact_component_ldl", f"{count} independently certified sparse components; work bound {work}")
        return result("unknown", "size_limit", "matrix exceeds the rational LDL dimension budget")
    values = [[Fraction(float(v)) for v in row] for row in a.toarray()]
    n = len(values)
    _charge_verification(n**3)
    for k in range(n):
        pivot = values[k][k]
        if pivot < 0 or (pivot == 0 and any(values[i][k] for i in range(k + 1, n))):
            return result("not_psd", "exact_ldl", "negative Schur-complement direction")
        if pivot == 0:
            continue
        for i in range(k + 1, n):
            for j in range(i, n):
                v = values[i][j] - values[i][k] * values[j][k] / pivot
                if max(v.numerator.bit_length(), v.denominator.bit_length()) > max_fraction_bits:
                    return result(
                        "unknown", "arithmetic_limit", "rational LDL exceeded its bit budget"
                    )
                values[i][j] = values[j][i] = v
    return result(
        "positive_semidefinite",
        "exact_ldl",
        "exact nonnegative LDL pivots including singular blocks",
    )


def certify_psd_factor(matrix, factor, *, weights=None, diagonal=None,
                       max_product_terms=1_000_000, max_nonzeros=1_000_000,
                       max_fraction_bits=8192):
    """Verify exact represented-data equality P=R.T diag(d) R+diag(s), d,s>=0.

    A floating allclose or rounded Gram product is insufficient. Work/fill/bit
    exhaustion returns unknown; a mismatched witness says nothing about PSD.
    """
    for name,value in [('max_product_terms',max_product_terms),('max_nonzeros',max_nonzeros),('max_fraction_bits',max_fraction_bits)]:
        if type(value) is not int or value<0: raise ValueError(f'{name} must be nonnegative')
    a=sparse.csr_matrix(matrix,dtype=float,copy=True); a.sum_duplicates(); a.eliminate_zeros(); a.sort_indices()
    r=sparse.csr_matrix(factor,dtype=float,copy=True); r.sum_duplicates(); r.eliminate_zeros(); r.sort_indices()
    if a.shape[0]!=a.shape[1] or r.shape[1]!=a.shape[0] or not np.isfinite(a.data).all() or not np.isfinite(r.data).all():
        raise ValueError('finite factor and square matrix dimensions must agree')
    d=np.ones(r.shape[0]) if weights is None else np.array(weights,dtype=float,copy=True)
    s=np.zeros(a.shape[0]) if diagonal is None else np.array(diagonal,dtype=float,copy=True)
    if d.shape!=(r.shape[0],) or s.shape!=(a.shape[0],) or not np.isfinite(d).all() or not np.isfinite(s).all() or np.any(d<0) or np.any(s<0):
        raise ValueError('factor weights and diagonal must be finite nonnegative vectors')
    digest=hash_sparse_matrix(a)
    def result(method,reason,certified=False):
        return PSDEvidence('positive_semidefinite' if certified else 'unknown',method,digest,reason)
    for original in (matrix,factor):
        if sparse.issparse(original):
            limit=max_nonzeros if original is matrix else max_product_terms
            if original.nnz>limit: return result('factor_work_limit','sparse input exceeds preparation budget')
            coo=original.tocoo()
            if len(set(zip(coo.row.tolist(),coo.col.tolist())))!=coo.nnz:
                return result('factor_representation','duplicate sparse entries require explicit canonicalization before supplying an exact witness')
    if (a!=a.T).nnz: return result('symmetry','matrix is not exactly symmetric')
    terms=sum(int(count)*(int(count)+1)//2 for count in np.diff(r.indptr))
    if terms>max_product_terms or a.nnz>max_nonzeros:
        return result('factor_work_limit','factor product exceeds work/nonzero budget')
    _charge_verification(terms+a.nnz+r.nnz+len(s))
    product={}
    for row in range(r.shape[0]):
        if d[row]==0: continue
        start,stop=r.indptr[row:row+2]; weight=Fraction(float(d[row]))
        values=[Fraction(float(v)) for v in r.data[start:stop]]; indices=r.indices[start:stop]
        for local,i in enumerate(indices):
            for other in range(local,len(indices)):
                key=(int(i),int(indices[other])); value=product.get(key,Fraction())+weight*values[local]*values[other]
                if max(value.numerator.bit_length(),value.denominator.bit_length())>max_fraction_bits:
                    return result('factor_bit_limit','factor arithmetic exceeded bit budget')
                product[key]=value
                if len(product)>max_nonzeros: return result('factor_fill_limit','factor fill exceeded nonzero budget')
    for i,value in enumerate(s):
        if value:
            product[i,i]=product.get((i,i),Fraction())+Fraction(float(value))
            number=product[i,i]
            if max(number.numerator.bit_length(),number.denominator.bit_length())>max_fraction_bits:
                return result('factor_bit_limit','factor diagonal exceeded bit budget')
            if len(product)>max_nonzeros: return result('factor_fill_limit','factor diagonal fill exceeded nonzero budget')
    expected={}
    for i in range(a.shape[0]):
        for k in range(a.indptr[i],a.indptr[i+1]):
            if a.indices[k]>=i: expected[i,int(a.indices[k])]=Fraction(float(a.data[k]))
    if any(value!=expected.get(key,Fraction()) for key,value in product.items()) or any(value!=product.get(key,Fraction()) for key,value in expected.items()):
        return result('factor_mismatch','exact represented matrix differs from the proposed Gram witness')
    return result('exact_gram','exact sparse nonnegative weighted Gram plus diagonal identity',True)
