"""Block affine lowering. Coefficients remain CSR until scalar row emission."""

from dataclasses import dataclass
from collections import OrderedDict
import numpy as np
from scipy import sparse


@dataclass
class AffineBlock:
    shape: tuple
    constant: np.ndarray
    coefficients: sparse.csr_matrix


class AffineEvaluator:
    def __init__(self, model, offsets):
        self.model, self.offsets = model, offsets
        self.n = max((stop for _, stop in offsets.values()), default=0)
        self.cache = {}
        self.shared = model._compiler_cache.setdefault("affine-blocks", OrderedDict())
        self.layout = tuple(offsets.items())

    def _constant(self, node):
        if node.kind == "constant":
            return node.payload
        if node.kind == "parameter":
            return self.model._parameters[node.payload].value
        value = self.evaluate(node)
        return value.constant.reshape(value.shape)

    def evaluate(self, node):
        if id(node) in self.cache:
            return self.cache[id(node)]
        key = (
            node,
            self.layout,
            tuple(
                (p, self.model._parameter_versions[p]) for p in sorted(node.parameter_dependencies)
            ),
        )
        if key in self.shared:
            self.shared.move_to_end(key)
            self.cache[id(node)] = self.shared[key]
            return self.cache[id(node)]
        shape = node.shape
        size = int(np.prod(shape)) if shape else 1
        if node.kind in ("constant", "parameter"):
            value = self._constant(node)
            value = value.toarray() if sparse.issparse(value) else value
            result = AffineBlock(
                shape, np.asarray(value).reshape(-1), sparse.csr_matrix((size, self.n))
            )
        elif node.kind == "variable":
            start, stop = self.offsets[node.payload]
            result = AffineBlock(
                shape,
                np.zeros(size),
                sparse.csr_matrix(
                    (np.ones(size), (np.arange(size), np.arange(start, stop))), shape=(size, self.n)
                ),
            )
        elif node.kind == "matmul" and any(a.degree == 0 for a in node.args):
            left, right = node.args
            if left.degree == 0:
                matrix = sparse.csr_matrix(self._constant(left))
                value = self.evaluate(right)
                columns = right.shape[1] if len(right.shape) == 2 else 1
                transform = sparse.kron(matrix, sparse.eye(columns), format="csr")
            else:
                matrix = self._constant(right)
                if len(right.shape) == 1:
                    matrix = np.asarray(matrix).reshape(-1, 1)
                matrix = sparse.csr_matrix(matrix)
                value = self.evaluate(left)
                rows = left.shape[0] if len(left.shape) == 2 else 1
                transform = sparse.kron(sparse.eye(rows), matrix.T, format="csr")
            result = AffineBlock(shape, transform @ value.constant, transform @ value.coefficients)
        elif node.kind in ("add", "mul"):
            left, right = [self.evaluate(a) for a in node.args]

            def broadcast(value):
                indices = np.broadcast_to(
                    np.arange(value.constant.size).reshape(value.shape), shape
                ).reshape(-1)
                return value.constant[indices], value.coefficients[indices]

            a, A = broadcast(left)
            b, B = broadcast(right)
            result = (
                AffineBlock(shape, a + b, A + B)
                if node.kind == "add"
                else AffineBlock(shape, a * b, A.multiply(b[:, None]) + B.multiply(a[:, None]))
            )
        elif node.kind == "neg":
            value = self.evaluate(node.args[0])
            result = AffineBlock(shape, -value.constant, -value.coefficients)
        elif node.kind in ("index", "transpose", "sum"):
            value = self.evaluate(node.args[0])
            indices = np.arange(value.constant.size).reshape(value.shape)
            if node.kind == "sum":
                if node.payload is None:
                    destinations = np.zeros(value.constant.size, dtype=int)
                else:
                    destinations = np.broadcast_to(
                        np.expand_dims(np.arange(size).reshape(shape), node.payload), value.shape
                    ).reshape(-1)
                transform = sparse.csr_matrix(
                    (np.ones(value.constant.size), (destinations, np.arange(value.constant.size))),
                    shape=(size, value.constant.size),
                )
                result = AffineBlock(
                    shape, transform @ value.constant, transform @ value.coefficients
                )
            else:
                selected = (indices[node.payload] if node.kind == "index" else indices.T).reshape(
                    -1
                )
                result = AffineBlock(shape, value.constant[selected], value.coefficients[selected])
        elif node.kind == "concat":
            values = [self.evaluate(a) for a in node.args]
            result = AffineBlock(
                shape,
                np.concatenate([v.constant for v in values]),
                sparse.vstack([v.coefficients for v in values], format="csr"),
            )
        else:
            raise NotImplementedError(f"no affine lowering for {node.kind}")
        result.coefficients = result.coefficients.tocsr()
        result.coefficients.eliminate_zeros()
        self.cache[id(node)] = result
        self.shared[key] = result
        if len(self.shared) > 4096:
            self.shared.popitem(last=False)
        return result
