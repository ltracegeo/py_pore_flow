"""Weighted Quick Union with Path Compression (WQUPC).

A union-find data structure used to track connected components in the pore
network. Elements are indexed in a single flat array where throats occupy
indices [0, nT) and pores occupy [nT, nT+nP). The `find` operation uses
full path compression and `union` uses weighted (by subtree size) merging,
giving near-amortized O(1) per operation.
"""

import numba as nb
import numpy as np
from numpy.typing import NDArray


class Wqupc:
    """Weighted Quick Union with Path Compression for pore/throat networks.

    Manages a union-find structure over nT throats + nP pores.  Throats are
    stored at indices [0, nT) and pores at [nT, nT+nP) in the internal arrays.
    """

    def __init__(self, nP: int, nT: int) -> None:
        self.nP: int = nP
        self.nT: int = nT
        self.parent_array: NDArray[np.uint32] = np.arange(nP + nT, dtype=np.uint32)
        self.size_array: NDArray[np.float64] = np.ones(nP + nT)

    def union(self, p: int, q: int) -> None:
        """Merge the component containing throat *p* with the one containing pore *q*."""
        wqupc_union(p, q + self.nT, self.parent_array, self.size_array)

    def find_throat(self, p: int) -> int:
        """Return the root representative for throat *p*."""
        return wqupc_find(p, self.parent_array)

    def find_pore(self, p: int) -> int:
        """Return the root representative for pore *p*."""
        return wqupc_find(p + self.nT, self.parent_array)

    def reset(self) -> None:
        """Reset every element to be its own root (all components size 1)."""
        wqupc_reset(self.parent_array, self.size_array, self.nP, self.nT)


@nb.njit(cache=True)
def wqupc_find(p: int, parent_array: NDArray[np.uint32]) -> int:
    """Find root of *p* with full path compression."""
    root = p
    while root != parent_array[root]:
        root = parent_array[root]

    # Path compression: point every node on the path directly to root.
    while p != root:
        next_p = parent_array[p]
        parent_array[p] = root
        p = next_p

    return root


@nb.njit(cache=True)
def wqupc_union(
    p: int,
    q: int,
    parent_array: NDArray[np.uint32],
    size_array: NDArray[np.float64],
) -> None:
    """Merge components containing *p* and *q* using weighted union."""
    rootP = wqupc_find(p, parent_array)
    rootQ = wqupc_find(q, parent_array)

    if rootP == rootQ:
        return

    # Attach smaller tree under larger tree's root.
    if size_array[rootP] < size_array[rootQ]:
        parent_array[rootP] = rootQ
        size_array[rootQ] += size_array[rootP]
    else:
        parent_array[rootQ] = rootP
        size_array[rootP] += size_array[rootQ]


@nb.njit(cache=True)
def wqupc_reset(
    parent_array: NDArray[np.uint32],
    size_array: NDArray[np.float64],
    nP: int,
    nT: int,
) -> None:
    """Reset all elements to be their own root with size 1."""
    for i in range(nP + nT):
        parent_array[i] = i
        size_array[i] = 1
