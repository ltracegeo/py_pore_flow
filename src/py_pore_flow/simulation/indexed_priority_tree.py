"""Red-Black Tree backed indexed priority queue for border elements.

Provides O(log N) insert, remove, update, contains, get_pc, pop_min,
and merge operations using two coordinated Red-Black trees:

  - **Index tree**: keyed by element index for fast membership and value lookup
  - **PC tree**: keyed by (capillary_pressure, element_index) for min-extraction

All tree operations use Numba ``@njit(cache=True)`` for maximum performance.
Data is stored in pre-allocated numpy arrays to avoid Python object overhead
and repeated memory allocation.

Usage::

    tree = IndexedPriorityTree(max_size=10000)
    tree.insert(42, 1500.0)   # elem_index=42, pc=1500.0
    tree.insert(99, 1200.0)
    tree.contains(42)          # True
    tree.get_pc(42)            # 1500.0
    tree.update_pc(42, 1100.0)
    idx, pc = tree.pop_min()   # (42, 1100.0)
"""

import numpy as np
from numba import njit


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

_NIL = np.int64(-1)
_RED = np.int8(0)
_BLACK = np.int8(1)


# ══════════════════════════════════════════════════════════════════════════════
# Generic Red-Black tree structural operations
# ══════════════════════════════════════════════════════════════════════════════
# These operate on abstract (L, R, P, C, root) arrays and work for both the
# index tree and the PC tree.


@njit(cache=True)
def _rot_left(L, R, P, root, x):
    """Left-rotate around node *x*."""
    y = R[x]
    R[x] = L[y]
    if L[y] != _NIL:
        P[L[y]] = x
    P[y] = P[x]
    if P[x] == _NIL:
        root[0] = y
    elif x == L[P[x]]:
        L[P[x]] = y
    else:
        R[P[x]] = y
    L[y] = x
    P[x] = y


@njit(cache=True)
def _rot_right(L, R, P, root, y):
    """Right-rotate around node *y*."""
    x = L[y]
    L[y] = R[x]
    if R[x] != _NIL:
        P[R[x]] = y
    P[x] = P[y]
    if P[y] == _NIL:
        root[0] = x
    elif y == L[P[y]]:
        L[P[y]] = x
    else:
        R[P[y]] = x
    R[x] = y
    P[y] = x


@njit(cache=True)
def _fix_insert(L, R, P, C, root, z):
    """Restore Red-Black properties after inserting *z*."""
    while z != root[0] and C[P[z]] == _RED:
        gp = P[P[z]]
        if P[z] == L[gp]:
            u = R[gp]  # uncle
            if u != _NIL and C[u] == _RED:
                C[P[z]] = _BLACK
                C[u] = _BLACK
                C[gp] = _RED
                z = gp
            else:
                if z == R[P[z]]:
                    z = P[z]
                    _rot_left(L, R, P, root, z)
                C[P[z]] = _BLACK
                C[P[P[z]]] = _RED
                _rot_right(L, R, P, root, P[P[z]])
        else:
            u = L[gp]  # uncle
            if u != _NIL and C[u] == _RED:
                C[P[z]] = _BLACK
                C[u] = _BLACK
                C[gp] = _RED
                z = gp
            else:
                if z == L[P[z]]:
                    z = P[z]
                    _rot_right(L, R, P, root, z)
                C[P[z]] = _BLACK
                C[P[P[z]]] = _RED
                _rot_left(L, R, P, root, P[P[z]])
    C[root[0]] = _BLACK


@njit(cache=True)
def _transplant(L, R, P, root, u, v):
    """Replace subtree rooted at *u* with subtree rooted at *v*."""
    if P[u] == _NIL:
        root[0] = v
    elif u == L[P[u]]:
        L[P[u]] = v
    else:
        R[P[u]] = v
    if v != _NIL:
        P[v] = P[u]


@njit(cache=True)
def _subtree_min(L, x):
    """Find the minimum (leftmost) node in the subtree rooted at *x*."""
    while L[x] != _NIL:
        x = L[x]
    return x


@njit(cache=True)
def _fix_delete(L, R, P, C, root, x, xp):
    """Restore Red-Black properties after deleting a BLACK node.

    *x* is the replacement node (may be NIL).  *xp* is x's parent (tracked
    explicitly because x may be NIL and cannot carry a parent pointer).
    """
    while x != root[0] and (x == _NIL or C[x] == _BLACK):
        if x == L[xp]:
            w = R[xp]
            if C[w] == _RED:
                C[w] = _BLACK
                C[xp] = _RED
                _rot_left(L, R, P, root, xp)
                w = R[xp]
            wl = L[w]
            wr = R[w]
            if (wl == _NIL or C[wl] == _BLACK) and (wr == _NIL or C[wr] == _BLACK):
                C[w] = _RED
                x = xp
                xp = P[x]
            else:
                if wr == _NIL or C[wr] == _BLACK:
                    if wl != _NIL:
                        C[wl] = _BLACK
                    C[w] = _RED
                    _rot_right(L, R, P, root, w)
                    w = R[xp]
                C[w] = C[xp]
                C[xp] = _BLACK
                if R[w] != _NIL:
                    C[R[w]] = _BLACK
                _rot_left(L, R, P, root, xp)
                x = root[0]
        else:
            w = L[xp]
            if C[w] == _RED:
                C[w] = _BLACK
                C[xp] = _RED
                _rot_right(L, R, P, root, xp)
                w = L[xp]
            wl = L[w]
            wr = R[w]
            if (wr == _NIL or C[wr] == _BLACK) and (wl == _NIL or C[wl] == _BLACK):
                C[w] = _RED
                x = xp
                xp = P[x]
            else:
                if wl == _NIL or C[wl] == _BLACK:
                    if wr != _NIL:
                        C[wr] = _BLACK
                    C[w] = _RED
                    _rot_left(L, R, P, root, w)
                    w = L[xp]
                C[w] = C[xp]
                C[xp] = _BLACK
                if L[w] != _NIL:
                    C[L[w]] = _BLACK
                _rot_right(L, R, P, root, xp)
                x = root[0]
    if x != _NIL:
        C[x] = _BLACK


@njit(cache=True)
def _rb_delete(L, R, P, C, root, z):
    """Delete node *z* from the tree (CLRS algorithm, preserves node positions).

    After deletion, the node at array position ``z`` is no longer part of this
    tree's structure.  Its pointers in *this* tree's arrays are stale, but
    pointers in *other* tree arrays (for the dual-tree) are unaffected.
    """
    y = z
    y_oc = C[y]
    x = _NIL
    xp = _NIL

    if L[z] == _NIL:
        x = R[z]
        xp = P[z]
        _transplant(L, R, P, root, z, R[z])
    elif R[z] == _NIL:
        x = L[z]
        xp = P[z]
        _transplant(L, R, P, root, z, L[z])
    else:
        y = _subtree_min(L, R[z])
        y_oc = C[y]
        x = R[y]
        if P[y] == z:
            xp = y
        else:
            xp = P[y]
            _transplant(L, R, P, root, y, R[y])
            R[y] = R[z]
            P[R[y]] = y
        _transplant(L, R, P, root, z, y)
        L[y] = L[z]
        P[L[y]] = y
        C[y] = C[z]

    if y_oc == _BLACK and root[0] != _NIL:
        _fix_delete(L, R, P, C, root, x, xp)


# ══════════════════════════════════════════════════════════════════════════════
# Index tree: BST ordered by element index
# ══════════════════════════════════════════════════════════════════════════════


@njit(cache=True)
def _idx_insert(eidx, L, R, P, C, root, z):
    """Insert node *z* into the index tree (keyed by ``eidx[z]``)."""
    key = eidx[z]
    p = _NIL
    cur = root[0]
    while cur != _NIL:
        p = cur
        if key < eidx[cur]:
            cur = L[cur]
        else:
            cur = R[cur]
    P[z] = p
    if p == _NIL:
        root[0] = z
    elif key < eidx[p]:
        L[p] = z
    else:
        R[p] = z
    L[z] = _NIL
    R[z] = _NIL
    C[z] = _RED
    _fix_insert(L, R, P, C, root, z)


# ══════════════════════════════════════════════════════════════════════════════
# PC tree: BST ordered by (capillary pressure, element index)
# ══════════════════════════════════════════════════════════════════════════════


@njit(cache=True)
def _pc_insert(pcd, eidx, L, R, P, C, root, z):
    """Insert node *z* into the PC tree (composite key ``(pcd[z], eidx[z])``)."""
    pc_z = pcd[z]
    idx_z = eidx[z]
    p = _NIL
    cur = root[0]
    while cur != _NIL:
        p = cur
        if pc_z < pcd[cur] or (pc_z == pcd[cur] and idx_z < eidx[cur]):
            cur = L[cur]
        else:
            cur = R[cur]
    P[z] = p
    if p == _NIL:
        root[0] = z
    elif pc_z < pcd[p] or (pc_z == pcd[p] and idx_z < eidx[p]):
        L[p] = z
    else:
        R[p] = z
    L[z] = _NIL
    R[z] = _NIL
    C[z] = _RED
    _fix_insert(L, R, P, C, root, z)


# ══════════════════════════════════════════════════════════════════════════════
# Node pool management
# ══════════════════════════════════════════════════════════════════════════════


@njit(cache=True)
def _alloc(free, state):
    """Allocate a node from the free list.  Returns the node position."""
    top = state[1]
    pos = free[top]
    state[1] = top - 1
    state[0] += 1
    return pos


@njit(cache=True)
def _release(free, state, pos):
    """Return node *pos* to the free list."""
    state[1] += 1
    free[state[1]] = pos
    state[0] -= 1


# ══════════════════════════════════════════════════════════════════════════════
# Dual-tree composite operations
# ══════════════════════════════════════════════════════════════════════════════


@njit(cache=True)
def _dt_insert(eidx, pcd, pos_of, free, state,
               il, ir, ip, ic, irt,
               pl, pr, pp, pcc, prt,
               elem_index, pc_val):
    """Insert ``(elem_index, pc_val)`` into both trees.  Returns True if inserted."""
    if pos_of[elem_index] != _NIL:
        return False
    pos = _alloc(free, state)
    eidx[pos] = elem_index
    pcd[pos] = pc_val
    pos_of[elem_index] = pos
    _idx_insert(eidx, il, ir, ip, ic, irt, pos)
    _pc_insert(pcd, eidx, pl, pr, pp, pcc, prt, pos)
    return True


@njit(cache=True)
def _dt_remove(eidx, pcd, pos_of, free, state,
               il, ir, ip, ic, irt,
               pl, pr, pp, pcc, prt,
               elem_index):
    """Remove *elem_index* from both trees.  Returns True if removed."""
    pos = pos_of[elem_index]
    if pos == _NIL:
        return False
    _rb_delete(il, ir, ip, ic, irt, pos)
    _rb_delete(pl, pr, pp, pcc, prt, pos)
    pos_of[elem_index] = _NIL
    _release(free, state, pos)
    return True


@njit(cache=True)
def _dt_update_pc(eidx, pcd, pos_of,
                  pl, pr, pp, pcc, prt,
                  elem_index, new_pc):
    """Update the capillary pressure of *elem_index* in the PC tree."""
    pos = pos_of[elem_index]
    if pos == _NIL:
        return
    if pcd[pos] == new_pc:
        return
    # Remove from PC tree, update value, re-insert.
    _rb_delete(pl, pr, pp, pcc, prt, pos)
    pcd[pos] = new_pc
    _pc_insert(pcd, eidx, pl, pr, pp, pcc, prt, pos)


@njit(cache=True)
def _dt_pop_min(eidx, pcd, pos_of, free, state,
                il, ir, ip, ic, irt,
                pl, pr, pp, pcc, prt):
    """Pop the element with the lowest capillary pressure.

    Returns ``(elem_index, pc)``.  Returns ``(-1, 0.0)`` if the tree is empty.
    """
    if prt[0] == _NIL:
        return _NIL, np.float64(0.0)
    pos = _subtree_min(pl, prt[0])
    elem_index = eidx[pos]
    pc_val = pcd[pos]
    _rb_delete(il, ir, ip, ic, irt, pos)
    _rb_delete(pl, pr, pp, pcc, prt, pos)
    pos_of[elem_index] = _NIL
    _release(free, state, pos)
    return elem_index, pc_val


@njit(cache=True)
def _dt_peek_min(eidx, pcd, pl, prt):
    """Return ``(elem_index, pc)`` of the minimum-PC element without removing.

    Returns ``(-1, 0.0)`` if the tree is empty.
    """
    if prt[0] == _NIL:
        return _NIL, np.float64(0.0)
    pos = _subtree_min(pl, prt[0])
    return eidx[pos], pcd[pos]


@njit(cache=True)
def _dt_bulk_insert(eidx, pcd, pos_of, free, state,
                    il, ir, ip, ic, irt,
                    pl, pr, pp, pcc, prt,
                    indices, pcs, n):
    """Insert *n* elements from *indices* and *pcs* arrays.  Skips duplicates."""
    inserted = np.int64(0)
    for i in range(n):
        if _dt_insert(eidx, pcd, pos_of, free, state,
                      il, ir, ip, ic, irt,
                      pl, pr, pp, pcc, prt,
                      indices[i], pcs[i]):
            inserted += 1
    return inserted

@njit(cache=True)
def _dt_bulk_update_pc(eidx, pcd, pos_of,
                       pl, pr, pp, pcc, prt,
                       indices, pcs, n):
    """Update *n* elements from *indices* and *pcs* arrays in the PC tree."""
    for i in range(n):
        _dt_update_pc(eidx, pcd, pos_of, pl, pr, pp, pcc, prt, indices[i], pcs[i])

@njit(cache=True)
def _dt_bulk_remove(eidx, pcd, pos_of, free, state,
                    il, ir, ip, ic, irt,
                    pl, pr, pp, pcc, prt,
                    indices, n):
    """Remove *n* elements from *indices* array."""
    removed = np.int64(0)
    for i in range(n):
        if _dt_remove(eidx, pcd, pos_of, free, state,
                      il, ir, ip, ic, irt,
                      pl, pr, pp, pcc, prt,
                      indices[i]):
            removed += 1
    return removed


# ══════════════════════════════════════════════════════════════════════════════
# Utility — traversal and validation
# ══════════════════════════════════════════════════════════════════════════════


@njit(cache=True)
def _collect_inorder(eidx, pcd, L, R, root, out_idx, out_pc):
    """In-order traversal of tree rooted at ``root[0]``.

    Writes element indices and PCs into *out_idx* / *out_pc* (pre-allocated).
    Returns the number of elements written.
    """
    if root[0] == _NIL:
        return np.int64(0)
    stack = np.empty(128, np.int64)
    top = np.int64(-1)
    cur = root[0]
    n = np.int64(0)
    while cur != _NIL or top >= 0:
        while cur != _NIL:
            top += 1
            stack[top] = cur
            cur = L[cur]
        cur = stack[top]
        top -= 1
        out_idx[n] = eidx[cur]
        out_pc[n] = pcd[cur]
        n += 1
        cur = R[cur]
    return n


@njit(cache=True)
def _validate_rb(L, R, P, C, root):
    """Check Red-Black tree invariants.  Returns True if all hold."""
    if root[0] == _NIL:
        return True
    # Root must be black.
    if C[root[0]] != _BLACK:
        return False
    # DFS checking red-property and equal black-heights.
    stack = np.empty(256, np.int64)
    bh_stack = np.empty(256, np.int64)
    top = np.int64(0)
    stack[0] = root[0]
    bh_stack[0] = np.int64(0)
    ref_bh = np.int64(-1)

    while top >= 0:
        node = stack[top]
        bh = bh_stack[top]
        top -= 1

        if C[node] == _BLACK:
            bh += 1

        # Red node must not have a red child.
        if C[node] == _RED:
            if L[node] != _NIL and C[L[node]] == _RED:
                return False
            if R[node] != _NIL and C[R[node]] == _RED:
                return False

        # Check parent pointers.
        if L[node] != _NIL and P[L[node]] != node:
            return False
        if R[node] != _NIL and P[R[node]] != node:
            return False

        # Check black-height at NIL children (NIL counts as black).
        if L[node] == _NIL:
            leaf_bh = bh + 1
            if ref_bh == -1:
                ref_bh = leaf_bh
            elif leaf_bh != ref_bh:
                return False
        if R[node] == _NIL:
            leaf_bh = bh + 1
            if ref_bh == -1:
                ref_bh = leaf_bh
            elif leaf_bh != ref_bh:
                return False

        # Push children (right first so left is processed first).
        if R[node] != _NIL:
            top += 1
            stack[top] = R[node]
            bh_stack[top] = bh
        if L[node] != _NIL:
            top += 1
            stack[top] = L[node]
            bh_stack[top] = bh

    return True


# ══════════════════════════════════════════════════════════════════════════════
# Python wrapper class
# ══════════════════════════════════════════════════════════════════════════════


class IndexedPriorityTree:
    """Dual Red-Black tree indexed priority queue.

    Stores ``(element_index, capillary_pressure)`` pairs with O(log N) insert,
    remove, update, and min-extraction.  Element indices must be integers in
    ``[0, max_size)``.

    Internally two RB-trees share the same node pool:

    * **Index tree** — keyed by element index, for fast membership/lookup.
    * **PC tree** — keyed by ``(pc, element_index)``, for O(log N) min-extraction.

    A direct-address array ``pos_of`` additionally provides O(1) ``contains``
    and ``get_pc``.

    Parameters
    ----------
    max_size : int
        Capacity *and* upper bound on element indices (exclusive).
    """

    def __init__(self, max_size: int) -> None:
        n = max(int(max_size), 1)
        self._max_size = n

        # Shared node data.
        self._eidx = np.empty(n, np.int64)
        self._pcd = np.empty(n, np.float64)
        self._pos_of = np.full(n, -1, np.int64)

        # Free-list (stack, initially contains 0 … n-1 in LIFO order).
        self._free = np.arange(n - 1, -1, -1, dtype=np.int64)
        # state[0] = count, state[1] = free_top
        self._state = np.array([np.int64(0), np.int64(n - 1)])

        # Index tree arrays.
        self._il = np.full(n, -1, np.int64)
        self._ir = np.full(n, -1, np.int64)
        self._ip = np.full(n, -1, np.int64)
        self._ic = np.zeros(n, np.int8)
        self._irt = np.array([np.int64(-1)])

        # PC tree arrays.
        self._pl = np.full(n, -1, np.int64)
        self._pr = np.full(n, -1, np.int64)
        self._pp = np.full(n, -1, np.int64)
        self._pcc = np.zeros(n, np.int8)
        self._prt = np.array([np.int64(-1)])

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _all_args(self):
        """All 15 arrays, in the order expected by ``_dt_*`` functions."""
        return (
            self._eidx, self._pcd, self._pos_of, self._free, self._state,
            self._il, self._ir, self._ip, self._ic, self._irt,
            self._pl, self._pr, self._pp, self._pcc, self._prt,
        )

    # ------------------------------------------------------------------ #
    # Core operations
    # ------------------------------------------------------------------ #

    def insert(self, elem_index: int, pc: float) -> bool:
        """Insert ``(elem_index, pc)``.  Returns True if inserted, False if duplicate."""
        return bool(
            _dt_insert(*self._all_args(), np.int64(elem_index), np.float64(pc))
        )

    def remove(self, elem_index: int) -> bool:
        """Remove *elem_index*.  Returns True if removed, False if not found."""
        return bool(
            _dt_remove(*self._all_args(), np.int64(elem_index))
        )

    def contains(self, elem_index: int) -> bool:
        """O(1) membership test."""
        return self._pos_of[elem_index] != -1

    def get_pc(self, elem_index: int) -> float:
        """O(1) capillary-pressure lookup.  Raises ``KeyError`` if absent."""
        pos = self._pos_of[elem_index]
        if pos == -1:
            raise KeyError(f"Element {elem_index} not in tree")
        return float(self._pcd[pos])

    def update_pc(self, elem_index: int, new_pc: float) -> None:
        """Update capillary pressure.  O(log N) (PC-tree remove + re-insert)."""
        _dt_update_pc(
            self._eidx, self._pcd, self._pos_of,
            self._pl, self._pr, self._pp, self._pcc, self._prt,
            np.int64(elem_index), np.float64(new_pc),
        )

    def pop_min(self) -> tuple:
        """Remove and return ``(elem_index, pc)`` with the lowest PC.

        Raises ``IndexError`` on an empty tree.
        """
        idx, pc = _dt_pop_min(*self._all_args())
        if idx == -1:
            raise IndexError("pop_min from empty tree")
        return int(idx), float(pc)

    def peek_min(self) -> tuple:
        """Return ``(elem_index, pc)`` with the lowest PC without removing.

        Raises ``IndexError`` on an empty tree.
        """
        idx, pc = _dt_peek_min(self._eidx, self._pcd, self._pl, self._prt)
        if idx == -1:
            raise IndexError("peek_min on empty tree")
        return int(idx), float(pc)

    # ------------------------------------------------------------------ #
    # Bulk operations
    # ------------------------------------------------------------------ #

    def merge(self, other: "IndexedPriorityTree") -> int:
        """Merge all elements from *other* into this tree (skips duplicates).

        Returns the number of elements actually inserted.
        """
        if other.is_empty:
            return 0
        src_idx, src_pc = other.get_all()
        return int(
            _dt_bulk_insert(
                *self._all_args(), src_idx, src_pc, np.int64(len(src_idx))
            )
        )

    def set_all(self, indices: np.ndarray, pcs: np.ndarray) -> None:
        """Clear and bulk-insert ``(indices[i], pcs[i])`` pairs."""
        self.clear()
        n = len(indices)
        if n > 0:
            _dt_bulk_insert(
                *self._all_args(),
                indices.astype(np.int64), pcs.astype(np.float64), np.int64(n),
            )

    def insert_all(self, indices: np.ndarray, pcs: np.ndarray) -> None:
        """Bulk-insert ``(indices[i], pcs[i])`` pairs without clearing."""
        n = len(indices)
        if n > 0:
            _dt_bulk_insert(
                *self._all_args(),
                indices.astype(np.int64), pcs.astype(np.float64), np.int64(n),
            )

    def update_all(self, indices: np.ndarray, pcs: np.ndarray) -> None:
        """Update multiple capillary pressures in bulk."""
        n = len(indices)
        if n > 0:
            _dt_bulk_update_pc(
                self._eidx, self._pcd, self._pos_of,
                self._pl, self._pr, self._pp, self._pcc, self._prt,
                indices.astype(np.int64), pcs.astype(np.float64), np.int64(n),
            )

    def remove_all(self, indices: np.ndarray) -> None:
        """Remove multiple elements in bulk."""
        n = len(indices)
        if n > 0:
            _dt_bulk_remove(
                *self._all_args(),
                indices.astype(np.int64), np.int64(n),
            )

    # ------------------------------------------------------------------ #
    # Query / iteration
    # ------------------------------------------------------------------ #

    def get_all(self) -> tuple:
        """Return ``(indices, pcs)`` sorted by element index."""
        n = self._max_size
        if n == 0:
            return np.empty(0, np.int64), np.empty(0, np.float64)
        out_idx = np.empty(n, np.int64)
        out_pc = np.empty(n, np.float64)
        actual_n = _collect_inorder(
            self._eidx, self._pcd,
            self._il, self._ir, self._irt,
            out_idx, out_pc,
        )
        return out_idx[:actual_n], out_pc[:actual_n]

    def get_all_ids(self) -> np.ndarray:
        """Return element indices as a sorted int64 array."""
        idx, _ = self.get_all()
        return idx

    def get_all_sorted_by_pc(self) -> tuple:
        """Return ``(indices, pcs)`` sorted by capillary pressure (ascending)."""
        n = self.count
        if n == 0:
            return np.empty(0, np.int64), np.empty(0, np.float64)
        out_idx = np.empty(n, np.int64)
        out_pc = np.empty(n, np.float64)
        _collect_inorder(
            self._eidx, self._pcd,
            self._pl, self._pr, self._prt,
            out_idx, out_pc,
        )
        return out_idx, out_pc

    # ------------------------------------------------------------------ #
    # Housekeeping
    # ------------------------------------------------------------------ #

    def clear(self) -> None:
        """Remove all elements and reset internal state."""
        n = self._max_size
        self._pos_of[:] = -1
        self._state[0] = 0
        self._state[1] = n - 1
        self._free[:] = np.arange(n - 1, -1, -1, dtype=np.int64)
        self._irt[0] = -1
        self._prt[0] = -1

    @property
    def count(self) -> int:
        """Number of elements currently stored."""
        return int(self._state[0])

    @property
    def is_empty(self) -> bool:
        return self._state[0] == 0

    @property
    def max_size(self) -> int:
        return self._max_size

    def __len__(self) -> int:
        return self.count

    def __contains__(self, elem_index: int) -> bool:
        return self.contains(elem_index)

    # ------------------------------------------------------------------ #
    # Validation (for testing)
    # ------------------------------------------------------------------ #

    def validate(self) -> bool:
        """Check that both Red-Black trees satisfy all invariants."""
        return bool(
            _validate_rb(self._il, self._ir, self._ip, self._ic, self._irt)
            and _validate_rb(self._pl, self._pr, self._pp, self._pcc, self._prt)
        )
