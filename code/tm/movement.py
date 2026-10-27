"""Live-to-live movement matrix P_move: transition counts plus a Dirichlet prior on each row."""

import numpy as np
import scipy.sparse as sp


def counts(grid, pr):
    a = grid.index(pr["lat0"], pr["lon0"])
    b = grid.index(pr["lat1"], pr["lon1"])
    ok = (a >= 0) & (b >= 0)
    C = sp.csr_matrix((np.ones(ok.sum()), (a[ok], b[ok])), shape=(grid.n, grid.n))
    C.sum_duplicates()
    return C, int((~ok).sum())


def _rownorm(M):
    tot = np.asarray(M.sum(axis=1)).ravel()
    return sp.diags(np.divide(1.0, tot, out=np.zeros_like(tot), where=tot > 0)) @ M, tot


def prior_neighbors(grid, C, radius=1):
    return _rownorm(grid.window(radius))[0]


def prior_pooled(grid, C, radius=2, max_offset=None, q=0.999):
    """Each cell's prior is the displacement distribution pooled over cells within `radius`, applied from the cell."""
    C = C.tocoo()
    dy, dx = grid.offsets(C.row, C.col)
    K = int(max_offset if max_offset is not None else np.ceil(np.quantile(np.maximum(abs(dy), abs(dx)), q)))
    m = (np.abs(dy) <= K) & (np.abs(dx) <= K)
    w = 2 * K + 1
    O = sp.csr_matrix((C.data[m], (C.row[m], (dy[m] + K) * w + dx[m] + K)), shape=(grid.n, w * w))
    Op = (grid.window(radius) @ O).tocoo()
    dest = grid.shift(Op.row, Op.col // w - K, Op.col % w - K)
    ok = dest >= 0
    P0 = sp.csr_matrix((Op.data[ok], (Op.row[ok], dest[ok])), shape=(grid.n, grid.n))
    P0.sum_duplicates()
    return _rownorm(P0)[0]


PRIORS = {"none": None, "neighbors": prior_neighbors, "pooled": prior_pooled}


def movement_matrix(grid, C, prior="pooled", alpha=1.0, empty="absorb", **kw):
    """Posterior-mean rows (C_b + alpha P0_b) / (n_b + alpha). Rows with no data and no prior mass are
    zero ("absorb": mass entering them is lost) or the identity ("stay")."""
    n_b = np.asarray(C.sum(axis=1)).ravel()
    M = C.astype(float)
    if PRIORS[prior] is not None and alpha > 0:
        M = M + alpha * PRIORS[prior](grid, C, **kw)
    P, tot = _rownorm(M.tocsr())
    if empty == "stay":
        P = P + sp.diags((tot == 0).astype(float))
    return P.tocsr(), {"n_b": n_b, "empty": tot == 0}
