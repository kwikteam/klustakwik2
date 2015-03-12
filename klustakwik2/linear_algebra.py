'''
Linear algebra routines
'''

from numpy import *
from numpy.linalg import cholesky, LinAlgError
from scipy.linalg import cho_factor, cho_solve

from .data import BlockPlusDiagonalMatrix

__all__ = ['bpd_cholesky', 'bpd_trisolve']

def bpd_cholesky(M):
    M_chol = M.new_with_same_masks()
    lower = None
    if M.block.size:
        M_chol.block, lower = cho_factor(M.block)
    if (M.diagonal<=0).any():
        raise LinAlgError
    M_chol.diagonal = sqrt(M.diagonal)
    return M_chol, lower


def bpd_trisolve(M, lower, x):
    out = zeros(len(x))
    out[M.unmasked] = cho_solve((M.block, lower), x[M.unmasked])
    out[M.masked] = -x[M.masked]/M.diagonal
    return out
