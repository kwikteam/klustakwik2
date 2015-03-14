'''
Linear algebra routines
'''

from numpy import *
from numpy.linalg import cholesky, LinAlgError
from scipy.linalg import cho_factor, cho_solve

__all__ = ['BlockPlusDiagonalMatrix']

class BlockPlusDiagonalMatrix(object):
    def __init__(self, masked, unmasked, block=None, diagonal=None, lower=None):
        self.masked = masked
        self.unmasked = unmasked
        self.num_masked = len(masked)
        self.num_unmasked = len(unmasked)
        if block is None:
            block = zeros((self.num_unmasked, self.num_unmasked))
        if diagonal is None:
            diagonal = zeros(self.num_masked)
        self.block = block
        self.diagonal = diagonal
        self.lower = lower
    
    def new_with_same_masks(self, block=None, diagonal=None, lower=None):
        return BlockPlusDiagonalMatrix(self.masked, self.unmasked, block=block, diagonal=diagonal, lower=lower)

    def cholesky(self):
        M_chol = self.new_with_same_masks()
        lower = None
        if self.block.size:
            block, lower = cho_factor(self.block)
        if (self.diagonal<=0).any():
            raise LinAlgError
        diagonal = sqrt(self.diagonal)
        return self.new_with_same_masks(block, diagonal, lower)
    
    def trisolve(self, x):
        if self.lower is None:
            raise ValueError("Matrix is not in Cholesky format")
        out = zeros(len(x))
        out[self.unmasked] = cho_solve((self.block, self.lower), x[self.unmasked])
        out[self.masked] = -x[self.masked]/self.diagonal
        return out
