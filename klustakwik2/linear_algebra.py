'''
Linear algebra routines
'''

from numpy import *
from numpy.linalg import cholesky, LinAlgError
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular

__all__ = ['BlockPlusDiagonalMatrix']

class BlockPlusDiagonalMatrix(object):
    def __init__(self, masked, unmasked, block=None, diagonal=None):
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
    
    def new_with_same_masks(self, block=None, diagonal=None):
        if block is not None:
            block = block.copy()
        if diagonal is not None:
            diagonal = diagonal.copy()
        return BlockPlusDiagonalMatrix(self.masked, self.unmasked, block=block, diagonal=diagonal)

    def cholesky(self):
        M_chol = self.new_with_same_masks()
        lower = None
        if self.block.size:
            block = cholesky(self.block, lower=True)
        else:
            block = zeros_like(self.block)
        if (self.diagonal<=0).any():
            raise LinAlgError
        diagonal = sqrt(self.diagonal)
        return self.new_with_same_masks(block, diagonal)
    
    def trisolve(self, x):
        out = zeros(len(x))
        if len(self.unmasked):
            out[self.unmasked] = solve_triangular(self.block, x[self.unmasked], lower=True)
        out[self.masked] = x[self.masked]/self.diagonal
        return out
