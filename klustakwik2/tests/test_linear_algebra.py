from numpy import *
from klustakwik2 import *
from numpy.testing import assert_raises, assert_array_almost_equal, assert_array_equal
from nose import with_setup
from nose.tools import nottest
from klustakwik2.linear_algebra import BlockPlusDiagonalMatrix
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from klustakwik2.numerics.cylib.e_step import trisolve
from numpy.random import randn
from numpy.linalg import inv
from six.moves import range

def test_cholesky_trisolve():
    M = array([[3.5, 0,   0,   0],
               [0,   1.2, 0,   0.1],
               [0,   0,   6.2, 0],
               [0,   0.1,   0,   11]])
    masked = array([0, 2])
    unmasked = array([1, 3])
    cov = BlockPlusDiagonalMatrix(masked, unmasked)
    cov.block[:, :] = M[ix_(unmasked, unmasked)]
    cov.diagonal[:] = M[masked, masked]
    # test cholesky decomposition
    chol = cov.cholesky()
    M_chol_from_block = zeros_like(M)
    M_chol_from_block[ix_(unmasked, unmasked)] = chol.block
    M_chol_from_block[masked, masked] = chol.diagonal
    M_chol = cholesky(M, lower=True)
    assert_array_almost_equal(M_chol_from_block, M_chol)
    assert_array_almost_equal(M, M_chol.dot(M_chol.T))
    assert_array_almost_equal(cov.block, chol.block.dot(chol.block.T))
    # test trisolve
    x = randn(M.shape[0])
    y1 = solve_triangular(M_chol, x, lower=True)
    y2 = chol.trisolve(x)
    y3 = zeros(len(x))
    trisolve(chol.block, chol.diagonal, chol.masked, chol.unmasked, len(chol.masked), len(chol.unmasked), x, y3)
    assert_array_almost_equal(y1, y2)
    assert_array_almost_equal(y1, y3)
    assert_array_almost_equal(y2, y3)
    # test compute diagonal of inverse of cov matrix used in E-step
    inv_cov_diag = zeros(len(x))
    basis_vector = zeros(len(x))
    for i in range(len(x)):
        basis_vector[i] = 1.0
        root = chol.trisolve(basis_vector)
        inv_cov_diag[i] = sum(root**2)
        basis_vector[:] = 0
    M_inv_diag = diag(inv(M))
    assert_array_almost_equal(M_inv_diag, inv_cov_diag)


if __name__=='__main__':
    test_cholesky_trisolve()

