from numpy import *
from klustakwik2 import *
from numpy.testing import assert_raises, assert_array_almost_equal, assert_array_equal
from nose import with_setup
from nose.tools import nottest
from numpy.random import randint, rand

# we use the version that is used in klustakwik2 rather than separately testing the numba/cython
# versions
from klustakwik2.clustering import (accumulate_cluster_mask_sum, compute_cluster_means,
                                    compute_covariance_matrix)

from test_compute_cluster_masks import generate_simple_test_kk
from test_compute_covariance_matrix import test_compute_covariance_matrix

def test_m_step():
    '''
    This test doesn't add much because it's more or less the same as the M_step() code, but
    I guess it guards against future errors that might be introduced.
    '''
    kk = generate_simple_test_kk(points_for_cluster_mask=0.5)

    num_cluster_members = kk.num_cluster_members
    num_clusters = kk.num_clusters_alive
    num_features = kk.num_features

    kk.M_step()
    
    cov_matrices = test_compute_covariance_matrix()
    
    for cluster, cov in enumerate(cov_matrices):
        cluster += 1        
        if cluster==1:
            point = kk.mua_point
        else:
            point = kk.prior_point

        for i, j in enumerate(cov.unmasked):
            cov.block[i, i] += point*kk.data.noise_variance[j]
        for i, j in enumerate(cov.masked):
            cov.diagonal[i] += point*kk.data.noise_variance[j]
            
        factor = 1.0/(num_cluster_members[cluster]+point-1)
        cov.block *= factor
        cov.diagonal *= factor
        
        assert_array_almost_equal(cov.block, kk.covariance[cluster].block)
        assert_array_almost_equal(cov.diagonal, kk.covariance[cluster].diagonal)
    
    
if __name__=='__main__':
    test_m_step()
