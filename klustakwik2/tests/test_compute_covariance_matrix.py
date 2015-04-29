from numpy import *
from klustakwik2 import *
from numpy.testing import assert_raises, assert_array_almost_equal, assert_array_equal
from nose import with_setup
from nose.tools import nottest
from numpy.random import randint, rand
from copy import deepcopy

# we use the version that is used in klustakwik2 rather than separately testing the numba/cython
# versions
from klustakwik2.clustering import (accumulate_cluster_mask_sum, compute_cluster_means,
                                    compute_covariance_matrices)

from test_compute_cluster_masks import generate_simple_test_kk

def test_compute_covariance_matrix():
    kk = generate_simple_test_kk(points_for_cluster_mask=0.5)

    num_cluster_members = kk.num_cluster_members
    num_clusters = kk.num_clusters_alive
    num_features = kk.num_features

    kk.compute_cluster_masks()
    kk.cluster_mean = compute_cluster_means(kk)

    cov_matrices = []
    
    compute_covariance_matrices(kk)

    for cluster in xrange(1, num_clusters):
        cov = kk.covariance[cluster]
        f2m = kk.orig_features-kk.cluster_mean[cluster, :][newaxis, :]
        ct = kk.orig_correction_terms
        spikes = kk.get_spikes_in_cluster(cluster)
        block = zeros_like(cov.block)
        unmasked = cov.unmasked
        for spike in spikes:
            if cluster==1:
                for i in xrange(len(unmasked)):
                    block[i, i] += f2m[spike, unmasked[i]]**2
            else:
                block[:, :] += f2m[spike, unmasked][:, newaxis]*f2m[spike, unmasked][newaxis, :]
            for i in xrange(len(unmasked)):
                block[i, i] += ct[spike, unmasked[i]]
        if cluster==1:
            point = kk.mua_point
        else:
            point = kk.noise_point
        for i in xrange(len(cov.unmasked)):
            block[i, i] += point*kk.data.noise_variance[cov.unmasked[i]]
        factor = 1.0/(num_cluster_members[cluster]+point-1)
        block *= factor
        assert_array_almost_equal(block, cov.block)
        cov_matrices.append(deepcopy(cov))
        
    return cov_matrices


if __name__=='__main__':
    test_compute_covariance_matrix()
