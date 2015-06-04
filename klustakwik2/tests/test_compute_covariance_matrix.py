from numpy import *
from klustakwik2 import *
from numpy.testing import assert_raises, assert_array_almost_equal, assert_array_equal
from nose import with_setup
from nose.tools import nottest
from numpy.random import randint, rand
from copy import deepcopy
from six.moves import range
from klustakwik2.linear_algebra import BlockPlusDiagonalMatrix

# we use the version that is used in klustakwik2 rather than separately testing the numba/cython
# versions
from klustakwik2.clustering import (accumulate_cluster_mask_sum, compute_cluster_mean,
                                    compute_covariance_matrix)

from .test_compute_cluster_masks import generate_simple_test_kk

def test_compute_covariance_matrix():
    kk = generate_simple_test_kk(points_for_cluster_mask=0.5)

    num_cluster_members = kk.num_cluster_members
    num_clusters = kk.num_clusters_alive
    num_features = kk.num_features

    cluster_mask_sum = zeros((num_clusters, num_features))
    for cluster in range(kk.num_special_clusters, num_clusters):
        accumulate_cluster_mask_sum(kk, cluster_mask_sum[cluster, :], kk.get_spikes_in_cluster(cluster))
    cluster_mask_sum[:kk.num_special_clusters, :] = -1 # ensure that special clusters are masked

    cov_matrices = []

    for cluster in range(1, num_clusters):
        cluster_mean = compute_cluster_mean(kk, cluster)
        curmask = cluster_mask_sum[cluster, :]
        unmasked, = (curmask>=kk.points_for_cluster_mask).nonzero()
        masked, = (curmask<kk.points_for_cluster_mask).nonzero()
        unmasked = array(unmasked, dtype=int)
        masked = array(masked, dtype=int)
        cov = BlockPlusDiagonalMatrix(masked, unmasked)
        compute_covariance_matrix(kk, cluster, cluster_mean, cov)
        f2m = kk.orig_features-cluster_mean[newaxis, :]
        ct = kk.orig_correction_terms
        spikes = kk.get_spikes_in_cluster(cluster)
        block = zeros_like(cov.block)
        unmasked = cov.unmasked
        for spike in spikes:
            if cluster==1:
                for i in range(len(unmasked)):
                    block[i, i] += f2m[spike, unmasked[i]]**2
            else:
                block[:, :] += f2m[spike, unmasked][:, newaxis]*f2m[spike, unmasked][newaxis, :]
            for i in range(len(unmasked)):
                block[i, i] += ct[spike, unmasked[i]]
        if cluster==1:
            point = kk.mua_point
        else:
            point = kk.noise_point
        for i in range(len(cov.unmasked)):
            block[i, i] += point*kk.data.noise_variance[cov.unmasked[i]]
        factor = 1.0/(num_cluster_members[cluster]+point-1)
        block *= factor
        assert_array_almost_equal(block, cov.block)
        cov_matrices.append(deepcopy(cov))

    return cov_matrices


if __name__=='__main__':
    test_compute_covariance_matrix()
