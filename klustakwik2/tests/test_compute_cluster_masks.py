from numpy import *
from klustakwik2 import *
from numpy.testing import assert_raises, assert_array_almost_equal, assert_array_equal
from nose import with_setup
from nose.tools import nottest
from numpy.random import randint, rand

# we use the version that is used in klustakwik2 rather than separately testing the numba/cython
# versions
from klustakwik2.clustering import accumulate_cluster_mask_sum

from test_io import generate_simple_test_raw_data

@nottest
def generate_simple_test_kk(**params):
    raw_data, fet, fmask, features, correction_terms = generate_simple_test_raw_data()
    data = raw_data.to_sparse_data()
    kk = KK(data, **params)
    kk.orig_fet = fet
    kk.orig_fmask = fmask
    kk.orig_features = features
    kk.orig_correction_terms = correction_terms
    clusters = array([2, 3, 3, 4])
    kk.initialise_clusters(clusters)
    return kk

def test_accumulate_cluster_mask_sum():
    kk = generate_simple_test_kk()
    fet = kk.orig_fet
    fmask = kk.orig_fmask
    
    assert kk.num_clusters_alive==5

    num_clusters = kk.num_clusters_alive
    num_features = kk.num_features
    
    cluster_mask_sum = zeros((num_clusters, num_features))
    cluster_mask_sum[:2, :] = -1 # ensure that clusters 0 and 1 are masked
    
    accumulate_cluster_mask_sum(kk, cluster_mask_sum)
    
    # cluster 2 has only point 0 in it, so the cluster_mask sum should be just the corresponding
    # fmask line
    assert_array_almost_equal(cluster_mask_sum[2, :], fmask[0, :])
    # similarly for the others
    assert_array_almost_equal(cluster_mask_sum[3, :], fmask[1, :]+fmask[2, :])
    assert_array_almost_equal(cluster_mask_sum[4, :], fmask[3, :])
    assert (cluster_mask_sum[0, :]==-1).all()
    assert (cluster_mask_sum[1, :]==-1).all()


def test_compute_cluster_masks():
    kk = generate_simple_test_kk(points_for_cluster_mask=1)
    kk.compute_cluster_masks()
    assert_array_equal(kk.cluster_unmasked_features[0], [])
    assert_array_equal(kk.cluster_masked_features[0], [0, 1, 2, 3, 4])
    assert_array_equal(kk.cluster_unmasked_features[1], [])
    assert_array_equal(kk.cluster_masked_features[1], [0, 1, 2, 3, 4])
    assert_array_equal(kk.cluster_unmasked_features[2], [0])
    assert_array_equal(kk.cluster_masked_features[2], [1, 2, 3, 4])
    assert_array_equal(kk.cluster_unmasked_features[3], [1, 2])
    assert_array_equal(kk.cluster_masked_features[3], [0, 3, 4])
    assert_array_equal(kk.cluster_unmasked_features[4], [3, 4])
    assert_array_equal(kk.cluster_masked_features[4], [0, 1, 2])
    
    kk.points_for_cluster_mask = 0.1
    kk.compute_cluster_masks()
    assert_array_equal(kk.cluster_unmasked_features[0], [])
    assert_array_equal(kk.cluster_masked_features[0], [0, 1, 2, 3, 4])
    assert kk.covariance[0].block.shape==(0, 0)
    assert kk.covariance[0].diagonal.shape==(5,)
    assert_array_equal(kk.cluster_unmasked_features[1], [])
    assert_array_equal(kk.cluster_masked_features[1], [0, 1, 2, 3, 4])
    assert kk.covariance[1].block.shape==(0, 0)
    assert kk.covariance[1].diagonal.shape==(5,)
    assert_array_equal(kk.cluster_unmasked_features[2], [0, 1])
    assert_array_equal(kk.cluster_masked_features[2], [2, 3, 4])
    assert kk.covariance[2].block.shape==(2, 2)
    assert kk.covariance[2].diagonal.shape==(3,)
    assert_array_equal(kk.cluster_unmasked_features[3], [1, 2])
    assert_array_equal(kk.cluster_masked_features[3], [0, 3, 4])
    assert kk.covariance[3].block.shape==(2, 2)
    assert kk.covariance[3].diagonal.shape==(3,)
    assert_array_equal(kk.cluster_unmasked_features[4], [3, 4])
    assert_array_equal(kk.cluster_masked_features[4], [0, 1, 2])
    assert kk.covariance[4].block.shape==(2, 2)
    assert kk.covariance[4].diagonal.shape==(3,)

    kk.points_for_cluster_mask = 1.5
    kk.compute_cluster_masks()
    assert_array_equal(kk.cluster_unmasked_features[0], [])
    assert_array_equal(kk.cluster_masked_features[0], [0, 1, 2, 3, 4])
    assert_array_equal(kk.cluster_unmasked_features[1], [])
    assert_array_equal(kk.cluster_masked_features[1], [0, 1, 2, 3, 4])
    assert_array_equal(kk.cluster_unmasked_features[2], [])
    assert_array_equal(kk.cluster_masked_features[2], [0, 1, 2, 3, 4])
    assert_array_equal(kk.cluster_unmasked_features[3], [1, 2])
    assert_array_equal(kk.cluster_masked_features[3], [0, 3, 4])
    assert_array_equal(kk.cluster_unmasked_features[4], [])
    assert_array_equal(kk.cluster_masked_features[4], [0, 1, 2, 3, 4])
    

if __name__=='__main__':
    test_accumulate_cluster_mask_sum()
    test_compute_cluster_masks()
    