from numpy import *
from klustakwik2 import *
from numpy.testing import assert_raises, assert_array_almost_equal, assert_array_equal
from nose import with_setup
from nose.tools import nottest
from numpy.random import randint, rand
from six.moves import range

from .test_mask_starts import generate_multimask_test_data

def test_partitioning():
    for _ in range(100):
        # actually we only need to generate some arbitrary data, the values don't matter much
        data = generate_multimask_test_data(10, 10, 5)
        kk = KK(data)
        clusters = [0, 1, 0, 2, 0, 2, 0, 3, 0, 3]
        kk.initialise_clusters(clusters)
        assert_array_equal(kk.num_cluster_members, [5, 1, 2, 2])
        assert_array_equal(kk.get_spikes_in_cluster(0), [0, 2, 4, 6, 8])
        assert_array_equal(kk.get_spikes_in_cluster(1), [1])
        assert_array_equal(kk.get_spikes_in_cluster(2), [3, 5])
        assert_array_equal(kk.get_spikes_in_cluster(3), [7, 9])
        assert_array_equal(kk.spikes_in_cluster, [0, 2, 4, 6, 8, 1, 3, 5, 7, 9])
        assert_array_equal(kk.spikes_in_cluster_offset, [0, 5, 6, 8, 10])
    

if __name__=='__main__':
    test_partitioning()
