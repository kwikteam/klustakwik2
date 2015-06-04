from numpy import *
from klustakwik2 import *
from numpy.testing import assert_raises, assert_array_almost_equal, assert_array_equal
from nose import with_setup
from nose.tools import nottest
from numpy.random import randint, rand
from six.moves import range

@nottest
def generate_multimask_test_data(num_masks, num_points, num_features):
    # step 1: generate masks
    found = set()
    all_masks = []
    for i in range(num_masks):
        while True:
            u, = randint(2, size=num_features).nonzero()
            h = u.tostring()
            if h not in found:
                found.add(h)
                all_masks.append(u)
                break
    assert len(all_masks)==num_masks
    # step 2: generate data
    fet = []
    fmask = []
    offsets = []
    unmasked = []
    n = 0
    offsets = [n]
    for i in range(num_points):
        u = all_masks[randint(num_masks)]
        fet.append(rand(len(u)))
        fmask.append(0.5+0.5*rand(len(u)))
        unmasked.append(u)
        n += len(u)
        offsets.append(n)
    assert len(offsets)==num_points+1
    fet = hstack(fet)
    fmask = hstack(fmask)
    unmasked = array(hstack(unmasked), dtype=int)
    offsets = array(offsets)
    return RawSparseData(full(num_features, 0.5), full(num_features, 1./12), # mean/var of rand()
                         fet, fmask, unmasked, offsets).to_sparse_data()


def test_mask_starts():
    data = generate_multimask_test_data(100, 1000, 10)

    assert data.num_spikes==1000
    assert data.num_features==10
    assert 90<=data.num_masks<=110 # a bit of leeway because it's random

    # too few clusters raises an error
    assert_raises(ValueError, mask_starts, data, 0, 2)
    assert_raises(ValueError, mask_starts, data, 2, 2)

    # too many clusters logs a warning and gives us as many as possible
    clusters = mask_starts(data, 150, 2)
    assert amin(clusters)==2
    # 102 because we aim for 150 but only have 100 masks, which is 102 clusters (0,1=noise,mua)
    assert 95<amax(clusters)<102

    # half the maximum number of clusters should be ok and give us something to test
    clusters = mask_starts(data, 50, 2)
    assert amin(clusters)==2
    assert 45<amax(clusters)<50 # a bit of leeway because it's random

    # Basic sanity check on the generated clusters
    for p in range(len(clusters)):
        cluster = clusters[p]
        unmasked = data.unmasked[data.unmasked_start[p]:data.unmasked_end[p]]
        for p2 in range(len(clusters)):
            cluster2 = clusters[p2]
            #unmasked2 = data.unmasked[data.unmasked_start[p2]:data.unmasked_end[p2]]
            # this says that if the masks are the same they ought to be assigned to the same
            # cluster
            if data.unmasked_start[p]==data.unmasked_start[p2]:
                assert cluster2==cluster


if __name__=='__main__':
    test_mask_starts()
