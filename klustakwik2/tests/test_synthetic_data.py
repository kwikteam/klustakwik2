'''
Test clustering performance on synthetic dataset.
'''

from numpy import *
from klustakwik2 import *
from numpy.testing import assert_raises, assert_array_almost_equal, assert_array_equal
from nose import with_setup
from nose.tools import nottest
from numpy.random import randint, rand, randn

@nottest
def generate_synthetic_data(num_features, spikes_per_centre, centres):
    '''
    Generates data that comes from a distribution with multiple centres. centres is a list of 
    tuples (fet_mean, fet_std, fmask_mean, fmask_std) and features and masks are generated via
    normal random numbers (fmask is clipped between 0 and 1).
    '''
    fet = []
    fmask = []
    offsets = []
    unmasked = []
    n = 0
    num_spikes = len(centres)*spikes_per_centre
    offsets = [n]
    fsum = zeros(num_features)
    fsum2 = zeros(num_features)
    for c, s, fmc, fms in centres:
        c = array(c, dtype=float)
        s = array(s, dtype=float)
        fmc = array(fmc, dtype=float)
        fms = array(fms, dtype=float)
        for i in xrange(spikes_per_centre):
            f = randn(num_features)*s+c
            fm = clip(randn(num_features)*fms+fmc, 0, 1)
            u, = (fm>0).nonzero()
            u = array(u, dtype=int)
            fet.append(f[u])
            fmask.append(fm[u])
            unmasked.append(u)
            n += len(u)
            offsets.append(n)
            fsum += f
            fsum2 += f**2
    fet = hstack(fet)
    fmask = hstack(fmask)
    unmasked = hstack(unmasked)
    offsets = array(offsets)
    return RawSparseData(fsum/num_spikes, fsum2/num_spikes-(fsum/num_spikes)**2,
                         fet, fmask, unmasked, offsets).to_sparse_data()
    

def test_synthetic_trivial():
    '''
    This is the most trivial clustering problem, two well separated clusters in 2D with almost
    no noise and perfect starting masks. All the algorithm has to do is not do anything. We
    therefore test that it gives perfect results.
    '''
    data = generate_synthetic_data(2, 100, [
       # fet mean, fet var,      fmask mean, fmask var                                     
        ((1, 0),   (0.01, 0.01), (1, 0),     (0.0, 0.0)),
        ((0, 1),   (0.01, 0.01), (0, 1),     (0.0, 0.0)),
        ])
    kk = KK(data)
    kk.cluster(10)
    assert len(unique(kk.clusters[:100]))==1
    assert len(unique(kk.clusters[100:]))==1
    assert len(unique(kk.clusters))==2
    assert amin(kk.clusters)==2


def test_synthetic_easy():
    data = generate_synthetic_data(4, 1000, [
        ((1, 1, 0, 0), (0.2, 0.2, 0.2, 0.2), (1, 1, 0, 0), (0.0, 0.1, 0.0, 0.0)),
        ((0, 0, 1, 1), (0.2, 0.2, 0.2, 0.2), (0, 0, 1, 1), (0.0, 0.0, 0.1, 0.0)),
        ])
    kk = KK(data)
    kk.cluster(10)
    print bincount(kk.clusters[:1000])
    print bincount(kk.clusters[1000:])
    # todo: what to make of this?

if __name__=='__main__':
    console_log_level('debug')
#     test_synthetic_trivial()
    test_synthetic_easy()
