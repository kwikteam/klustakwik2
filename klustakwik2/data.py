from numpy import *
from .hashing import hash_array

__all__ = ['SparseArray1D', 'Spike', 'DataContainer']

class SparseArray1D(object):
    def __init__(self, vals, inds, n):
        self.vals = vals
        self.inds = inds
        self.n = n


class Spike(object):
    def __init__(self, features, mask, num_features):
        self.features = features
        self.mask = mask
        self.num_features = num_features


# Probably rename/remove/refactor this class
class DataContainer(object):
    def __init__(self, num_features, masks=None, spikes=None):
        if masks is None:
            masks = {}
        if spikes is None:
            spikes = []
        self.num_features = num_features
        self.masks = masks
        self.spikes = spikes
        
    def add_from_dense(self, fetvals, fmaskvals):
        num_features = len(fetvals)
        if not hasattr(self, 'fetsum'):
            self.fetsum = zeros(num_features)
            self.fet2sum = zeros(num_features)
            self.nsum = zeros(num_features)
        inds, = (fmaskvals>0).nonzero()
        masked, = (fmaskvals==0).nonzero()
        indhash = hash_array(inds)
        if indhash in self.masks:
            inds = self.masks[indhash]
        else:
            self.masks[indhash] = inds
        spike = Spike(SparseArray1D(fetvals[inds], inds, num_features),
                      SparseArray1D(fmaskvals[inds], inds, num_features),
                      num_features=num_features)
        self.spikes.append(spike)
        self.fetsum[masked] += fetvals[masked]
        self.fet2sum[masked] += fetvals[masked]**2
        self.nsum[masked] += 1
        
    def do_initial_precomputations(self):
        self.compute_noise_mean_and_variance()
        self.compute_correction_term_and_replace_data()
        
    def compute_noise_mean_and_variance(self):
        self.nsum[self.nsum==0] = 1
        mu = self.noise_mean = self.fetsum/self.nsum
        self.noise_variance = self.fet2sum/self.nsum-mu**2

    def compute_correction_term_and_replace_data(self):
        self.correction_terms = []
        for spike in self.spikes:
            I = spike.features.inds
            x = spike.features.vals
            w = spike.mask.vals
            nu = self.noise_mean[I]
            sigma2 = self.noise_variance[I]
            y = w*x+(1-w)*nu
            z = w*x*x+(1-w)*(nu*nu+sigma2)
            correction_term = SparseArray1D(z-y*y, I, self.num_features)
            self.correction_terms.append(correction_term)
            spike.features.vals = y
