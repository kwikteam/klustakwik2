from numpy import *
from .hashing import hash_array

__all__ = ['SparseArray1D', 'Spike', 'DataContainer']

class SparseArray1D(object):
    def __init__(self, vals, inds, n):
        self.vals = vals
        self.inds = inds
        self.n = n


class Spike(object):
    def __init__(self, features, num_features):
        self.features = features
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
                      num_features=num_features)
        self.spikes.append(spike)
        self.fetsum[masked] += fetvals[masked]
        self.fet2sum[masked] += fetvals[masked]**2
        self.nsum[masked] += 1
        
    @property
    def mean(self):
        if hasattr(self, '_mean'):
            return self._mean
        self.nsum[self.nsum==0] = 1
        self._mean = self.fetsum/self.nsum
        return self._mean
    
    @property
    def var(self):
        if hasattr(self, '_var'):
            return self._var
        mu = self.mean
        self._var = self.fet2sum/self.nsum-(self.fetsum/self.nsum)**2
        return self._var
