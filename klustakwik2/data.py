__all__ = ['SparseArray1D', 'Spike']

class SparseArray1D(object):
    def __init__(self, vals, inds):
        self.vals = vals
        self.inds = inds

class Spike(object):
    def __init__(self, features):
        self.features = features
