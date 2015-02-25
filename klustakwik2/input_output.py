from numpy import *
from itertools import izip
from klustakwik2.hashing import hash_array
from klustakwik2.data import SparseArray1D, Spike
import time

def load_fet_fmask(fname, shank):
    fet_fname = fname+'.fet.'+str(shank)
    fmask_fname = fname+'.fmask.'+str(shank)
    # read files
    fet_file = open(fet_fname, 'r')
    fmask_file = open(fmask_fname, 'r')
    # read first line of fmask file
    nchannels = int(fmask_file.readline())
    # Stage 1: read min/max of fet values for normalisation
    fet_file.readline() # skip first line (num channels)
    # we normalise channel-by-channel
    vmin = ones(nchannels)*inf
    vmax = ones(nchannels)*-inf
    for line in fet_file:
        vals = fromstring(line, dtype=float, sep=' ')
        vmin = minimum(vals, vmin)
        vmax = maximum(vals, vmax)
    # Stage 2: read data line by line, normalising
    fet_file.close()
    fet_file = open(fet_fname, 'r')
    fet_file.readline()
    data_loader = DataLoader()
    vdiff = vmax-vmin
    vdiff[vdiff==0] = 1
    for fetline, fmaskline in izip(fet_file, fmask_file):
        # this normalisation could be refactored
        fetvals = (fromstring(fetline, dtype=float, sep=' ')-vmin)/vdiff
        fmaskvals = fromstring(fmaskline, dtype=float, sep=' ')
        data_loader.add_from_dense(fetvals, fmaskvals)
    return data_loader

# Probably rename/remove/refactor this class
class DataLoader(object):
    def __init__(self):
        self.masks = {}
        self.spikes = []
        
    def add_from_dense(self, fetvals, fmaskvals):
        nchannels = len(fetvals)
        if not hasattr(self, 'fetsum'):
            self.fetsum = zeros(nchannels)
            self.fet2sum = zeros(nchannels)
            self.nsum = zeros(nchannels)    
        inds, = (fmaskvals>0).nonzero()
        masked, = (fmaskvals==0).nonzero()
        indhash = hash_array(inds)
        if indhash in self.masks:
            inds = self.masks[indhash]
        else:
            self.masks[indhash] = inds
        spike = Spike(SparseArray1D(fetvals[inds], inds))
        self.spikes.append(spike)
        self.fetsum[masked] += fetvals[masked]
        self.fet2sum[masked] += fetvals[masked]**2
        self.nsum[masked] += 1
        
    @property
    def mean(self):
        self.nsum[self.nsum==0] = 1
        return self.fetsum/self.nsum
    
    @property
    def var(self):
        mu = self.mean
        return self.fet2sum/self.nsum-(self.fetsum/self.nsum)**2


if __name__=='__main__':
    from pylab import *
    fname, shank = '../temp/testsmallish', 4
    start = time.time()
    data = load_fet_fmask(fname, shank)
    print time.time()-start
    plot(data.mean)
    plot(data.var)
    show()
    