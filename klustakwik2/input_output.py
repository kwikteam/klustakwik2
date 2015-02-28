from numpy import *
from itertools import izip
from .hashing import hash_array
from .data import SparseArray1D, Spike, DataContainer


__all__ = ['load_fet_fmask',
           'DataContainer',
           ]

def load_fet_fmask(fname, shank):
    fet_fname = fname+'.fet.'+str(shank)
    fmask_fname = fname+'.fmask.'+str(shank)
    # read files
    fet_file = open(fet_fname, 'r')
    fmask_file = open(fmask_fname, 'r')
    # read first line of fmask file
    num_features = int(fmask_file.readline())
    # Stage 1: read min/max of fet values for normalisation
    fet_file.readline() # skip first line (num channels)
    # we normalise channel-by-channel
    vmin = ones(num_features)*inf
    vmax = ones(num_features)*-inf
    for line in fet_file:
        vals = fromstring(line, dtype=float, sep=' ')
        vmin = minimum(vals, vmin)
        vmax = maximum(vals, vmax)
    # Stage 2: read data line by line, normalising
    fet_file.close()
    fet_file = open(fet_fname, 'r')
    fet_file.readline()
    data = DataContainer(num_features)
    vdiff = vmax-vmin
    vdiff[vdiff==0] = 1
    for fetline, fmaskline in izip(fet_file, fmask_file):
        # this normalisation could be refactored
        fetvals = (fromstring(fetline, dtype=float, sep=' ')-vmin)/vdiff
        fmaskvals = fromstring(fmaskline, dtype=float, sep=' ')
        data.add_from_dense(fetvals, fmaskvals)
    return data
