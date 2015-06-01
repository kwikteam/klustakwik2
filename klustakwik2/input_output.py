from numpy import *
import time

from .data import RawSparseData
from .logger import log_message
from six.moves import zip

__all__ = ['load_fet_fmask_to_raw', 'save_clu', 'load_clu', 'SaveCluEvery']


def load_fet_fmask_to_raw(fname, shank, use_features=None, drop_last_n_features=0):
    if use_features is None and drop_last_n_features>0:
        use_features = slice(None, -drop_last_n_features)
    else:
        use_features = slice(None)
    fet_fname = fname+'.fet.'+str(shank)
    fmask_fname = fname+'.fmask.'+str(shank)
    # read files
    fet_file = open(fet_fname, 'r')
    fmask_file = open(fmask_fname, 'r')
    # read first line of fmask file
    num_features = int(fmask_file.readline())
    features_to_use = arange(num_features)[use_features]
    num_features = len(features_to_use)
    # Stage 1: read min/max of fet values for normalisation
    # and count total number of unmasked features
    fet_file.readline() # skip first line (num channels)
    # we normalise channel-by-channel
    vmin = ones(num_features)*inf
    vmax = ones(num_features)*-inf
    total_unmasked_features = 0
    num_spikes = 0
    for fetline, fmaskline in zip(fet_file, fmask_file):
        vals = fromstring(fetline, dtype=float, sep=' ')[use_features]
        fmaskvals = fromstring(fmaskline, dtype=float, sep=' ')[use_features]
        inds, = (fmaskvals>0).nonzero()
        total_unmasked_features += len(inds)
        vmin = minimum(vals, vmin)
        vmax = maximum(vals, vmax)
        num_spikes += 1
    # Stage 2: read data line by line, normalising
    fet_file.close()
    fet_file = open(fet_fname, 'r')
    fet_file.readline()
    fmask_file.close()
    fmask_file = open(fmask_fname, 'r')
    fmask_file.readline()
    vdiff = vmax-vmin
    vdiff[vdiff==0] = 1
    fetsum = zeros(num_features)
    fet2sum = zeros(num_features)
    nsum = zeros(num_features)
    all_features = zeros(total_unmasked_features)
    all_fmasks = zeros(total_unmasked_features)
    all_unmasked = zeros(total_unmasked_features, dtype=int)
    offsets = zeros(num_spikes+1, dtype=int)
    curoff = 0
    for i, (fetline, fmaskline) in enumerate(zip(fet_file, fmask_file)):
        fetvals = (fromstring(fetline, dtype=float, sep=' ')[use_features]-vmin)/vdiff
        fmaskvals = fromstring(fmaskline, dtype=float, sep=' ')[use_features]
        inds, = (fmaskvals>0).nonzero()
        masked_inds, = (fmaskvals==0).nonzero()
        all_features[curoff:curoff+len(inds)] = fetvals[inds]
        all_fmasks[curoff:curoff+len(inds)] = fmaskvals[inds]
        all_unmasked[curoff:curoff+len(inds)] = inds
        offsets[i] = curoff
        curoff += len(inds)
        fetsum[masked_inds] += fetvals[masked_inds]
        fet2sum[masked_inds] += fetvals[masked_inds]**2
        nsum[masked_inds] += 1
    offsets[-1] = curoff

    nsum[nsum==0] = 1
    noise_mean = fetsum/nsum
    noise_variance = fet2sum/nsum-noise_mean**2

    return RawSparseData(noise_mean, noise_variance, all_features, all_fmasks, all_unmasked, offsets)


def save_clu(kk, fname, shank):
    savetxt(fname+'.clu.'+str(shank), kk.clusters+1, '%d', header=str(amax(kk.clusters)), comments='')


def load_clu(fname):
    return loadtxt(fname, skiprows=1, dtype=int)-1


class SaveCluEvery(object):
    '''
    Callback to save the clu file every fixed number of iterations.

    fname can be a simple string, or a formattable string that takes the kk object as an argument,
    so that you can write, e.g. fname='testdata.{kk.name}.{kk.current_iteration}'.

    every is the number of minutes between saves

    save_all=True will append .iterN for N the iteration number to the filename, and therefore save
    each clu file produced.
    '''
    def __init__(self, fname, shank, every=1, save_all=False):
        self.fname = fname
        self.shank = shank
        self.every = every*60
        self.save_all = save_all
        self.t_next = time.time()+self.every

    def __call__(self, kk):
        t_cur = time.time()
        if kk.name=='' and t_cur>self.t_next and not kk.is_subset:
            shank = str(self.shank)
            if self.save_all:
                shank = str(shank)+'.iter'+str(kk.current_iteration)
            fname = self.fname.format(kk=kk)
            log_message('info', 'Saving clu to file '+fname+'.clu.'+shank)
            save_clu(kk, fname, shank)
            self.t_next = time.time()+self.every

