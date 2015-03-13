from numpy import *
from .hashing import hash_array

__all__ = ['SparseArray1D', 'Spike',
           'BlockPlusDiagonalMatrix',
           'DataContainer',
           ]

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


class BlockPlusDiagonalMatrix(object):
    def __init__(self, masked, unmasked):
        self.masked = masked
        self.unmasked = unmasked
        self.num_masked = len(masked)
        self.num_unmasked = len(unmasked)
        self.block = zeros((self.num_unmasked, self.num_unmasked))
        self.diagonal = zeros(self.num_masked)
    
    def new_with_same_masks(self):
        return BlockPlusDiagonalMatrix(self.masked, self.unmasked)


# Probably rename/remove/refactor this class
class DataContainer(object):
    def __init__(self, num_features, num_spikes, total_unmasked_features,
                 num_masks, total_mask_length,
                 masks=None, spikes=None):
        if masks is None:
            masks = {}
        if spikes is None:
            spikes = []
        self.masks = masks
        self.spikes = spikes
        self.num_features = num_features
        self.num_spikes = num_spikes
        self.total_unmasked_features = total_unmasked_features
        self.num_masks = num_masks
        self.total_mask_length = total_mask_length
        self.flat_data = FlatDataContainer(num_features, num_spikes,
                                           total_unmasked_features,
                                           num_masks, total_mask_length)
        self.fetsum = zeros(num_features)
        self.fet2sum = zeros(num_features)
        self.nsum = zeros(num_features)
        
    def add_from_dense(self, fetvals, fmaskvals):
        num_features = len(fetvals)
        inds, = (fmaskvals>0).nonzero()
        masked, = (fmaskvals==0).nonzero()
        self.fetsum[masked] += fetvals[masked]
        self.fet2sum[masked] += fetvals[masked]**2
        self.nsum[masked] += 1
        spike = self.flat_data.add_from_sparse(fetvals[inds],
                                               fmaskvals[inds],
                                               inds)
        self.spikes.append(spike)
        indhash = hash_array(inds)
        self.masks[indhash] = spike.features.inds
        
    def do_initial_precomputations(self):
        self.compute_noise_mean_and_variance()
        self.compute_correction_term_and_replace_data()
        
    def compute_noise_mean_and_variance(self):
        self.nsum[self.nsum==0] = 1
        mu = self.noise_mean = self.fetsum/self.nsum
        self.noise_variance = self.fet2sum/self.nsum-mu**2

    def compute_correction_term_and_replace_data(self):
        for spike in self.spikes:
            I = spike.features.inds
            x = spike.features.vals
            w = spike.mask.vals
            nu = self.noise_mean[I]
            sigma2 = self.noise_variance[I]
            y = w*x+(1-w)*nu
            z = w*x*x+(1-w)*(nu*nu+sigma2)
            spike.correction_term.vals[:] = z-y*y
            spike.features.vals[:] = y


class FlatDataContainer(object):
    def __init__(self, num_features, num_spikes, total_unmasked_features,
                 num_masks, total_mask_length):
        self.num_features = num_features
        self.num_spikes = num_spikes
        self.total_unmasked_features = total_unmasked_features
        self.num_masks = num_masks
        self.total_mask_length = total_mask_length
        # allocate sufficient memory
        self.all_features = zeros(total_unmasked_features)
        self.all_fmasks = zeros(total_unmasked_features)
        self.all_correction_terms = zeros(total_unmasked_features)
        self.all_maskinds = zeros(total_mask_length, dtype=int)
        self.feature_start_indices = zeros(num_spikes, dtype=int)
        self.feature_end_indices = zeros(num_spikes, dtype=int)
        self.maskind_start_indices = zeros(num_spikes, dtype=int)
        self.maskind_end_indices = zeros(num_spikes, dtype=int)
        # tracking variables
        self.current_spike = 0
        self.current_feature_offset = 0
        self.current_maskind_offset = 0
        self.masks = dict()

    def add_from_sparse(self, fetvals, fmaskvals, inds):
        num_features = self.num_features
        indhash = hash_array(inds)
        if indhash in self.masks:
            mask_start_idx, mask_end_idx = self.masks[indhash]
        else:
            mask_start_idx = self.current_maskind_offset
            mask_end_idx = mask_start_idx+len(inds)
            self.current_maskind_offset = mask_end_idx
            self.all_maskinds[mask_start_idx:mask_end_idx] = inds
            self.masks[indhash] = (mask_start_idx, mask_end_idx)
        fet_start_idx = self.current_feature_offset
        fet_end_idx = fet_start_idx+len(inds)
        self.all_features[fet_start_idx:fet_end_idx] = fetvals
        self.all_fmasks[fet_start_idx:fet_end_idx] = fmaskvals
        spike = Spike(SparseArray1D(self.all_features[fet_start_idx:fet_end_idx],
                                    self.all_maskinds[mask_start_idx:mask_end_idx],
                                    num_features),
                      SparseArray1D(self.all_fmasks[fet_start_idx:fet_end_idx],
                                    self.all_maskinds[mask_start_idx:mask_end_idx],
                                    num_features),
                      num_features=num_features)
        spike.correction_term = SparseArray1D(
                        self.all_correction_terms[fet_start_idx:fet_end_idx],
                        self.all_maskinds[mask_start_idx:mask_end_idx],
                        num_features)
        return spike
