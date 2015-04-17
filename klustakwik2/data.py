from numpy import *
import time

from .precomputations import sort_masks, compute_correction_terms_and_replace_data
from .logger import log_message

__all__ = ['RawSparseData', 'SparseData',
           ]


class RawSparseData(object):
    '''
    Contains raw, sparse data used for clustering
    
    This object is only used as an intermediate step, not directly by the
    other algorithms. It should be converted into a SparseData object.
    
    Note that for .fet/.fmask data there already has to be some non-trivial
    processing to get this far (to sparsify the data and extract the
    noise mean and variance). However, in the future this step may have been
    computed by SpikeDetekt which will output sparse data.
    
    Also note that the indices can be a smaller array than the features and
    fmasks because there may be a much smaller unique set.
    TODO: method to carry this out
    
    Consists of the following:

    - noise_mean, noise_variance: arrays of length the number of features
    - features: array of feature values
    - masks: array of mask values
    - unmasked: array of unmasked indices
    - offset: array of offset indices (of length the number of spikes + 1)
    
    This is a sparse array such that the feature vector for spike i has
    non-default values at indices unmasked[offset[i]:offset[i+1]]
    with values features[offset[i]:offset[i+1]]. masks has the same structure. 
    '''
    def __init__(self,
                 noise_mean, noise_variance,
                 features, masks, unmasked,
                 offsets,
                 ):
        # Raw data
        self.noise_mean = noise_mean
        self.noise_variance = noise_variance
        self.features = features
        self.masks = masks
        self.unmasked = unmasked
        self.offsets = offsets
        
    def to_sparse_data(self):
        values_start = self.offsets[:-1]
        values_end = self.offsets[1:]
        log_message('debug', 'Starting compute_correction_terms_and_replace_data', name='data')
        features, correction_terms = compute_correction_terms_and_replace_data(self)
        log_message('debug', 'Finished compute_correction_terms_and_replace_data', name='data')
        log_message('debug', 'Starting sort_masks', name='data')
        order, unmasked, unmasked_start, unmasked_end = sort_masks(self)
        log_message('debug', 'Finished sort_masks', name='data')
        return SparseData(self.noise_mean, self.noise_variance,
                          features, self.masks,
                          values_start[order], values_end[order],
                          unmasked,
                          unmasked_start, unmasked_end,
                          correction_terms,
                          )


class SparseData(object):
    '''
    Notes:
    - Assumes that the spikes are in sorted mask order, can use unmasked_start
      as a proxy for the identity of the mask
    '''
    def __init__(self,
                 noise_mean, noise_variance,
                 features, masks,
                 values_start, values_end,
                 unmasked,
                 unmasked_start, unmasked_end,
                 correction_terms,
                 ):
        # Data arrays
        self.noise_mean = noise_mean
        self.noise_variance = noise_variance
        self.features = features
        self.masks = masks
        self.values_start = values_start
        self.values_end = values_end
        self.unmasked = unmasked
        self.unmasked_start = unmasked_start
        self.unmasked_end = unmasked_end
        self.correction_terms = correction_terms
        # Derived data
        self.num_spikes = len(self.values_start)
        self.num_features = len(self.noise_mean)
        
        self.num_masks = len(unique(self.unmasked_start))
        
    def subset(self, spikes):
        return SparseData(self.noise_mean, self.noise_variance,
                          self.features, self.masks,
                          self.values_start[spikes], self.values_end[spikes],
                          self.unmasked,
                          self.unmasked_start[spikes], self.unmasked_end[spikes],
                          self.correction_terms,
                          )
