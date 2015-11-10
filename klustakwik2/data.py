from numpy import *
import time

from .precomputations import (reduce_masks, compute_correction_terms_and_replace_data,
                              compute_float_num_unmasked, reduce_masks_from_arrays,
                              compute_float_num_unmasked_from_arrays)
from .logger import log_message
from six.moves import range

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
    
    These arrays have to be normalised in the following fashion:
    - For each channel, the features have to be first normalised between 0 and 1 (the linear
      transformation is different for each channel).
    - The noise mean and noise variance are computed on this normalised data.
    - Unmasked data is where fmask>0, masked data is where fmask==0
    - Only masked data is used for computing the noise mean and noise variance.
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
        unmasked, unmasked_start, unmasked_end = reduce_masks(self)
        log_message('debug', 'Finished sort_masks', name='data')
        float_num_unmasked = compute_float_num_unmasked(self)
        return SparseData(self.noise_mean, self.noise_variance,
                          features, self.masks,
                          values_start, values_end,
                          unmasked,
                          unmasked_start, unmasked_end,
                          correction_terms,
                          float_num_unmasked,
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
                 float_num_unmasked,
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
        self.float_num_unmasked = float_num_unmasked
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
                          self.float_num_unmasked[spikes],
                          )

    def subset_features(self, feature_indices):
        '''
        Returns a pair data, spikes where spikes are those spikes that are not masked on every channel.
        '''
        feature_indices = array(feature_indices, dtype=int)
        # do the easy parts
        noise_mean = self.noise_mean[feature_indices]
        noise_variance = self.noise_variance[feature_indices]

        new_feature_indices = full(self.num_features, -1, dtype=self.unmasked.dtype)
        new_feature_indices[feature_indices] = arange(len(feature_indices))

        # inefficient method, loop through all spikes
        spikes = []
        features = []
        masks = []
        offsets = [0]
        total_features = 0
        unmasked = []
        correction_terms = []
        for p in range(self.num_spikes):
            U = self.unmasked[self.unmasked_start[p]:self.unmasked_end[p]]
            U_in = in1d(U, feature_indices)
            if sum(U_in):
                F = self.features[self.values_start[p]:self.values_end[p]]
                M = self.masks[self.values_start[p]:self.values_end[p]]
                CT = self.correction_terms[self.values_start[p]:self.values_end[p]]
                U2, = U_in.nonzero()
                F2 = F[U2]
                M2 = M[U2]
                CT2 = CT[U2]
                spikes.append(p)
                features.append(F2)
                masks.append(M2)
                correction_terms.append(CT2)
                unmasked.append(new_feature_indices[U[U_in]])
                total_features += len(F2)
                offsets.append(total_features)
        features = array(hstack(features), dtype=self.features.dtype)
        masks = array(hstack(masks), dtype=self.masks.dtype)
        correction_terms = array(hstack(correction_terms), dtype=self.correction_terms.dtype)
        unmasked = array(hstack(unmasked), dtype=self.unmasked.dtype)
        spikes = array(spikes, dtype=int)
        offsets = array(offsets, dtype=int)
        values_start = offsets[:-1]
        values_end = offsets[1:]

        assert amin(unmasked)>=0

        unmasked, unmasked_start, unmasked_end = reduce_masks_from_arrays(values_start, values_end, unmasked)
        float_num_unmasked = compute_float_num_unmasked_from_arrays(masks, offsets)

        data = SparseData(noise_mean, noise_variance,
                          features, masks,
                          values_start, values_end,
                          unmasked, unmasked_start, unmasked_end,
                          correction_terms,
                          float_num_unmasked,
                          )

        return data, spikes
