from numpy import *
from .masks_cy import *
from random import shuffle

__all__ = ['sort_masks', 'mask_start_clusters']

def sort_masks(raw_data):
    # step 1: sort into lexicographical order of masks
    O = raw_data.offsets
    I = raw_data.unmasked
    start = zeros(len(O)-1, dtype=int)
    end = zeros(len(O)-1, dtype=int)
    new_indices = zeros_like(I)

    num_new_indices = do_sort_masks(I, O, start, end, new_indices)
    
    new_indices = array(new_indices[:num_new_indices], copy=True)

    return None, new_indices, start, end


def mask_start_clusters(data, num_clusters):
    clusters = full(data.num_spikes, -1, dtype=int) # start with -1 to make sure we don't miss any
    num_features = data.num_features

    # go through the spikes in random order so as not to bias the selection of masks
    allspikes = arange(data.num_spikes)
    shuffle(allspikes)
    
    do_mask_starts(clusters,
                   data.unmasked, data.unmasked_start, data.unmasked_end,
                   allspikes, num_features, num_clusters)

    return clusters
