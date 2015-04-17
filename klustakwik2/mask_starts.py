from numpy import *
from .logger import log_message
from random import shuffle
from numpy.random import randint

__all__ = ['mask_starts']

def mask_difference(A, B, n):
    X = zeros(n, dtype=int)
    Y = zeros(n, dtype=int)
    X[A] = 1
    Y[B] = 1
    return sum(abs(X-Y))


def mask_starts(data, num_clusters):
    log_message('info', 'Using mask starts with %d clusters' % num_clusters)
    if num_clusters<=2:
        raise ValueError("Number of starting clusters must be at least 3 to accomodate noise and mua clusters.")
    num_clusters -= 2 # noise and mua clusters are not generated, we add 2 at the end
    if data.num_masks<num_clusters:
        log_message('warning', ('Not enough masks (%d) for specified number of starting '
                                'clusters (%d)') % (data.num_masks, num_clusters+2))
        num_clusters = data.num_masks
    clusters = full(data.num_spikes, -1, dtype=int) # start with -1 to make sure we don't miss any
    found = dict()
    end = dict()
    cur_cluster = 0
    num_features = data.num_features
    
    # go through the spikes in random order so as not to bias the selection of masks
    allspikes = arange(data.num_spikes)
    shuffle(allspikes)
    
    for p in allspikes:
        unmasked = data.unmasked[data.unmasked_start[p]:data.unmasked_end[p]]
        mask_id = data.unmasked_start[p]
        if mask_id in found:
            # we've already used this mask
            clusters[p] = found[mask_id]
        else:
            if cur_cluster<num_clusters:
                # use a new mask
                found[mask_id] = cur_cluster
                end[mask_id] = data.unmasked_end[p]
                clusters[p] = cur_cluster
                cur_cluster += 1
            else:
                # we have to find the closest mask
                best_distance = num_features+1
                best_ids = []
                for candidate_id in found.iterkeys():
                    candidate_mask = data.unmasked[candidate_id:end[candidate_id]]
                    d = mask_difference(candidate_mask, unmasked, num_features)
                    if d==best_distance:
                        best_ids.append(candidate_id)
                    elif d<best_distance:
                        best_distance = d
                        best_ids = [candidate_id]
                # select a random id from the set of equally closest matches to avoid bias
                best_id = best_ids[randint(len(best_ids))]
                clusters[p] = found[best_id]
                found[mask_id] = found[best_id]
                end[mask_id] = end[best_id]
    return clusters+2 # +2 because 0 and 1 are noise and mua cluster
