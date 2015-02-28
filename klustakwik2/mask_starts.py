from numpy import *
from .hashing import hash_array

__all__ = ['mask_starts']

def mask_difference(A, B, n):
    X = zeros(n, dtype=int)
    Y = zeros(n, dtype=int)
    return sum(abs(X-Y))


def mask_starts(data, num_clusters):
    if len(data.masks)<num_clusters:
        print ('Not enough masks (%d) for that many starting '
               'clusters (%d)') % (len(data.masks), num_clusters)
        num_clusters = len(data.masks)
    clusters = zeros(len(data.spikes), dtype=int)
    found = dict()
    cur_cluster = 0
    num_features = data.num_features
    for i, spike in enumerate(data.spikes):
        unmasked = spike.features.inds
        unmasked_hash = hash_array(unmasked)
        if unmasked_hash in found:
            # we've already used this mask
            clusters[i] = found[unmasked_hash]
        else:
            if cur_cluster<num_clusters:
                # use a new mask
                found[unmasked_hash] = cur_cluster
                clusters[i] = cur_cluster
                cur_cluster += 1
            else:
                # we have to find the closest mask
                best_distance = num_features+1
                best_hash = None
                for candidate_hash in found.iterkeys():
                    candidate_mask = data.masks[candidate_hash].features.inds
                    d = mask_difference(candidate_mask, unmasked)
                    if d<best_distance:
                        best_distance = d
                        best_hash = candidate_hash
                clusters[i] = found[best_hash]
                found[unmasked_hash] = found[best_hash]
    return clusters
