from numpy import *

__all__ = ['mask_starts']

def mask_difference(A, B, n):
    X = zeros(n, dtype=int)
    Y = zeros(n, dtype=int)
    X[A] = 1
    Y[B] = 1
    return sum(abs(X-Y))


def mask_starts(data, num_clusters):
    if data.num_masks<num_clusters:
        print ('Not enough masks (%d) for that many starting '
               'clusters (%d)') % (data.num_masks, num_clusters)
        num_clusters = data.num_masks
    clusters = zeros(data.num_spikes, dtype=int)
    found = dict()
    end = dict()
    cur_cluster = 0
    num_features = data.num_features
    for p in xrange(data.num_spikes):
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
                best_id = None
                for candidate_id in found.iterkeys():
                    candidate_mask = data.unmasked[mask_id:end[mask_id]]
                    d = mask_difference(candidate_mask, unmasked)
                    if d<best_distance:
                        best_distance = d
                        best_id = candidate_id
                clusters[p] = found[best_id]
                found[mask_id] = found[best_id]
    return clusters
