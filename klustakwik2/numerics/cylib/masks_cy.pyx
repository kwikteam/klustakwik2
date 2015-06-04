#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: infer_types=True
# distutils: language = c++

import numpy
cimport numpy
from numpy.random import randint

from cython cimport integral, floating

from libcpp.vector cimport vector

cdef integral mask_difference(integral[:] unmasked, integral start1, integral end1, integral start2, integral end2):
    cdef integral i1, i2, u1, u2, d
    cdef integral n1 = end1-start1
    cdef integral n2 = end2-start2
    i1 = start1
    i2 = start2
    d = 0
    while i1<end1 or i2<end2:
        if i1>=end1:
            # there now cannot be a match to u2, so we increase distance by 1
            d += 1
            i2 += 1
            continue
        if i2>=end2:
            # there now cannot be a match to u1, so we increase distance by 1
            d += 1
            i1 += 1
            continue
        u1 = unmasked[i1]
        u2 = unmasked[i2]
        if u1==u2:
            # we match u1 and u2 so increment both pointers to put them out of the pool and since they're matched
            # we don't increase the distance
            i1 += 1
            i2 += 1
        elif u1<u2:
            # we remove u1 from the pool and it didn't match, so we increase the distance
            d += 1
            i1 += 1
        elif u2<u1:
            # we remove u2 from the pool and it didn't match, so we increase the distance
            d += 1
            i2 += 1
    return d


cpdef do_mask_starts(integral[:] clusters,
                     integral[:] unmasked,
                     integral[:] ustart,
                     integral[:] uend,
                     integral[:] allspikes,
                     integral num_features,
                     integral num_clusters,
                     ):
    cdef vector[integral] best_ids
    cdef vector[integral] candidate_ids
    cdef vector[integral] candidate_ends
    cdef integral cur_cluster = 0
    cdef integral p, mask_id, best_distance, candidate_id, candidate_end, c_idx, d, p_idx
    found = dict()
    end = dict()
    for p_idx in range(len(allspikes)):
        p = allspikes[p_idx]
        mask_id = ustart[p]
        if mask_id in found:
            # we've already used this mask
            clusters[p] = found[mask_id]
        else:
            if cur_cluster<num_clusters:
                # use a new mask
                found[mask_id] = cur_cluster
                end[mask_id] = uend[p]
                clusters[p] = cur_cluster
                cur_cluster += 1
                candidate_ids.push_back(mask_id)
                candidate_ends.push_back(uend[p])
            else:
                # we have to find the closest mask (this is the computationally intensive bit!)
                best_distance = num_features+1
                best_ids.clear()
                for c_idx in range(candidate_ids.size()):
                    candidate_id = candidate_ids[c_idx]
                    candidate_end = candidate_ends[c_idx]
                    d = mask_difference(unmasked, ustart[p], uend[p], candidate_id, candidate_end)
                    if d==best_distance:
                        best_ids.push_back(candidate_id)
                    elif d<best_distance:
                        best_distance = d
                        best_ids.clear()
                        best_ids.push_back(candidate_id)
                best_id = best_ids[randint(best_ids.size())]
                clusters[p] = found[best_id]
                found[mask_id] = found[best_id]
                end[mask_id] = end[best_id]
