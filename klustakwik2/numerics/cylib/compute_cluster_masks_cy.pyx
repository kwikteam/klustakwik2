#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: infer_types=True

import numpy
cimport numpy

from cython cimport integral, floating

cpdef doaccum(integral[:] clusters,
              integral[:] unmasked,
              integral[:] ustart,
              integral[:] uend,
              floating[:] masks,
              integral[:] vstart,
              integral[:] vend,
              floating[:, :] cluster_mask_sum,
              integral num_special_clusters,
             ):
    cdef integral p, c, num_unmasked
    for p in range(len(clusters)):
        c = clusters[p]
        if c<num_special_clusters:
            continue
        num_unmasked = uend[p]-ustart[p]
        for i in range(num_unmasked):
            cluster_mask_sum[c, unmasked[ustart[p]+i]] += masks[vstart[p]+i]
