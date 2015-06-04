#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: infer_types=True

import numpy
cimport numpy

from cython cimport integral, floating

cpdef doaccum(integral[:] spikes,
              integral[:] unmasked,
              integral[:] ustart,
              integral[:] uend,
              floating[:] masks,
              integral[:] vstart,
              integral[:] vend,
              floating[:] cluster_mask_sum,
             ):
    cdef integral pp, p, c, num_unmasked, i, j, k
    for pp in range(len(spikes)):
        p = spikes[pp]
        num_unmasked = uend[p]-ustart[p]
        for i in range(num_unmasked):
            j = unmasked[ustart[p]+i]
            k = vstart[p]+i
            cluster_mask_sum[j] += masks[k]
