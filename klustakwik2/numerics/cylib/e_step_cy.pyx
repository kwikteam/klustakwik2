#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: infer_types=True
# distutils: language = c++

import numpy
cimport numpy

from cython cimport integral, floating

from libc.math cimport log

import math
cdef double pi = math.pi

# TODO: check that the memory layout assumptions below are correct
cdef trisolve(
            floating[:, :] chol_block,
            floating[:] chol_diagonal,
            integral[:] chol_masked,
            integral[:] chol_unmasked,
            integral num_masked, integral num_unmasked,
            floating[:] x,
            floating[:] root,
            ):
    cdef integral ii, jj, i, j
    cdef floating s
    for ii in range(num_unmasked):
        i = chol_unmasked[ii]
        s = x[i]
        for jj in range(ii):
            j = chol_unmasked[jj]
            s += chol_block[ii, jj]
        root[i] = -s/chol_block[ii, ii]
    for ii in range(num_masked):
        i = chol_masked[ii]
        root[i] = -x[i]/chol_diagonal[ii]


cpdef int do_log_p_assign_computations(
            floating[:] noise_mean,
            floating[:, :] cluster_mean,
            floating[:] correction_terms,
            floating[:] log_p_best,
            floating[:] log_p_second_best,
            integral[:] clusters,
            integral[:] clusters_second_best,
            integral[:] old_clusters,
            char full_step,
            floating[:] inv_cov_diag,
            floating[:] weight,
            integral[:] unmasked,
            integral[:] ustart,
            integral[:] uend,
            floating[:] features,
            integral[:] vstart,
            integral[:] vend,
            floating[:] root,
            floating[:] f2cm,
            integral num_features, integral num_spikes,
            floating log_addition,
            integral cluster,
            floating[:, :] chol_block,
            floating[:] chol_diagonal,
            integral[:] chol_masked,
            integral[:] chol_unmasked,
            ):
    cdef integral i, j, ii, p, num_unmasked
    cdef floating mahal, log_p, cur_log_p_best, cur_log_p_second_best
    cdef integral chol_num_unmasked = len(chol_unmasked)
    cdef integral chol_num_masked = len(chol_masked)
    cdef integral num_skipped = 0

    for p in range(num_spikes):
        # to save time, only recalculate if the last one was close
        # TODO: replace this with something that doesn't require keeping all of log_p 2D array
#         if not full_step and clusters[p]==old_clusters[p] and log_p[cluster, p]-log_p[clusters[p], p]>dist_thresh:
#             num_skipped += 1
#             continue
        for i in range(num_features):
            f2cm[i] = noise_mean[i]-cluster_mean[cluster, i]
        num_unmasked = uend[p]-ustart[p]
        for ii in range(num_unmasked):
            i = unmasked[ustart[p]+ii]
            j = vstart[p]+ii
            f2cm[i] = features[j]-cluster_mean[cluster, i]
        
        trisolve(chol_block, chol_diagonal,
                 chol_masked, chol_unmasked,
                 chol_num_masked, chol_num_unmasked,
                 f2cm, root)
        
        mahal = 0
        for i in range(num_features):
            mahal += root[i]*root[i]
        num_unmasked = uend[p]-ustart[p]
        for ii in range(num_unmasked):
            i = unmasked[ustart[p]+ii]
            mahal += inv_cov_diag[i]*correction_terms[vstart[p]+ii]
            
        log_p = mahal/2.0+log_addition
        
        cur_log_p_best = log_p_best[p]
        cur_log_p_second_best = log_p_second_best[p]
        
        if log_p<cur_log_p_best:
            log_p_second_best[p] = cur_log_p_best
            clusters_second_best[p] = clusters[p]
            log_p_best[p] = log_p
            clusters[p] = cluster
        elif log_p<cur_log_p_second_best:
            log_p_second_best[p] = log_p
            clusters_second_best[p] = cluster
        
    return num_skipped
