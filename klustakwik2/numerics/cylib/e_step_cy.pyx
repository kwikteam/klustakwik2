#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: infer_types=True
# distutils: language = c++

import numpy
cimport numpy

from cython cimport integral, floating

from cython.parallel import prange, threadid

from libc.math cimport log

import math
cdef double pi = math.pi

# This function assumes a lower triangular matrix
cpdef trisolve(
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
            s -= chol_block[ii, jj]*root[j]
        root[i] = s/chol_block[ii, ii]
    for ii in range(num_masked):
        i = chol_masked[ii]
        root[i] = x[i]/chol_diagonal[ii]


cpdef do_log_p_assign_computations(
            floating[:] noise_mean,
            floating[:] noise_variance,
            floating[:] cluster_mean,
            floating[:] correction_terms,
            floating[:] log_p_best,
            floating[:] log_p_second_best,
            integral[:] clusters,
            integral[:] clusters_second_best,
            integral[:] old_clusters,
            char full_step,
            floating[:] inv_cov_diag,
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
            integral n_cpu,
            floating[:] cluster_log_p,
            integral[:] candidates,
            char only_evaluate_current_clusters,
            ):
    cdef integral i, j, ii, jj, p, pp, num_unmasked
    cdef floating mahal, log_p, cur_log_p_best, cur_log_p_second_best, s
    cdef integral chol_num_unmasked = len(chol_unmasked)
    cdef integral chol_num_masked = len(chol_masked)
    cdef integral thread_idx
    cdef integral vo # vector offset for root and f2cm

    for pp in prange(num_spikes, nogil=True, num_threads=n_cpu):
        thread_idx = threadid()
        vo = num_features*thread_idx
        
        if full_step and not only_evaluate_current_clusters:
            p = pp
        else:
            p = candidates[pp]
        
        for i in range(num_features):
            f2cm[i+vo] = noise_mean[i]-cluster_mean[i]
        num_unmasked = uend[p]-ustart[p]
        for ii in range(num_unmasked):
            i = unmasked[ustart[p]+ii]
            j = vstart[p]+ii
            f2cm[i+vo] = features[j]-cluster_mean[i]
        
        # TriSolve step, inlined for Cython OpenMP support
        # This code is adapted from the function above, see there for details
        for ii in range(chol_num_unmasked):
            i = chol_unmasked[ii]
            s = f2cm[i+vo]
            for jj in range(ii):
                j = chol_unmasked[jj]
                s = s-chol_block[ii, jj]*root[j+vo]
            root[i+vo] = s/chol_block[ii, ii]
        for ii in range(chol_num_masked):
            i = chol_masked[ii]
            root[i+vo] = f2cm[i+vo]/chol_diagonal[ii]
        
        mahal = 0
        for i in range(num_features):
            mahal = mahal+root[i+vo]*root[i+vo]
        num_unmasked = uend[p]-ustart[p]
#         for ii in range(num_unmasked):
#             i = unmasked[ustart[p]+ii]
#             mahal = mahal+inv_cov_diag[i]*correction_terms[vstart[p]+ii]
        # the above misses out the contribution of the masked terms. To compute this, we note that
        # correction_terms = noise_variance[i] for masked terms. We can also compute inv_cov_diag but rather
        # than do this, we just mimic the computation above to be more sure of getting it right.
        # However! We don't have access to the set of masked points so we have to invert the list of unmasked points
        jj = -1
        j = -1
        for i in range(num_features):
            if i>j and jj+1<num_unmasked:
                jj = jj+1
                j = unmasked[ustart[p]+jj]
            if i==j:
                mahal = mahal+inv_cov_diag[i]*correction_terms[vstart[p]+jj]
            else:
                mahal = mahal+inv_cov_diag[i]*noise_variance[i]
            
        log_p = mahal/2.0+log_addition
        
        cluster_log_p[p] = log_p

    if only_evaluate_current_clusters:
        return

    for pp in range(num_spikes):

        if full_step:
            p = pp
        else:
            p = candidates[pp]
        
        log_p = cluster_log_p[p]
        
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
