#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: infer_types=True

import numpy
cimport numpy

from libc.math cimport log

import math
cdef double pi = math.pi

# TODO: check that the memory layout assumptions below are correct
cdef trisolve(
            double *chol_block,
            double *chol_diagonal,
            int *chol_masked,
            int *chol_unmasked,
            int num_masked, int num_unmasked,
            double *x,
            double *root,
            ):
    cdef int ii, jj, i, j
    cdef double s
    for ii in range(num_unmasked):
        i = chol_unmasked[ii]
        s = x[i]
        for jj in range(ii):
            j = chol_unmasked[jj]
            s += chol_block[ii*num_unmasked+jj]
        root[i] = -s/chol_block[ii*num_unmasked+ii]
    for ii in range(num_masked):
        i = chol_masked[ii]
        root[i] = -x[i]/chol_diagonal[ii]


def compute_log_p_and_assign(kk, cluster, inv_cov_diag, log_root_det, chol):
    num_clusters = len(kk.num_cluster_members)
    num_features = kk.num_features
    num_spikes = kk.num_spikes

    weight = kk.weight
    log_addition = log_root_det-log(weight[cluster])+0.5*log(2*pi)*num_features

    data = kk.data
    unmasked = data.unmasked
    ustart = data.unmasked_start
    uend = data.unmasked_end
    features = data.features
    vstart = data.values_start
    vend = data.values_end
    
    f2cm = numpy.zeros(num_features)
    root = numpy.zeros(num_features)
    
    log_p_best = kk.log_p_best
    log_p_second_best = kk.log_p_second_best
    noise_mean = data.noise_mean
    cluster_mean = kk.cluster_mean
    correction_terms = data.correction_terms
    
    clusters = kk.clusters
    clusters_second_best = kk.clusters_second_best
    old_clusters = kk.old_clusters
    full_step = kk.full_step

    num_skipped = do_log_p_assign_computations(
                                  noise_mean, cluster_mean, correction_terms,
                                  log_p_best, log_p_second_best,
                                  clusters, clusters_second_best, old_clusters,
                                  full_step,
                                  inv_cov_diag, weight,
                                  unmasked, ustart, uend, features, vstart, vend,
                                  root, f2cm, num_features, num_spikes, log_addition, cluster,
                                  chol.block, chol.diagonal, chol.masked, chol.unmasked,
                                  )
    
    return num_skipped


cdef int do_log_p_assign_computations(
            numpy.ndarray[double, ndim=1] noise_mean,
            numpy.ndarray[double, ndim=2] cluster_mean,
            numpy.ndarray[double, ndim=1] correction_terms,
            numpy.ndarray[double, ndim=1] log_p_best,
            numpy.ndarray[double, ndim=1] log_p_second_best,
            numpy.ndarray[int, ndim=1] clusters,
            numpy.ndarray[int, ndim=1] clusters_second_best,
            numpy.ndarray[int, ndim=1] old_clusters,
            char full_step,
            numpy.ndarray[double, ndim=1] inv_cov_diag,
            numpy.ndarray[double, ndim=1] weight,
            numpy.ndarray[int, ndim=1] unmasked,
            numpy.ndarray[int, ndim=1] ustart,
            numpy.ndarray[int, ndim=1] uend,
            numpy.ndarray[double, ndim=1] features,
            numpy.ndarray[int, ndim=1] vstart,
            numpy.ndarray[int, ndim=1] vend,
            numpy.ndarray[double, ndim=1] root,
            numpy.ndarray[double, ndim=1] f2cm,
            int num_features, int num_spikes,
            double log_addition,
            int cluster,
            numpy.ndarray[double, ndim=2] chol_block,
            numpy.ndarray[double, ndim=1] chol_diagonal,
            numpy.ndarray[int, ndim=1] chol_masked,
            numpy.ndarray[int, ndim=1] chol_unmasked,
            ):
    cdef int i, j, ii, p, num_unmasked
    cdef double mahal, log_p, cur_log_p_best, cur_log_p_second_best
    cdef double * chol_block_ptr = &(chol_block[0, 0])
    cdef double * chol_diagonal_ptr = &(chol_diagonal[0])
    cdef int * chol_masked_ptr = &(chol_masked[0])
    cdef int * chol_unmasked_ptr = &(chol_unmasked[0])
    cdef int chol_num_unmasked = len(chol_unmasked)
    cdef int chol_num_masked = len(chol_masked)
    cdef double * f2cm_ptr = &(f2cm[0])
    cdef double * root_ptr = &(root[0])
    cdef int num_skipped = 0
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
        
        trisolve(chol_block_ptr, chol_diagonal_ptr,
                 chol_masked_ptr, chol_unmasked_ptr,
                 chol_num_masked, chol_num_unmasked,
                 f2cm_ptr, root_ptr)
        
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
