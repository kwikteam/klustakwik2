#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: infer_types=True

import numpy
cimport numpy

def compute_cluster_means(kk):
    num_clusters = len(kk.num_cluster_members)
    num_features = kk.num_features
    
    noise_mean = kk.data.noise_mean
    num_cluster_members = kk.num_cluster_members
    
    cluster_mean = numpy.zeros((num_clusters, num_features))
    num_added = numpy.zeros((num_clusters, num_features), dtype=int)

    data = kk.data
    clusters = kk.clusters
    unmasked = data.unmasked
    ustart = data.unmasked_start
    uend = data.unmasked_end
    masks = data.masks
    vstart = data.values_start
    vend = data.values_end

    doaccum(clusters, unmasked, ustart, uend, masks, vstart, vend, cluster_mean, num_added)
        
    for cluster in range(num_clusters):
        prior = 0
        if cluster==1:
            prior = kk.mua_point
        elif cluster>=2:
            prior = kk.prior_point
        cluster_mean[cluster, :] += noise_mean*(num_cluster_members[cluster]-num_added[cluster, :]) 
        cluster_mean[cluster, :] += prior*noise_mean
        cluster_mean[cluster, :] /= num_cluster_members[cluster]+prior
                
    return cluster_mean

cdef doaccum(numpy.ndarray[int, ndim=1] clusters,
             numpy.ndarray[int, ndim=1] unmasked,
             numpy.ndarray[int, ndim=1] ustart,
             numpy.ndarray[int, ndim=1] uend,
             numpy.ndarray[double, ndim=1] features,
             numpy.ndarray[int, ndim=1] vstart,
             numpy.ndarray[int, ndim=1] vend,
             numpy.ndarray[double, ndim=2] cluster_mean,
             numpy.ndarray[int, ndim=2] num_added,
             ):
    cdef int p
    cdef int c
    cdef int num_unmasked
    cdef int j, k
    for p in range(len(clusters)):
        c = clusters[p]
        num_unmasked = uend[p]-ustart[p]
        for i in range(num_unmasked):
            j = unmasked[ustart[p]+i]
            k = vstart[p]+i
            cluster_mean[c, j] += features[k]
            num_added[c, j] += 1
            