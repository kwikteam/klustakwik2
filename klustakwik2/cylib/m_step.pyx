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
    features = data.features
    vstart = data.values_start
    vend = data.values_end

    do_mean_accum(clusters, unmasked, ustart, uend, features, vstart, vend, cluster_mean, num_added)
        
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


cdef do_mean_accum(
             numpy.ndarray[int, ndim=1] clusters,
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


def compute_covariance_matrix(kk, cluster):
    num_clusters = len(kk.num_cluster_members)
    num_features = kk.num_features

    cov = kk.covariance[cluster]
    block = cov.block
    
    spikes_in_cluster = kk.spikes_in_cluster
    spikes_in_cluster_offset = kk.spikes_in_cluster_offset
    spike_indices = spikes_in_cluster[spikes_in_cluster_offset[cluster]:spikes_in_cluster_offset[cluster+1]]

    data = kk.data
    clusters = kk.clusters
    unmasked = data.unmasked
    ustart = data.unmasked_start
    uend = data.unmasked_end
    features = data.features
    vstart = data.values_start
    vend = data.values_end
    
    f2m = numpy.zeros(num_features)
    ct = numpy.zeros(num_features)
    
    if cluster==1:
        do_var_accum_mua(spike_indices, kk.cluster_mean[cluster, :], cov.unmasked, block,
                         unmasked, ustart, uend, features, vstart, vend,
                         f2m, ct, data.correction_terms, num_features,
                         )
    else:
        do_var_accum(spike_indices, kk.cluster_mean[cluster, :], cov.unmasked, block,
                     unmasked, ustart, uend, features, vstart, vend,
                     f2m, ct, data.correction_terms, num_features,
                     )
    

def do_var_accum(
            numpy.ndarray[int, ndim=1] spike_indices,
            numpy.ndarray[double, ndim=1] cluster_mean,
            numpy.ndarray[int, ndim=1] cov_unmasked,
            numpy.ndarray[double, ndim=2] block,
            numpy.ndarray[int, ndim=1] unmasked,
            numpy.ndarray[int, ndim=1] ustart,
            numpy.ndarray[int, ndim=1] uend,
            numpy.ndarray[double, ndim=1] features,
            numpy.ndarray[int, ndim=1] vstart,
            numpy.ndarray[int, ndim=1] vend,
            numpy.ndarray[double, ndim=1] f2m,
            numpy.ndarray[double, ndim=1] ct,
            numpy.ndarray[double, ndim=1] correction_term,
            int num_features,
            ):
    cdef int i, j, k, ip
    cdef int num_unmasked
    cdef int p
    cdef int num_spikes = len(spike_indices)
    for ii in range(num_spikes):
        p = spike_indices[ii]
        for i in range(num_features):
            f2m[i] = 0
        num_unmasked = uend[p]-ustart[p]
        for i in range(num_unmasked):
            j = unmasked[ustart[p]+i]
            k = vstart[p]+i
            f2m[j] = features[k]-cluster_mean[j]
            
        num_unmasked = len(cov_unmasked)
        for i in range(num_unmasked):
            for j in range(num_unmasked):
                block[i, j] += f2m[cov_unmasked[i]]*f2m[cov_unmasked[j]]
        
        for i in range(num_features):
            ct[i] = 0
        num_unmasked = uend[p]-ustart[p]
        for i in range(num_unmasked):
            ct[unmasked[ustart[p]+i]] = correction_term[vstart[p]+i]
        
        num_unmasked = len(cov_unmasked)
        for i in range(num_unmasked):
            block[i, i] += ct[cov_unmasked[i]]

def do_var_accum_mua(
            numpy.ndarray[int, ndim=1] spike_indices,
            numpy.ndarray[double, ndim=1] cluster_mean,
            numpy.ndarray[int, ndim=1] cov_unmasked,
            numpy.ndarray[double, ndim=2] block,
            numpy.ndarray[int, ndim=1] unmasked,
            numpy.ndarray[int, ndim=1] ustart,
            numpy.ndarray[int, ndim=1] uend,
            numpy.ndarray[double, ndim=1] features,
            numpy.ndarray[int, ndim=1] vstart,
            numpy.ndarray[int, ndim=1] vend,
            numpy.ndarray[double, ndim=1] f2m,
            numpy.ndarray[double, ndim=1] ct,
            numpy.ndarray[double, ndim=1] correction_term,
            int num_features,
            ):
    cdef int i, j, k, ip
    cdef int num_unmasked
    cdef int p
    cdef int num_spikes = len(spike_indices)
    for ii in range(num_spikes):
        p = spike_indices[ii]
        for i in range(num_features):
            f2m[i] = 0
        num_unmasked = uend[p]-ustart[p]
        for i in range(num_unmasked):
            j = unmasked[ustart[p]+i]
            k = vstart[p]+i
            f2m[j] = features[k]-cluster_mean[j]
            
        num_unmasked = len(cov_unmasked)
        for i in range(num_unmasked):
            block[i, i] += f2m[cov_unmasked[i]]*f2m[cov_unmasked[i]]
        
        for i in range(num_features):
            ct[i] = 0
        num_unmasked = uend[p]-ustart[p]
        for i in range(num_unmasked):
            ct[unmasked[ustart[p]+i]] = correction_term[vstart[p]+i]
        
        num_unmasked = len(cov_unmasked)
        for i in range(num_unmasked):
            block[i, i] += ct[cov_unmasked[i]]
        