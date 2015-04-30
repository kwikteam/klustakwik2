#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: infer_types=True

import numpy
cimport numpy

from cython cimport integral, floating

cpdef do_compute_cluster_means(
             integral[:] clusters,
             integral[:] unmasked,
             integral[:] ustart,
             integral[:] uend,
             floating[:] features,
             integral[:] vstart,
             integral[:] vend,
             floating[:, :] cluster_mean,
             integral[:, :] num_added,
             integral[:] num_cluster_members,
             floating[:] noise_mean,
             floating mua_point, floating prior_point,
             integral mua_cluster, integral num_special_clusters,
             ):
    cdef integral p, c, j, k, num_unmasked
    cdef integral num_clusters = num_cluster_members.shape[0]
    cdef integral num_features = cluster_mean.shape[1]
    cdef floating prior
    for p in range(len(clusters)):
        c = clusters[p]
        num_unmasked = uend[p]-ustart[p]
        for i in range(num_unmasked):
            j = unmasked[ustart[p]+i]
            k = vstart[p]+i
            cluster_mean[c, j] += features[k]
            num_added[c, j] += 1
    for cluster in range(num_clusters):
        if num_cluster_members[cluster]==0:
            continue
        prior = 0
        if cluster==mua_cluster:
            prior = mua_point
        elif cluster>=num_special_clusters:
            prior = prior_point
        for i in range(num_features):
            cluster_mean[cluster, i] += noise_mean[i]*(num_cluster_members[cluster]-num_added[cluster, i]) 
            cluster_mean[cluster, i] += prior*noise_mean[i]
            cluster_mean[cluster, i] /= num_cluster_members[cluster]+prior


cpdef do_var_accum(
            integral[:] spike_indices,
            floating[:] cluster_mean,
            integral[:] cov_unmasked,
            floating[:, :] block,
            integral[:] unmasked,
            integral[:] ustart,
            integral[:] uend,
            floating[:] features,
            integral[:] vstart,
            integral[:] vend,
            floating[:] f2m,
            floating[:] ct,
            floating[:] correction_term,
            integral num_features,
            ):
    cdef integral i, j, k, ip, num_unmasked, p
    cdef integral num_spikes = len(spike_indices)
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


cpdef do_var_accum_mua(
            integral[:] spike_indices,
            floating[:] cluster_mean,
            integral[:] cov_unmasked,
            floating[:, :] block,
            integral[:] unmasked,
            integral[:] ustart,
            integral[:] uend,
            floating[:] features,
            integral[:] vstart,
            integral[:] vend,
            floating[:] f2m,
            floating[:] ct,
            floating[:] correction_term,
            integral num_features,
            ):
    cdef integral i, j, k, ip, num_unmasked, p
    cdef integral num_spikes = len(spike_indices)
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
        