#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: infer_types=True

import numpy
cimport numpy

from cython cimport integral, floating

cpdef do_compute_cluster_mean(
             integral[:] spikes,
             integral[:] unmasked,
             integral[:] ustart,
             integral[:] uend,
             floating[:] features,
             integral[:] vstart,
             integral[:] vend,
             floating[:] cluster_mean,
             integral[:] num_added,
             floating[:] noise_mean,
             floating prior,
             ):
    cdef integral pp, p, c, j, k, num_unmasked
    cdef integral num_features = cluster_mean.shape[0]
    cdef integral num_cluster_members
    num_cluster_members = len(spikes)
    for pp in range(num_cluster_members):
        p = spikes[pp]
        num_unmasked = uend[p]-ustart[p]
        for i in range(num_unmasked):
            j = unmasked[ustart[p]+i]
            k = vstart[p]+i
            cluster_mean[j] += features[k]
            num_added[j] += 1
    if num_cluster_members==0:
        return
    for i in range(num_features):
        cluster_mean[i] += noise_mean[i]*(num_cluster_members-num_added[i]) 
        cluster_mean[i] += prior*noise_mean[i]
        cluster_mean[i] /= num_cluster_members+prior


cpdef do_var_accum(
            integral[:] spike_indices,
            floating[:] cluster_mean,
            floating[:] noise_mean,
            floating[:] noise_variance,
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
            f2m[i] = noise_mean[i]-cluster_mean[i]
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
            ct[i] = noise_variance[i]
        num_unmasked = uend[p]-ustart[p]
        for i in range(num_unmasked):
            ct[unmasked[ustart[p]+i]] = correction_term[vstart[p]+i]
        
        num_unmasked = len(cov_unmasked)
        for i in range(num_unmasked):
            block[i, i] += ct[cov_unmasked[i]]


cpdef do_var_accum_mua(
            integral[:] spike_indices,
            floating[:] cluster_mean,
            floating[:] noise_mean,
            floating[:] noise_variance,
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
            f2m[i] = noise_mean[i]-cluster_mean[i]
        num_unmasked = uend[p]-ustart[p]
        for i in range(num_unmasked):
            j = unmasked[ustart[p]+i]
            k = vstart[p]+i
            f2m[j] = features[k]-cluster_mean[j]
            
        num_unmasked = len(cov_unmasked)
        for i in range(num_unmasked):
            block[i, i] += f2m[cov_unmasked[i]]*f2m[cov_unmasked[i]]
        
        for i in range(num_features):
            ct[i] = noise_variance[i]
        num_unmasked = uend[p]-ustart[p]
        for i in range(num_unmasked):
            ct[unmasked[ustart[p]+i]] = correction_term[vstart[p]+i]
        
        num_unmasked = len(cov_unmasked)
        for i in range(num_unmasked):
            block[i, i] += ct[cov_unmasked[i]]
        