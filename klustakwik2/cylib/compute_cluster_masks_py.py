from .compute_cluster_masks import doaccum

def accumulate_cluster_mask_sum(kk, cluster_mask_sum):
    data = kk.data
    clusters = kk.clusters
    unmasked = data.unmasked
    ustart = data.unmasked_start
    uend = data.unmasked_end
    masks = data.masks
    vstart = data.values_start
    vend = data.values_end
    num_spikes = data.num_spikes
    doaccum(clusters, unmasked, ustart, uend, masks, vstart, vend, cluster_mask_sum)
