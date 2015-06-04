from .compute_cluster_masks_cy import doaccum

__all__ = ['accumulate_cluster_mask_sum']

def accumulate_cluster_mask_sum(kk, cluster_mask_sum, spikes):
    data = kk.data
    doaccum(spikes, data.unmasked, data.unmasked_start, data.unmasked_end,
            data.masks, data.values_start, data.values_end, cluster_mask_sum)
