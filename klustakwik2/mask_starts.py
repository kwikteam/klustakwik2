from numpy import *
from .logger import log_message
from random import shuffle
from numpy.random import randint
from .numerics import mask_start_clusters

__all__ = ['mask_starts']

def mask_starts(data, num_clusters, num_special_clusters):
    log_message('info', 'Using mask starts with %d clusters, data has '
                        '%d unique masks' % (num_clusters, data.num_masks))
    if num_clusters<=num_special_clusters:
        raise ValueError("Number of starting clusters must be at least 3 to accomodate noise and mua clusters.")
    num_clusters -= num_special_clusters # special clusters are not generated, we add them at the end
    if data.num_masks<num_clusters:
        log_message('warning', ('Not enough masks (%d) for specified number of starting '
                                'clusters (%d)') % (data.num_masks, num_clusters+2))
        num_clusters = data.num_masks
         
    return mask_start_clusters(data, num_clusters)+num_special_clusters
