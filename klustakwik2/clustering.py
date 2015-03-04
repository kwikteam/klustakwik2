from numpy import *
from itertools import izip

from .mask_starts import mask_starts

import time

__all__ = ['KK']

class KK(object):
    def __init__(self, data,
                 prior_point=1,
                 mua_point=1,
                 noise_point=1,
                 ):
        
        self.data = data
        
        self.prior_point = prior_point
        self.mua_point = mua_point
        self.noise_point = noise_point
    
    def cluster(self, num_starting_clusters):
        start = time.time()
        self.clusters = mask_starts(self.data, num_starting_clusters)
        print 'Mask starts:', time.time()-start
        self.CEM()
    
    def CEM(self, recurse=True):
        start = time.time()
        self.M_step()
        print 'M_step:', time.time()-start
        start = time.time()
        self.E_step()
        print 'E_step:', time.time()-start
        self.C_step()
        self.compute_cluster_penalties()
        if recurse:
            self.consider_deletion()
        self.compute_score()
    
    def M_step(self):
        # eliminate any clusters with 0 members
        self.reindex_clusters()
        num_cluster_members = self.num_cluster_members
        num_clusters_alive = len(self.num_cluster_members)
        num_features = self.num_features
        
        # Normalize by total number of points to give class weight
        denom = float(self.num_points+self.noise_point+self.mua_point+
                                      self.prior_point*(num_clusters_alive-2))
        weight = (num_cluster_members+self.prior_point)/denom
        # different calculation for clusters 0 and 1
        weight[0] = (num_cluster_members[0]+self.noise_point)/denom
        weight[1] = (num_cluster_members[1]+self.mua_point)/denom
        
        # Compute means for each cluster
        cluster_mean = zeros((num_clusters_alive, num_features))
        F = zeros(num_features)
        for cluster, spike in izip(self.clusters, self.data.spikes):
            features = spike.features
            F[:] = self.data.noise_mean
            F[features.inds] = features.vals
            cluster_mean[cluster, :] += F
        for cluster in xrange(num_clusters_alive):
            prior = 0
            if cluster==1:
                prior = self.mua_point
            elif cluster>=2:
                prior = self.prior_point
            cluster_mean[cluster, :] += prior*self.data.noise_mean
            cluster_mean[cluster, :] /= num_cluster_members[cluster]+prior
        
    
    def E_step(self):
        pass
    
    def C_step(self):
        pass
    
    def compute_cluster_penalties(self):
        pass
    
    def consider_deletion(self):
        pass
    
    def compute_score(self):
        pass

    @property
    def num_cluster_members(self):
        return bincount(self.clusters)
    
    @property
    def num_points(self):
        return len(self.data.spikes)
    
    @property
    def num_features(self):
        return self.data.num_features
    
    def reindex_clusters(self):
        '''
        Remove any clusters with 0 members (except for clusters 0 and 1)
        '''
        num_cluster_members = self.num_cluster_members
        I = num_cluster_members>0
        I[0:2] = True # we keep clusters 0 and 1
        remapping = hstack((0, cumsum(I)))[:-1]
        self.clusters = remapping[self.clusters]
