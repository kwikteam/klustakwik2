from numpy import *
from numpy.linalg import LinAlgError
from itertools import izip

from .mask_starts import mask_starts
from .linear_algebra import BlockPlusDiagonalMatrix

from .cylib.compute_cluster_masks import accumulate_cluster_mask_sum
from .cylib.m_step import compute_cluster_means, compute_covariance_matrix
from .cylib.e_step import compute_log_p

import time

__all__ = ['KK']

def get_diagonal(x):
    '''
    Return a writeable view of the diagonal of x
    '''
    return x.reshape(-1)[::x.shape[0]+1]


class KK(object):
    def __init__(self, data,
                 prior_point=1,
                 mua_point=1,
                 noise_point=1,
                 points_for_cluster_mask=10,
                 dist_thresh=log(1000),
                 penalty_k=0.0,
                 penalty_k_log_n=1.0,
                 ):
        
        self.data = data
        
        self.prior_point = prior_point
        self.mua_point = mua_point
        self.noise_point = noise_point
        self.points_for_cluster_mask = points_for_cluster_mask
        self.dist_thresh = dist_thresh
        self.penalty_k = penalty_k
        self.penalty_k_log_n = penalty_k_log_n
    
    def cluster(self, num_starting_clusters):
        start = time.time()
        self.clusters = mask_starts(self.data, num_starting_clusters)
        self.old_clusters = -1*ones(len(self.clusters), dtype=int)
        print 'Mask starts:', time.time()-start
        start = time.time()
        self.reindex_clusters()
        print 'Reindex clusters:', time.time()-start
        self.CEM()
    
    def CEM(self, recurse=True):
        
        self.full_step = True
        
        start = time.time()
        self.M_step()
        print 'M_step:', time.time()-start
        start = time.time()
        self.E_step()
        print 'E_step:', time.time()-start
        start = time.time()
        self.C_step()
        print 'C_step:', time.time()-start
        start = time.time()
        self.compute_cluster_penalties()
        print 'compute_cluster_penalties:', time.time()-start
        if recurse:
            start = time.time()
            self.consider_deletion()
            print 'consider_deletion:', time.time()-start
        self.compute_score()
    
    def M_step(self):
        # eliminate any clusters with 0 members, compute the list of spikes
        # in each cluster, compute the cluster masks and allocate space for
        # covariance matrices
        start = time.time()
        self.reindex_clusters()
        self.compute_cluster_masks()
        print 'compute_cluster_masks:', time.time()-start  

        num_cluster_members = self.num_cluster_members
        num_clusters = len(self.num_cluster_members)
        num_features = self.num_features
        
        # Normalize by total number of points to give class weight
        denom = float(self.num_spikes+self.noise_point+self.mua_point+
                                      self.prior_point*(num_clusters-2))
        self.weight = weight = (num_cluster_members+self.prior_point)/denom
        # different calculation for clusters 0 and 1
        weight[0] = (num_cluster_members[0]+self.noise_point)/denom
        weight[1] = (num_cluster_members[1]+self.mua_point)/denom
        
        # Compute means for each cluster
        # Note that we do this densely at the moment, might want to switch
        # that to a sparse structure later
        start = time.time()
        self.cluster_mean = compute_cluster_means(self)
        print 'Compute cluster_mean:', time.time()-start  
                
        # Compute covariance matrices
        start = time.time()
        for cluster in xrange(2, num_clusters):
            compute_covariance_matrix(self, cluster)
            cov = self.covariance[cluster]
            block_diagonal = get_diagonal(cov.block)
            
            # Add prior
            block_diagonal[:] += self.prior_point*self.data.noise_variance[cov.unmasked]
            cov.diagonal[:] += self.prior_point*self.data.noise_variance[cov.masked]
            
            # Normalise
            factor = 1.0/(num_cluster_members[cluster]+self.prior_point-1)
            cov.block *= factor
            cov.diagonal *= factor
                        
        print 'Compute covariance matrices:', time.time()-start
            
    def E_step(self):
        num_spikes = self.num_spikes
        num_clusters = len(self.num_cluster_members)
        num_features = self.num_features
        
        weight = self.weight

        self.log_p = log_p = zeros((num_clusters, num_spikes))
        num_skipped = 0
        
        # start with cluster 0 - uniform distribution over space
        # because we have normalized all dims to 0...1, density will be 1.
        log_p[0, :] = -log(weight[0])
        
        clusters_to_kill = []
        
        for cluster in xrange(1, num_clusters):
            cov = self.covariance[cluster]
            try:
                chol = cov.cholesky()
            except LinAlgError:
                clusters_to_kill.append(cluster)
                continue

            # LogRootDet is given by log of product of diagonal elements
            log_root_det = sum(log(chol.diagonal))+sum(log(chol.block.diagonal()))

            # compute diagonal of inverse of cov matrix
            inv_cov_diag = zeros(num_features)
            basis_vector = zeros(num_features)
            for i in xrange(num_features):
                basis_vector[i] = 1.0
                root = chol.trisolve(basis_vector)
                inv_cov_diag[i] = sum(root**2)
                basis_vector[i] = 0.0
                
            compute_log_p(self, cluster, inv_cov_diag, log_root_det, chol)
        
    def C_step(self, allow_assign_to_noise=True):
        if not allow_assign_to_noise:
            cstart = 2
        else:
            cstart = 0
        log_p = self.log_p[cstart:, :]
        self.clusters = argmin(log_p, axis=0)+cstart
        R = arange(len(self.clusters))
        best_p = self.log_p[self.clusters, R]
        self.log_p[self.clusters, R] = inf
        self.clusters_second_best = argmin(log_p, axis=0)+cstart
        self.log_p[self.clusters, R] = best_p
        # We have changed clusters so now we need to reindex
        # no, we shouldn't reindex because if we do that invalidates log_p
        #self.reindex_clusters()
    
    def compute_cluster_penalties(self):
        num_cluster_members = self.num_cluster_members
        num_clusters = len(self.num_cluster_members)
        self.cluster_penalty = cluster_penalty = zeros(num_clusters)
        sic = self.spikes_in_cluster
        sico = self.spikes_in_cluster_offset
        ustart = self.data.unmasked_start
        uend = self.data.unmasked_end
        penalty_k = self.penalty_k
        penalty_k_log_n = self.penalty_k_log_n
        for cluster in xrange(num_clusters):
            curspikes = sic[sico[cluster]:sico[cluster+1]]
            num_spikes = len(curspikes)
            if num_spikes>0:
                num_unmasked = uend[curspikes]-ustart[curspikes]
                num_params = sum(num_unmasked*(num_unmasked+1)/2+num_unmasked+1)
                mean_params = float(num_params)/num_spikes
                cluster_penalty[cluster] = penalty_k*mean_params*2+penalty_k_log_n*mean_params*log(mean_params)/2    
    
    def consider_deletion(self):
        num_cluster_members = self.num_cluster_members
        num_clusters = len(self.num_cluster_members)
        sic = self.spikes_in_cluster
        sico = self.spikes_in_cluster_offset
        log_p = self.log_p
        
        deletion_loss = zeros(num_clusters)
        I = arange(self.num_spikes)
        add.at(deletion_loss, self.clusters, log_p[self.clusters_second_best, I]-log_p[self.clusters, I])
        candidate_cluster = 2+argmin((deletion_loss-self.cluster_penalty)[2:])
        loss = deletion_loss[candidate_cluster]
        delta_pen = self.cluster_penalty[candidate_cluster]
        
        if loss<0:
            # delete this cluster
            # reassign points
            cursic = sic[sico[candidate_cluster]:sico[candidate_cluster+1]]
            self.clusters[cursic] = self.clusters_second_best[cursic]
            # recompute penalties
            self.compute_cluster_penalties()
        
        self.reindex_clusters() # this clobbers log_p
    
    def compute_score(self):
        pass

    @property
    def num_spikes(self):
        return self.data.num_spikes
    
    @property
    def num_features(self):
        return self.data.num_features
    
    def reindex_clusters(self):
        '''
        Remove any clusters with 0 members (except for clusters 0 and 1),
        and recompute the list of spikes in each cluster. After this function is
        run, you can use the attributes:
        
        - num_cluster_members (of length the number of clusters)
        - spikes_in_cluster, spikes_in_cluster_offset

        spikes_in_cluster[spikes_in_cluster_offset[c]:spikes_in_cluster_offset[c+1]] will be in the indices
        of all the spikes in cluster c. 
        '''
        num_cluster_members = array(bincount(self.clusters), dtype=int)
        I = num_cluster_members>0
        I[0:2] = True # we keep clusters 0 and 1
        remapping = hstack((0, cumsum(I)))[:-1]
        self.clusters = remapping[self.clusters]
        self.num_cluster_members = num_cluster_members = array(bincount(self.clusters), dtype=int)
        I = array(argsort(self.clusters), dtype=int)
        y = self.clusters[I]
        n = amax(y)+2
        J = searchsorted(y, arange(n))
        self.spikes_in_cluster = I
        self.spikes_in_cluster_offset = J
        
    def compute_cluster_masks(self):
        '''
        Computes the masked and unmasked indices for each cluster based on the
        masks for each point in that cluster. Allocates space for covariance
        matrices.
        '''
        num_clusters = len(self.num_cluster_members)
        num_features = self.num_features
        
        # Compute the sum of 
        cluster_mask_sum = zeros((num_clusters, num_features))
        cluster_mask_sum[:2, :] = -1 # ensure that clusters 0 and 1 are masked
        # Use efficient version
        accumulate_cluster_mask_sum(self, cluster_mask_sum)
        
        # Compute the masked and unmasked sets
        self.cluster_masked_features = []
        self.cluster_unmasked_features = []
        self.covariance = []
        for cluster in xrange(num_clusters):
            curmask = cluster_mask_sum[cluster, :]
            unmasked, = (curmask>=self.points_for_cluster_mask).nonzero()
            masked, = (curmask<self.points_for_cluster_mask).nonzero()
            unmasked = array(unmasked, dtype=int)
            masked = array(masked, dtype=int)
            self.cluster_masked_features.append(masked)
            self.cluster_unmasked_features.append(unmasked)
            self.covariance.append(BlockPlusDiagonalMatrix(masked, unmasked))
            