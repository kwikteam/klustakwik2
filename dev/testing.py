from klustakwik2 import *
from pylab import *
import time
import cPickle as pickle
import os

fname, shank = '../temp/testsmallish', 4

log_to_file(fname+'.klg', 'debug')
log_suppress_hierarchy('klustakwik', inclusive=False)

if os.path.exists(fname+'.pickle'):
    start_time = time.time()
    data = pickle.load(open(fname+'.pickle', 'rb'))
    print 'load from pickle:', time.time()-start_time
else:
    start_time = time.time()
    raw_data = loat_fet_fmask_to_raw(fname, shank)
    print 'load_fet_fmask_to_raw:', time.time()-start_time
    start_time = time.time()
    data = raw_data.to_sparse_data()
    print 'raw_data.to_sparse_data():', time.time()-start_time
    pickle.dump(data, open(fname+'.pickle', 'wb'), -1)
    
print 'Number of spikes:', data.num_spikes
print 'Number of unique masks:', data.num_masks

kk = KK(data)

if os.path.exists(fname+'.clu.pickle'):
    print 'Loading clusters from file'
    clusters = pickle.load(open(fname+'.clu.pickle', 'rb'))
else:
    print 'Generating clusters'
    kk.cluster(100)
    clusters = kk.clusters
    pickle.dump(clusters, open(fname+'.clu.pickle', 'wb'), -1)

kk.clusters = clusters
kk.reindex_clusters()

for cluster in xrange(kk.num_clusters_alive):
    if cluster % 4 == 0:
        figure()
    maskimg = []
    for spike in kk.get_spikes_in_cluster(cluster):
        row = zeros(kk.num_features)
        unmasked = data.unmasked[data.unmasked_start[spike]:data.unmasked_end[spike]]
        row[unmasked] = data.masks[data.values_start[spike]:data.values_end[spike]]
        maskimg.append(row)
    if len(maskimg)==0:
        continue
    maskimg = array(maskimg)
    print maskimg.shape
    subplot(2, 2, cluster%4 + 1)
    imshow(maskimg, origin='lower left', aspect='auto', interpolation='nearest')
    gray()
    title(cluster)
show()
