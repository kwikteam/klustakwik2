from pylab import *
from klustakwik2 import *
import cPickle as pickle
import os
from ipyparallel import Client

fname, shank = '../temp/testsmallish', 4
params = dict(
    max_iterations=1000,
    use_mua_cluster=False,
#             split_every=1, split_first=1, # for debugging splits
#             split_every=1000000, split_first=1000000, # disable splitting
#             points_for_cluster_mask=1e-100, # don't use reduced cluster masks
#             full_step_every=1,
#             always_split_bimodal=True,
#             dist_thresh=15,
#             subset_break_fraction=0.01,
#             break_fraction=0.01,
#             fast_split=True,
#             max_split_iterations=10,
    consider_cluster_deletion=True,
    )
num_starting_masks = 100

################################################
iterations = []
scores = []
num_clusters = []

log_to_file(fname+'.klg.'+str(shank), 'debug')
log_suppress_hierarchy('klustakwik', inclusive=False)

if os.path.exists(fname+'.pickle'):
    data = pickle.load(open(fname+'.pickle', 'rb'))
else:
    raw_data = load_fet_fmask_to_raw(fname, shank, drop_last_n_features=1)
    data = raw_data.to_sparse_data()
    pickle.dump(data, open(fname+'.pickle', 'wb'), -1)

client = Client()
distributer = IPythonDistributer(client)
#distributer = None

kk = KK(data, distributer=distributer, **params)
kk.register_callback(SaveCluEvery(fname, shank, every=5))

if False:
    clusters = loadtxt(fname+'.clu.'+str(shank), skiprows=1, dtype=int)
    kk.cluster_from(clusters)
else:
    #kk.cluster_with_subset_schedule(100, [0.99, 1.0])
    kk.cluster_mask_starts()

save_clu(kk, fname, shank)
