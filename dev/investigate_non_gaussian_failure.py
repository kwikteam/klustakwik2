from numpy import *
from klustakwik2 import *
from klustakwik2.tests.test_synthetic_data import generate_synthetic_data

log_to_file('ngf.klg', 'debug')

data = load_fet_fmask_to_raw('test', 1).to_sparse_data()
# data = generate_synthetic_data(4, 1000, [
#     ((1, 1, 0, 0), (0.1,)*4, (1.5, 0.5, 0, 0), (0.05, 0.05, 0.01, 0)),
#     ((0, 1, 1, 0), (0.1,)*4, (0, 0.5, 1.5, 0), (0, 0.05, 0.05, 0.01)),
#     ((0, 0, 1, 1), (0.1,)*4, (0, 0, 0.5, 1.5), (0.01, 0, 0.05, 0.05)),
#     ((1, 0, 0, 1), (0.1,)*4, (1.5, 0, 0, 1.5), (0.05, 0, 0.01, 0.05)),
#     ], save_to_fet='test')
kk = KK(data, use_mua_cluster=False,
        split_every=1000000, split_first=1000000, # disable splitting
        )
# dump_variable(kk, 'num_cluster_members', slot='end_M_step')
# dump_variable(kk, 'num_clusters_alive', slot='end_M_step')
# dump_variable(kk, 'cluster_mean', slot='end_M_step')
# dump_variable(kk, 'weight', slot='end_M_step')
dump_variable(kk, 'bincount(kk.clusters)', iscode=True, slot='end_M_step')
dump_variable(kk, 'bincount(kk.clusters)', iscode=True, slot='end_EC_steps')
# dump_variable(kk, 'cluster_penalty', slot='end_compute_cluster_penalties')
# dump_covariance_matrices(kk)
dump_variable(kk, 'log_p_best', slot='end_EC_steps')
dump_variable(kk,
              '(amin(kk.log_p_best), amax(kk.log_p_best), mean(kk.log_p_best), std(kk.log_p_best))',
              iscode=True, slot='end_EC_steps', suffix='log_p_dist')

kk.cluster(20)

print bincount(kk.clusters)
print
print bincount(kk.clusters[0:1000])
print bincount(kk.clusters[1000:2000])
print bincount(kk.clusters[2000:3000])
print bincount(kk.clusters[3000:4000])
