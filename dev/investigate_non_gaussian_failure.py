from numpy import *
from klustakwik2 import *
from klustakwik2.tests.test_synthetic_data import generate_synthetic_data

console_log_level('debug')

data = generate_synthetic_data(4, 1000, [
    ((1, 1, 0, 0), (0.1,)*4, (1.5, 0.5, 0, 0), (0.05, 0.05, 0.01, 0)),
    ((0, 1, 1, 0), (0.1,)*4, (0, 0.5, 1.5, 0), (0, 0.05, 0.05, 0.01)),
    ((0, 0, 1, 1), (0.1,)*4, (0, 0, 0.5, 1.5), (0.01, 0, 0.05, 0.05)),
    ((1, 0, 0, 1), (0.1,)*4, (1.5, 0, 0, 1.5), (0.05, 0, 0.01, 0.05)),
    ])
kk = KK(data)
dump_covariance_matrices(kk)

kk.cluster(20)

print bincount(kk.clusters[0:1000])
print bincount(kk.clusters[1000:2000])
print bincount(kk.clusters[2000:3000])
print bincount(kk.clusters[3000:4000])
