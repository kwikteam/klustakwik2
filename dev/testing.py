from klustakwik2 import *
from pylab import *
import time

fname, shank = '../temp/testsmallish', 4

start = time.time()
data = load_fet_fmask(fname, shank)
print 'Loading time:', time.time()-start

start = time.time()
data.do_initial_precomputations()
print 'Initial precomputation time:', time.time()-start

kk = KK(data)
kk.cluster(num_starting_clusters=100)

show()
