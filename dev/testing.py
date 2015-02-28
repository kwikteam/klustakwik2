from klustakwik2 import *
from pylab import *
import time

fname, shank = '../temp/testsmallish', 4

start = time.time()
data = load_fet_fmask(fname, shank)
print 'Loading time:', time.time()-start

#plot(data.noise_mean)
#plot(data.noise_variance)

start = time.time()
clusters = mask_starts(data, 100)
print 'Min cluster', amin(clusters), 'max cluster', amax(clusters)
print 'Mask starts time:', time.time()-start

start = time.time()
data.do_initial_precomputations()
print 'Initial precomputation time:', time.time()-start

show()
