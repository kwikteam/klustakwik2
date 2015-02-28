from klustakwik2 import *
from pylab import *
import time

fname, shank = '../temp/testsmallish', 4
start = time.time()
data = load_fet_fmask(fname, shank)
print 'Loading time:', time.time()-start
#plot(data.mean)
#plot(data.var)
start = time.time()
print mask_starts(data, 100)
print 'Mask starts time:', time.time()-start

show()
