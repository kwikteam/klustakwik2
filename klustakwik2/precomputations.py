from numpy import *

__all__ = ['compute_correction_terms_and_replace_data',
           'sort_masks', 'float_num_unmasked',
           ]

def compute_correction_terms_and_replace_data(raw_data):
    I = raw_data.unmasked
    x = raw_data.features
    w = raw_data.masks
    nu = raw_data.noise_mean[I]
    sigma2 = raw_data.noise_variance[I]
    y = w*x+(1-w)*nu
    z = w*x*x+(1-w)*(nu*nu+sigma2)
    correction_terms = z-y*y
    features = y
    return features, correction_terms


def sort_masks(raw_data):
    # step 1: sort into lexicographical order of masks
    O = raw_data.offsets
    I = raw_data.unmasked
    x = arange(len(O)-1)
    # converting the array to a string allows for a lexicographic compare
    # the details of the comparison are irrelevant as long as it is
    # consistent (for sorting) and never equal if the underlying arrays
    # are unequal
    x = array(sorted(x, key=lambda p: I[O[p]:O[p+1]].tostring()), dtype=int)
    # step 2: iterate through all indices and add to collection if the
    # indices have changed
    oldstr = None
    new_indices = []
    start = zeros(len(O)-1, dtype=int)
    end = zeros(len(O)-1, dtype=int)
    curstart = 0
    curend = 0
    for i, p in enumerate(x):
        curind = I[O[p]:O[p+1]]
        curstr = curind.tostring()
        if curstr!=oldstr:
            new_indices.append(curind)
            oldstr = curstr
            curstart = curend
            curend += len(curind)
        start[i] = curstart
        end[i] = curend
    # step 3: convert into start, end
    new_indices = hstack(new_indices)
    return x, new_indices, start, end


def compute_float_num_unmasked(data):
    return add.reduceat(data.masks, data.offsets[:-1])
