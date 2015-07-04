from numpy import *
from bisect import bisect_left

__all__ = ['compute_correction_terms_and_replace_data',
           'reduce_masks', 'reduce_masks_from_arrays', 'float_num_unmasked',
           'compute_float_num_unmasked_from_arrays',
           'compute_float_num_unmasked,'
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


def reduce_masks_from_arrays(Ostart, Oend, I):
    x = arange(len(Ostart))
    # converting the array to a string allows for a lexicographic compare
    # the details of the comparison are irrelevant as long as it is
    # consistent (for sorting) and never equal if the underlying arrays
    # are unequal
    x = array(sorted(x, key=lambda p: I[Ostart[p]:Oend[p]].tostring()), dtype=int)
    y = empty_like(x)
    y[x] = arange(len(x)) # y is the inverse of x as a permutation
    # step 2: iterate through all indices and add to collection if the
    # indices have changed
    oldstr = None
    new_indices = []
    start = zeros(len(Ostart), dtype=int)
    end = zeros(len(Ostart), dtype=int)
    curstart = 0
    curend = 0
    for i, p in enumerate(x):
        curind = I[Ostart[p]:Oend[p]]
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
    return new_indices, start[y], end[y]


def reduce_masks(raw_data):
    # step 1: sort into lexicographical order of masks
    start = raw_data.offsets[:-1]
    end = raw_data.offsets[1:]
    I = raw_data.unmasked
    return reduce_masks_from_arrays(start, end, I)


def compute_float_num_unmasked(data):
    M = data.masks
    O = data.offsets
    return compute_float_num_unmasked_from_arrays(M, O)


def compute_float_num_unmasked_from_arrays(M, O):
    # Rather than just doing add.reduceat(data.masks, data.offset[:-1]) which is what we'd like to
    # do, we have to avoid any offsets equal to the length of masks because even though
    # the calculation is correct if you think of it as slices, numpy raises an error if any indices
    # are equal to the length of the array. So we just cut those out and add some zeros at the end.
    O = O[:-1]
    n = bisect_left(O, len(M))-1
    O2 = O[:n]
    U = hstack((add.reduceat(M, O2), zeros(len(O)-n)))
    return U
