import numpy as np
cimport numpy as np

DTYPE = np.int
DTYPE_F = np.float

ctypedef np.int_t DTYPE_t

def top_pairs_arr(np.ndarray[DTYPE_t, ndim=2] pairs):
    cdef set pairs1 = set()
    cdef set pairs2 = set()
    cdef int xmax = pairs.shape[0]
    cdef int out_size = 0
    cdef np.ndarray[DTYPE_t, ndim=1] out_ind = np.zeros([xmax], dtype=DTYPE)

    for i in range(xmax):
        p1 = pairs[i, 0]
        p2 = pairs[i, 1]
        if (p1 in pairs1) or (p2 in pairs2):
            continue
        pairs1.add(p1)
        pairs2.add(p2)
        out_ind[out_size] = i
        out_size += 1
    return out_ind[:out_size]


def get_distance_values_arr(
    np.ndarray[DTYPE_t, ndim=1] inds1,
    np.ndarray[DTYPE_t, ndim=1] inds2,
    dict distance_dok,
    np.ndarray[object, ndim=2] df_arr1,
    np.ndarray[object, ndim=2] df_arr2,
    list distance_functions):

    cdef int dcount = inds1.shape[0]
    cdef int colcount = len(distance_functions)
    cdef int ind1, ind2, i, c
    cdef list d
    cdef tuple key
    cdef np.ndarray[object, ndim=1] arr1, arr2

    cdef np.ndarray[double, ndim=2] out_raw = np.zeros([dcount, colcount], dtype=DTYPE_F)

    for i in range(dcount):
        ind1 = inds1[i]
        ind2 = inds2[i]
        key = (ind1, ind2)
        d = distance_dok.get(key, [])
        if len(d) == 0:
            for c in range(colcount):
                fun = distance_functions[c]
                d.append(fun(df_arr1[ind1, c], df_arr2[ind2, c]))
            distance_dok[key] = d
        for c in range(colcount):
            out_raw[i, c] = d[c]
    return out_raw