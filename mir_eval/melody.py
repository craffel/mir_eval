# CREATED:2014-03-07 by Justin Salamon <justin.salamon@nyu.edu>
'''
Melody extraction evaluation, based on the protocols used in MIREX since 2005.
'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def evaluate_melody(ref,est,interpolation_hop=0.010):

    '''
    Evaluate two melody (predominant f0) transcriptions. Each should be
    '''

    # For convenience, separate time and frequency arrays
    ref_time = ref[0,:]
    ref_freq = ref[1,:]
    est_time = est[0,:]
    est_freq = est[1,:]

    # in case the sequence doesn't start at time 0
    ref_time = np.insert(ref_time,0,0)
    ref_freq = np.insert(ref_freq,0,ref_freq[0])
    est_time = np.insert(est_time,0,0)
    est_freq = np.insert(est_freq,0,ref_freq[0])

    # sample to common hop size using linear interpolation
    ref_interp_func = sp.interpolate.interp1d(ref_time,ref_freq)
    ref_time_grid = np.arange(0, ref_time[-1], interpolation_hop)
    ref_freq_interp = ref_interp_func(ref_time_grid)

    est_interp_func = sp.interpolate.interp1d(est_time,est_freq)
    est_time_grid = np.arange(0, est_time[-1], interpolation_hop)
    est_freq_interp = est_interp_func(est_time_grid)

    # debug
    # plt.plot(ref_time_grid,ref_freq_interp,est_time_grid,est_freq_interp)
    # plt.show()
    # print ref_time_grid[0:10]
    # print ref_freq_interp[0:10]
    # print est_time_grid[0:10]
    # print est_freq_interp[0:10]

    # ensure both sequences are of same length
    len_diff = len(ref_freq_interp) - len(est_freq_interp)
    if len_diff >= 0:
        est_freq_interp = np.append(est_freq_interp,np.zeros(len_diff))
    else:
        est_freq_interp = np.resize(est_freq_interp,len(ref_freq_interp))

    # debug
    print len(ref_freq_interp)
    print len(est_freq_interp)




