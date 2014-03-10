# CREATED:2014-03-07 by Justin Salamon <justin.salamon@nyu.edu>
'''
Melody extraction evaluation, based on the protocols used in MIREX since 2005.
'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys


def evaluate_melody(ref, est, interpolation_hop=0.010):
    '''
    Evaluate two melody (predominant f0) transcriptions. Each should be
    '''

    # For convenience, separate time and frequency arrays
    ref_time = ref[0]
    ref_freq = ref[1]
    est_time = est[0]
    est_freq = est[1]

    # in case the sequence doesn't start at time 0
    if ref_time[0] > 0:
        ref_time = np.insert(ref_time, 0, 0)
        ref_freq = np.insert(ref_freq, 0, ref_freq[0])
    if est_time[0] > 0:
        est_time = np.insert(est_time, 0, 0)
        est_freq = np.insert(est_freq, 0, ref_freq[0])

    # sample to common hop size using linear interpolation
    ref_interp_func = sp.interpolate.interp1d(ref_time, ref_freq)
    # replace with linspace?
    ref_time_grid = np.arange(0, ref_time[-1] + sys.float_info.epsilon,
                              interpolation_hop)
    ref_freq_interp = ref_interp_func(ref_time_grid)

    est_interp_func = sp.interpolate.interp1d(est_time, est_freq)
    # replace with linspace?
    est_time_grid = np.arange(0, est_time[-1] + sys.float_info.epsilon,
                              interpolation_hop)
    est_freq_interp = est_interp_func(est_time_grid)

    # debug
    # print ref_time_grid, ":", ref_freq_interp
    # print est_time_grid, ":", est_freq_interp
    # plt.plot(ref_time_grid, ref_freq_interp, 'bo', est_time_grid,
    #         est_freq_interp, 'rx')
    # plt.show()
    # print ref_time_grid[0:10]
    # print ref_freq_interp[0:10]
    # print est_time_grid[0:10]
    # print est_freq_interp[0:10]

    # ensure both sequences are of same length
    len_diff = len(ref_freq_interp) - len(est_freq_interp)
    if len_diff >= 0:
        est_freq_interp = np.append(est_freq_interp, np.zeros(len_diff))
    else:
        est_freq_interp = np.resize(est_freq_interp, len(ref_freq_interp))

    # debug
    # print len(ref_freq_interp)
    # print len(est_freq_interp)

    v_ref = (ref_freq_interp > 0)
    v_est = (est_freq_interp > 0)

    uv_ref = (ref_freq_interp <= 0)
    uv_est = (est_freq_interp <= 0)

    # How voicing is computed
    #        | v_ref | uv_ref |
    # -------|-------|--------|
    # v_est  |  TP   |   FP   |
    # -------|-------|------- |
    # uv_est |  FN   |   TN   |
    # -------------------------

    TP = sum(v_ref * v_est)
    FP = sum(uv_ref * v_est)
    FN = sum(v_ref * uv_est)
    TN = sum(uv_ref * uv_est)

    # Voicing recall = fraction of voiced frames according the reference that
    # are declared as voiced by the algorithm
    vx_recall = TP / float(TP + FN)

    # Voicing false alarm = fraction of unvoiced frames according to the
    # reference that are declared as voiced by the algorithm
    vx_false_alm = FP / float(FP + TN + sys.float_info.epsilon)

    # debug
    # print "vx_recall:", vx_recall
    # print "vx_false_alm:", vx_false_alm

    # convert voiced sequences into cent scale (currently using 10Hz as 0
    # cents)
    v_ref_ind = np.nonzero(v_ref)[0]
    ref_cent = np.zeros(len(ref_freq_interp))
    ref_cent[v_ref_ind] = 1200 * np.log2(ref_freq_interp[v_ref_ind] / 10.0)

    v_est_ind = np.nonzero(v_est)[0]
    est_cent = np.zeros(len(est_freq_interp))
    est_cent[v_est_ind] = 1200 * np.log2(est_freq_interp[v_est_ind] / 10.0)

    # debug
    # print ref_cent
    # print est_cent

    # Calculate number of correct pitch estimates
    cent_diff = np.abs(ref_cent - est_cent)
    raw_pitch = sum(cent_diff[v_ref] <= 50) / float(sum(v_ref))

    # debug
    # print cent_diff[v_ref]
    # print "raw_pitch:",raw_pitch

    # Map distances to single octave
    cent_diff_chroma = abs(cent_diff - 1200 * np.floor(cent_diff/1200.0 + 0.5))
    raw_chroma = sum(cent_diff_chroma[v_ref] <= 50) / float(sum(v_ref))

    # debug
    # print cent_diff_chroma[v_ref]
    # print "raw_chroma:",raw_chroma

    # Calculate the overall accuracy
    overall_accuracy = (sum(cent_diff[v_ref] <= 50) + TN) / float(len(
        ref_cent))

    # debug
    # print "overall_accuracy:",overall_accuracy

    return vx_recall, vx_false_alm, raw_pitch, raw_chroma, overall_accuracy








