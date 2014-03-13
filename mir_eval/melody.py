# CREATED:2014-03-07 by Justin Salamon <justin.salamon@nyu.edu>
'''
Melody extraction evaluation, based on the protocols used in MIREX since 2005.
'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys


def evaluate_melody(ref, est, hop=0.010):
    '''
    Evaluate two melody (predominant f0) transcriptions, where the first is
    treated as the reference (ground truth) and the second as the estimate to
    be evaluated.

    Each melody should be a numpy array with 2 elements, the first being a
    numpy array of timestamps and the second a numpy array of frequency values
    in Hz. Unvoiced frames are indicated either by 0Hz or by a negative
    Hz value - negative values represent the algorithm's pitch estimate for
    frames it has determined as unvoiced, in case they are in fact voiced.

    For a frame-by-frame comparison, both sequences are resampled using the
    provided hop size (in seconds), the default being 10 ms. The frequency
    values of the resampled sequences are obtained via linear interpolation
    of the original values converted to a cent scale.

     The output consists of five evaluation measures:
     - voicing recall
     - voicing false alarm rate
     - raw pitch
     - raw chroma
     - overall accuracy

    For a detailed explanation of the measures please refer to:
    J. Salamon, E. Gomez, D. P. W. Ellis and G. Richard, "Melody Extraction
    from Polyphonic Music Signals: Approaches, Applications and Challenges",
    IEEE Signal Processing Magazine, 31(2):118-134, Mar. 2014.
    '''

    # STEP 1
    # For convenience, separate time and frequency arrays
    ref_time = ref[0]
    ref_freq = ref[1]
    est_time = est[0]
    est_freq = est[1]

    # STEP 2
    # take absolute values (since negative values are allowed) and convert
    # non-zero Hz values into cents (using 10Hz for 0 cents)
    ref_cent = np.zeros(len(ref_freq))
    ref_nonz_ind = np.nonzero(ref_freq)[0]
    ref_cent[ref_nonz_ind] = 1200 * np.log2(np.abs(ref_freq[ref_nonz_ind]) /
                                            10.0)
    # sign now restored after interpolation, so comment out next 3 lines
    # ref_neg_ind = np.nonzero(ref_freq < 0)[0]
    # ref_cent[ref_neg_ind] = np.negative(ref_cent[ref_neg_ind])
    # ref_freq = ref_cent

    est_cent = np.zeros(len(est_freq))
    est_nonz_ind = np.nonzero(est_freq)[0]
    est_cent[est_nonz_ind] = 1200 * np.log2(np.abs(est_freq[est_nonz_ind]) /
                                            10.0)
    # sign now restored after interpolation, so comment out next 3 lines
    # est_neg_ind = np.nonzero(est_freq < 0)[0]
    # est_cent[est_neg_ind] = np.negative(est_cent[est_neg_ind])
    # est_freq = est_cent

    # STEP 3
    # in case the sequences don't start at time 0
    if ref_time[0] > 0:
        ref_time = np.insert(ref_time, 0, 0)
        ref_cent = np.insert(ref_cent, 0, ref_cent[0])
    if est_time[0] > 0:
        est_time = np.insert(est_time, 0, 0)
        est_cent = np.insert(est_cent, 0, est_cent[0])

    # STEP 4
    # sample to common hop size using linear interpolation
    ref_interp_func = sp.interpolate.interp1d(ref_time, ref_cent)
    ref_time_grid = np.linspace(0, hop * np.floor(ref_time[-1] / hop),
                                np.floor(ref_time[-1] / hop) + 1)
    ref_cent_interp = ref_interp_func(ref_time_grid)

    est_interp_func = sp.interpolate.interp1d(est_time, est_cent)
    est_time_grid = np.linspace(0, hop * np.floor(est_time[-1] / hop),
                                np.floor(est_time[-1] / hop) + 1)
    est_cent_interp = est_interp_func(est_time_grid)

    # STEP 5
    # fix interpolated values between non-zero/zero transitions:
    # interpolating these values doesn't make sense, so replace with value
    # of start point.
    index_orig = 0
    index_interp = 0
    while index_orig < len(ref_cent)-1:
        if np.logical_xor(ref_cent[index_orig]>0,ref_cent[index_orig+1]>0):
            while index_interp < len(ref_time_grid) and ref_time_grid[
                index_interp] <= ref_time[index_orig]:
                index_interp += 1
            while index_interp < len(ref_time_grid) and ref_time_grid[
                index_interp] < ref_time[index_orig+1]:
                ref_cent_interp[index_interp] = ref_cent[index_orig]
                index_interp += 1
        index_orig += 1

    index_orig = 0
    index_interp = 0
    while index_orig < len(est_cent)-1:
        if np.logical_xor(est_cent[index_orig]>0,est_cent[index_orig+1]>0):
            while index_interp < len(est_time_grid) and est_time_grid[
                index_interp] <= est_time[index_orig]:
                index_interp += 1
            while index_interp < len(est_time_grid) and est_time_grid[
                index_interp] < est_time[index_orig+1]:
                est_cent_interp[index_interp] = est_cent[index_orig]
                index_interp += 1
        index_orig += 1

    # STEP 6
    # restore original sign to interpolated sequences
    index_orig = 0
    index_interp = 0
    while index_orig < len(ref_freq)-1:
        if ref_freq[index_orig]<0:
            while index_interp < len(ref_time_grid) and ref_time_grid[
                index_interp] < ref_time[index_orig]:
                index_interp += 1
            while index_interp < len(ref_time_grid) and ref_time_grid[
                index_interp] < ref_time[index_orig+1]:
                ref_cent_interp[index_interp] *= -1
                index_interp += 1
        index_orig += 1

    index_orig = 0
    index_interp = 0
    while index_orig < len(est_freq)-1:
        if est_freq[index_orig]<0:
            while index_interp < len(est_time_grid) and est_time_grid[
                index_interp] < est_time[index_orig]:
                index_interp += 1
            while index_interp < len(est_time_grid) and est_time_grid[
                index_interp] < est_time[index_orig+1]:
                est_cent_interp[index_interp] *= -1
                index_interp += 1
        index_orig += 1

    # STEP 7
    # ensure the estimated sequence is the same length as the reference
    len_diff = len(ref_cent_interp) - len(est_cent_interp)
    if len_diff >= 0:
        est_cent_interp = np.append(est_cent_interp, np.zeros(len_diff))
    else:
        est_cent_interp = np.resize(est_cent_interp, len(ref_cent_interp))


    # STEP 8
    # calculate voicing measures
    v_ref = (ref_cent_interp > 0)
    v_est = (est_cent_interp > 0)

    uv_ref = (ref_cent_interp <= 0)
    uv_est = (est_cent_interp <= 0)

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
    # are declared as voiced by the estimate
    vx_recall = TP / float(TP + FN)

    # Voicing false alarm = fraction of unvoiced frames according to the
    # reference that are declared as voiced by the estimate
    vx_false_alm = FP / float(FP + TN + sys.float_info.epsilon)

    # debug
    # print "vx_recall:", vx_recall
    # print "vx_false_alm:", vx_false_alm

    # STEP 9
    # Raw pitch = the number of voiced frames in the reference for which the
    # estimate provides a correct frequency value (within 50 cents).
    # NB: voicing estimation is ignored in this measure (hence the abs)
    cent_diff = np.abs(np.abs(ref_cent_interp) - np.abs(est_cent_interp))
    raw_pitch = sum(cent_diff[v_ref] <= 50) / float(sum(v_ref))

    # debug
    # print cent_diff[v_ref]
    # print "raw_pitch:",raw_pitch

    # STEP 10
    # Raw chroma = same as raw pitch except that octave errors are ignored.
    cent_diff_chroma = abs(cent_diff - 1200 * np.floor(cent_diff/1200.0 + 0.5))
    raw_chroma = sum(cent_diff_chroma[v_ref] <= 50) / float(sum(v_ref))

    # debug
    # print cent_diff_chroma[v_ref]
    # print "raw_chroma:",raw_chroma

    # STEP 11
    # Overall accuracy = combine voicing and raw pitch to give an overall
    # performance measure
    cent_diff_overall = np.abs(ref_cent_interp - est_cent_interp)
    overall_accuracy = (sum(cent_diff_overall[v_ref] <= 50) + TN) / float(len(
        ref_cent_interp))

    # debug
    # print "overall_accuracy:",overall_accuracy

    # debug
    plt.figure()
    plt.plot(ref_time,ref_freq,'-bo',est_time,est_freq,'-ro')

    plt.figure()
    plt.plot(ref_time_grid,ref_cent_interp,'-bo',est_time_grid,est_cent_interp,
             '-ro')

    return vx_recall, vx_false_alm, raw_pitch, raw_chroma, overall_accuracy








