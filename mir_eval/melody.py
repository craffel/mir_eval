# CREATED:2014-03-07 by Justin Salamon <justin.salamon@nyu.edu>
'''
Melody extraction evaluation, based on the protocols used in MIREX since 2005.
'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys


def evaluate_melody(ref, est, hop=0.010, plotmatch=False):
    '''
    Evaluate two melody (predominant f0) transcriptions, where the first is
    treated as the reference (ground truth) and the second as the estimate to
    be evaluated.

    Input:
        ref - array of shape (2,x), ref[0] contains timestamps and ref[1]
              the corresponding reference frequency values in Hz (see *).
        est - array of shape (2,x), ref[0] contains timestamps and ref[1]
              the corresponding estimate frequency values in Hz (see **).
        hop - the desired hop size (in seconds) to compare the reference and
              estimate sequences (see ***)
        plotmatch - when True, will plot the reference and estimate sequences
                    (see ****)

    Output:
        voicing recall -            Fraction of voiced frames in ref estimated as voiced in est
        voicing false alarm rate -  Fraction of unvoiced frames in ref estimated as voiced in est
        raw pitch -                 Fraction of voiced frames in ref for which est gives a correct pitch estimate (within 50 cents)
        raw chroma -                Same as raw pitch, but ignores octave errors
        overall accuracy -          Overall performance measure combining pitch and voicing

        For a detailed explanation of the measures please refer to:
        J. Salamon, E. Gomez, D. P. W. Ellis and G. Richard, "Melody Extraction
        from Polyphonic Music Signals: Approaches, Applications and Challenges",
        IEEE Signal Processing Magazine, 31(2):118-134, Mar. 2014.

    *    Unvoiced frames should be indicated by 0 Hz.
    **   Unvoiced frames can be indicated either by 0 Hz or by a negative Hz
         value - negative values represent the algorithm's pitch estimate for
         frames it has determined as unvoiced, in case they are in fact voiced.
    ***  For a frame-by-frame comparison, both sequences are resampled using
         the provided hop size (in seconds), the default being 10 ms. The
         frequency values of the resampled sequences are obtained via linear
         interpolation of the original frequency values converted using a cent
         scale.
    **** Two plots will be generated: the first simply displays the original
         sequences (ref in blue, est in red). The second will display the
         resampled sequences, ref in blue, and est in 3 possible colours:
         red = mismatch, yellow = chroma match, green = pitch match.
    '''

    # STEP 0
    # Cast to numpy arrays and run safety checks
    try:
        ref = np.asarray(ref, dtype=np.float64)
    except ValueError:
        print 'Error: ref could not be read, ' \
              'are the time and frequency sequences of the same length?'
        return None
    try:
        est = np.asarray(est, dtype=np.float64)
    except ValueError:
        print 'Error: est could not be read, ' \
              'are the time and frequency sequences of the same length?'
        return None

    if ref.shape[0] != 2:
        print 'Error: ref should be of dimension (2,x), but is of dimension',\
            ref.shape
        return None
    if est.shape[0] != 2:
        print 'Error: est should of dimension (2,x), but is of dimension', \
            est.shape
        return None

    if len(ref[0])==0 or len(est[0])==0:
        print 'Error: one of the inputs seems to be empty?'
        return None


    # STEP 1
    # For convenience, separate time and frequency arrays
    ref_time = ref[0]
    ref_freq = ref[1]
    est_time = est[0]
    est_freq = est[1]

    # STEP 2
    # take absolute values (since negative values are allowed) and convert
    # non-zero Hz values into cents (using 10Hz for 0 cents)
    base_frequency = 10.0
    ref_cent = np.zeros(len(ref_freq))
    ref_nonz_ind = np.nonzero(ref_freq)[0]
    ref_cent[ref_nonz_ind] = 1200 * np.log2(np.abs(ref_freq[ref_nonz_ind]) /
                                            base_frequency)
    # sign now restored after interpolation, so comment out next 3 lines
    # ref_neg_ind = np.nonzero(ref_freq < 0)[0]
    # ref_cent[ref_neg_ind] = np.negative(ref_cent[ref_neg_ind])
    # ref_freq = ref_cent

    est_cent = np.zeros(len(est_freq))
    est_nonz_ind = np.nonzero(est_freq)[0]
    est_cent[est_nonz_ind] = 1200 * np.log2(np.abs(est_freq[est_nonz_ind]) /
                                            base_frequency)
    # sign now restored after interpolation, so comment out next 3 lines
    # est_neg_ind = np.nonzero(est_freq < 0)[0]
    # est_cent[est_neg_ind] = np.negative(est_cent[est_neg_ind])
    # est_freq = est_cent

    # STEP 3
    # Add sample in case the sequences don't start at time 0
    # first keep backup of original time sequences for plotting
    ref_time_plt = ref_time
    est_time_plt = est_time
    # then check if missing sample at time 0
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
    index_interp = 0
    for index_orig in range(len(ref_cent) - 1):
        if np.logical_xor(ref_cent[index_orig] > 0,
                          ref_cent[index_orig + 1] > 0):
            while index_interp < len(ref_time_grid) and ref_time_grid[
                index_interp] <= ref_time[index_orig]:
                index_interp += 1
            while index_interp < len(ref_time_grid) and ref_time_grid[
                index_interp] < ref_time[index_orig + 1]:
                ref_cent_interp[index_interp] = ref_cent[index_orig]
                index_interp += 1
        # index_orig += 1

    index_interp = 0
    for index_orig in range(len(est_cent) - 1):
        if np.logical_xor(est_cent[index_orig] > 0,
                          est_cent[index_orig + 1] > 0):
            while index_interp < len(est_time_grid) and est_time_grid[
                index_interp] <= est_time[index_orig]:
                index_interp += 1
            while index_interp < len(est_time_grid) and est_time_grid[
                index_interp] < est_time[index_orig + 1]:
                est_cent_interp[index_interp] = est_cent[index_orig]
                index_interp += 1

    # STEP 6
    # restore original sign to interpolated sequences
    index_interp = 0
    for index_orig in range(len(ref_freq) - 1):
        if ref_freq[index_orig] < 0:
            while index_interp < len(ref_time_grid) and ref_time_grid[
                index_interp] < ref_time[index_orig]:
                index_interp += 1
            while index_interp < len(ref_time_grid) and ref_time_grid[
                index_interp] < ref_time[index_orig + 1]:
                ref_cent_interp[index_interp] *= -1
                index_interp += 1

    index_interp = 0
    for index_orig in range(len(est_freq) - 1):
        if est_freq[index_orig] < 0:
            while index_interp < len(est_time_grid) and est_time_grid[
                index_interp] < est_time[index_orig]:
                index_interp += 1
            while index_interp < len(est_time_grid) and est_time_grid[
                index_interp] < est_time[index_orig + 1]:
                est_cent_interp[index_interp] *= -1
                index_interp += 1

    # STEP 7
    # ensure the estimated sequence is the same length as the reference
    est_time_grid = ref_time_grid
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
    cent_diff_chroma = abs(
        cent_diff - 1200 * np.floor(cent_diff / 1200.0 + 0.5))
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
    # print ref_cent_interp
    # print est_cent_interp
    # print cent_diff

    if plotmatch:
        green = (0, 1, 0)
        yellow = (1, 1, 0)

        plt.figure()
        plt.plot(ref_time_plt, ref_freq, 'b.', est_time_plt, est_freq, 'r.')
        plt.title("Original sequences")

        plt.figure()
        p = plt.plot(ref_time_grid, ref_cent_interp, 'b.', est_time_grid,
                     est_cent_interp,
                     'r.', est_time_grid[cent_diff_chroma <= 50],
                     est_cent_interp[cent_diff_chroma <= 50], 'y.',
                     est_time_grid[
                         cent_diff <= 50], est_cent_interp[cent_diff <= 50],
                     'g.')
        plt.title("Resampled sequences")
        plt.setp(p[0], 'color', 'b', 'markeredgecolor', 'b')
        plt.setp(p[1], 'color', 'r', 'markeredgecolor', 'r')
        plt.setp(p[2], 'color', yellow, 'markeredgecolor', yellow)
        plt.setp(p[3], 'color', green, 'markeredgecolor', green)

    return vx_recall, vx_false_alm, raw_pitch, raw_chroma, overall_accuracy








