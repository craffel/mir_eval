# CREATED:2014-03-07 by Justin Salamon <justin.salamon@nyu.edu>
'''
Melody extraction evaluation, based on the protocols used in MIREX since 2005.

For a detailed explanation of the measures please refer to:
J. Salamon, E. Gomez, D. P. W. Ellis and G. Richard, "Melody Extraction
from Polyphonic Music Signals: Approaches, Applications and Challenges",
IEEE Signal Processing Magazine, 31(2):118-134, Mar. 2014.
'''

import numpy as np
import sys
import scipy.interpolate


def hz2cents(freq_hz, base_frequency=10.0):
    '''
    Convert an array of frequency values in Hz to cents
    0 values are left in place.

    :parameters:
        - freq_hz : ndarray
            Array of frequencies in Hz.
        - base_frequency : float
            Base frequency for conversion.
    :returns:
        - cent : ndarray
            Array of frequencies in cents, relative to base_frequency
    '''
    freq_cent = np.zeros(freq_hz.shape[0])
    freq_nonz_ind = np.flatnonzero(freq_hz)
    freq_cent[freq_nonz_ind] = 1200*np.log2(np.abs(freq_hz[freq_nonz_ind])/base_frequency)

    return freq_cent


def freq_to_voicing(frequencies):
    '''Convert from an array of frequency values to frequency array + voice/unvoiced array

    :parameters:
        - frequencies : ndarray
            Array of frequencies.  A frequency <= 0 indicates "unvoiced".
    :returns:
        - frequencies : ndarray
            Array of frequencies, all >= 0.
        - voiced : ndarray
            Boolean array, same length as frequencies, which indicates voiced or unvoiced
    '''
    return np.abs(frequencies), frequencies > 0


def resample_melody_series(times, frequencies, voicing, hop=0.01):
    '''Resamples frequency and voicing time series to a new timescale.
    Maintains any zero ("unvoiced") values in frequencies.

    :parameters:
        - times : ndarray
            Times of each frequency value
        - frequencies : ndarray
            Array of frequency values, >= 0
        - voicing : ndarray
            Boolean array which indicates voiced or unvoiced
        - hop : float
            Hop size for resampling.  Default .01

    :returns:
        - times_new : ndarray
            Times of each resampled frequency value
        - frequencies_resampled : ndarray
            Frequency array resampled to new timebase
        - voicing_resampled : ndarray
            Boolean voicing array resampled to new timebase
    '''
    # Fill in zero values with the last reported frequency
    # to avoid erroneous values when resampling
    frequencies_held = np.array(frequencies)
    for n, frequency in enumerate(frequencies[1:]):
        if frequency == 0:
            frequencies_held[n + 1] = frequencies_held[n]
    # Compute new timebase.  Rounding/linspace is to avoid float problems.
    times = np.round(times*1e10)*1e-10
    times_new = np.linspace(0, hop*int(np.floor(times[-1]/hop)), int(np.floor(times[-1]/hop)) + 1)
    times_new = np.round(times_new*1e10)*1e-10
    # Linearly interpolate frequencies
    frequencies_resampled = scipy.interpolate.interp1d(times, frequencies_held)(times_new)
    # Retain zeros
    frequency_mask = scipy.interpolate.interp1d(times, frequencies, 'zero')(times_new)
    frequencies_resampled *= (frequency_mask != 0)
    # Nearest-neighbor interpolate voicing
    voicing_resampled = scipy.interpolate.interp1d(times, voicing, 'zero')(times_new)
    return times_new, frequencies_resampled, voicing_resampled.astype(np.bool)

def to_cent_voicing(ref_time, ref_freq, est_time, est_freq, **kwargs):
    '''Converts reference and estimated time/frequency (Hz) annotations to
    sampled frequency (cent)/voicing arrays.

    A zero frequency indicates "unvoiced".

    A negative frequency indicates "Predicted as unvoiced,
    but if it's voiced, this is the frequency estimate".

    :parameters:
        - ref_time : ndarray
            Time of each reference frequency value
        - ref_freq : ndarray
            Array of reference frequency values
        - est_time : ndarray
            Time of each reference frequency value
        - est_freq : ndarray
            Array of reference frequency values
        - base_frequency : float
            Base frequency in Hz for conversion to cents, default 10.0
        - hop : float
            Hop size, in seconds, to resample, default .01

    :returns:
        - ref_voicing : ndarray
            Resampled reference boolean voicing array
        - est_voicing : ndarray
            Resampled estimated boolean voicing array
        - ref_cent : ndarray
            Resampled reference frequency (cent) array
        - est_cent : ndarray
            Resampled estimated frequency (cent) array
    '''
    # Set default kwargs parameters
    base_frequency = kwargs.get('base_frequency', 10.)
    hop = kwargs.get('hop', .01)
    # Check if missing sample at time 0 and if so add one
    if ref_time[0] > 0:
        ref_time = np.insert(ref_time, 0, 0)
        ref_freq = np.insert(ref_freq, 0, ref_voicing[0])
    if est_time[0] > 0:
        est_time = np.insert(est_time, 0, 0)
        est_freq = np.insert(est_freq, 0, est_voicing[0])
    # Get separated frequency array and voicing boolean array
    ref_freq, ref_voicing = freq_to_voicing(ref_freq)
    est_freq, est_voicing = freq_to_voicing(est_freq)
    # convert both sequences to cents
    ref_cent = hz2cents(ref_freq)
    est_cent = hz2cents(est_freq)
    # Resample to common time base
    ref_time_grid, ref_cent, ref_voicing = resample_melody_series(ref_time,
                                                                  ref_cent,
                                                                  ref_voicing,
                                                                  hop)
    est_time_grid, est_cent, est_voicing = resample_melody_series(est_time,
                                                                  est_cent,
                                                                  est_voicing,
                                                                  hop)
    # ensure the estimated sequence is the same length as the reference
    len_diff = ref_cent.shape[0] - est_cent.shape[0]
    if len_diff >= 0:
        est_cent = np.append(est_cent, np.zeros(len_diff))
        est_voicing = np.append(est_voicing, np.zeros(len_diff))
    else:
        est_cent = est_cent[:ref_cent.shape[0]]
        est_voicing = est_voicing[ref_voicing.shape[0]]

    return ref_voicing, est_voicing, ref_cent, est_cent

def voicing_measures(ref_voicing, est_voicing):
    '''
    Compute the voicing recall and false alarm rates given two voicing indicator
    sequences, one as reference (truth) and the other as the estimate (prediction).
    The sequences must be of the same length.

    Input:
    - ref_voicing : np.array or list
    Reference voicing indicator where val>0 indicates voiced, val<=0 indicates unvoiced

    - est_voicing : np.array or list
    Estimate voicing indicator where val>0 indicates voiced, val<=0 indicates unvoiced

    Output:
    - vx_recall: float
    Voicing recall rate, the fraction of voiced frames in ref indicated as voiced in est

    - vx_false_alarm : float
    Voicing false alarm rate, the fraction of unvoiced frames in ref indicated as voiced in est
    '''

    # check for equal length
    if len(ref_voicing) != len(est_voicing):
        print "Error: inputs must be arrays or lists of the same length"
        return None

    # convert to booleans
    v_ref = np.asarray(ref_voicing) > 0
    v_est = np.asarray(est_voicing) > 0

    uv_ref = np.asarray(ref_voicing) <= 0
    uv_est = np.asarray(est_voicing) <= 0

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

    return vx_recall, vx_false_alm


def raw_pitch_accuracy(ref_cent, ref_voicing, est_cent, est_voicing):
    '''
    Compute the raw pitch accuracy given two pitch (frequency) sequences in cents
    and matching voicing indicator sequences. The first pitch and voicing arrays
    are treated as the reference (truth), and the second two as the estimate (prediction).
    All 4 sequences must be of the same length.

    Input:
    - ref_cent : np.array
    Reference pitch sequence in cents

    - ref_voicing : np.array or list
    Reference voicing indicator where val>0 indicates voiced, val<=0 indicates unvoiced

    - est_cent : np.array
    Estimate pitch sequence in cents

    - est_voicing : np.array or list
    Estimate voicing indicator where val>0 indicates voiced, val<=0 indicates unvoiced

    Output:
    - raw_pitch: float
    Raw pitch accuracy, the fraction of voiced frames in ref_cent for which est_cent
    provides a correct frequency values (within 50 cents).
    '''

    l1,l2,l3,l4 = len(ref_cent),len(ref_voicing),len(est_cent),len(est_voicing)
    if l1 != l2 or l1 != l3 or l1 != l4:
        print "Error: all 4 sequences must be of the same length"
        return None

    # convert to booleans
    v_ref = np.asarray(ref_voicing) > 0

    # Raw pitch = the number of voiced frames in the reference for which the
    # estimate provides a correct frequency value (within 50 cents).
    # NB: voicing estimation is ignored in this measure
    cent_diff = np.abs(ref_cent - est_cent)
    raw_pitch = sum(cent_diff[v_ref] <= 50) / float(sum(v_ref))

    return raw_pitch


def raw_chroma_accuracy(ref_cent, ref_voicing, est_cent, est_voicing):
    '''
    Compute the raw chroma accuracy given two pitch (frequency) sequences in cents
    and matching voicing indicator sequences. The first pitch and voicing arrays
    are treated as the reference (truth), and the second two as the estimate (prediction).
    All 4 sequences must be of the same length.

    Input:
    - ref_cent : np.array
    Reference pitch sequence in cents

    - ref_voicing : np.array or list
    Reference voicing indicator where val>0 indicates voiced, val<=0 indicates unvoiced

    - est_cent : np.array
    Estimate pitch sequence in cents

    - est_voicing : np.array or list
    Estimate voicing indicator where val>0 indicates voiced, val<=0 indicates unvoiced

    Output:
    - raw_chroma: float
    Raw chroma accuracy, the fraction of voiced frames in ref_cent for which est_cent
    provides a correct frequency values (within 50 cents), ignoring octave errors
    '''

    l1,l2,l3,l4 = len(ref_cent),len(ref_voicing),len(est_cent),len(est_voicing)
    if l1 != l2 or l1 != l3 or l1 != l4:
        print "Error: all 4 sequences must be of the same length"
        return None

    # convert to booleans
    v_ref = np.asarray(ref_voicing) > 0

    # Raw chroma = same as raw pitch except that octave errors are ignored.
    cent_diff = np.abs(ref_cent - est_cent)
    cent_diff_chroma = abs(cent_diff - 1200 * np.floor(cent_diff / 1200.0 + 0.5))
    raw_chroma = sum(cent_diff_chroma[v_ref] <= 50) / float(sum(v_ref))

    return raw_chroma


def overall_accuracy(ref_cent, ref_voicing, est_cent, est_voicing):
    '''
    Compute the overall accuracy given two pitch (frequency) sequences in cents
    and matching voicing indicator sequences. The first pitch and voicing arrays
    are treated as the reference (truth), and the second two as the estimate (prediction).
    All 4 sequences must be of the same length.

    Input:
    - ref_cent : np.array
    Reference pitch sequence in cents

    - ref_voicing : np.array or list
    Reference voicing indicator where val>0 indicates voiced, val<=0 indicates unvoiced

    - est_cent : np.array
    Estimate pitch sequence in cents

    - est_voicing : np.array or list
    Estimate voicing indicator where val>0 indicates voiced, val<=0 indicates unvoiced

    Output:
    - overall_accuracy: float
    Overall accuracy, the total fraction of correctly estimates frames, where
    provides a correct frequency values (within 50 cents).
    '''

    l1,l2,l3,l4 = len(ref_cent),len(ref_voicing),len(est_cent),len(est_voicing)
    if l1 != l2 or l1 != l3 or l1 != l4:
        print "Error: all 4 sequences must be of the same length"
        return None

    # Compute boolean voicing indicators
    v_ref = np.asarray(ref_voicing) > 0
    v_est = np.asarray(est_voicing) > 0
    uv_ref = np.asarray(ref_voicing) <= 0
    uv_est = np.asarray(est_voicing) <= 0

    # True negatives = frames correctly estimates as unvoiced
    TN = sum(uv_ref * uv_est)

    cent_diff = np.abs(ref_cent - est_cent)
    overall_accuracy = (sum(cent_diff[v_ref * v_est] <= 50) + TN) / float(len(ref_cent))

    return overall_accuracy
