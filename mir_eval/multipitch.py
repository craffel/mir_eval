'''
The goal of multiple f0 (multipitch) estimation and tracking is to identify
the active fundamental frequencies in each time frame in a complex music
signal.

For a detailed explanation of the measures please refer to:
    G. E. Poliner, and D. P. W. Ellis, "A Discriminative Model for Polyphonic
    Piano Transription", EURASIP Journal on Advances in Signal Processing,
    2007(1):154-163, Jan. 2007.

Conventions
-----------

Metrics
-------

'''

import numpy as np
import collections
from scipy.interpolate import interp1d
from . import util
import warnings


# The maximum allowable time stamp (seconds)
MAX_TIME = 30000.

# The maximum allowable frequency (Hz)
MAX_FREQ = 5000.
MIN_FREQ = 20.


def validate(ref_time, ref_freqs, est_time, est_freqs):
    """Checks that the time and frequency inputs are well-formed.

    Parameters
    ----------
    ref_time : np.ndarray
        reference time stamps in seconds
    ref_freqs : list of np.ndarrays
        reference frequencies in Hz
    est_time : np.ndarray
        estimate time stamps in seconds
    est_freqs : list of np.ndarrays
        estimated frequencies in Hz

    """
    if ref_time.size == 0:
        raise ValueError("Reference frequencies are empty.")
    if len(ref_freqs) == 0:
        raise ValueError("Reference frequencies are empty.")
    if est_time.size == 0:
        raise ValueError("Estimated times are empty.")
    if len(est_freqs) == 0:
        raise ValueError("Estimated frequencies are empty.")
    if ref_time.size != len(ref_freqs):
        raise ValueError('Reference times and frequencies have unequal '
                         'lengths.')
    if est_time.size != len(est_freqs):
        raise ValueError('Estimate times and frequencies have unequal '
                         'lengths.')

    util.validate_events(ref_time, max_time=MAX_TIME)
    util.validate_events(est_time, max_time=MAX_TIME)

    for freq in ref_freqs:
        util.validate_frequencies(freq, max_freq=MAX_FREQ, min_freq=MIN_FREQ)

    for freq in est_freqs:
        util.validate_frequencies(freq)


def resample_multipitch(times, frequencies, target_times):
    """Resamples multipitch time series to a new timescale. Values in
    'target_times' outside the range of 'times' return no pitch estimate.

    Parameters
    ----------
    times : np.ndarrayre
        Array of time stamps
    frequencies : list of np.ndarrays
        List of np.ndarrays of frequency values
    target_times : np.ndarray
        Array of target time stamps

    Returns
    -------
    frequencies_resampled : list of lists
        Frequency list of lists resampled to new timebase
    """
    n_times = len(frequencies)

    frequency_index = np.arange(0, n_times)
    new_frequency_index = interp1d(
        times, frequency_index, kind='nearest', bounds_error=False,
        assume_sorted=True, fill_value=n_times)(target_times)

    freq_vals = frequencies + [np.array([])]
    frequencies_resampled = [
        freq_vals[i] for i in new_frequency_index.astype(int)]

    return frequencies_resampled


def frequencies_to_logscale(frequencies):
    """Converts frequencies to semitone log scale.

    Parameters
    ----------
    frequencies : list of np.ndarrays
        Original frequency values

    Returns
    -------
    frequencies_logscale : list of np.ndarrays
        Frequency values in semitone log scale.
    """
    return [12.0*np.log2(freqs) for freqs in frequencies]


def logscale_to_single_octave(frequencies_logscale):
    return [np.mod(freqs, 12) for freqs in frequencies_logscale]


def compute_num_freqs(frequencies):
    return np.array([f.size for f in frequencies])


def compute_num_true_positives(ref_freqs, est_freqs, window=0.5):
    """Compute the number of true positives in an estimate given a reference.
    A frequency is correct if it is within a semitone of the
    correct frequency.

    Parameters
    ----------
    ref_freqs : list of np.ndarrays
        reference frequencies in semitones
    est_freqs : list of np.ndarrays
        estimated frequencies in semitones
    window : float
        Window size, in semitones

    Returns
    -------
    true_positives : np.ndarray
        Array the same length as ref_freqs containing the number of true
        positives.

    """
    n_frames = len(ref_freqs)
    true_positives = np.zeros((n_frames, ))

    for i, (ref_frame, est_frame) in enumerate(zip(ref_freqs, est_freqs)):
        if len(ref_frame) == 0 or len(est_frame) == 0:
            true_positives[i] = 0.0
        else:
            matching = util.match_events(ref_frame, est_frame, window)
            true_positives[i] = len(matching)

    return true_positives


def compute_accuracy_metrics(true_positives, n_ref, n_est):
    true_positive_sum = float(true_positives.sum())

    n_est_sum = n_est.sum()
    if n_est_sum > 0:
        precision = true_positive_sum/n_est.sum()
    else:
        warnings.warn("Estimate frequencies are all empty.")
        precision = 0.0

    n_ref_sum = n_ref.sum()
    if n_ref_sum > 0:
        recall = true_positive_sum/n_ref.sum()
    else:
        warnings.warn("Reference frequencies are all empty.")
        recall = 0.0

    acc_denom = (n_est + n_ref - true_positives).sum()
    if acc_denom > 0:
        accuracy = true_positive_sum/acc_denom
    else:
        accuracy = 0.0

    return precision, recall, accuracy


def compute_error_score_metrics(true_positives, n_ref, n_est):
    n_ref_sum = float(n_ref.sum())

    if n_ref_sum == 0:
        warnings.warn("Reference frequencies are all empty.")
        return 0., 0., 0., 0.

    e_sub = (np.min([n_ref, n_est], axis=0) - true_positives).sum()/n_ref_sum

    e_miss_numerator = n_ref - n_est
    e_miss_numerator[e_miss_numerator < 0] = 0
    e_miss = e_miss_numerator.sum()/n_ref_sum

    e_fa_numerator = n_est - n_ref
    e_fa_numerator[e_fa_numerator < 0] = 0
    e_fa = e_fa_numerator.sum()/n_ref_sum

    e_tot = (np.max([n_ref, n_est], axis=0) - true_positives).sum()/n_ref_sum

    return e_sub, e_miss, e_fa, e_tot


def evaluate(ref_time, ref_freqs, est_time, est_freqs, **kwargs):

    validate(ref_time, ref_freqs, est_time, est_freqs)

    est_freqs_resampled = resample_multipitch(est_time, est_freqs, ref_time)

    ref_freqs_log = frequencies_to_logscale(ref_freqs)
    est_freqs_log = frequencies_to_logscale(est_freqs_resampled)

    ref_freqs_log_chroma = logscale_to_single_octave(ref_freqs_log)
    est_freqs_log_chroma = logscale_to_single_octave(est_freqs_log)

    n_ref = compute_num_freqs(ref_freqs_log)
    n_est = compute_num_freqs(est_freqs_log)

    true_positives = util.filter_kwargs(
        compute_num_true_positives, ref_freqs_log, est_freqs_log, **kwargs)

    true_positives_chroma = util.filter_kwargs(
        compute_num_true_positives, ref_freqs_log_chroma, est_freqs_log_chroma,
        **kwargs)

    scores = collections.OrderedDict()

    (scores['Precision'],
     scores['Recall'],
     scores['Accuracy']) = compute_accuracy_metrics(
         true_positives, n_ref, n_est)

    (scores['Chroma Precision'],
     scores['Chroma Recall'],
     scores['Chroma Accuracy']) = compute_accuracy_metrics(
         true_positives_chroma, n_ref, n_est)

    (scores['Substitution Error'],
     scores['Miss Error'],
     scores['False Alarm Error'],
     scores['Total Error']) = compute_error_score_metrics(
         true_positives, n_ref, n_est)

    (scores['Chroma Substitution Error'],
     scores['Chroma Miss Error'],
     scores['Chroma False Alarm Error'],
     scores['Chroma Total Error']) = compute_error_score_metrics(
         true_positives_chroma, n_ref, n_est)

    return scores
