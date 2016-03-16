'''
The goal of multiple f0 (multipitch) estimation and tracking is to identify
all of the active fundamental frequencies in each time frame in a complex music
signal.

Conventions
-----------
Multipitch estimates are represented by a timebase and a corresponding list
of arrays of frequency estimates. Frequency estimates may have any number of
frequency values, including 0 (represented by an empty array). Time values are
in units of seconds and frequency estimates are in units of Hz.

Estimate time series should ideally be equal to reference time series, but if
this is not the case, the estimate time series is resampled using a nearest
neighbor interpolation to match the estimate. Time values in the estimate time
series that are outside of the range of the reference time series are given
null (empty array) frequencies.

By default, a frequency is "correct" if it is within 0.5 semitones of a
reference frequency. Frequency values are compared by first mapping them to
log-2 semitone space, where the distance between semitones is constant.
Chroma-wrapped frequency values are computed by taking the log-2 frequency
values modulo 12 to map them down to a single octave. A chroma-wrapped
frequency estimate is correct if it's single-octave value is within 0.5
semitones of the single-octave reference frequency.

The metrics are based on those described in
[#poliner2007]_ and [#bay2009]_.

Metrics
-------
* :func:`mir_eval.multipitch.accuracy`: Precision, Recall, and Accuracy scores
  based on the number of esimated frequencies which are sufficiently close to
  the reference frequencies.

* :func:`mir_eval.multipitch.error_score`: Substitution, Miss, False Alarm and
  Total Error scores based on the number of esimated frequencies which are
  sufficiently close to the reference frequencies.


:references:
    .. [#poliner2007] G. E. Poliner, and D. P. W. Ellis, "A Discriminative
    Model for Polyphonic Piano Transription", EURASIP Journal on Advances in
    Signal Processing, 2007(1):154-163, Jan. 2007.
    .. [#bay2009] Bay, M., Ehmann, A. F., & Downie, J. S. (2009). Evaluation of
    Multiple-F0 Estimation and Tracking Systems. In ISMIR (pp. 315-320).
'''

import numpy as np
import collections
import scipy.interpolate
from . import util
import warnings


MAX_TIME = 30000.  # The maximum allowable time stamp (seconds)
MAX_FREQ = 5000.  # The maximum allowable frequency (Hz)
MIN_FREQ = 20.  # The minimum allowable frequency (Hz)


def validate(ref_time, ref_freqs, est_time, est_freqs):
    """Checks that the time and frequency inputs are well-formed.

    Parameters
    ----------
    ref_time : np.ndarray
        reference time stamps in seconds
    ref_freqs : list of np.ndarray
        reference frequencies in Hz
    est_time : np.ndarray
        estimate time stamps in seconds
    est_freqs : list of np.ndarray
        estimated frequencies in Hz

    """

    util.validate_events(ref_time, max_time=MAX_TIME)
    util.validate_events(est_time, max_time=MAX_TIME)

    if ref_time.size == 0:
        warnings.warn("Reference times are empty.")
    if ref_time.ndim != 1:
        raise ValueError("Reference times have invalid dimension")
    if len(ref_freqs) == 0:
        warnings.warn("Reference frequencies are empty.")
    if est_time.size == 0:
        warnings.warn("Estimated times are empty.")
    if est_time.ndim != 1:
        raise ValueError("Estimated times have invalid dimension")
    if len(est_freqs) == 0:
        warnings.warn("Estimated frequencies are empty.")
    if ref_time.size != len(ref_freqs):
        raise ValueError('Reference times and frequencies have unequal '
                         'lengths.')
    if est_time.size != len(est_freqs):
        raise ValueError('Estimate times and frequencies have unequal '
                         'lengths.')

    for freq in ref_freqs:
        util.validate_frequencies(freq, max_freq=MAX_FREQ, min_freq=MIN_FREQ,
                                  allow_negatives=False)

    for freq in est_freqs:
        util.validate_frequencies(freq, max_freq=MAX_FREQ, min_freq=MIN_FREQ,
                                  allow_negatives=False)


def resample_multipitch(times, frequencies, target_times):
    """Resamples multipitch time series to a new timescale. Values in
    `target_times` outside the range of `times` return no pitch estimate.

    Parameters
    ----------
    times : np.ndarray
        Array of time stamps
    frequencies : list of np.ndarray
        List of np.ndarrays of frequency values
    target_times : np.ndarray
        Array of target time stamps

    Returns
    -------
    frequencies_resampled : list of lists
        Frequency list of lists resampled to new timebase
    """
    if target_times.size == 0:
        return []

    if times.size == 0:
        return [np.array([])]*len(target_times)

    n_times = len(frequencies)

    # scipy's interpolate doesn't handle ragged arrays. Instead, we interpolate
    # the frequency index and then map back to the frequency values.
    # This only works because we're using a nearest neighbor interpolator!
    frequency_index = np.arange(0, n_times)

    # assume_sorted=True for efficiency
    # since we're interpolating the index, fill_value is set to the first index
    # that is out of range. We handle this in the next line.
    new_frequency_index = scipy.interpolate.interp1d(
        times, frequency_index, kind='nearest', bounds_error=False,
        assume_sorted=True, fill_value=n_times)(target_times)

    # create array of frequencies plus additional empty element at the end for
    # target time stamps that are out of the interpolation range
    freq_vals = frequencies + [np.array([])]

    # map interpolated indices back to frequency values
    frequencies_resampled = [
        freq_vals[i] for i in new_frequency_index.astype(int)]

    return frequencies_resampled


def frequencies_to_logscale(frequencies):
    """Converts frequencies to semitone log scale.

    Parameters
    ----------
    frequencies : list of np.ndarray
        Original frequency values

    Returns
    -------
    frequencies_logscale : list of np.ndarray
        Frequency values in semitone log scale.
    """
    return [12.0*np.log2(freqs) for freqs in frequencies]


def logscale_to_single_octave(frequencies_logscale):
    """Wrap log scale frequencies to a single octave.

    Parameters
    ----------
    frequencies_logscale : list of np.ndarray
        Log scale frequency values

    Returns
    -------
    frequencies_logscale_chroma : list of np.ndarray
        Log scale frequency values wrapped to one octave.

    """
    return [np.mod(freqs, 12) for freqs in frequencies_logscale]


def compute_num_freqs(frequencies):
    """Computes the number of frequencies for each time point.

    Parameters
    ----------
    frequencies : list of np.ndarray
        Frequency values

    Returns
    -------
    num_freqs : np.ndarray
        Number of frequencies at each time point.
    """
    return np.array([f.size for f in frequencies])


def compute_num_true_positives(ref_freqs, est_freqs, window=0.5):
    """Compute the number of true positives in an estimate given a reference.
    A frequency is correct if it is within a quartertone of the
    correct frequency.

    Parameters
    ----------
    ref_freqs : list of np.ndarray
        reference frequencies in semitones
    est_freqs : list of np.ndarray
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
        # match frequency events within tolerance window in semitones
        matching = util.match_events(ref_frame, est_frame, window)
        true_positives[i] = len(matching)

    return true_positives


def accuracy(true_positives, n_ref, n_est):
    """Compute accuracy metrics.

    Parameters
    ----------
    true_positives : np.ndarray
        Array containing the number of true positives at each time point.
    n_ref : np.ndarray
        Array containing the number of reference frequencies at each time
        point.
    n_est : np.ndarray
        Array containing the number of estimate frequencies at each time point.

    Returns
    -------
    precision : float
        sum(true_positives)/sum(n_est)
    recall : float
        sum(true_positives)/sum(n_ref)
    acc : float
        sum(true_positives)/sum(n_est + n_ref - true_positives)

    """
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
        acc = true_positive_sum/acc_denom
    else:
        acc = 0.0

    return precision, recall, acc


def error_score(true_positives, n_ref, n_est):
    """Compute error score metrics.

    Parameters
    ----------
    true_positives : np.ndarray
        Array containing the number of true positives at each time point.
    n_ref : np.ndarray
        Array containing the number of reference frequencies at each time
        point.
    n_est : np.ndarray
        Array containing the number of estimate frequencies at each time point.

    Returns
    -------
    e_sub : float
        Substitution error
    e_miss : float
        Miss error
    e_fa : float
        False alarm error
    e_tot : float
        Total error

    """
    n_ref_sum = float(n_ref.sum())

    if n_ref_sum == 0:
        warnings.warn("Reference frequencies are all empty.")
        return 0., 0., 0., 0.

    # Substitution error
    e_sub = (np.min([n_ref, n_est], axis=0) - true_positives).sum()/n_ref_sum

    # compute the max of (n_ref - n_est) and 0
    e_miss_numerator = n_ref - n_est
    e_miss_numerator[e_miss_numerator < 0] = 0
    # Miss error
    e_miss = e_miss_numerator.sum()/n_ref_sum

    # compute the max of (n_est - n_ref) and 0
    e_fa_numerator = n_est - n_ref
    e_fa_numerator[e_fa_numerator < 0] = 0
    # False alarm error
    e_fa = e_fa_numerator.sum()/n_ref_sum

    # total error
    e_tot = (np.max([n_ref, n_est], axis=0) - true_positives).sum()/n_ref_sum

    return e_sub, e_miss, e_fa, e_tot


def evaluate(ref_time, ref_freqs, est_time, est_freqs, **kwargs):
    """Evaluate two multipitch (multi-f0) transcriptions, where the first is
    treated as the reference (ground truth) and the second as the estimate to
    be evaluated (prediction).

    Examples
    --------
    >>> ref_time, ref_freq = mir_eval.io.load_ragged_time_series('ref.txt')
    >>> est_time, est_freq = mir_eval.io.load_ragged_time_series('est.txt')
    >>> scores = mir_eval.multipitch.evaluate(ref_time, ref_freq,
    ...                                       est_time, est_freq)

    Parameters
    ----------
    ref_time : np.ndarray
        Time of each reference frequency value
    ref_freqs : list of np.ndarray
        List of np.ndarrays of reference frequency values
    est_time : np.ndarray
        Time of each estimated frequency value
    est_freqs : list of np.ndarray
        List of np.ndarrays of estimate frequency values
    kwargs
        Additional keyword arguments which will be passed to the
        appropriate metric or preprocessing functions.

    Returns
    -------
    scores : dict
        Dictionary of scores, where the key is the metric name (str) and
        the value is the (float) score achieved.

    """
    validate(ref_time, ref_freqs, est_time, est_freqs)

    if est_time.size != ref_time.size or not np.allclose(est_time, ref_time):
        warnings.warn("Estimate times not equal to reference times. "
                      "Resampling to common time base.")
        est_freqs = resample_multipitch(est_time, est_freqs, ref_time)

    ref_freqs_log = frequencies_to_logscale(ref_freqs)
    est_freqs_log = frequencies_to_logscale(est_freqs)

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
     scores['Accuracy']) = accuracy(
         true_positives, n_ref, n_est)

    (scores['Chroma Precision'],
     scores['Chroma Recall'],
     scores['Chroma Accuracy']) = accuracy(
         true_positives_chroma, n_ref, n_est)

    (scores['Substitution Error'],
     scores['Miss Error'],
     scores['False Alarm Error'],
     scores['Total Error']) = error_score(
         true_positives, n_ref, n_est)

    (scores['Chroma Substitution Error'],
     scores['Chroma Miss Error'],
     scores['Chroma False Alarm Error'],
     scores['Chroma Total Error']) = error_score(
         true_positives_chroma, n_ref, n_est)

    return scores
