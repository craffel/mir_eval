"""
Alignment models are given a sequence of events along with a piece of audio, and then return a
sequence of timestamps, with one timestamp for each event, indicating the position of this event
in the audio. The events are listed in order of occurrence in the audio, so that output
timestamps have to be monotonically increasing.
Evaluation usually involves taking the series of predicted and ground truth timestamps and
comparing their distance, usually on a pair-wise basis, e.g. taking the median absolute error in
seconds.

Conventions
-----------
Timestamps should be provided in the form of a 1-dimensional array of onset
times in seconds in increasing order.

Metrics
-------
* :func:`mir_eval.alignment.ae`: Median absolute error and average absolute error
* :func:`mir_eval.alignment.pc`: Percentage of correct timestamps, where a timestamp is counted
as correct if it lies within a certain tolerance window around the ground truth timestamp
* :func:`mir_eval.alignment.pcs`: Percentage of correct segments: Percentage of overlap between
predicted segments and ground truth segments, where segments are defined by (start time,
end time) pairs
* :func:`mir_eval.alignment.perceptual_metric`: metric based on human synchronicity perception as
measured in the paper "User-centered evaluation of lyrics to audio alignment",
N. Lizé-Masclef, A. Vaglio, M. Moussallam, ISMIR 2021

References
----------
  .. [#lizemasclef2021] N. Lizé-Masclef, A. Vaglio, M. Moussallam.
    "User-centered evaluation of lyrics to audio alignment",
    International Society for Music Information Retrieval (ISMIR) conference,
    2021.

  .. [#mauch2010] M. Mauch, F: Hiromasa, M. Goto.
    "Lyrics-to-audio alignment and phrase-level segmentation using incomplete internet-style chord annotations",
    Frontiers in Proceedings of the Sound Music Computing Conference (SMC), 2010.

  .. [#dzhambazov2017] G. Dzhambazov.
    "Knowledge-Based Probabilistic Modeling For Tracking Lyrics In Music Audio Signals",
    PhD Thesis, 2017.

  .. [#fujihara2011] H. Fujihara, M. Goto, J. Ogata, H. Okuno.
    "LyricSynchronizer: Automatic synchronization system between musical audio signals and lyrics",
    IEEE Journal of Selected Topics in Signal Processing, VOL. 5, NO. 6, 2011

"""

import collections

import numpy as np
from scipy.stats import skewnorm

from mir_eval.util import filter_kwargs


def validate(reference_timestamps: np.ndarray, estimated_timestamps: np.ndarray):
    """Checks that the input annotations to a metric look like valid onset time
    arrays, and throws helpful errors if not.
    Parameters
    ----------
    reference_timestamps : np.ndarray
        reference timestamp locations, in seconds
    estimated_timestamps : np.ndarray
        estimated timestamp locations, in seconds
    """
    # We need to have 1D numpy arrays
    if not isinstance(reference_timestamps, np.ndarray):
        raise ValueError(
            f"Reference timestamps need to be a numpy array, but got {type(reference_timestamps)}"
        )
    if not isinstance(estimated_timestamps, np.ndarray):
        raise ValueError(
            f"Estimated timestamps need to be a numpy array, but got {type(estimated_timestamps)}"
        )
    if reference_timestamps.ndim != 1:
        raise ValueError(
            "Reference timestamps need to be a one-dimensional vector, but got"
            f" {reference_timestamps.ndim} dimensions"
        )
    if estimated_timestamps.ndim != 1:
        raise ValueError(
            "Estimated timestamps need to be a one-dimensional vector, but got"
            f" {estimated_timestamps.ndim} dimensions"
        )

    # If reference or estimated timestamps are empty, cannot compute metric
    if reference_timestamps.size == 0:
        raise ValueError("Reference timestamps are empty.")
    if estimated_timestamps.size != reference_timestamps.size:
        raise ValueError(
            "Number of timestamps must be the same in prediction and ground truth, "
            f"but found {estimated_timestamps.size} in prediction and "
            f"{reference_timestamps.size} in ground truth"
        )

    # Check monotonicity
    if not np.all(reference_timestamps[1:] - reference_timestamps[:-1] >= 0):
        raise ValueError("Reference timestamps are not monotonically increasing!")
    if not np.all(estimated_timestamps[1:] - estimated_timestamps[:-1] >= 0):
        raise ValueError("Estimated timestamps are not monotonically increasing!")

    # Check positivity (need for correct PCS metric calculation)
    if not np.all(reference_timestamps >= 0):
        raise ValueError("Reference timestamps can not be below 0!")
    if not np.all(estimated_timestamps >= 0):
        raise ValueError("Estimated timestamps can not be below 0!")


def ae(reference_timestamps, estimated_timestamps):
    """Compute the absolute deviations between predicted and reference timestamps,
    and then takes the median or average over all events

    Examples
    --------
    >>> reference_timestamps = mir_eval.io.load_events('reference.txt')
    >>> estimated_timestamps = mir_eval.io.load_events('estimated.txt')
    >>> mae, aae = mir_eval.align.ae(reference_onsets, estimated_timestamps)

    Parameters
    ----------
    reference_timestamps : np.ndarray
        reference timestamps, in seconds
    estimated_timestamps : np.ndarray
        estimated timestamps, in seconds
    Returns
    -------
    mae : float
        Median absolute error
    aae: float
        Average absolute error
    """
    validate(reference_timestamps, estimated_timestamps)
    deviations = np.abs(reference_timestamps - estimated_timestamps)
    return np.median(deviations), np.mean(deviations)


def pc(reference_timestamps, estimated_timestamps, window=0.3):
    """Compute the percentage of correctly predicted timestamps. A timestamp is predicted
    correctly if its position doesn't deviate more than the window parameter from the ground
    truth timestamp.

    Examples
    --------
    >>> reference_timestamps = mir_eval.io.load_events('reference.txt')
    >>> estimated_timestamps = mir_eval.io.load_events('estimated.txt')
    >>> pc = mir_eval.align.pc(reference_onsets, estimated_timestamps, window=0.2)

    Parameters
    ----------
    reference_timestamps : np.ndarray
        reference timestamps, in seconds
    estimated_timestamps : np.ndarray
        estimated timestamps, in seconds
    window : float
        Window size, in seconds
        (Default value = .1)
    Returns
    -------
    pc : float
        Percentage of correct timestamps
    """
    validate(reference_timestamps, estimated_timestamps)
    deviations = np.abs(reference_timestamps - estimated_timestamps)
    return np.mean(deviations <= window)


def pcs(reference_timestamps, estimated_timestamps, duration: float):
    """Constructs segments out of predicted and estimated timestamps separately
    out of each given timestamp vector with entries (t1,t2, ... tN), yielding segments with
    the following (start, end) boundaries: (0, t1), (t1, t2), ... (tN, duration).
    The metric then calculates the percentage of overlap between correct segments compared to the
    total duration.

    This metric is based on the paper
    "LyricSynchronizer: Automatic synchronization system between musical audio signals and lyrics"

    Examples
    --------
    >>> reference_timestamps = mir_eval.io.load_events('reference.txt')
    >>> estimated_timestamps = mir_eval.io.load_events('estimated.txt')
    >>> duration = max(np.max(reference_timestamps), np.max(estimated_timestamps)) + 10
    >>> pcs = mir_eval.align.pcs(reference_onsets, estimated_timestamps, duration)

    Parameters
    ----------
    reference_timestamps : np.ndarray
        reference timestamps, in seconds
    estimated_timestamps : np.ndarray
        estimated timestamps, in seconds
    duration : float
        Total duration of audio (seconds) - this is needed to compute the final segments properly!
    Returns
    -------
    pcs : float
        Percentage of time where ground truth and predicted segments overlap
    """
    validate(reference_timestamps, estimated_timestamps)
    duration = float(duration)
    assert np.max(reference_timestamps) <= duration, (
        "Expected largest reference timestamp "
        f"{np.max(reference_timestamps)} to not be "
        f"larger than duration {duration}"
    )
    assert np.max(estimated_timestamps) <= duration, (
        "Expected largest estimated timestamp "
        f"{np.max(estimated_timestamps)} to not be "
        f"larger than duration {duration}"
    )

    ref_starts = np.concatenate([[0], reference_timestamps])
    ref_ends = np.concatenate([reference_timestamps, [duration]])
    est_starts = np.concatenate([[0], estimated_timestamps])
    est_ends = np.concatenate([estimated_timestamps, [duration]])
    overlap_starts = np.maximum(ref_starts, est_starts)
    overlap_ends = np.minimum(ref_ends, est_ends)
    overlap_duration = np.sum(np.maximum(overlap_ends - overlap_starts, 0))
    return overlap_duration / duration


def perceptual_synchrony(offset):
    """function perceptually weighting the deviations of onsets"""
    skewness = 1.12244251
    localisation = -0.22270315
    scale = 0.29779424
    normalisation_factor = 1.6857
    return (1.0 / normalisation_factor) * skewnorm.pdf(
        offset, skewness, loc=localisation, scale=scale
    )


def perceptual_metric(reference_timestamps, estimated_timestamps):
    """metric based on human synchronicity perception as measured in the paper

    "User-centered evaluation of lyrics to audio alignment",
    N. Lizé-Masclef, A. Vaglio, M. Moussallam, ISMIR 2021

    The parameters of this function were tuned on data collected through a user Karaoke-like
    experiment
    It reflects human judgment of how "synchronous" lyrics and audio stimuli are perceived
    in that setup.
    Beware that this metric is non-symmetrical and by construction it is also not equal to 1 at 0.

    Examples
    --------
    >>> reference_timestamps = mir_eval.io.load_events('reference.txt')
    >>> estimated_timestamps = mir_eval.io.load_events('estimated.txt')
    >>> perceptual = mir_eval.align.perceptual_metric(reference_onsets, estimated_timestamps)

    Parameters
    ----------
    reference_timestamps : np.ndarray
        reference timestamps, in seconds
    estimated_timestamps : np.ndarray
        estimated timestamps, in seconds
    Returns
    -------
    perceptual_score : float
        Perceptual score, averaged over all timestamps
    """
    validate(reference_timestamps, estimated_timestamps)
    perceptual_score = perceptual_synchrony(estimated_timestamps - reference_timestamps)
    return np.mean(perceptual_score)


def evaluate(reference_timestamps, estimated_timestamps, duration: float, **kwargs):
    """Compute all metrics for the given reference and estimated annotations.
    Examples
    --------
    >>> reference_timestamps = mir_eval.io.load_events('reference.txt')
    >>> estimated_timestamps = mir_eval.io.load_events('estimated.txt')
    >>> duration = max(np.max(reference_timestamps), np.max(estimated_timestamps)) + 10
    >>> scores = mir_eval.align.evaluate(reference_onsets, estimated_timestamps, duration)

    Parameters
    ----------
    reference_timestamps : np.ndarray
        reference timestamp locations, in seconds
    estimated_timestamps : np.ndarray
        estimated timestamp locations, in seconds
    duration : float
        Overall length of the input in seconds, needed to determine end of segments in the PCS
        metric
    kwargs
        Additional keyword arguments which will be passed to the
        appropriate metric or preprocessing functions.
    Returns
    -------
    scores : dict
        Dictionary of scores, where the key is the metric name (str) and
        the value is the (float) score achieved.
    """
    # Compute all metrics
    scores = collections.OrderedDict()

    scores["pc"] = filter_kwargs(pc, reference_timestamps, estimated_timestamps, **kwargs)
    scores["mae"], scores["aae"] = ae(reference_timestamps, estimated_timestamps)
    scores["pcs"] = pcs(reference_timestamps, estimated_timestamps, duration)
    scores["perceptual"] = perceptual_metric(reference_timestamps, estimated_timestamps)

    return scores
