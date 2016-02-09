'''
The aim of a transcription algorithm is to produce a symbolic representation of
a recorded piece of music in the form of a set of discrete notes. There are
different ways to represent notes symbolically. Here we use the piano-roll
convention, meaning each notes has a start time, a duration (or end time), and
a single, constant, pitch value. Pitch values can be quantized (e.g. to a
semitone grid tuned to 440 Hz), but do not have to be. Also, the transcription
can contain the notes of a single instrument or voice (for example the melody),
or the notes of all instruments/voices in the recording. This module is
instrument agnostic: all notes in the estimate are compared against all notes
in the reference.

There are many metrics for evaluating transcription algorithms. Here we limit
ourselves to the most simple and commonly used: given two sets of notes, we
count how many estimate notes match the reference, and how many do not. Based
on these counts we compute the precision, recall, and f-measure of the estimate
given the reference. The default criteria for considering two notes to be a
match are adopted from the MIREX "Multiple fundamental frequency estimation and
tracking, task 2" criteria:
(http://www.music-ir.org/mirex/wiki/2015:Multiple_Fundamental_Frequency_
Estimation_%26_Tracking#Evaluation):

"A ground truth note is assumed to be correctly transcribed if the system
returns a note that is within a half semitone of that note AND the returned
note's onset is within a 100 ms range (+- 50ms) of the onset of the ground
truth note, and its offset is within 20% [of the ground truth note's duration]
range of the ground truth note's offset. Again, one ground truth note can only
be associated with one transcribed note."

Since note offsets are considerably harder to estimate than onsets, an option
is provided to only consider onsets for determining matches in the evaluation.
Similarly, the user can adjust any of the three thresholds (pitch, onset and
offset) to obtain a stricter or more lax evaluation.

For further details see Salamon, 2013 (page 186), and references therein:

    Salamon, J. (2013). Melody Extraction from Polyphonic Music Signals.
    Ph.D. thesis, Universitat Pompeu Fabra, Barcelona, Spain, 2013.

Conventions
-----------

Notes should be provided in the form of three 1-dimensional arrays: the first
containing note onset times in seconds, the second containing the corresponding
note offset times in seconds, and third containing the corresponding note pitch
values, represented by their fundamental frequency (f0) in Hertz.

Metrics
-------

* :func:`mir_eval.transcription.precision_recall_f1`: The precision, recall,
  and F-measure of the note transcription, where an estimated note is
  considered correct if its pitch, onset and offset are sufficiently close to
  a reference note

'''

import numpy as np
import collections
from . import util
import warnings


def validate(ref_intervals, ref_pitches, est_intervals, est_pitches):
    """Checks that the input annotations to a metric look like time intervals
    and a pitch list, and throws helpful errors if not.

    Parameters
    ----------
    ref_intervals : np.ndarray, shape=(n,2)
        Array of reference notes time intervals (onset and offset times)
    ref_pitches: list, len=n
        List of reference pitch values in Hertz
    est_intervals : np.ndarray, shape=(m,2)
        Array of estimated notes time intervals (onset and offset times)
    est_pitches : list, len=m
        List of estimated pitch values in Hertz
    """
    # If reference or estimated notes are empty, warn
    if ref_intervals.size == 0:
        warnings.warn("Reference note intervals are empty.")
    if len(ref_pitches) == 0:
        warnings.warn("Reference note pitches are empty.")
    if est_intervals.size == 0:
        warnings.warn("Estimate note intervals are empty.")
    if len(est_pitches) == 0:
        warnings.warn("Estimate note pitches are empty.")

    # Make sure intervals and pitches match in length
    if not len(ref_intervals)==len(ref_pitches):
        warnings.warn("Reference intervals and pitches have different "
                      "lengths.")
    if not len(est_intervals)==len(est_pitches):
        warnings.warn("Estimate intervals and pitches have different lengths.")

    # Make sure all pitch values are positive
    if np.min(ref_pitches) <= 0:
        warnings.warn("Reference contains at least one non-positive pitch "
                      "value")
    if np.min(est_pitches) <= 0:
        warnings.warn("Estimate contains at least one non-positive pitch "
                      "value")


def precision_recall_f1(ref_intervals, ref_pitches, est_intervals, est_pitches,
        onset_tolerance=0.05, offset_ratio=0.2, pitch_tolerance=50.0):
    """Compute the Precision, Recall and F-measure of correct vs incorrectly
    transcribed notes. "Correctness" is determined based on note onset, offset
    and pitch as detailed at the top of this document.

    Examples
    --------
    >>> ref_intervals, ref_pitches = mir_eval.io.load_valued_intervals(
    ... 'reference.txt')
    >>> est_intervals, est_pitches = mir_eval.io.load_valued_intervals(
    ... 'estimated.txt')
    >>> precision, recall, f_measure =
    ... mir_eval.transcription.precision_recall_f1(ref_intervals, ref_pitches,
    ... est_intervals, est_pitches)

    Parameters
    ----------
    ref_intervals : np.ndarray, shape=(n,2)
        Array of reference notes time intervals (onset and offset times)
    ref_pitches: list, len=n
        List of reference pitch values in Hertz
    est_intervals : np.ndarray, shape=(m,2)
        Array of estimated notes time intervals (onset and offset times)
    est_pitches : list, len=m
        List of estimated pitch values in Hertz
    onset_tolerance : float > 0
        The tolerance for an estimated note's onset deviating from the
        reference note's onset, in seconds. Default is 0.05 (50 ms).
    offset_ratio: float > 0
        The ratio of the reference note's duration used to define the
        offset_tolerance. Default is 0.2 (20%), meaning the offset_tolerance
        will equal the ref_duration * 0.2 * 0.5 (0.5 since the window is
        centered on the reference offset).
    pitch_tolerance: float > 0
        The tolerance for an estimated note's pitch deviating from the
        reference note's pitch, in cents. Default is 50.0 (50 cents).

    Returns (TODO)
    -------
    f_score : float
        The computed F-measure score

    """
    validate(ref_intervals, ref_pitches, est_intervals, est_pitches)
    # When reference notes are empty, metrics are undefined, return 0's
    if len(ref_pitches) == 0 or len(est_pitches) == 0:
        return 0., 0., 0.

    matching = util.match_notes(ref_intervals, ref_pitches, est_intervals,
                                est_pitches, onset_tolerance=onset_tolerance,
                                offset_ratio=offset_ratio,
                                pitch_tolerance=pitch_tolerance)

    precision = float(len(matching))/len(est_pitches)
    recall = float(len(matching))/len(ref_pitches)
    f_measure = util.f_measure(precision, recall)
    return precision, recall, f_measure


def evaluate(ref_intervals, ref_pitches, est_intervals, est_pitches, **kwargs):
    """Compute all metrics for the given reference and estimated annotations.

    Examples
    --------
    >>> ref_intervals, ref_pitches = mir_eval.io.load_valued_intervals(
    ... 'reference.txt')
    >>> est_intervals, est_pitches = mir_eval.io.load_valued_intervals(
    ... 'estimate.txt')
    >>> scores = mir_eval.transcription.evaluate(ref_intervals, ref_pitches,
    ... est_intervals, est_pitches)

    Parameters
    ----------
    ref_intervals : np.ndarray, shape=(n,2)
        Array of reference notes time intervals (onset and offset times)
    ref_pitches: list, len=n
        List of reference pitch values in Hertz
    est_intervals : np.ndarray, shape=(m,2)
        Array of estimated notes time intervals (onset and offset times)
    est_pitches : list, len=m
        List of estimated pitch values in Hertz
    kwargs
        Additional keyword arguments which will be passed to the
        appropriate metric or preprocessing functions.

    Returns
    -------
    scores : dict
        Dictionary of scores, where the key is the metric name (str) and
        the value is the (float) score achieved.

    """

    # Now compute all the metrics

    scores = collections.OrderedDict()

    # All metrics
    (scores['Precision'],
     scores['Recall'],
     scores['F-measure']) = util.filter_kwargs(precision_recall_f1,
                                               ref_intervals, ref_pitches,
                                               est_intervals, est_pitches,
                                               **kwargs)

    return scores
