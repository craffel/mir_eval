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
count how many estimate notes match the reference, and how many do not. Based on
these counts we compute the precision, recall, and f-measure of the estimate
given the reference. The default criteria for considering two notes to be a
match are adopted from the MIREX "Multiple fundamental frequency estimation and
tracking, task 2" criteria:
(http://www.music-ir.org/mirex/wiki/2015:Multiple_Fundamental_Frequency_Estimation_%26_Tracking#Evaluation):

"A ground truth note is assumed to be correctly transcribed if the system
returns a note that is within a half semitone of that note AND the returned
note's onset is within a 100 ms range (+- 50ms) of the onset of the ground
truth note, and its offset is within 20% [of the note's duration] range of the
ground truth note's offset. Again, one ground truth note can only be associated
with one transcribed note."

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

* :func:`mir_eval.transcription.prf`: The precision, recall, and F-measure of
  the note transcription, where an estimated note is considered correct if its
  pitch, onset and offset are sufficiently close to a reference note

'''

import numpy as np
import collections
from . import util
import warnings


def validate(ref_onsets, ref_offsets, ref_pitches, est_onsets, est_offsets,
             est_pitches):
    """Checks that the input annotations to a metric look like note onsets,
    offsets and pitch arrays, and throws helpful errors if not.

    Parameters
    ----------
    ref_onsets: np.ndarray
        reference note onset times, in seconds
    ref_offsets: np.ndarray
        reference note offset times, in seconds
    ref_pitches: np.ndarray
        reference note pitch values, in Hertz
    est_onsets : np.ndarray
        estimated note onset times, in seconds
    est_offsets: np.ndarray
        estimated note offset times, in seconds
    est_pitches: np.ndarray
        estimated note pitch values, in Hertz
    """
    # If reference or estimated beats are empty, warn
    if ref_onsets.size == 0:
        warnings.warn("Reference note onsets are empty.")
    if ref_offsets.size == 0:
        warnings.warn("Reference note offsets are empty.")
    if ref_pitches.size == 0:
        warnings.warn("Reference note pitches are empty.")
    if est_onsets.size == 0:
        warnings.warn("Estimated note onsets are empty.")
    if est_offsets.size == 0:
        warnings.warn("Estimated note offsets are empty.")
    if est_pitches.size == 0:
        warnings.warn("Estimated note pitches are empty.")

    # Make sure all three arrays of each transcription match in length
    if not (len(ref_onsets)==len(ref_offsets) and
            len(ref_onsets)==len(ref_pitches)):
        warnings.warn("Reference arrays have different lengths.")
    if not (len(est_onsets)==len(est_offsets) and
            len(est_onsets)==len(est_pitches)):
        warnings.warn("Estimate arrays have different lengths.")

    # Check for notes with negative duration
    negative_duration_note = False
    for onset, offset in zip(ref_onsets, ref_offsets):
        if offset < onset:
            negative_duration_note = True
    if negative_duration_note:
        warnings.warn("Reference contains at least one note with negative "
                      "duration")
    negative_duration_note = False
    for onset, offset in zip(est_onsets, est_offsets):
        if offset < onset:
            negative_duration_note = True
    if negative_duration_note:
        warnings.warn("Estimate contains at least one note with negative "
                      "duration")

    # Make sure all pitch values are positive
    if np.min(ref_pitches) <= 0:
        warnings.warn("Reference contains at least one non-positive pitch "
                      "value")
    if np.min(est_pitches) <= 0:
        warnings.warn("Estimate contains at least one non-positive pitch "
                      "value")


def prf(ref_onsets, ref_offsets, ref_pitches, est_onsets, est_offsets,
        est_pitches):
    """Compute the Precision, Recall and F-measure of correct vs incorrectly
    transcribed notes. "Correctness" is determined based on note onset, offset
    and pitch as detailed at the top of this document.

    Examples
    --------
    >>> ref_onsets, ref_offsets, ref_pitches = mir_eval.io.load_delimited(
    ... 'reference.txt', [float, float, float], '\t')
    >>> est_onsets, est_offsets, est_pitches = mir_eval.io.load_delimited(
    ... 'estimated.txt', [float, float, float], '\t')
    >>> precision, recall, f_measure = mir_eval.transcription.prf(ref_onests,
    ... ref_offsets, ref_pitches, est_onsets, est_offsets, est_pitches)

    Parameters (TODO)
    ----------
    reference_beats : np.ndarray
        reference beat times, in seconds
    estimated_beats : np.ndarray
        estimated beat times, in seconds
    f_measure_threshold : float
        Window size, in seconds
        (Default value = 0.07)

    Returns (TODO)
    -------
    f_score : float
        The computed F-measure score

    """
    validate(ref_onsets, ref_offsets, ref_pitches, est_onsets, est_offsets,
             est_pitches)
    # # When estimated beats are empty, no beats are correct; metric is 0
    # if ref_onsets.size == 0 or reference_beats.size == 0:
    #     return 0.
    # # Compute the best-case matching between reference and estimated locations
    # matching = util.match_events(reference_beats,
    #                              estimated_beats,
    #                              f_measure_threshold)
    #
    # precision = float(len(matching))/len(estimated_beats)
    # recall = float(len(matching))/len(reference_beats)
    # return util.f_measure(precision, recall)


def evaluate(ref_onsets, ref_offsets, ref_pitches, est_onsets, est_offsets,
             est_pitches, **kwargs):
    """Compute all metrics for the given reference and estimated annotations.

    Examples
    --------
    >>> ref_onsets, ref_offsets, ref_pitches = mir_eval.io.load_delimited(
    ... 'reference.txt', [float, float, float], '\t')
    >>> est_onsets, est_offsets, est_pitches = mir_eval.io.load_delimited(
    ... 'estimate.txt', [float, float, float], '\t')
    >>> scores = mir_eval.transcription.evaluate(ref_onsets, ref_offsets,
    ... ref_pitches, est_onsets, est_offsets, est_pitches)

    Parameters (TODO)
    ----------
    reference_beats : np.ndarray
        Reference beat times, in seconds
    estimated_beats : np.ndarray
        Query beat times, in seconds
    kwargs
        Additional keyword arguments which will be passed to the
        appropriate metric or preprocessing functions.

    Returns (TODO)
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
     scores['F-measure']) = util.filter_kwargs(prf, ref_onsets, ref_offsets,
                                               ref_pitches, est_onsets,
                                               est_offsets, est_pitches,
                                               **kwargs)

    return scores
