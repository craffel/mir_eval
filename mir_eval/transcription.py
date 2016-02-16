'''
The aim of a transcription algorithm is to produce a symbolic representation of
a recorded piece of music in the form of a set of discrete notes. There are
different ways to represent notes symbolically. Here we use the piano-roll
convention, meaning each note has a start time, a duration (or end time), and
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
match are adopted from the `MIREX Multiple fundamental frequency estimation and
tracking, Note Tracking subtask (task 2) <http://www.music-ir.org/mirex/wiki/\
2015:Multiple_Fundamental_Frequency_Estimation_%26_Tracking_Results_-_MIREX_\
Dataset#Task_2:Note_Tracking_.28NT.29>`_:

"This subtask is evaluated in two different ways. In the first setup , a
returned note is assumed correct if its onset is within +-50ms of a ref note
and its F0 is within +- quarter tone of the corresponding reference note,
ignoring the returned offset values. In the second setup, on top of the above
requirements, a correct returned note is required to have an offset value
within 20% of the ref notes duration around the ref note's offset, or within
50ms whichever is larger."

In short, we compute precision, recall and f-measure, once without taking
offsets into account, and the second time with.

For further details see Salamon, 2013 (page 186), and references therein:

    Salamon, J. (2013). Melody Extraction from Polyphonic Music Signals.
    Ph.D. thesis, Universitat Pompeu Fabra, Barcelona, Spain, 2013.

Note: two different evaluation scripts have been used in MIREX over the years,
where one uses ``<`` for matching onsets, offsets, and pitch values, whilst
the other uses ``<=`` for these checks. That is, if the distance between two
onsets is exactly equal to the defined threshold (e.g. 0.05), then the former
script would not consider the two notes to have matching onsets, whilst the
latter would. `mir_eval` provides both options: by default the latter
(``<=``) is used, but you can set ``strict=True`` when calling
:func:`mir_eval.transcription.precision_recall_f1()` in which case ``<`` will
be used. The default value (``strict=False``) matches the evaluation code that
was used to produce the results reported on the MIREX website for the "Su"
dataset in 2015.


Conventions
-----------

Notes should be provided in the form of an interval array and a pitch array.
The interval array contains two columns, one for note onsets and the second
for note offsets (each row represents a single note). The pitch array contains
one column with the corresponding note pitch values (one value per note),
represented by their fundamental frequency (f0) in Hertz.

Metrics
-------

* :func:`mir_eval.transcription.precision_recall_f1`: The precision,
  recall, and F-measure of the note transcription, where an estimated note is
  considered correct if its pitch, onset and (optionally) offset are
  sufficiently close to a reference note

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
    ref_pitches: np.ndarray, shape=(n,)
        Array of reference pitch values in Hertz
    est_intervals : np.ndarray, shape=(m,2)
        Array of estimated notes time intervals (onset and offset times)
    est_pitches : np.ndarray, shape=(m,)
        Array of estimated pitch values in Hertz
    """
    # If reference or estimated notes are empty, warn
    if ref_intervals.size == 0:
        warnings.warn("Reference notes are empty.")
    if est_intervals.size == 0:
        warnings.warn("Estimate notes are empty.")

    # Make sure intervals and pitches match in length
    if not ref_intervals.shape[0] == ref_pitches.shape[0]:
        raise ValueError('Reference intervals and pitches have different '
                         'lengths.')
    if not est_intervals.shape[0] == est_pitches.shape[0]:
        raise ValueError('Estimate intervals and pitches have different '
                         'lengths.')

    # Make sure all pitch values are positive
    if np.min(ref_pitches) <= 0:
        raise ValueError("Reference contains at least one non-positive pitch "
                         "value")
    if np.min(est_pitches) <= 0:
        raise ValueError("Estimate contains at least one non-positive pitch "
                         "value")


def match_notes(ref_intervals, ref_pitches, est_intervals, est_pitches,
                onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=0.2,
                offset_min_tolerance=0.05, strict=False):
    """Compute a maximum matching between reference and estimated notes,
    subject to onset, pitch and (optionally) offset constraints.

    Given two note sequences represented by ``ref_intervals``, ``ref_pitches``,
    ``est_intervals`` and ``est_pitches`` (see ``io.load_valued_intervals``),
    we seek the largest set of correspondences ``(i, j)`` such that:

    1. The onset of ref note i is within ``onset_tolerance`` of the onset of
       est note j.
    2. The pitch of ref note i is within ``pitch_tolerance`` of the pitch of
       est note j.
    3. If ``offset_ratio`` is not ``None``, the offset of ref note i has to be
       within ``offset_tolerance`` of the offset of est note j, where
       ``offset_tolerance`` is equal to half the window given by taking
       ``offset_ratio`` of the ref note's duration, i.e. ``0.5 * offset_ratio *
       ref_duration[i]`` where ``ref_duration[i] = ref_intervals[i, 1] -
       ref_intervals[i, 0]``. If the resulting ``offset_tolerance`` is less
       than 0.05 (50 ms), 0.05 is used instead.
    4. If ``offset_ratio`` is ``None``, note offsets are ignored, and only
       criteria 1 and 2 are taken into consideration.

    Every ref note is matched against at most one est note.

    This is useful for computing precision/recall metrics for note
    transcription.

    Parameters
    ----------
    ref_intervals : np.ndarray, shape=(n,2)
        Array of reference notes time intervals (onset and offset times)
    ref_pitches: np.ndarray, shape=(n,)
        Array of reference pitch values in Hertz
    est_intervals : np.ndarray, shape=(m,2)
        Array of estimated notes time intervals (onset and offset times)
    est_pitches : np.ndarray, shape=(m,)
        Array of estimated pitch values in Hertz
    onset_tolerance : float > 0
        The tolerance for an estimated note's onset deviating from the
        reference note's onset, in seconds. Default is 0.05 (50 ms).
    pitch_tolerance: float > 0
        The tolerance for an estimated note's pitch deviating from the
        reference note's pitch, in cents. Default is 50.0 (50 cents).
    offset_ratio: float > 0 or None
        The ratio of the reference note's duration used to define the
        offset_tolerance. Default is 0.2 (20%), meaning the offset_tolerance
        will equal the ref_duration * 0.2 * 0.5 (0.5 since the window is
        centered on the reference offset), or 0.05 (50 ms), whichever is
        greater. If ``offset_ratio`` is set to ``None``, offsets are ignored in
        the matching.
    offset_min_tolerance: float > 0
        The minimum tolerance for offset matching. See offset_ratio description
        for an explanation of how the offset tolerance is determined. Note:
        this parameter only influences the results if ``offset_ratio`` is not
        ``None``.
    strict: bool
        If ``strict=False`` (the default), threshold checks for onset, offset,
        and pitch matching are performed using ``<=`` (less than or equal). If
        ``strict=True``, the threshold checks are performed using ``<`` (less
        than).

    Returns
    -------
    matching : list of tuples
        A list of matched reference and estimated notes.
        ``matching[i] == (i, j)`` where reference note i matches estimate note
        j.
    """
    # set the comparison function
    if strict:
        cmp = np.less
    else:
        cmp = np.less_equal

    # check for onset matches
    onset_distances = np.abs(np.subtract.outer(ref_intervals[:, 0],
                                               est_intervals[:, 0]))
    onset_hit_matrix = cmp(onset_distances, onset_tolerance)

    # check for pitch matches
    pitch_distances = np.abs(1200*np.log2(np.divide.outer(ref_pitches,
                                                          est_pitches)))
    pitch_hit_matrix = cmp(pitch_distances, pitch_tolerance)

    # check for offset matches if offset_ratio is not None
    if offset_ratio is not None:
        offset_distances = np.abs(np.subtract.outer(ref_intervals[:, 1],
                                                    est_intervals[:, 1]))
        ref_durations = util.intervals_to_durations(ref_intervals)
        offset_tolerances = 0.5 * offset_ratio * ref_durations
        offset_tolerance_matrix = np.tile(offset_tolerances,
                                          (offset_distances.shape[1], 1)).T
        min_tolerance_inds = offset_tolerance_matrix < offset_min_tolerance
        offset_tolerance_matrix[min_tolerance_inds] = offset_min_tolerance
        offset_hit_matrix = cmp(offset_distances, offset_tolerance_matrix)
    else:
        offset_hit_matrix = np.ones_like(onset_hit_matrix)

    # check for overall matches
    note_hit_matrix = onset_hit_matrix * pitch_hit_matrix * offset_hit_matrix
    hits = np.where(note_hit_matrix)

    # Construct the graph input
    # Flip graph so that 'matching' is a list of tuples where the first item
    # in each tuple is the reference note index, and the second item is the
    # estimate note index.
    G = {}
    for ref_i, est_i in zip(*hits):
        if est_i not in G:
            G[est_i] = []
        G[est_i].append(ref_i)

    # Compute the maximum matching
    matching = sorted(util._bipartite_match(G).items())

    return matching


def precision_recall_f1(ref_intervals, ref_pitches, est_intervals, est_pitches,
                        onset_tolerance=0.05, pitch_tolerance=50.0,
                        offset_ratio=0.2, offset_min_tolerance=0.05,
                        strict=False):
    """Compute the Precision, Recall and F-measure of correct vs incorrectly
    transcribed notes. "Correctness" is determined based on note onset, pitch
    and (optionally) offset: an estimated note is assumed correct if its onset
    is within +-50ms of a ref note and its pitch (F0) is within +- quarter tone
    (50 cents) of the corresponding reference note. If with_offset is False,
    note offsets are ignored in the comparison. It with_offset is True,
    on top of the above requirements, a correct returned note is required to
    have an offset value within 20% (by default, adjustable via the
    offset_ratio parameter) of the ref note's duration around the ref note's
    offset, or within offset_min_tolerance (50ms by default), whichever is
    larger.

    Examples
    --------
    >>> ref_intervals, ref_pitches = mir_eval.io.load_valued_intervals(
    ...     'reference.txt')
    >>> est_intervals, est_pitches = mir_eval.io.load_valued_intervals(
    ...     'estimated.txt')
    >>> precision, recall, f_measure =
    ...     mir_eval.transcription.precision_recall_f1(ref_intervals,
    ...     ref_pitches, est_intervals, est_pitches)
    >>> precision_no_offset, recall_no_offset, f_measure_no_offset =
    ...     mir_eval.transcription.precision_recall_f1(ref_intervals,
    ...     ref_pitches, est_intervals, est_pitches, offset_ratio=None)

    Parameters
    ----------
    ref_intervals : np.ndarray, shape=(n,2)
        Array of reference notes time intervals (onset and offset times)
    ref_pitches: np.ndarray, shape=(n,)
        Array of reference pitch values in Hertz
    est_intervals : np.ndarray, shape=(m,2)
        Array of estimated notes time intervals (onset and offset times)
    est_pitches : np.ndarray, shape=(m,)
        Array of estimated pitch values in Hertz
    onset_tolerance : float > 0
        The tolerance for an estimated note's onset deviating from the
        reference note's onset, in seconds. Default is 0.05 (50 ms).
    pitch_tolerance: float > 0
        The tolerance for an estimated note's pitch deviating from the
        reference note's pitch, in cents. Default is 50.0 (50 cents).
    offset_ratio: float > 0 or None
        The ratio of the reference note's duration used to define the
        offset_tolerance. Default is 0.2 (20%), meaning the offset_tolerance
        will equal the ref_duration * 0.2 * 0.5 (*0.5 since the window is
        centered on the reference offset), or min_offset_tolerance (0.05 by
        default, i.e. 50 ms), whichever is greater. If ``offset_ratio`` is set
        to ``None``, offsets are ignored in the evaluation.
    offset_min_tolerance: float > 0
        The minimum tolerance for offset matching. See offset_ratio description
        for an explanation of how the offset tolerance is determined. Note:
        this parameter only influences the results if offset_ratio is not
        ``None``.
    strict: bool
        If ``strict=False`` (the default), threshold checks for onset, offset,
        and pitch matching are performed using ``<=`` (less than or equal). If
        ``strict=True``, the threshold checks are performed using ``<`` (less
        than).

    Returns
    -------
    precision : float
        The computed precision score
    recall : float
        The computed recall score
    f_measure : float
        The computed F-measure score
    """
    validate(ref_intervals, ref_pitches, est_intervals, est_pitches)
    # When reference notes are empty, metrics are undefined, return 0's
    if len(ref_pitches) == 0 or len(est_pitches) == 0:
        return 0., 0., 0.

    matching = match_notes(ref_intervals, ref_pitches, est_intervals,
                           est_pitches, onset_tolerance=onset_tolerance,
                           pitch_tolerance=pitch_tolerance,
                           offset_ratio=offset_ratio,
                           offset_min_tolerance=offset_min_tolerance,
                           strict=strict)

    precision = float(len(matching))/len(est_pitches)
    recall = float(len(matching))/len(ref_pitches)
    f_measure = util.f_measure(precision, recall)
    return precision, recall, f_measure


def evaluate(ref_intervals, ref_pitches, est_intervals, est_pitches, **kwargs):
    """Compute all metrics for the given reference and estimated annotations.

    Examples
    --------
    >>> ref_intervals, ref_pitches = mir_eval.io.load_valued_intervals(
    ...     'reference.txt')
    >>> est_intervals, est_pitches = mir_eval.io.load_valued_intervals(
    ...     'estimate.txt')
    >>> scores = mir_eval.transcription.evaluate(ref_intervals, ref_pitches,
    ...     est_intervals, est_pitches)

    Parameters
    ----------
    ref_intervals : np.ndarray, shape=(n,2)
        Array of reference notes time intervals (onset and offset times)
    ref_pitches: np.ndarray, shape=(n,)
        Array of reference pitch values in Hertz
    est_intervals : np.ndarray, shape=(m,2)
        Array of estimated notes time intervals (onset and offset times)
    est_pitches : np.ndarray, shape=(m,)
        Array of estimated pitch values in Hertz
    kwargs
        Additional keyword arguments which will be passed to the
        appropriate metric or preprocessing functions.

    Returns
    -------
    scores : dict
        Dictionary of scores, where the key is the metric name (str) and
        the value is the (float) score achieved.
    """
    # Compute all the metrics
    scores = collections.OrderedDict()

    # Precision, recall and f-measure taking note offsets into account
    offset_ratio = kwargs.get('offset_ratio', .2)
    if offset_ratio is not None:
        (scores['Precision'],
         scores['Recall'],
         scores['F-measure']) = util.filter_kwargs(
            precision_recall_f1, ref_intervals, ref_pitches, est_intervals,
            est_pitches, offset_ratio=offset_ratio, **kwargs)

    # Precision, recall and f-measure NOT taking note offsets into account
    kwargs['offset_ratio'] = None
    (scores['Precision_no_offset'],
     scores['Recall_no_offset'],
     scores['F-measure_no_offset']) = \
        util.filter_kwargs(precision_recall_f1, ref_intervals, ref_pitches,
                           est_intervals, est_pitches, **kwargs)

    return scores
