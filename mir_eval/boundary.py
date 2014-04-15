# CREATED:2013-08-13 12:02:42 by Brian McFee <brm2132@columbia.edu>
'''Structural segmentation evaluation, following the protocols of MIREX2012.
    - Boundary detection
        - (precision, recall, f-measure)
        - median distance to nearest boundary
'''

import numpy as np
import functools

from . import util

def __validate_intervals(intervals):

    # Validate interval shape
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError('Segment intervals should be n-by-2 numpy ndarray')

    # Make sure no beat times are negative
    if (intervals < 0).any():
        raise ValueError('Negative interval times found')

def validate(metric):
    '''Decorator which checks that the input annotations to a metric
    look like valid segment times, and throws helpful errors if not.

    :parameters:
        - metric : function
            Evaluation metric function.  First two arguments must be
            reference_intervals and estimated_intervals.

    :returns:
        - metric_validated : function
            The function with the segment intervals are validated
    '''
    @functools.wraps(metric)
    def metric_validated(reference_intervals, estimated_intervals, *args, **kwargs):
        for intervals in [reference_intervals, estimated_intervals]:
            __validate_intervals(intervals)

        return metric(reference_intervals, estimated_intervals, *args, **kwargs)

    return metric_validated

@validate
def detection(reference_intervals, estimated_intervals, window=0.5, beta=1.0, trim=True):
    '''Boundary detection hit-rate.

    A hit is counted whenever an reference boundary is within ``window`` of a estimated
    boundary.

    :usage:
        >>> # With 0.5s windowing
        >>> reference, true_labels = mir_eval.io.load_annotation('truth.lab')
        >>> estimated, pred_labels = mir_eval.io.load_annotation('prediction.lab')
        >>> P05, R05, F05 = mir_eval.segment.boundary_detection(reference, estimated, window=0.5)
        >>> # With 3s windowing
        >>> P3, R3, F3 = mir_eval.segment.boundary_detection(reference, estimated, window=3)


    :parameters:
        - reference_intervals : np.ndarray, shape=(n, 2)
            reference segment intervals, as returned by `mir_eval.io.load_annotation`

        - estimated_intervals : np.ndarray, shape=(m, 2)
            estimated segment intervals, as returned by `mir_eval.io.load_annotation`

        - window : float > 0
            size of the window of 'correctness' around ground-truth beats (in seconds)

        - beta : float > 0
            weighting constant for F-measure.

        - trim : boolean
            if ``True``, the first and last boundary times are ignored.
            Typically, these denote start (0) and end-markers.

    :returns:
        - precision : float
            precision of predictions

        - recall : float
            recall of ground-truth beats

        - f_measure : float
            F-measure (weighted harmonic mean of ``precision`` and ``recall``)
    '''

    # Convert intervals to boundaries
    reference_boundaries = util.intervals_to_boundaries(reference_intervals)
    estimated_boundaries = util.intervals_to_boundaries(estimated_intervals)

    # Suppress the first and last intervals
    if trim:
        reference_boundaries = reference_boundaries[1:-1]
        estimated_boundaries = estimated_boundaries[1:-1]

    # If we have no boundaries, we get no score.
    if len(reference_boundaries) == 0 or len(estimated_boundaries) == 0:
        return 0.0, 0.0, 0.0

    # Compute the hits
    dist        = np.abs( np.subtract.outer(reference_boundaries, estimated_boundaries)) <= window

    # Precision: how many estimated intervals were hits?
    precision   = np.mean(dist.max(axis=0))

    # Recall: how many of the intervals did we catch?
    recall      = np.mean(dist.max(axis=1))

    # And the f-measure
    f_measure   = util.f_measure(precision, recall, beta=beta)

    return precision, recall, f_measure

@validate
def deviation(reference_intervals, estimated_intervals, trim=True):
    '''Compute the median deviations between reference and estimated boundary times.

    :usage:
        >>> reference, true_labels = mir_eval.io.load_annotation('truth.lab')
        >>> estimated, pred_labels = mir_eval.io.load_annotation('prediction.lab')
        >>> t_to_p, p_to_t = mir_eval.segment.boundary_deviation(reference, estimated)

    :parameters:
        - reference_intervals : np.ndarray, shape=(n, 2)
            reference segment intervals, as returned by `mir_eval.io.load_annotation`

        - estimated_intervals : np.ndarray, shape=(m, 2)
            estimated segment intervals, as returned by `mir_eval.io.load_annotation`

        - trim : boolean
            if ``True``, the first and last intervals are ignored.
            Typically, these denote start (0) and end-markers.

    :returns:
        - true_to_estimated : float
            median time from each true boundary to the closest estimated boundary

        - estimated_to_true : float
            median time from each estimated boundary to the closest true boundary
    '''

    # Convert intervals to boundaries
    reference_boundaries = util.intervals_to_boundaries(reference_intervals)
    estimated_boundaries = util.intervals_to_boundaries(estimated_intervals)

    # Suppress the first and last intervals
    if trim:
        reference_boundaries = reference_boundaries[1:-1]
        estimated_boundaries = estimated_boundaries[1:-1]

    # If we have no boundaries, we get no score.
    if len(reference_boundaries) == 0 or len(estimated_boundaries) == 0:
        return 0.0, 0.0, 0.0

    dist = np.abs( np.subtract.outer(reference_boundaries, estimated_boundaries) )

    true_to_estimated = np.median(np.sort(dist, axis=1)[:, 0])
    estimated_to_true = np.median(np.sort(dist, axis=0)[0, :])

    return true_to_estimated, estimated_to_true
