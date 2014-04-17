# CREATED:2013-08-13 12:02:42 by Brian McFee <brm2132@columbia.edu>
'''Structural segmentation evaluation, following the protocols of MIREX2012.
    - Boundary detection
        - (precision, recall, f-measure)
        - median distance to nearest boundary
'''

import numpy as np
import functools
import collections

from . import util

def __validate_intervals(intervals):
    '''Internal validation function for interval arrays'''

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
        '''Validate both reference and estimated intervals'''
        for intervals in [reference_intervals, estimated_intervals]:
            __validate_intervals(intervals)

        return metric(reference_intervals, estimated_intervals, *args, **kwargs)

    return metric_validated

@validate
def detection(reference_intervals, estimated_intervals, window=0.5, beta=1.0, trim=False):
    '''Boundary detection hit-rate.

    A hit is counted whenever an reference boundary is within ``window`` of a estimated
    boundary.  Note that each boundary is matched at most once: this is achieved by computing
    the size of a maximal matching between reference and estimated boundary points, subject
    to the window constraint.

    :usage:
        >>> ref_intervals, ref_labels = mir_eval.io.load_annotation('reference.lab')
        >>> est_intervals, est_labels = mir_eval.io.load_annotation('estimate.lab')
        >>> # With 0.5s windowing
        >>> P05, R05, F05 = mir_eval.boundary.detection(ref_intervals, est_intervals, window=0.5)
        >>> # With 3s windowing
        >>> P3, R3, F3 = mir_eval.boundary.detection(ref_intervals, est_intervals, window=3)
        >>> # Ignoring hits for the beginning and end of track
        >>> P, R, F = mir_eval.boundary.detection(ref_intervals, est_intervals, window=0.5, trim=True)


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
            precision of estimated predictions

        - recall : float
            recall of reference reference boundaries

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

    n_ref, n_est = len(reference_boundaries), len(estimated_boundaries)
    
    window_match    = np.abs(np.subtract.outer(reference_boundaries, estimated_boundaries)) <= window
    window_match    = window_match.astype(int)
    
    # L. Lovasz On determinants, matchings and random algorithms. 
    # In L. Budach, editor, Fundamentals of Computation Theory, pages 565-574. Akademie-Verlag, 1979.
    #
    # If we build the skew-symmetric adjacency matrix 
    # D[i, n_ref+j] = 1 <=> ref[i] within window of est[j]
    # D[n_ref + j, i] = -1 <=> same
    #
    # then rank(D) = 2 * maximum matching
    #
    # This way, we find the optimal assignment of reference and annotation boundaries.
    #
    skew_adjacency  = np.zeros((n_ref + n_est, n_ref + n_est), dtype=np.int32)
    skew_adjacency[:n_ref, n_ref:] = window_match
    skew_adjacency[n_ref:, :n_ref] = -window_match.T
    
    matching_size = np.linalg.matrix_rank(skew_adjacency) / 2.0
    
    precision   = matching_size / len(estimated_boundaries)
    recall      = matching_size / len(reference_boundaries)
    
    f_measure   = util.f_measure(precision, recall, beta=beta)
    
    return precision, recall, f_measure

@validate
def deviation(reference_intervals, estimated_intervals, trim=False):
    '''Compute the median deviations between reference and estimated boundary times.

    :usage:
        >>> ref_intervals, ref_labels = mir_eval.io.load_annotation('reference.lab')
        >>> est_intervals, est_labels = mir_eval.io.load_annotation('estimate.lab')
        >>> r_to_e, e_to_r = mir_eval.boundary.deviation(ref_intervals, est_intervals)

    :parameters:
        - reference_intervals : np.ndarray, shape=(n, 2)
            reference segment intervals, as returned by `mir_eval.io.load_annotation`

        - estimated_intervals : np.ndarray, shape=(m, 2)
            estimated segment intervals, as returned by `mir_eval.io.load_annotation`

        - trim : boolean
            if ``True``, the first and last intervals are ignored.
            Typically, these denote start (0.0) and end-of-track markers.

    :returns:
        - reference_to_estimated : float
            median time from each reference boundary to the closest estimated boundary

        - estimated_to_reference : float
            median time from each estimated boundary to the closest reference boundary
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
        return np.nan, np.nan

    dist = np.abs( np.subtract.outer(reference_boundaries, estimated_boundaries) )

    reference_to_estimated = np.median(dist.min(axis=0))
    estimated_to_reference = np.median(dist.min(axis=1))

    return reference_to_estimated, estimated_to_reference


metrics = collections.OrderedDict()

# Create an ordered dict mapping metric names to functions
metrics = collections.OrderedDict()
metrics['detection'] = detection
metrics['deviation'] = deviation

