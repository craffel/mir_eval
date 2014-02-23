# CREATED:2013-08-13 12:02:42 by Brian McFee <brm2132@columbia.edu>
'''Structural segmentation evaluation, following the protocols of MIREX2012.
    - Boundary detection
        - (precision, recall, f-measure)
        - median distance to nearest boundary
    - Frame clustering
'''

import numpy as np
import scipy.stats
import sklearn.metrics.cluster as metrics

from . import util

def boundary_detection(reference_intervals, estimated_intervals, window=0.5, beta=1.0, trim=True):
    '''Boundary detection hit-rate.  

    A hit is counted whenever an reference boundary is within ``window`` of a estimated
    boundary.

    :usage:
        >>> # With 0.5s windowing
        >>> reference, true_labels = mir_eval.util.import_segments('truth.csv')
        >>> estimated, pred_labels = mir_eval.util.import_segments('prediction.csv')
        >>> P05, R05, F05 = mir_eval.segment.boundary_detection(reference, estimated, window=0.5)
        >>> # With 3s windowing
        >>> P3, R3, F3 = mir_eval.segment.boundary_detection(reference, estimated, window=3)


    :parameters:
        - reference_intervals : list-like, float
            ground-truth segment boundary times (in seconds)

        - estimated_intervals : list-like, float
            estimated segment boundary times (in seconds)

        - window : float > 0
            size of the window of 'correctness' around ground-truth beats (in seconds)

        - beta : float > 0
            weighting constant for F-measure.

        - trim : boolean
            if ``True``, the first and last intervals are ignored.
            Typically, these denote start (0) and end-markers.

    :returns:
        - precision : float
            precision of predictions

        - recall : float
            recall of ground-truth beats

        - f_measure : float
            F-measure (weighted harmonic mean of ``precision`` and ``recall``)
    '''

    # Suppress the first and last intervals
    if trim:
        reference_intervals = reference_intervals[1:-1]
        estimated_intervals = estimated_intervals[1:-1]

    # Compute the hits
    dist        = np.abs( np.subtract.outer(reference_intervals, estimated_intervals)) <= window

    # Precision: how many estimated intervals were hits?
    precision   = np.mean(dist.max(axis=0))

    # Recall: how many of the intervals did we catch?
    recall      = np.mean(dist.max(axis=1))

    # And the f-measure
    f_measure   = util.f_measure(precision, recall, beta=beta)

    return precision, recall, f_measure

def boundary_deviation(reference_intervals, estimated_intervals, trim=True):
    '''Compute the median deviations between reference and estimated boundary times.

    :usage:
        >>> reference, true_labels = mir_eval.util.import_segments('truth.csv')
        >>> estimated, pred_labels = mir_eval.util.import_segments('prediction.csv')
        >>> t_to_p, p_to_t = mir_eval.segment.boundary_deviation(reference, estimated)

    :parameters:
        - reference_intervals : list-like, float
            ground-truth segment boundary times (in seconds)

        - estimated_intervals : list-like, float
            estimated segment boundary times (in seconds)

        - trim : boolean
            if ``True``, the first and last intervals are ignored.
            Typically, these denote start (0) and end-markers.

    :returns:
        - true_to_estimated : float
            median time from each true boundary to the closest estimated boundary

        - estimated_to_true : float
            median time from each estimated boundary to the closest true boundary
    '''

    # Suppress the first and last intervals
    if trim:
        reference_intervals = reference_intervals[1:-1]
        estimated_intervals = estimated_intervals[1:-1]

    dist = np.abs( np.subtract.outer(reference_intervals, estimated_intervals) )

    true_to_estimated = np.median(np.sort(dist, axis=1)[:, 0])
    estimated_to_true = np.median(np.sort(dist, axis=0)[0, :])

    return true_to_estimated, estimated_to_true

def _intervals_to_frames(intervals, frame_size=0.1):
    '''Convert a sequence of intervals to frame-level segment annotations.
    
    :parameters:
        - intervals : list-like float
            segment boundary times (in seconds).

        - frame_size : float > 0
            duration of each frame (in seconds)

    :returns:
        - y : np.array, dtype=int
            array of segment labels for each frame

    ..note::
        It is assumed that ``intervals[-1]` == length of song

    ..note::
        Segment intervals will be rounded down to the nearest multiple 
        of ``frame_size``.
    '''
    
    intervals = np.sort(frame_size * np.round(intervals / frame_size))
    intervals = np.unique(np.concatenate(([0], intervals)))

    # Build the frame label array
    y = np.zeros(int(intervals[-1] / frame_size))

    for (i, (start, end)) in enumerate(zip(intervals[:-1], intervals[1:])):
        y[int(start / frame_size):int(end / frame_size)] = i

    return y

def frame_clustering_pairwise(reference_intervals, estimated_intervals, frame_size=0.1, beta=1.0):
    '''Frame-clustering segmentation evaluation by pair-wise agreement.

    :parameters:
        - reference_intervals : list-like, float
            ground-truth segment boundary times (in seconds)

        - estimated_intervals : list-like, float
            estimated segment boundary times (in seconds)

        - frame_size : float > 0
            length (in seconds) of frames for clustering

        - beta : float > 0
            beta value for F-measure

    :returns:
        - Pair_precision : float > 0
        - Pair_recall   : float > 0
        - Pair_F        : float > 0
            Precision/recall/f-measure of detecting whether
            frames belong in the same cluster

    :raises:
        - ValueError
            If ``reference_intervals`` and ``estimated_intervals`` do not span the
            same time duration.

    ..note::
        It is assumed that ``intervals[-1]`` == length of song

    ..note::
        Segment intervals will be rounded down to the nearest multiple 
        of frame_size.

    ..seealso:: mir_eval.util.adjust_intervals
    '''

    # Generate the cluster labels
    y_true = _intervals_to_frames(reference_intervals, frame_size=frame_size)
    y_pred = _intervals_to_frames(estimated_intervals, frame_size=frame_size)
    # Make sure we have the same number of frames
    if len(y_true) != len(y_pred):
        raise ValueError('Timing mismatch: %.3f vs %.3f' % (reference_intervals[-1], estimated_intervals[-1]))

    # Construct the label-agreement matrices
    agree_true  = np.triu(np.equal.outer(y_true, y_true))
    agree_pred  = np.triu(np.equal.outer(y_pred, y_pred))
    
    matches     = float((agree_true & agree_pred).sum())
    precision   = matches / agree_true.sum()
    recall      = matches / agree_pred.sum()
    f_measure   = util.f_measure(precision, recall, beta=beta)

    return precision, recall, f_measure

def frame_clustering_ari(reference_intervals, estimated_intervals, frame_size=0.1):
    '''Adjusted Rand Index (ARI) for frame clustering segmentation evaluation.

    :parameters:
        - reference_intervals : list-like, float
            ground-truth segment boundary times (in seconds)

        - estimated_intervals : list-like, float
            estimated segment boundary times (in seconds)

        - frame_size : float > 0
            length (in seconds) of frames for clustering

    :returns:
        - ARI : float > 0
            Adjusted Rand index between segmentations.

    ..note::
        It is assumed that ``intervals[-1]`` == length of song

    ..note::
        Segment intervals will be rounded down to the nearest multiple 
        of frame_size.
    '''
    # Generate the cluster labels
    y_true = _intervals_to_frames(reference_intervals, frame_size=frame_size)
    y_pred = _intervals_to_frames(estimated_intervals, frame_size=frame_size)
    # Make sure we have the same number of frames
    if len(y_true) != len(y_pred):
        raise ValueError('Timing mismatch: %.3f vs %.3f' % (reference_intervals[-1], estimated_intervals[-1]))

    return metrics.adjusted_rand_score(y_true, y_pred)

def frame_clustering_mi(reference_intervals, estimated_intervals, frame_size=0.1):
    '''Frame-clustering segmentation: mutual information metrics.

    :parameters:
    - reference_intervals : list-like, float
        ground-truth segment boundary times (in seconds)

    - estimated_intervals : list-like, float
        estimated segment boundary times (in seconds)

    - frame_size : float > 0
        length (in seconds) of frames for clustering

    :returns:
    - MI : float >0
        Mutual information between segmentations
    - AMI : float 
        Adjusted mutual information between segmentations.
    - NMI : float > 0
        Normalize mutual information between segmentations

    ..note::
        It is assumed that `intervals[-1] == length of song`

    ..note::
        Segment intervals will be rounded down to the nearest multiple 
        of frame_size.
    '''
    # Generate the cluster labels
    y_true = _intervals_to_frames(reference_intervals, frame_size=frame_size)
    y_pred = _intervals_to_frames(estimated_intervals, frame_size=frame_size)

    # Make sure we have the same number of frames
    if len(y_true) != len(y_pred):
        raise ValueError('Timing mismatch: %.3f vs %.3f' % (reference_intervals[-1], estimated_intervals[-1]))

    # Mutual information
    mutual_info         = metrics.mutual_info_score(y_true, y_pred)

    # Adjusted mutual information
    adj_mutual_info     = metrics.adjusted_mutual_info_score(y_true, y_pred)
    
    # Normalized mutual information
    norm_mutual_info    = metrics.normalized_mutual_info_score(y_true, y_pred)

    return mutual_info, adj_mutual_info, norm_mutual_info
    
def frame_clustering_nce(reference_intervals, estimated_intervals, frame_size=0.1, beta=1.0):
    '''Frame-clustering segmentation: normalized conditional entropy

    Computes cross-entropy of cluster assignment, normalized by the max-entropy.

    :parameters:
        - reference_intervals : list-like, float
            ground-truth segment boundary times (in seconds)

        - estimated_intervals : list-like, float
            estimated segment boundary times (in seconds)

        - frame_size : float > 0
            length (in seconds) of frames for clustering

        - beta : float > 0
            beta for F-measure

    :returns:
        - S_over
            Over-clustering score: 
            ``1 - H(y_pred | y_true) / log(|y_pred|)``

        - S_under
            Under-clustering score:
            ``1 - H(y_true | y_pred) / log(|y_true|)``

        - F
            F-measure for (S_over, S_under)
        
    ..note:: Towards quantitative measures of evaluating song segmentation.
        Lukashevich, H. ISMIR 2008.
    '''

    # Generate the cluster labels
    y_true = _intervals_to_frames(reference_intervals, frame_size=frame_size)
    y_pred = _intervals_to_frames(estimated_intervals, frame_size=frame_size)

    # Make sure we have the same number of frames
    if len(y_true) != len(y_pred):
        raise ValueError('Timing mismatch: %.3f vs %.3f' % (reference_intervals[-1], estimated_intervals[-1]))

    # Make the contingency table: shape = (n_true, n_pred)
    contingency = metrics.contingency_matrix(y_true, y_pred).astype(float)

    # Compute the marginals
    p_pred = contingency.sum(axis=0) / len(y_true) 
    p_true = contingency.sum(axis=1) / len(y_true)

    true_given_pred = p_pred.dot(scipy.stats.entropy(contingency,   base=2))
    pred_given_true = p_true.dot(scipy.stats.entropy(contingency.T, base=2))

    score_over = 0.0
    if contingency.shape[1] > 1:
        score_over  = 1. - pred_given_true / np.log2(contingency.shape[1])

    score_under = 0.0
    if contingency.shape[0] > 1:
        score_under = 1. - true_given_pred / np.log2(contingency.shape[0])

    f_measure = util.f_measure(score_over, score_under, beta=beta)

    return score_over, score_under, f_measure
