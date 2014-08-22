# CREATED:2013-08-13 12:02:42 by Brian McFee <brm2132@columbia.edu>
'''Structural segmentation evaluation, following the protocols of MIREX2012.

   Frame clustering metrics:
        - pairwise classification
        - adjusted rand index
        - mutual information
        - normalized conditional entropy
'''

import numpy as np
import scipy.stats
import sklearn.metrics.cluster as skmetrics
import collections
import warnings

from . import util


def validate(reference_intervals, reference_labels, estimated_intervals,
             estimated_labels):
    '''Checks that the input annotations to a metric look like valid segment
    times, and throws helpful errors if not.

    :parameters:
        - reference_intervals : np.ndarray, shape=(n, 2)
            reference segment intervals, in the format returned by
            :func:`mir_eval.io.load_labeled_intervals`.

        - reference_labels : list, shape=(n,)
            reference segment labels, in the format returned by
            :func:`mir_eval.io.load_labeled_intervals`.

        - estimated_intervals : np.ndarray, shape=(m, 2)
            estimated segment intervals, in the format returned by
            :func:`mir_eval.io.load_labeled_intervals`.

        - estimated_labels : list, shape=(m,)
            estimated segment labels, in the format returned by
            :func:`mir_eval.io.load_labeled_intervals`.
    '''
    for (intervals, labels) in [(reference_intervals, reference_labels),
                                (estimated_intervals, estimated_labels)]:

        util.validate_intervals(intervals)
        if intervals.shape[0] != len(labels):
            raise ValueError('Number of intervals does not match number '
                             'of labels')

        # Make sure beat times are increasing
        if not np.allclose(intervals[0, 0], 0.0):
            raise ValueError('Segment intervals do not start at 0')

    if reference_intervals.size == 0:
        warnings.warn("Reference intervals are empty.")
    if estimated_intervals.size == 0:
        warnings.warn("Estimated intervals are empty.")
    if not np.allclose(reference_intervals[-1, 1], estimated_intervals[-1, 1]):
        raise ValueError('End times do not match')


def pairwise(reference_intervals, reference_labels,
             estimated_intervals, estimated_labels,
             frame_size=0.1, beta=1.0):
    '''Frame-clustering segmentation evaluation by pair-wise agreement.

    :usage:
        >>> (ref_intervals,
             ref_labels) = mir_eval.io.load_labeled_intervals('ref.lab')
        >>> (est_intervals,
             est_labels) = mir_eval.io.load_labeled_intervals('est.lab')
        >>> # Trim or pad the estimate to match reference timing
        >>> (ref_intervals,
             ref_labels) = mir_eval.util.adjust_intervals(ref_intervals,
                                                          ref_labels,
                                                          t_min=0)
        >>> (est_intervals,
             est_labels) = mir_eval.util.adjust_intervals(est_intervals,
                                                          est_labels,
                                                          t_min=0,
                                                          t_max=ref_intervals.max())
        >>> precision, recall, f = mir_eval.structure.pairwise(ref_intervals,
                                                               ref_labels,
                                                               est_intervals,
                                                               est_labels)

    :parameters:
        - reference_intervals : np.ndarray, shape=(n, 2)
            reference segment intervals, in the format returned by
            :func:`mir_eval.io.load_labeled_intervals`.

        - reference_labels : list, shape=(n,)
            reference segment labels, in the format returned by
            :func:`mir_eval.io.load_labeled_intervals`.

        - estimated_intervals : np.ndarray, shape=(m, 2)
            estimated segment intervals, in the format returned by
            :func:`mir_eval.io.load_labeled_intervals`.

        - estimated_labels : list, shape=(m,)
            estimated segment labels, in the format returned by
            :func:`mir_eval.io.load_labeled_intervals`.

        - frame_size : float > 0
            length (in seconds) of frames for clustering

        - beta : float > 0
            beta value for F-measure

    :returns:
        - precision : float > 0
            Precision of detecting whether frames belong in the same cluster
        - recall : float > 0
            Recall of detecting whether frames belong in the same cluster
        - f : float > 0
            F-measure of detecting whether frames belong in the same cluster

    :raises:
        - ValueError
            If ``reference_intervals`` and ``estimated_intervals`` do not span
            the same time duration.

    .. seealso:: :func:`mir_eval.util.adjust_intervals`
    '''

    validate(reference_intervals, reference_labels, estimated_intervals,
             estimated_labels)
    # Generate the cluster labels
    y_ref = util.intervals_to_samples(reference_intervals,
                                      reference_labels,
                                      sample_size=frame_size)[-1]

    y_ref = util.index_labels(y_ref)[0]

    # Map to index space
    y_est = util.intervals_to_samples(estimated_intervals,
                                      estimated_labels,
                                      sample_size=frame_size)[-1]

    y_est = util.index_labels(y_est)[0]

    # Build the reference label agreement matrix
    agree_ref = np.equal.outer(y_ref, y_ref)
    # Count the unique pairs
    n_agree_ref = (agree_ref.sum() - len(y_ref)) / 2.0

    # Repeat for estimate
    agree_est = np.equal.outer(y_est, y_est)
    n_agree_est = (agree_est.sum() - len(y_est)) / 2.0

    # Find where they agree
    matches = np.logical_and(agree_ref, agree_est)
    n_matches = (matches.sum() - len(y_ref)) / 2.0

    precision = n_matches / n_agree_est
    recall = n_matches / n_agree_ref
    f_measure = util.f_measure(precision, recall, beta=beta)

    return precision, recall, f_measure


def rand_index(reference_intervals, reference_labels,
               estimated_intervals, estimated_labels,
               frame_size=0.1, beta=1.0):
    '''(Non-adjusted) Rand index.

    :usage:
        >>> (ref_intervals,
             ref_labels) = mir_eval.io.load_labeled_intervals('ref.lab')
        >>> (est_intervals,
             est_labels) = mir_eval.io.load_labeled_intervals('est.lab')
        >>> # Trim or pad the estimate to match reference timing
        >>> (ref_intervals,
             ref_labels) = mir_eval.util.adjust_intervals(ref_intervals,
                                                          ref_labels,
                                                          t_min=0)
        >>> (est_intervals,
             est_labels) = mir_eval.util.adjust_intervals(est_intervals,
                                                          est_labels,
                                                          t_min=0,
                                                          t_max=ref_intervals.max())
        >>> rand_index = mir_eval.structure.rand_index(ref_intervals,
                                                       ref_labels,
                                                       est_intervals,
                                                       est_labels)

    :parameters:
        - reference_intervals : np.ndarray, shape=(n, 2)
            reference segment intervals, in the format returned by
            :func:`mir_eval.io.load_labeled_intervals`.

        - reference_labels : list, shape=(n,)
            reference segment labels, in the format returned by
            :func:`mir_eval.io.load_labeled_intervals`.

        - estimated_intervals : np.ndarray, shape=(m, 2)
            estimated segment intervals, in the format returned by
            :func:`mir_eval.io.load_labeled_intervals`.

        - estimated_labels : list, shape=(m,)
            estimated segment labels, in the format returned by
            :func:`mir_eval.io.load_labeled_intervals`.

        - frame_size : float > 0
            length (in seconds) of frames for clustering

        - beta : float > 0
            beta value for F-measure

    :returns:
        - rand_index : float > 0
            Rand index

    :raises:
        - ValueError
            If ``reference_intervals`` and ``estimated_intervals`` do not span
            the same time duration.

    .. seealso:: :func:`mir_eval.util.adjust_intervals`
    '''

    validate(reference_intervals, reference_labels, estimated_intervals,
             estimated_labels)
    # Generate the cluster labels
    y_ref = util.intervals_to_samples(reference_intervals,
                                      reference_labels,
                                      sample_size=frame_size)[-1]

    y_ref = util.index_labels(y_ref)[0]

    # Map to index space
    y_est = util.intervals_to_samples(estimated_intervals,
                                      estimated_labels,
                                      sample_size=frame_size)[-1]

    y_est = util.index_labels(y_est)[0]

    # Build the reference label agreement matrix
    agree_ref = np.equal.outer(y_ref, y_ref)

    # Repeat for estimate
    agree_est = np.equal.outer(y_est, y_est)

    # Find where they agree
    matches_pos = np.logical_and(agree_ref, agree_est)

    # Find where they disagree
    matches_neg = np.logical_and(~agree_ref, ~agree_est)

    n_pairs = len(y_ref) * (len(y_ref) - 1) / 2.0

    n_matches_pos = (matches_pos.sum() - len(y_ref)) / 2.0
    n_matches_neg = matches_neg.sum() / 2.0
    rand = (n_matches_pos + n_matches_neg) / n_pairs

    return rand


def ari(reference_intervals, reference_labels,
        estimated_intervals, estimated_labels,
        frame_size=0.1):
    '''Adjusted Rand Index (ARI) for frame clustering segmentation evaluation.

    :usage:
        >>> (ref_intervals,
             ref_labels) = mir_eval.io.load_labeled_intervals('ref.lab')
        >>> (est_intervals,
             est_labels) = mir_eval.io.load_labeled_intervals('est.lab')
        >>> # Trim or pad the estimate to match reference timing
        >>> (ref_intervals,
             ref_labels) = mir_eval.util.adjust_intervals(ref_intervals,
                                                          ref_labels,
                                                          t_min=0)
        >>> (est_intervals,
             est_labels) = mir_eval.util.adjust_intervals(est_intervals,
                                                          est_labels,
                                                          t_min=0,
                                                          t_max=ref_intervals.max())
        >>> ari_score = mir_eval.structure.ari(ref_intervals, ref_labels,
                                               est_intervals, est_labels)

    :parameters:
        - reference_intervals : np.ndarray, shape=(n, 2)
            reference segment intervals, in the format returned by
            :func:`mir_eval.io.load_labeled_intervals`.

        - reference_labels : list, shape=(n,)
            reference segment labels, in the format returned by
            :func:`mir_eval.io.load_labeled_intervals`.

        - estimated_intervals : np.ndarray, shape=(m, 2)
            estimated segment intervals, in the format returned by
            :func:`mir_eval.io.load_labeled_intervals`.

        - estimated_labels : list, shape=(m,)
            estimated segment labels, in the format returned by
            :func:`mir_eval.io.load_labeled_intervals`.

        - frame_size : float > 0
            length (in seconds) of frames for clustering

    :returns:
        - ari_score : float > 0
            Adjusted Rand index between segmentations.

    .. note::
        It is assumed that ``intervals[-1]`` == length of song

    .. note::
        Segment intervals will be rounded down to the nearest multiple
        of frame_size.
    '''
    validate(reference_intervals, reference_labels, estimated_intervals,
             estimated_labels)
    # Generate the cluster labels
    y_ref = util.intervals_to_samples(reference_intervals,
                                      reference_labels,
                                      sample_size=frame_size)[-1]

    y_ref = util.index_labels(y_ref)[0]

    # Map to index space
    y_est = util.intervals_to_samples(estimated_intervals,
                                      estimated_labels,
                                      sample_size=frame_size)[-1]

    y_est = util.index_labels(y_est)[0]

    return skmetrics.adjusted_rand_score(y_ref, y_est)


def mutual_information(reference_intervals, reference_labels,
                       estimated_intervals, estimated_labels,
                       frame_size=0.1):
    '''Frame-clustering segmentation: mutual information metrics.

    :usage:
        >>> (ref_intervals,
             ref_labels) = mir_eval.io.load_labeled_intervals('ref.lab')
        >>> (est_intervals,
             est_labels) = mir_eval.io.load_labeled_intervals('est.lab')
        >>> # Trim or pad the estimate to match reference timing
        >>> (ref_intervals,
             ref_labels) = mir_eval.util.adjust_intervals(ref_intervals,
                                                          ref_labels,
                                                          t_min=0)
        >>> (est_intervals,
             est_labels) = mir_eval.util.adjust_intervals(est_intervals,
                                                          est_labels,
                                                          t_min=0,
                                                          t_max=ref_intervals.max())
        >>> mi, ami, nmi = mir_eval.structure.mutual_information(ref_intervals,
                                                                 ref_labels,
                                                                 est_intervals,
                                                                 est_labels)

    :parameters:
        - reference_intervals : np.ndarray, shape=(n, 2)
            reference segment intervals, in the format returned by
            :func:`mir_eval.io.load_labeled_intervals`.

        - reference_labels : list, shape=(n,)
            reference segment labels, in the format returned by
            :func:`mir_eval.io.load_labeled_intervals`.

        - estimated_intervals : np.ndarray, shape=(m, 2)
            estimated segment intervals, in the format returned by
            :func:`mir_eval.io.load_labeled_intervals`.

        - estimated_labels : list, shape=(m,)
            estimated segment labels, in the format returned by
            :func:`mir_eval.io.load_labeled_intervals`.

        - frame_size : float > 0
            length (in seconds) of frames for clustering

    :returns:
        - MI : float > 0
            Mutual information between segmentations
        - AMI : float
            Adjusted mutual information between segmentations.
        - NMI : float > 0
            Normalize mutual information between segmentations

    .. note::
        It is assumed that `intervals[-1] == length of song`

    .. note::
        Segment intervals will be rounded down to the nearest multiple
        of frame_size.
    '''
    validate(reference_intervals, reference_labels, estimated_intervals,
             estimated_labels)
    # Generate the cluster labels
    y_ref = util.intervals_to_samples(reference_intervals,
                                      reference_labels,
                                      sample_size=frame_size)[-1]

    y_ref = util.index_labels(y_ref)[0]

    # Map to index space
    y_est = util.intervals_to_samples(estimated_intervals,
                                      estimated_labels,
                                      sample_size=frame_size)[-1]

    y_est = util.index_labels(y_est)[0]

    # Mutual information
    mutual_info = skmetrics.mutual_info_score(y_ref, y_est)

    # Adjusted mutual information
    adj_mutual_info = skmetrics.adjusted_mutual_info_score(y_ref, y_est)

    # Normalized mutual information
    norm_mutual_info = skmetrics.normalized_mutual_info_score(y_ref, y_est)

    return mutual_info, adj_mutual_info, norm_mutual_info


def nce(reference_intervals, reference_labels, estimated_intervals,
        estimated_labels, frame_size=0.1, beta=1.0):
    '''Frame-clustering segmentation: normalized conditional entropy

    Computes cross-entropy of cluster assignment, normalized by the
    max-entropy.

    :usage:
        >>> (ref_intervals,
             ref_labels) = mir_eval.io.load_labeled_intervals('ref.lab')
        >>> (est_intervals,
             est_labels) = mir_eval.io.load_labeled_intervals('est.lab')
        >>> # Trim or pad the estimate to match reference timing
        >>> (ref_intervals,
             ref_labels) = mir_eval.util.adjust_intervals(ref_intervals,
                                                          ref_labels,
                                                          t_min=0)
        >>> (est_intervals,
             est_labels) = mir_eval.util.adjust_intervals(est_intervals,
                                                          est_labels,
                                                          t_min=0,
                                                          t_max=ref_intervals.max())
        >>> S_over, S_under, S_F = mir_eval.structure.nce(ref_intervals,
                                                          ref_labels,
                                                          est_intervals,
                                                          est_labels)

    :parameters:
        - reference_intervals : np.ndarray, shape=(n, 2)
            reference segment intervals, in the format returned by
            :func:`mir_eval.io.load_labeled_intervals`.

        - reference_labels : list, shape=(n,)
            reference segment labels, in the format returned by
            :func:`mir_eval.io.load_labeled_intervals`.

        - estimated_intervals : np.ndarray, shape=(m, 2)
            estimated segment intervals, in the format returned by
            :func:`mir_eval.io.load_labeled_intervals`.

        - estimated_labels : list, shape=(m,)
            estimated segment labels, in the format returned by
            :func:`mir_eval.io.load_labeled_intervals`.

        - frame_size : float > 0
            length (in seconds) of frames for clustering

        - beta : float > 0
            beta for F-measure

    :returns:
        - S_over
            Over-clustering score:
            ``1 - H(y_est | y_ref) / log(|y_est|)``
            If `|y_est|==1`, then `S_over` will be 0.

        - S_under
            Under-clustering score:
            ``1 - H(y_ref | y_est) / log(|y_ref|)``
            If `|y_ref|==1`, then `S_under` will be 0.

        - S_F
            F-measure for (S_over, S_under)

    :references:
        .. [#] Hanna M. Lukashevich. "Towards Quantitative Measures of
            Evaluating Song Segmentation," in Proceedings of the 9th
            International Society for Music Information Retrieval Conference,
            2007, pp. 375-380.
    '''

    validate(reference_intervals, reference_labels, estimated_intervals,
             estimated_labels)
    # Generate the cluster labels
    y_ref = util.intervals_to_samples(reference_intervals,
                                      reference_labels,
                                      sample_size=frame_size)[-1]

    y_ref = util.index_labels(y_ref)[0]

    # Map to index space
    y_est = util.intervals_to_samples(estimated_intervals,
                                      estimated_labels,
                                      sample_size=frame_size)[-1]

    y_est = util.index_labels(y_est)[0]

    # Make the contingency table: shape = (n_ref, n_est)
    contingency = skmetrics.contingency_matrix(y_ref, y_est).astype(float)

    # Normalize by the number of frames
    contingency = contingency / len(y_ref)

    # Compute the marginals
    p_est = contingency.sum(axis=0)
    p_ref = contingency.sum(axis=1)

    # H(true | prediction) = sum_j P[estimated = j] *
    # sum_i P[true = i | estimated = j] log P[true = i | estimated = j]
    # entropy sums over axis=0, which is true labels
    true_given_est = p_est.dot(scipy.stats.entropy(contingency, base=2))
    pred_given_ref = p_ref.dot(scipy.stats.entropy(contingency.T, base=2))

    score_under = 0.0
    if contingency.shape[0] > 1:
        score_under = 1. - true_given_est / np.log2(contingency.shape[0])

    score_over = 0.0
    if contingency.shape[1] > 1:
        score_over = 1. - pred_given_ref / np.log2(contingency.shape[1])

    f_measure = util.f_measure(score_over, score_under, beta=beta)

    return score_over, score_under, f_measure


# Create an ordered dict mapping metric names to functions
METRICS = collections.OrderedDict()
METRICS['pairwise'] = pairwise
METRICS['ARI'] = ari
METRICS['MI'] = mutual_information
METRICS['NCE'] = nce
