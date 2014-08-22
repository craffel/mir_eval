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
import scipy.sparse
import scipy.misc
import scipy.special
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


def _contingency_matrix(reference_indices, estimated_indices):
    '''
    Computes the contingency matrix of a true labeling vs an estimated one.

    :parameters:
        - reference_indices : np.ndarray
            Array of reference indices

        - estimated_indices : np.ndarray
            Array of estimated indices

    :returns:
        - contingency_matrix : np.ndarray
            Contingency matrix, shape=(#reference indices, #estimated indices)

    .. note:: Based on sklearn.metrics.cluster.contingency_matrix
    '''
    ref_classes, ref_class_idx = np.unique(reference_indices,
                                           return_inverse=True)
    est_classes, est_class_idx = np.unique(estimated_indices,
                                           return_inverse=True)
    n_ref_classes = ref_classes.shape[0]
    n_est_classes = est_classes.shape[0]
    # Using coo_matrix is faster than histogram2d
    return scipy.sparse.coo_matrix((np.ones(ref_class_idx.shape[0]),
                                    (ref_class_idx, est_class_idx)),
                                   shape=(n_ref_classes, n_est_classes),
                                   dtype=np.int).toarray()


def _adjusted_rand_index(reference_indices, estimated_indices):
    '''
    Compute the Rand index, adjusted for change.

    :parameters:
        - reference_indices : np.ndarray
            Array of reference indices

        - estimated_indices : np.ndarray
            Array of estimated indices

    :returns:
        - ari : float
            Adjusted Rand index

    .. note:: Based on sklearn.metrics.cluster.adjusted_rand_score
    '''
    n_samples = len(reference_indices)
    ref_classes = np.unique(reference_indices)
    est_classes = np.unique(estimated_indices)
    # Special limit cases: no clustering since the data is not split;
    # or trivial clustering where each document is assigned a unique cluster.
    # These are perfect matches hence return 1.0.
    if (ref_classes.shape[0] == est_classes.shape[0] == 1
            or ref_classes.shape[0] == est_classes.shape[0] == 0
            or (ref_classes.shape[0] == est_classes.shape[0] ==
                len(reference_indices))):
        return 1.0

    contingency = _contingency_matrix(reference_indices, estimated_indices)

    # Compute the ARI using the contingency data
    sum_comb_c = sum(scipy.misc.comb(n_c, 2, exact=1) for n_c in
                     contingency.sum(axis=1))
    sum_comb_k = sum(scipy.misc.comb(n_k, 2, exact=1) for n_k in
                     contingency.sum(axis=0))

    sum_comb = sum((scipy.misc.comb(n_ij, 2, exact=1) for n_ij in
                    contingency.flatten()))
    prod_comb = (sum_comb_c * sum_comb_k)/float(scipy.misc.comb(n_samples, 2))
    mean_comb = (sum_comb_k + sum_comb_c)/2.
    return ((sum_comb - prod_comb)/(mean_comb - prod_comb))


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

    return _adjusted_rand_index(y_ref, y_est)


def _mutual_info_score(reference_indices, estimated_indices, contingency=None):
    '''
    Compute the mutual information between two sequence labelings.

    :parameters:
        - reference_indices : np.ndarray
            Array of reference indices

        - estimated_indices : np.ndarray
            Array of estimated indices

        - contingency : np.ndarray
            Pre-computed contingency matrix.  If None, one will be computed.

    :returns:
        - mi : float
            Mutual information

    .. note:: Based on sklearn.metrics.cluster.mutual_info_score
    '''
    if contingency is None:
        contingency = _contingency_matrix(reference_indices,
                                          estimated_indices).astype(float)
    contingency_sum = np.sum(contingency)
    pi = np.sum(contingency, axis=1)
    pj = np.sum(contingency, axis=0)
    outer = np.outer(pi, pj)
    nnz = contingency != 0.0
    # normalized contingency
    contingency_nm = contingency[nnz]
    log_contingency_nm = np.log(contingency_nm)
    contingency_nm /= contingency_sum
    # log(a / b) should be calculated as log(a) - log(b) for
    # possible loss of precision
    log_outer = -np.log(outer[nnz]) + np.log(pi.sum()) + np.log(pj.sum())
    mi = (contingency_nm * (log_contingency_nm - np.log(contingency_sum))
          + contingency_nm * log_outer)
    return mi.sum()


def _entropy(labels):
    '''
    Calculates the entropy for a labeling.

    :parameters:
        - labels : list-like
            List of labels.

    :returns:
        - entropy : float
            Entropy of the labeling.

    .. note:: Based on sklearn.metrics.cluster.entropy
    '''
    if len(labels) == 0:
        return 1.0
    label_idx = np.unique(labels, return_inverse=True)[1]
    pi = np.bincount(label_idx).astype(np.float)
    pi = pi[pi > 0]
    pi_sum = np.sum(pi)
    # log(a / b) should be calculated as log(a) - log(b) for
    # possible loss of precision
    return -np.sum((pi / pi_sum) * (np.log(pi) - np.log(pi_sum)))


def _expected_mutual_information(contingency, n_samples):
    '''
    Calculate the expected mutual information for two labelings.

    :parameters:
        - contingency : np.ndarray
            Contingency matrix.
        - n_samples : int
            Number of label samples

    :returns:
        - emi : float
            Expected mutual information score

    .. note:: Based on sklearn.metrics.cluster.expected_mutual_information
    '''
    R, C = contingency.shape
    N = float(n_samples)
    a = np.sum(contingency, axis=1).astype(np.int32)
    b = np.sum(contingency, axis=0).astype(np.int32)
    # There are three major terms to the EMI equation, which are multiplied to
    # and then summed over varying nij values.
    # While nijs[0] will never be used, having it simplifies the indexing.
    nijs = np.arange(0, max(np.max(a), np.max(b)) + 1, dtype='float')
    # Stops divide by zero warnings. As its not used, no issue.
    nijs[0] = 1
    # term1 is nij / N
    term1 = nijs / N
    # term2 is log((N*nij) / (a * b)) == log(N * nij) - log(a * b)
    # term2 uses the outer product
    log_ab_outer = np.log(np.outer(a, b))
    # term2 uses N * nij
    log_Nnij = np.log(N * nijs)
    # term3 is large, and involved many factorials. Calculate these in log
    # space to stop overflows.
    gln_a = scipy.special.gammaln(a + 1)
    gln_b = scipy.special.gammaln(b + 1)
    gln_Na = scipy.special.gammaln(N - a + 1)
    gln_Nb = scipy.special.gammaln(N - b + 1)
    gln_N = scipy.special.gammaln(N + 1)
    gln_nij = scipy.special.gammaln(nijs + 1)
    # start and end values for nij terms for each summation.
    start = np.array([[v - N + w for w in b] for v in a], dtype='int')
    start = np.maximum(start, 1)
    end = np.minimum(np.resize(a, (C, R)).T, np.resize(b, (R, C))) + 1
    # emi itself is a summation over the various values.
    emi = 0
    for i in range(R):
        for j in range(C):
            for nij in range(start[i, j], end[i, j]):
                term2 = log_Nnij[nij] - log_ab_outer[i, j]
                # Numerators are positive, denominators are negative.
                gln = (gln_a[i] + gln_b[j] + gln_Na[i] + gln_Nb[j]
                     - gln_N - gln_nij[nij]
                     - scipy.special.gammaln(a[i] - nij + 1)
                     - scipy.special.gammaln(b[j] - nij + 1)
                     - scipy.special.gammaln(N - a[i] - b[j] + nij + 1))
                term3 = np.exp(gln)
                emi += (term1[nij] * term2 * term3)
    return emi


def _adjusted_mutual_info_score(reference_indices, estimated_indices):
    '''
    Compute the mutual information between two sequence labelings, adjusted for
    chance.

    :parameters:
        - reference_indices : np.ndarray
            Array of reference indices

        - estimated_indices : np.ndarray
            Array of estimated indices

    :returns:
        - ami : float <= 1.0
            Mutual information

    .. note:: Based on sklearn.metrics.cluster.adjusted_mutual_info_score
    '''
    n_samples = len(reference_indices)
    ref_classes = np.unique(reference_indices)
    est_classes = np.unique(estimated_indices)
    # Special limit cases: no clustering since the data is not split.
    # This is a perfect match hence return 1.0.
    if (ref_classes.shape[0] == est_classes.shape[0] == 1
            or ref_classes.shape[0] == est_classes.shape[0] == 0):
        return 1.0
    contingency = _contingency_matrix(reference_indices,
                                      estimated_indices).astype(float)
    # Calculate the MI for the two clusterings
    mi = _mutual_info_score(reference_indices, estimated_indices,
                           contingency=contingency)
    # Calculate the expected value for the mutual information
    emi = _expected_mutual_information(contingency, n_samples)
    # Calculate entropy for each labeling
    h_true, h_pred = _entropy(reference_indices), _entropy(estimated_indices)
    ami = (mi - emi) / (max(h_true, h_pred) - emi)
    return ami


def _normalized_mutual_info_score(reference_indices, estimated_indices):
    '''
    Compute the mutual information between two sequence labelings, adjusted for
    chance.

    :parameters:
        - reference_indices : np.ndarray
            Array of reference indices

        - estimated_indices : np.ndarray
            Array of estimated indices

    :returns:
        - nmi : float <= 1.0
            Normalized mutual information

    .. note:: Based on sklearn.metrics.cluster.normalized_mutual_info_score
    '''
    ref_classes = np.unique(reference_indices)
    est_classes = np.unique(estimated_indices)
    # Special limit cases: no clustering since the data is not split.
    # This is a perfect match hence return 1.0.
    if (ref_classes.shape[0] == est_classes.shape[0] == 1
            or ref_classes.shape[0] == est_classes.shape[0] == 0):
        return 1.0
    contingency = _contingency_matrix(reference_indices,
                                      estimated_indices).astype(float)
    contingency = np.array(contingency, dtype='float')
    # Calculate the MI for the two clusterings
    mi = _mutual_info_score(reference_indices, estimated_indices,
                           contingency=contingency)
    # Calculate the expected value for the mutual information
    # Calculate entropy for each labeling
    h_true, h_pred = _entropy(reference_indices), _entropy(estimated_indices)
    nmi = mi / max(np.sqrt(h_true * h_pred), 1e-10)
    return nmi


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
    mutual_info = _mutual_info_score(y_ref, y_est)

    # Adjusted mutual information
    adj_mutual_info = _adjusted_mutual_info_score(y_ref, y_est)

    # Normalized mutual information
    norm_mutual_info = _normalized_mutual_info_score(y_ref, y_est)

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
    contingency = _contingency_matrix(y_ref, y_est).astype(float)

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
