# CREATED:2013-08-13 12:02:42 by Brian McFee <brm2132@columbia.edu>
'''Structural segmentation evaluation, following the protocols of MIREX2012:
    - Boundary detection
        - (precision, recall, f-measure)
        - median distance to nearest boundary
    - Frame clustering
'''

import numpy as np
import sklearn.metrics.cluster as metrics

def boundary_detection(annotated_boundaries, predicted_boundaries, window=0.5, beta=1.0):
    '''Boundary detection hit-rate.

    :parameters:
    - annotated_boundaries : list-like, float
        ground-truth segment boundary times (in seconds)

    - predicted_boundaries : list-like, float
        predicted segment boundary times (in seconds)

    - window : float > 0
        size of the window of 'correctness' around ground-truth beats (in seconds)

    - beta : float > 0
        'beta' for F-measure.

    :returns:
    - P : float
        precision of predictions
    - R : float
        recall of ground-truth beats
    - F : float
        F-measure (harmonic mean of P, R)
    '''

    # Compute the hits
    D = np.abs( np.subtract.outer(annotated_boundaries, predicted_boundaries)) <= window

    # Note: in these metrics, each boundary can only be counted at most once

    # Precision: how many predicted boundaries were hits?
    P = np.mean(D.max(axis=0))

    # Recall: how many of the boundaries did we catch?
    R = np.mean(D.max(axis=1))

    # And the f-measure
    F = 0.0

    if P > 0 or R > 0:
        F = (1 + beta**2) * P * R / ((beta**2) * P + R)

    return P, R, F

def boundary_deviation(annotated_boundaries, predicted_boundaries):
    '''Compute the median deviations between annotated and predicted boundary times.

    :parameters:
    - annotated_boundaries : list-like, float
        ground-truth segment boundary times (in seconds)

    - predicted_boundaries : list-like, float
        predicted segment boundary times (in seconds)

    :returns:
    - true_to_predicted : float
        median time from each true boundary to the closest predicted boundary

    - predicted_to_true : float
        median time from each predicted boundary to the closest true boundary
    '''

    D = np.abs( np.subtract.outer(annotated_boundaries, predicted_boundaries) )

    true_to_predicted = np.median(np.sort(D, axis=1)[:, 0])
    predicted_to_true = np.median(np.sort(D, axis=0)[0, :])

    return true_to_predicted, predicted_to_true


def frame_clustering(annotated_boundaries, predicted_boundaries, frame_size=0.1, beta=1.0):
    '''Frame-clustering segmentation evaluation.

    :parameters:
    - annotated_boundaries : list-like, float
        ground-truth segment boundary times (in seconds)

    - predicted_boundaries : list-like, float
        predicted segment boundary times (in seconds)

    - frame_size : float > 0
        length (in seconds) of frames for clustering

    - beta : float > 0
        beta value for F-measure

    :returns:
    - ARI : float > 0
        adjusted Rand index

    - AMI : float > 0
        adjusted mutual information

    - NMI : float > 0
        normalized mutual information
    - Pair_precision : float > 0
    - Pair_recall   : float > 0
    - Pair_F        : float > 0
        Precision/recall/f-measure of detecting whether
        frames belong in the same cluster

    - Completeness  : float > 0
    - Homogeneity   : float > 0
    - V             : float > 0
        V-measure is the harmonic mean of Completeness and Homogeneity


    .. note::
        Boundaries are assumed to include end-points (0, len(song))

    .. note::
        All boundaries will be quantized to the nearest multiple of the frame size.
    '''


    def frame_labels(B):
        
        # First, round the boundaries to be a multiple of the frame size
        B = B - np.mod(B, frame_size)

        # Build the frame label array
        n = int(B[-1] / frame_size)
        y = np.zeros(n)

        for (i, (start, end)) in enumerate(zip(B[:-1], B[1:]), 1):
            y[int(start / frame_size):int(end / frame_size)] = i

        return y


    # Generate the cluster labels
    y_true = frame_labels(annotated_boundaries)
    y_pred = frame_labels(predicted_boundaries)

    # Make sure we have the same number of frames
    assert(len(y_true) == len(y_pred))

    # pairwise precision-recall
    def _frame_pairwise_detection(Y1, Y2):
        
        # Construct the label-agreement matrices
        A1 = np.triu(np.equal.outer(Y1, Y1))
        A2 = np.triu(np.equal.outer(Y2, Y2))
        
        matches = float((A1 & A2).sum())
        P = matches / A1.sum()
        R = matches / A2.sum()
        F = 0.0
        if P > 0 or R > 0:
            F = (1 + beta**2) * P * R / ((beta**2) * P + R)
        return P, R, F

    P, R, F = _frame_pairwise_detection(y_true, y_pred)

    # Compute all the clustering metrics
    ## Adjusted rand index
    ARI = metrics.adjusted_rand_score(y_true, y_pred)

    ## Adjusted mutual information
    AMI = metrics.adjusted_mutual_info_score(y_true, y_pred)
    
    ## Normalized mutual information
    NMI = metrics.normalized_mutual_info_score(y_true, y_pred)

    ## Completeness
    Hom, Comp, V = metrics.homogeneity_completeness_v_measure(y_true, y_pred)

    return ARI, AMI, NMI, P, R, F, Comp, Hom, V

