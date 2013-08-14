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

    A hit is counted whenever an annotated boundary is within `window` of a predicted
    boundary.

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


def boundaries_to_frames(boundaries, frame_size=0.1):
    '''Convert a sequence of boundaries to frame-level segment annotations.
    
    :parameters:
    - boundaries : list-like float
        segment boundary times (in seconds).

    - frame_size : float > 0
        duration of each frame (in seconds)

    :returns:
    - y : np.array, dtype=int
        array of segment labels for each frame

    ..note::
        It is assumed that `boundaries[-1] == length of song`

    ..note::
        Segment boundaries will be rounded down to the nearest multiple 
        of frame_size.
    '''
    
    boundaries = np.sort(boundaries - np.mod(boundaries, frame_size))
    boundaries = np.unique(np.concatenate(([0], boundaries)))

    # Build the frame label array
    y = np.zeros(int(boundaries[-1] / frame_size))

    for (i, (start, end)) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        y[int(start / frame_size):int(end / frame_size)] = i

    return y

def frame_clustering_pairwise(annotated_boundaries, predicted_boundaries, frame_size=0.1, beta=1.0):
    '''Frame-clustering segmentation evaluation by pair-wise agreement.

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
    - Pair_precision : float > 0
    - Pair_recall   : float > 0
    - Pair_F        : float > 0
        Precision/recall/f-measure of detecting whether
        frames belong in the same cluster

    ..note::
        It is assumed that `boundaries[-1] == length of song`

    ..note::
        Segment boundaries will be rounded down to the nearest multiple 
        of frame_size.
    '''

    # Generate the cluster labels
    y_true = boundaries_to_frames(annotated_boundaries)
    y_pred = boundaries_to_frames(predicted_boundaries)
    # Make sure we have the same number of frames
    assert(len(y_true) == len(y_pred))

    # Construct the label-agreement matrices
    A1 = np.triu(np.equal.outer(y_true, y_true))
    A2 = np.triu(np.equal.outer(y_pred, y_pred))
    
    matches = float((A1 & A2).sum())
    P = matches / A1.sum()
    R = matches / A2.sum()
    F = 0.0
    if P > 0 or R > 0:
        F = (1 + beta**2) * P * R / ((beta**2) * P + R)

    return P, R, F


def frame_clustering_rand(annotated_boundaries, predicted_boundaries, frame_size=0.1):
    '''Frame-clustering segmentation via Rand index.

    :parameters:
    - annotated_boundaries : list-like, float
        ground-truth segment boundary times (in seconds)

    - predicted_boundaries : list-like, float
        predicted segment boundary times (in seconds)

    - frame_size : float > 0
        length (in seconds) of frames for clustering

    :returns:
    - ARI : float > 0
        Adjusted Rand index between segmentations.

    ..note::
        It is assumed that `boundaries[-1] == length of song`

    ..note::
        Segment boundaries will be rounded down to the nearest multiple 
        of frame_size.
    '''
    # Generate the cluster labels
    y_true = boundaries_to_frames(annotated_boundaries)
    y_pred = boundaries_to_frames(predicted_boundaries)
    # Make sure we have the same number of frames
    assert(len(y_true) == len(y_pred))

    # Compute all the clustering metrics
    ## Adjusted rand index

    return metrics.adjusted_rand_score(y_true, y_pred)

def frame_clustering_mutual_information(annotated_boundaries, predicted_boundaries, frame_size=0.1):
    '''Frame-clustering segmentation: mutual information metrics.

    :parameters:
    - annotated_boundaries : list-like, float
        ground-truth segment boundary times (in seconds)

    - predicted_boundaries : list-like, float
        predicted segment boundary times (in seconds)

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
        It is assumed that `boundaries[-1] == length of song`

    ..note::
        Segment boundaries will be rounded down to the nearest multiple 
        of frame_size.
    '''
    # Generate the cluster labels
    y_true = boundaries_to_frames(annotated_boundaries)
    y_pred = boundaries_to_frames(predicted_boundaries)
    # Make sure we have the same number of frames
    assert(len(y_true) == len(y_pred))


    ## Adjusted mutual information
    MI = metrics.mutual_info_score(y_true, y_pred)

    ## Adjusted mutual information
    AMI = metrics.adjusted_mutual_info_score(y_true, y_pred)
    
    ## Normalized mutual information
    NMI = metrics.normalized_mutual_info_score(y_true, y_pred)

    return MI, AMI, NMI
    
def frame_clustering_v_measure(annotated_boundaries, predicted_boundaries, frame_size=0.1):
    '''Frame-clustering segmentation: v-measure metrics.

    :parameters:
    - annotated_boundaries : list-like, float
        ground-truth segment boundary times (in seconds)

    - predicted_boundaries : list-like, float
        predicted segment boundary times (in seconds)

    - frame_size : float > 0
        length (in seconds) of frames for clustering

    :returns:
    - H : float 
        Homogeneity
    - C : float
        Completeness
    - V : float
        V-measure, harmonic mean of H and C
    
    ..note::
        It is assumed that `boundaries[-1] == length of song`

    ..note::
        Segment boundaries will be rounded down to the nearest multiple 
        of frame_size.
    '''

    # Generate the cluster labels
    y_true = boundaries_to_frames(annotated_boundaries)
    y_pred = boundaries_to_frames(predicted_boundaries)
    # Make sure we have the same number of frames
    assert(len(y_true) == len(y_pred))

    ## Completeness
    return metrics.homogeneity_completeness_v_measure(y_true, y_pred)

