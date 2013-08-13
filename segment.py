# CREATED:2013-08-13 12:02:42 by Brian McFee <brm2132@columbia.edu>
'''Structural segmentation evaluation, following the protocols of MIREX2012:
    - Boundary detection
        - (precision, recall, f-measure)
        - median distance to nearest boundary
    - Frame clustering
'''

import numpy as np

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
