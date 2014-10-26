'''
The goal of a tempo estimation algorithm is to automatically detect the tempo
of a piece of music, measured in beats per minute (BPM).  

Conventions
-----------


Metrics
-------

* :func:`mir_eval.tempo.detection`: 

'''

import numpy as np
import collections
from . import util
import warnings


def validate(reference_tempi, reference_weight, estimated_tempi):
    '''
    Checks that the input annotations to a metric look like valid tempo
    annotations.

    :parameters:
        - reference_onsets : np.ndarray
            reference tempo values, in bpm

        - reference_weight : float
            perceptual weight of slow vs fast in reference

        - estimated_onsets : np.ndarray
            estimated tempo values, in bpm

    :raises:
        - ValueError
            Thrown when the provided annotations are not valid.
    '''
    # If reference or estimated onsets are empty, warn because metric will be 0
    if reference_tempi.size != 2:
        raise ValueError("Reference tempi must have two values.")
    
    if estimated_tempi.size != 2:
        raise ValueError("Estimated tempi must have two values.")
    
    if np.any(reference_tempi <= 0):
        warnings.warn('Detected non-positive reference tempo')

    if reference_weight < 0 or reference_weight > 1:
        raise ValueError('Reference weight must lie in range [0, 1]')

def detection(reference_tempi, reference_weight, estimated_tempi, tol=0.08):
    '''
    Compute the tempo detection accuracy metric.
    Correctness of an estimated tempo is 

    :parameters:
      - reference_tempi : np.ndarray, shape=(2,)
        Two non-negative reference tempi, T1 and T2, such that T1 < T2.

      - reference_weight : float > 0
        The relative strength of T1 vs T2 in the reference.

      - estimated_tempi : np.ndarray, shape=(2,)
        Two non-negative estimated tempi, S1 and S2, such that S1 < S2

      - tol : float in [0, 1]:
        The maximum allowable deviation from a reference tempo to count as a hit.
        ``|S1 - T1| <= tol * T1``

    :returns:
      - relative_errors : np.ndarray, shape=(2,)
        The relative error of estimates vs reference tempi

      - hits : np.ndarray, shape=(2,)
        Boolean array counting whether each reference tempo was within tolerance 
        of an estimated tempo

      - p_score : float in [0, 1]
        Weighted average of recalls: 
        ``reference_weight * hits[0] + (1 - reference_weight) * hits[1]``
    '''

    validate(reference_tempi, reference_weight, estimated_tempi)

    relative_errors = []
    hits = []

    for ref_t in reference_tempi:
        # Compute the relative error for tihs reference tempo
        relative_errors.append(np.min(np.abs(ref_t - estimated_tempi) / float(ref_t)))

        # Count the hits
        hits.append(bool(relative_errors[-1] <= tol))

    p_score = reference_weight * hits[0] + (1.0-reference_weight) * hits[1]

    return relative_errors, hits, p_score

def evaluate(reference_tempi, reference_weight, estimated_tempi, **kwargs):
    '''
    Compute all metrics for the given reference and estimated annotations.

    :usage:

    :parameters:
        - kwargs
            Additional keyword arguments which will be passed to the
            appropriate metric or preprocessing functions.

    :returns:
        - scores : dict
            Dictionary of scores, where the key is the metric name (str) and
            the value is the (float) score achieved.
    '''
    # Compute all metrics
    scores = collections.OrderedDict()

    (scores['Relative-Error'],
     scores['Hits'],
     scores['P-score']) = util.filter_kwargs(detection, reference_tempi,
                                             reference_weight, estimated_tempi, 
                                             **kwargs)

    return scores
