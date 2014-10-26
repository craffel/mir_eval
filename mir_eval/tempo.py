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


def validate(reference_onsets, estimated_onsets):
    '''
    Checks that the input annotations to a metric look like valid onset time
    arrays, and throws helpful errors if not.

    :parameters:
        - reference_onsets : np.ndarray
            reference onset locations, in seconds
        - estimated_onsets : np.ndarray
            estimated onset locations, in seconds

    :raises:
        - ValueError
            Thrown when the provided annotations are not valid.
    '''
    # If reference or estimated onsets are empty, warn because metric will be 0
    if reference_onsets.size == 0:
        warnings.warn("Reference onsets are empty.")
    if estimated_onsets.size == 0:
        warnings.warn("Estimated onsets are empty.")
    for onsets in [reference_onsets, estimated_onsets]:
        util.validate_events(onsets, MAX_TIME)

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
