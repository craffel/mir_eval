'''Functions for evaluating the task of pattern discovery.

Input Format
============

Set of patterns including all their occurrences.

Metrics implemented
===================

Standard Precision, Recall and F1 Score
---------------------------------------

Strict metric in order to find the exact patterns. Used and described in this
paper:

Tom Collins, Jeremy Thurlow, Robin Laney, Alistair Willis, and Paul H.
Garthwaite. A comparative evaluation of algorithms for discovering
translational patterns in Baroque keyboard works. In J.S. Downie and R.
Veltkamp (Eds), Proc ISMIR, pp. 3-8, Utrecht, 2010.

Establishment Precision, Recall and F1 Score
--------------------------------------------






Metrics developed by Tom Colins and David Meredith.

Written by Oriol Nieto (oriol@nyu.edu), 2014
'''


import functools
import numpy as np
from . import util


def validate(metric):
    """Decorator which checks that the input annotations to a metric
    look like valid pattern lists, and throws helpful errors if not.

    :parameters:
        - metric : function
            Evaluation metric function.  First two arguments must be
            reference_patterns and estimated_patterns.

    :returns:
        - metric_validated : function
            The function with the pattern lists validated.
    """
    # Retain docstring, etc
    @functools.wraps(metric)
    def metric_validated(reference_patterns, estimated_patterns, *args,
                         **kwargs):
        '''
        Metric with input beat annotations validated
        '''
        for patterns in [reference_patterns, estimated_patterns]:
            # TODO:
            pass
            ## Make sure beat locations are 1-d np ndarrays
            #if beats.ndim != 1:
                #raise ValueError('Beat locations should be 1-d numpy ndarray')
            ## Make sure no beat times are huge
            #if (beats > 30000).any():
                #raise ValueError('A beat at time {}'.format(beats.max()) + \
                                 #' was found; should be in seconds.')
            ## Make sure no beat times are negative
            #if (beats < 0).any():
                #raise ValueError('Negative beat locations found')
            ## Make sure beat times are increasing
            #if (np.diff(beats) < 0).any():
                #raise ValueError('Beats should be in increasing order.')
        return metric(reference_patterns, estimated_patterns, *args, **kwargs)
    return metric_validated


@validate
def standard_FPR(reference_patterns, estimated_patterns, tol=1e-5):
    """Standard Precision, Recall and F1 Score.

    This metric checks if the prorotype patterns of the reference match
    possible translated patterns in the prototype patterns of the estimations.
    Since the sizes of these prototypes must be equal, this metric is quite
    restictive and it tends to be 0 in most of 2013 MIREX results.

    :param reference_patterns: The reference patterns using the same format as
        the one load_patterns in the input_output module returns.
    :type reference_patterns: list
    :param estimated_patterns: The estimated patterns using the same format as
        the one load_patterns in the input_output module returns.
    :type estimated_patterns: list
    :returns:
        - f_measure : float
            The standard F1 Score
        - precision : float
            The standard Precision
        - recall : float
            The standard Recall
    """
    nP = len(reference_patterns)    # Number of patterns in the reference
    nQ = len(estimated_patterns)    # Number of patterns in the estimation
    k = 0                           # Number of patterns that match

    # Find matches of the prototype patterns
    for ref_pattern in reference_patterns:
        P = np.asarray(ref_pattern[0])      # Get reference prototype
        for est_pattern in estimated_patterns:
            Q = np.asarray(est_pattern[0])  # Get estimation prototype

            if len(P) != len(Q):
                continue

            # Check transposition given a certain tolerance
            if np.sum(np.diff(P - Q, axis=0)) <= tol:
                k += 1
                break

    # Compute the standard measures
    precision = k / float(nQ)
    recall = k / float(nP)
    f_measure = util.f_measure(precision, recall)
    return f_measure, precision, recall
