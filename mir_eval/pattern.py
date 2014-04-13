"""Functions for evaluating the task of pattern discovery.

Input Format
============

Set of patterns including all their occurrences.

Metrics implemented
===================

Standard Precision, Recall and F1 Score
---------------------------------------

Strict metric in order to find the possibly transposed patterns of exact
length. Used and described in this paper:

Tom Collins, Jeremy Thurlow, Robin Laney, Alistair Willis, and Paul H.
Garthwaite. A comparative evaluation of algorithms for discovering
translational patterns in Baroque keyboard works. In J.S. Downie and R.
Veltkamp (Eds), Proc ISMIR, pp. 3-8, Utrecht, 2010.

Establishment Precision, Recall and F1 Score
--------------------------------------------







Written by Oriol Nieto (oriol@nyu.edu), 2014
"""


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
    """Standard F1 Score, Precision and Recall.

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


@validate
def establishment_FPR(reference_patterns, estimated_patterns):
    """Establishment F1 Score, Precision and Recall.
    """
    pass


@validate
def three_layer_FPR(reference_patterns, estimated_patterns):
    """Three Layer F1 Score, Precision and Recall. As described by Meridith.

    :param reference_patterns: The reference patterns using the same format as
        the one load_patterns in the input_output module returns.
    :type reference_patterns: list
    :param estimated_patterns: The estimated patterns using the same format as
        the one load_patterns in the input_output module returns.
    :type estimated_patterns: list
    :returns:
        - f_measure : float
            The three-layer F1 Score
        - precision : float
            The three-layer Precision
        - recall : float
            The three-layer Recall
    """

    def compute_first_layer_PR(ref_occs, est_occs):
        """Computes the first layer Precision and Recall values given the
        set of occurrences in the reference and the set of occurrences in the
        estimation."""
        # Find intersection between reference and estimation
        occ_P = set([tuple(midi_onset)
                    for midi_onset in ref_occs])
        occ_Q = set([tuple(midi_onset)
                    for midi_onset in est_occs])
        s = len(occ_P & occ_Q)    # Size of the intersection

        # Compute the first layer scores
        precision = s / float(len(ref_occs))
        recall = s / float(len(est_occs))
        return precision, recall

    def compute_second_layer_PR(ref_pattern, est_pattern):
        """Computes the second layer Precision and Recall values given the
        set of occurrences in the reference and the set of occurrences in the
        estimation."""
        # Compute the first layer scores
        F_1 = compute_layer(ref_pattern, est_pattern)

        # Compute the second layer scores
        precision = np.mean(np.max(F_1, axis=0))
        recall = np.mean(np.max(F_1, axis=1))
        return precision, recall

    def compute_layer(ref_elements, est_elements, layer=1):
        """Computes the F-measure matrix for a given layer. The reference and
        estimated elements can be either patters or occurrences, depending
        on the layer.

        For layer 1, the elements must be occurrences.
        For layer 2, the elements must be patterns.
        """
        nP = len(ref_elements)      # Number of elements in reference
        nQ = len(est_elements)      # Number of elements in estimation
        F = np.zeros((nP, nQ))      # F-measure matrix for the given layer
        for iP in xrange(nP):
            for iQ in xrange(nQ):
                if layer == 1:
                    func = compute_first_layer_PR
                elif layer == 2:
                    func = compute_second_layer_PR

                # Compute layer scores
                precision, recall = func(ref_elements[iP], est_elements[iQ])
                F[iP, iQ] = util.f_measure(precision, recall)
        return F

    # Compute the second layer (it includes the first layer)
    F_2 = compute_layer(reference_patterns, estimated_patterns, layer=2)

    # Compute the final scores (third layer)
    precision_3 = np.mean(np.max(F_2, axis=0))
    recall_3 = np.mean(np.max(F_2, axis=1))
    f_measure_3 = util.f_measure(precision_3, recall_3)
    return f_measure_3, precision_3, recall_3
