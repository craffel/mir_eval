# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

'''
Basic metrics for evaluating onset detection systems.
Based in part on this script:
    https://github.com/CPJKU/onset_detection/blob/master/onset_evaluation.py
'''

# <codecell>

import numpy as np
import functools
import collections
from . import util

# <codecell>

def validate(metric):
    '''Decorator which checks that the input annotations to a metric
    look like valid onset time arrays, and throws helpful errors if not.

    :parameters:
        - metric : function
            Evaluation metric function.  First two arguments must be
            reference_onsets and estimated_onsets.

    :returns:
        - metric_validated : function
            The function with the onset locations validated
    '''
    # Retain docstring, etc
    @functools.wraps(metric)
    def metric_validated(reference_onsets, estimated_onsets, *args, **kwargs):
        '''
        Metric with input onset annotations validated
        '''
        for onsets in [reference_onsets, estimated_onsets]:
            # Make sure beat locations are 1-d np ndarrays
            if onsets.ndim != 1:
                raise ValueError('Onset locations should be 1-d numpy ndarray')
            # Make sure no beat times are huge
            if (onsets > 30000).any():
                raise ValueError('An onset at time {}'.format(beats.max()) + \
                                 ' was found; should be in seconds.')
            # Make sure no beat times are negative
            if (onsets < 0).any():
                raise ValueError('Negative beat locations found')
            # Make sure beat times are increasing
            if (np.diff(onsets) < 0).any():
                raise ValueError('Onsets should be in increasing order.')
        return metric(reference_onsets, estimated_onsets, *args, **kwargs)
    return metric_validated

# <codecell>

@validate
def f_measure(reference_onsets, esimated_onsets, window=.05):
    '''
    Compute the F-measure of correct vs incorrectly predicted onsets.
    "Corectness" is determined over a small window.
    
    :usage:
        >>> reference_onsets = mir_eval.io.load_events('reference.txt')
        >>> estimated_onsets = mir_eval.io.load_events('estimated.txt')
        >>> f_measure = mir_eval.onset.f_measure(reference_beats, estimated_beats)

    :parameters:
        - reference_onsets : np.ndarray
            reference onset locations, in seconds
        - estimated_onsets : np.ndarray 
            estimated onset locations, in seconds
        - window : float
            Window size, in seconds, default 0.05
    
    :returns:
        - f_measure : float
            2*precision*recall/(precision + recall)
        - precision : float
            (# true positives)/(# true positives + # false positives)
        - recall : float
            (# true positives)/(# true positives + # false negatives)
    '''    
    # If both onset lists are empty, call it perfect accuracy
    if reference_onsets.size == 0 and esimated_onsets.size == 0:
        return 1., 1., 1.
    # If one list is empty and the other isn't, call it 0 accuracy
    elif reference_onsets.size == 0 or esimated_onsets.size == 0:
        return 0., 0., 0.
    # For accessing entries in each list
    reference_index = 0
    esimated_index = 0
    # Keep track of true/false positive/negatives
    true_positives = 0.0
    false_positives = 0.0
    false_negatives = 0.0
    while (reference_index < len(reference_onsets) and
          esimated_index < len(esimated_onsets)):
        # Get the current onsets
        reference_onset = reference_onsets[reference_index]
        esimated_onset = esimated_onsets[esimated_index]
        # Does the generated onset fall within window around the annotated one?
        if np.abs(reference_onset - esimated_onset) <= window:
            # Found a true positive!
            true_positives += 1
            # Look at the next onset time for both
            reference_index += 1
            esimated_index += 1
        # We're out of the window - are we before?
        elif esimated_onset < reference_onset:
            # Generated an extra onset - it's a false positive
            false_positives += 1
            # Next time, check if the next generated onset is correct
            esimated_index += 1
        # Or after?
        elif esimated_onset > reference_onset:
            # Must have missed the annotated onset - false negative
            false_negatives += 1
            # Next time, check this generated onset against the next annotated
            reference_index += 1
    # Any additional generated onsets are false positives
    false_positives += len(esimated_onsets) - esimated_index
    # Any additional annotated onsets are false negatives
    false_negatives += len(reference_onsets) - reference_index
    # Compute precision and recall
    precision = true_positives/(true_positives + false_positives)
    recall = true_positives/(true_positives + false_negatives)
    # Compute F-measure and return all statistics
    return util.f_measure(precision, recall), precision, recall

# <codecell>

# Create a dictionary which maps the name of each metric 
# to the function used to compute it
metrics = collections.OrderedDict()
metrics['F-measure'] = f_measure

