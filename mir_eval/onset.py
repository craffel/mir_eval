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
from . import util

# <codecell>

def f_measure(annotated_onsets, generated_onsets, window=.05):
    """
    Compute the F-measure, precision, and recall for a sequence of generated onsets
    
    Input:
        annotated_onsets - np.ndarray of reference onset times, in seconds
        generated_onsets - np.ndarray of generated onset times, in seconds
        window - An onset is correct if it is within +/-window seconds from an annotation, default .05
    Output:
        f_measure - 2*precision*recall/(precision + recall)
        precision - (# true positives)/(# true positives + # false positives)
        recall - (# true positives)/(# true positives + # false negatives)
    """
    # If both onset lists are empty, call it perfect accuracy
    if annotated_onsets.size == 0 and generated_onsets.size == 0:
        return 1., 1., 1.
    # If one list is empty and the other isn't, call it 0 accuracy
    elif annotated_onsets.size == 0 or generated_onsets.size == 0:
        return 0., 0., 0.
    # Counting the correct/incorrect onsets in this way requires sorting first
    annotated_onsets.sort()
    generated_onsets.sort()
    # For accessing entries in each list
    annotated_index = 0
    generated_index = 0
    # Keep track of true/false positive/negatives
    true_positives = 0.0
    false_positives = 0.0
    false_negatives = 0.0
    while (annotated_index < len(annotated_onsets) and
          generated_index < len(generated_onsets)):
        # Get the current onsets
        annotated_onset = annotated_onsets[annotated_index]
        generated_onset = generated_onsets[generated_index]
        # Does the generated onset fall within window around the annotated one?
        if np.abs(annotated_onset - generated_onset) <= window:
            # Found a true positive!
            true_positives += 1
            # Look at the next onset time for both
            annotated_index += 1
            generated_index += 1
        # We're out of the window - are we before?
        elif generated_onset < annotated_onset:
            # Generated an extra onset - it's a false positive
            false_positives += 1
            # Next time, check if the next generated onset is correct
            generated_index += 1
        # Or after?
        elif generated_onset > annotated_onset:
            # Must have missed the annotated onset - false negative
            false_negatives += 1
            # Next time, check this generated onset against the next annotated
            annotated_index += 1
    # Any additional generated onsets are false positives
    false_positives += len(generated_onsets) - generated_index
    # Any additional annotated onsets are false negatives
    false_negatives += len(annotated_onsets) - annotated_index
    # Compute precision and recall
    precision = true_positives/(true_positives + false_positives)
    recall = true_positives/(true_positives + false_negatives)
    # Compute F-measure and return all statistics
    return util.f_measure(precision, recall), precision, recall

