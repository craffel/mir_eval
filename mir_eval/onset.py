'''
Basic metrics for evaluating onset detection systems.

Based in part on this script:

    https://github.com/CPJKU/onset_detection/blob/master/onset_evaluation.py
'''

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
    '''
    # If reference or estimated onsets are empty, warn because metric will be 0
    if reference_onsets.size == 0:
        warnings.warn("Reference onsets are empty.")
    if estimated_onsets.size == 0:
        warnings.warn("Estimated onsets are empty.")
    for onsets in [reference_onsets, estimated_onsets]:
        util.validate_events(onsets)


def f_measure(reference_onsets, estimated_onsets, window=.05):
    '''
    Compute the F-measure of correct vs incorrectly predicted onsets.
    "Corectness" is determined over a small window.

    :usage:
        >>> reference_onsets = mir_eval.io.load_events('reference.txt')
        >>> estimated_onsets = mir_eval.io.load_events('estimated.txt')
        >>> F, P, R = mir_eval.onset.f_measure(reference_onsets,
                                               estimated_onsets)

    :parameters:
        - reference_onsets : np.ndarray
            reference onset locations, in seconds
        - estimated_onsets : np.ndarray
            estimated onset locations, in seconds
        - window : float
            Window size, in seconds

    :returns:
        - f_measure : float
            2*precision*recall/(precision + recall)
        - precision : float
            (# true positives)/(# true positives + # false positives)
        - recall : float
            (# true positives)/(# true positives + # false negatives)

    :references:
        .. [#] S. Dixon, "Onset detection revisited," in
            Proceedings of 9th International Conference on Digital Audio
            Effects (DAFx), Montreal, Canada, 2006, pp. 133-137.
        .. [#] Sebastian Bock, Florian Krebs, and Markus Schedl. "Evaluating
            the Online Capabilities of Onset Detection Methods", in Proceedings
            of the 13th International Society for Music Information Retrieval
            Conference, 2012, pp. 49-54.
    '''
    validate(reference_onsets, estimated_onsets)
    # If either list is empty, return 0s
    if reference_onsets.size == 0 or estimated_onsets.size == 0:
        return 0., 0., 0.
    # Compute the best-case matching between reference and estimated onset
    # locations
    matching = util.match_events(reference_onsets, estimated_onsets, window)

    precision = float(len(matching))/len(estimated_onsets)
    recall = float(len(matching))/len(reference_onsets)
    # Compute F-measure and return all statistics
    return util.f_measure(precision, recall), precision, recall


def evaluate(reference_onsets, estimated_onsets, **kwargs):
    '''
    Compute all metrics for the given reference and estimated annotations.

    :parameters:
        - reference_onsets : np.ndarray
            reference onset locations, in seconds
        - estimated_onsets : np.ndarray
            estimated onset locations, in seconds
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

    (scores['F-measure'],
     scores['Precision'],
     scores['Recall']) = util.filter_kwargs(f_measure, reference_onsets,
                                            estimated_onsets, **kwargs)

    return scores

# Create a dictionary which maps the name of each metric
# to the function used to compute it
METRICS = collections.OrderedDict()
METRICS['F-measure'] = f_measure
