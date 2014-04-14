'''
Unit tests for mir_eval.beat
'''

import numpy as np
import mir_eval
import pickle

def test_onset_functions():
    # Load in an example onset annotation
    reference_onsets = np.genfromtxt('data/onset/reference.onsets')
    # Load in an example onset detector output
    estimated_onsets = np.genfromtxt('data/onset/estimated.onsets')
    # Load in reference scores
    reference_scores = pickle.load(open('data/onset/cpjku_scores.pickle'))
    # List of functions in mir_eval.beat
    functions = [mir_eval.onset.f_measure]
    # Check each function output against beat evaluation toolbox
    for function in functions:
        my_score = function(reference_onsets, estimated_onsets)
        their_score = reference_scores[function.__name__]
        assert np.allclose(my_score, their_score)
