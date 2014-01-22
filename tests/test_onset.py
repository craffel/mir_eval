'''
Unit tests for mir_eval.beat
'''

import numpy as np
import mir_eval
import pickle

def test_beat_functions():
    # Load in an example onset annotation
    annotated_beats = np.genfromtxt('data/onset/annotated.onsets')
    # Load in an example onset detector output
    generated_beats = np.genfromtxt('data/onset/generated.onsets')
    # Load in reference scores
    bet_scores = pickle.load(open('data/onset/cpjku_scores.pickle'))
    # List of functions in mir_eval.beat
    functions = [mir_eval.onset.f_measure]
    # Check each function output against beat evaluation toolbox
    for function in functions:
        my_score = function(annotated_beats, generated_beats)
        their_score = bet_scores[function.__name__]
        assert np.allclose(my_score, their_score)