'''
Unit tests for mir_eval.beat
'''

import numpy as np
import mir_eval
import pickle

def test_beat_functions():
    # Load in an example beat annotation
    annotated_beats = np.genfromtxt('data/beat/annotated.beats')
    # Load in an example beat tracker output
    generated_beats = np.genfromtxt('data/beat/generated.beats')
    # Load in reference scores
    bet_scores = pickle.load(open('data/beat/bet_scores.pickle'))
    # List of functions in mir_eval.beat
    functions = [mir_eval.beat.f_measure,
                 mir_eval.beat.cemgil,
                 mir_eval.beat.goto,
                 mir_eval.beat.p_score,
                 mir_eval.beat.continuity,
                 mir_eval.beat.information_gain]
    # Check each function output against beat evaluation toolbox
    for function in functions:
        my_score = function(annotated_beats, generated_beats)
        their_score = bet_scores[function.__name__]
        assert np.allclose(my_score, their_score)