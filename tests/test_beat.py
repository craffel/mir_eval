'''
Unit tests for mir_eval.beat
'''

import numpy as np
import mir_eval
import pickle

def test_beat_functions():
    # Load in an example beat annotation
    reference_beats = np.genfromtxt('data/beat/reference.beats')
    # Load in an example beat tracker output
    estimated_beats = np.genfromtxt('data/beat/estimated.beats')
    # Trim the first 5 seconds off
    reference_beats = mir_eval.beat.trim_beats(reference_beats)
    estimated_beats = mir_eval.beat.trim_beats(estimated_beats)
    # Load in reference scores computed with the beat eval toolbox
    bet_scores = pickle.load(open('data/beat/bet_scores.pickle'))
    # List of functions in mir_eval.beat
    functions = {'f_measure':mir_eval.beat.f_measure,
                 'cemgil':mir_eval.beat.cemgil,
                 'goto':mir_eval.beat.goto,
                 'p_score':mir_eval.beat.p_score,
                 'continuity':mir_eval.beat.continuity,
                 'information_gain':mir_eval.beat.information_gain}
    # Check each function output against beat evaluation toolbox
    for name, function in functions.items():
        my_score = function(reference_beats, estimated_beats)
        their_score = bet_scores[name]
        assert np.allclose(my_score, their_score)
