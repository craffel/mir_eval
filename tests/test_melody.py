# CREATED: 4/15/14 9:42 AM by Justin Salamon <justin.salamon@nyu.edu>
'''
Unit tests for mir_eval.melody
'''

import numpy as np
import mir_eval

def test_melody_functions():

    songs = ['daisy1','daisy2','daisy3','daisy4','jazz1','jazz2','jazz3','jazz4','midi1','midi2','midi3','midi4','opera_fem2','opera_fem4','opera_male3','opera_male5','pop1','pop2','pop3','pop4']
    refpath = 'data/melody/mirex2011/adc2004_ref/'
    estpath = 'data/melody/mirex2011/adc2004_SG2/'
    resultspath = 'data/mirex2011/adc2004_results/SG2_per_track_results_mapped.csv'

    for song in songs:



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