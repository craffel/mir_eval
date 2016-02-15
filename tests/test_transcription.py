# CREATED: 2/9/16 2:27 PM by Justin Salamon <justin.salamon@nyu.edu>

import mir_eval
import json
import numpy as np

A_TOL = 1e-12


def test_match_notes():

    ref_int, ref_pitch = mir_eval.io.load_valued_intervals(
        'tests/data/transcription/ref00.txt')
    est_int, est_pitch = mir_eval.io.load_valued_intervals(
        'tests/data/transcription/est00.txt')

    matching = \
        mir_eval.transcription.match_notes(ref_int, ref_pitch, est_int,
                                           est_pitch)

    assert matching == [(0, 0), (3, 4)]

    matching = \
        mir_eval.transcription.match_notes(ref_int, ref_pitch, est_int,
                                           est_pitch, offset_ratio=None)

    assert matching == [(0, 0), (1, 1), (3, 3)]


def test_precision_recall_f1():

    ref_int, ref_pitch = mir_eval.io.load_valued_intervals(
        'tests/data/transcription/ref00.txt')
    est_int, est_pitch = mir_eval.io.load_valued_intervals(
        'tests/data/transcription/est00.txt')

    # load expected results
    scores = json.load(open('tests/data/transcription/output00.json', 'r'))

    precision, recall, f_measure = \
        mir_eval.transcription.precision_recall_f1(ref_int, ref_pitch, est_int,
                                                   est_pitch)

    scores_gen = np.array([precision, recall, f_measure])
    scores_exp = np.array([scores['Precision'], scores['Recall'],
                           scores['F-measure']])
    assert np.allclose(scores_exp, scores_gen, atol=A_TOL)

    precision, recall, f_measure = \
        mir_eval.transcription.precision_recall_f1(ref_int, ref_pitch, est_int,
                                                   est_pitch,
                                                   offset_ratio=None)

    scores_gen = np.array([precision, recall, f_measure])
    scores_exp = np.array([scores['Precision_no_offset'],
                           scores['Recall_no_offset'],
                           scores['F-measure_no_offset']])
    assert np.allclose(scores_exp, scores_gen, atol=A_TOL)
