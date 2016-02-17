# CREATED: 2/9/16 2:27 PM by Justin Salamon <justin.salamon@nyu.edu>

import mir_eval
import numpy as np

A_TOL = 1e-12

REF = np.array([
    [0.100, 0.300, 220.000],
    [0.300, 0.400, 246.942],
    [0.500, 0.600, 277.183],
    [0.550, 0.650, 293.665]])

EST = np.array([
    [0.120,   0.290,   225.000],
    [0.300,   0.340,   246.942],
    [0.500,   0.600,   500.000],
    [0.550,   0.600,   293.665],
    [0.560,   0.650,   293.665]])

SCORES = {
    "Precision": 0.4,
    "Recall": 0.5,
    "F-measure": 0.4444444444444445,
    "Precision_no_offset": 0.6,
    "Recall_no_offset": 0.75,
    "F-measure_no_offset": 0.6666666666666665
}


def test_match_notes():

    ref_int, ref_pitch = REF[:, :2], REF[:, 2]
    est_int, est_pitch = EST[:, :2], EST[:, 2]

    matching = \
        mir_eval.transcription.match_notes(ref_int, ref_pitch, est_int,
                                           est_pitch)

    assert matching == [(0, 0), (3, 3)]

    matching = \
        mir_eval.transcription.match_notes(ref_int, ref_pitch, est_int,
                                           est_pitch, offset_ratio=None)

    assert matching == [(0, 0), (1, 1), (3, 3)]


def test_precision_recall_f1():

    # load test data
    ref_int, ref_pitch = REF[:, :2], REF[:, 2]
    est_int, est_pitch = EST[:, :2], EST[:, 2]

    precision, recall, f_measure = \
        mir_eval.transcription.precision_recall_f1(ref_int, ref_pitch, est_int,
                                                   est_pitch)

    scores_gen = np.array([precision, recall, f_measure])
    scores_exp = np.array([SCORES['Precision'], SCORES['Recall'],
                           SCORES['F-measure']])
    assert np.allclose(scores_exp, scores_gen, atol=A_TOL)

    precision, recall, f_measure = \
        mir_eval.transcription.precision_recall_f1(ref_int, ref_pitch, est_int,
                                                   est_pitch,
                                                   offset_ratio=None)

    scores_gen = np.array([precision, recall, f_measure])
    scores_exp = np.array([SCORES['Precision_no_offset'],
                           SCORES['Recall_no_offset'],
                           SCORES['F-measure_no_offset']])
    assert np.allclose(scores_exp, scores_gen, atol=A_TOL)
