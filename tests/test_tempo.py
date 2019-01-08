#!/usr/bin/env python
'''
Unit tests for mir_eval.tempo
'''

import numpy as np
import mir_eval
from nose.tools import raises
import json
import glob


A_TOL = 1e-12


def _load_tempi(filename):

    values = mir_eval.io.load_delimited(filename, [float] * 3)

    return np.concatenate(values[:2]), values[-1][0]


def __check_score(sco_f, metric, score, expected_score):
    assert np.allclose(score, expected_score, atol=A_TOL)


def test_tempo_pass():
    good_ref = np.array([60, 120])
    good_weight = 0.5
    good_est = np.array([120, 180])
    good_tol = 0.08

    for good_tempo in [np.array([50, 50]), np.array([0, 50]), np.array([50, 0])]:
        yield mir_eval.tempo.detection, good_tempo, good_weight, good_est, good_tol
        yield mir_eval.tempo.detection, good_ref, good_weight, good_tempo, good_tol

    # allow both estimates to be zero
    yield mir_eval.tempo.detection, good_ref, good_weight, np.array([0, 0]), good_tol


def test_tempo_fail():

    @raises(ValueError)
    def __test(ref, weight, est, tol):
        mir_eval.tempo.detection(ref, weight, est, tol=tol)

    good_ref = np.array([60, 120])
    good_weight = 0.5
    good_est = np.array([120, 180])
    good_tol = 0.08

    for bad_tempo in [np.array([-1, -1]), np.array([-1, 0]),
                      np.array([-1, 50]), np.array([0, 1, 2]), np.array([0])]:
        yield __test, bad_tempo, good_weight, good_est, good_tol
        yield __test, good_ref, good_weight, bad_tempo, good_tol

    for bad_weight in [-1, 1.5]:
        yield __test, good_ref, bad_weight, good_est, good_tol

    for bad_tol in [-1, 0, 1.5]:
        yield __test, good_ref, good_weight, good_est, bad_tol

    # don't allow both references to be zero
    yield __test, np.array([0, 0]), good_weight, good_ref, good_tol


def test_tempo_regression():
    REF_GLOB = 'data/tempo/ref*.lab'
    EST_GLOB = 'data/tempo/est*.lab'
    SCORES_GLOB = 'data/tempo/output*.json'

    # Load in all files in the same order
    ref_files = sorted(glob.glob(REF_GLOB))
    est_files = sorted(glob.glob(EST_GLOB))
    sco_files = sorted(glob.glob(SCORES_GLOB))

    assert len(ref_files) == len(est_files) == len(sco_files)

    for ref_f, est_f, sco_f in zip(ref_files, est_files, sco_files):
        with open(sco_f, 'r') as fdesc:
            expected_scores = json.load(fdesc)

        ref_tempi, ref_weight = _load_tempi(ref_f)
        est_tempi, _ = _load_tempi(est_f)

        scores = mir_eval.tempo.evaluate(ref_tempi, ref_weight, est_tempi)

        for metric in scores:
            yield (__check_score, sco_f, metric, scores[metric],
                   expected_scores[metric])
