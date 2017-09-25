#!/usr/bin/env python
'''
Unit tests for mir_eval.tempo
'''
import nose
import warnings
import numpy as np
import mir_eval
from nose.tools import raises
import json
import glob


A_TOL = 1e-12


def test_load_tempo():
    tempi, weight = mir_eval.tempo.load('data/tempo/ref01.lab')
    assert np.allclose(tempi, [60, 120])
    assert weight == 0.5


@nose.tools.raises(ValueError)
def test_load_tempo_multiline():
    tempi, weight = mir_eval.tempo.load('data/tempo/bad00.lab')


@nose.tools.raises(ValueError)
def test_load_tempo_badweight():
    tempi, weight = mir_eval.tempo.load('data/tempo/bad01.lab')


def test_load_bad_tempi():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        tempi, weight = mir_eval.tempo.load('data/tempo/bad02.lab')
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert ('non-negative numbers' in str(w[-1].message))


def __check_score(sco_f, metric, score, expected_score):
    assert np.allclose(score, expected_score, atol=A_TOL)


def test_tempo_fail():

    @raises(ValueError)
    def __test(ref, weight, est, tol):
        mir_eval.tempo.detection(ref, weight, est, tol=tol)

    good_ref = np.array([60, 120])
    good_weight = 0.5
    good_est = np.array([120, 180])
    good_tol = 0.08

    for bad_tempo in [np.array([-1, 50]), np.array([0, 1, 2]), np.array([0])]:
        yield __test, bad_tempo, good_weight, good_est, good_tol
        yield __test, good_ref, good_weight, bad_tempo, good_tol

    for bad_weight in [-1, 1.5]:
        yield __test, good_ref, bad_weight, good_est, good_tol

    for bad_tol in [-1, 0, 1.5]:
        yield __test, good_ref, good_weight, good_est, bad_tol


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

        ref_tempi, ref_weight = mir_eval.tempo.load(ref_f)
        est_tempi, _ = mir_eval.tempo.load(est_f)

        scores = mir_eval.tempo.evaluate(ref_tempi, ref_weight, est_tempi)

        for metric in scores:
            yield (__check_score, sco_f, metric, scores[metric],
                   expected_scores[metric])
