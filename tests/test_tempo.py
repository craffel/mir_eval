#!/usr/bin/env python
'''
Unit tests for mir_eval.tempo
'''

import numpy as np
import mir_eval
from nose.tools import raises
import json
import glob

REF_GLOB = 'tests/data/tempo/ref*.lab'
EST_GLOB = 'tests/data/tempo/est*.lab'
SCORES_GLOB = 'tests/data/tempo/output*.json'


def test_fail():

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
