'''
Unit tests for mir_eval.beat
'''

import numpy as np
import json
import mir_eval
import glob
import warnings
import nose.tools

A_TOL = 1e-12

# Path to the fixture files
REF_GLOB = 'tests/data/beat/ref*.txt'
EST_GLOB = 'tests/data/beat/est*.txt'
SCORES_GLOB = 'tests/data/beat/output*.json'


def test_trim_beats():
    # Construct dummy beat times [0., 1., ...]
    dummy_beats = np.arange(10, dtype=np.float)
    # We expect trim_beats to remove all beats < 5s
    expected_beats = np.arange(5, 10, dtype=np.float)
    assert np.allclose(mir_eval.beat.trim_beats(dummy_beats), expected_beats)


def __unit_test_beat_function(metric):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        # First, test for a warning on empty beats
        metric(np.array([]), np.arange(10))
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert str(w[-1].message) == "Reference beats are empty."
        metric(np.arange(10), np.array([]))
        assert len(w) == 2
        assert issubclass(w[-1].category, UserWarning)
        assert str(w[-1].message) == "Estimated beats are empty."
        # And that the metric is 0
        assert np.allclose(metric(np.array([]), np.array([])), 0)

    # Now test validation function - beats must be 1d ndarray
    beats = np.array([[1., 2.]])
    nose.tools.assert_raises(ValueError, metric, beats, beats)
    # Beats must be in seconds (so not huge)
    beats = np.array([1e10, 1e11])
    nose.tools.assert_raises(ValueError, metric, beats, beats)
    # Beats must be sorted
    beats = np.array([2., 1.])
    nose.tools.assert_raises(ValueError, metric, beats, beats)

    # Valid beats which are the same produce a score of 1 for all metrics
    beats = np.arange(10, dtype=np.float)
    assert np.allclose(metric(beats, beats), 1)


def __check_score(sco_f, metric, score, expected_score):
    assert np.allclose(score, expected_score, atol=A_TOL)


def test_beat_functions():
    # Load in all files in the same order
    ref_files = sorted(glob.glob(REF_GLOB))
    est_files = sorted(glob.glob(EST_GLOB))
    sco_files = sorted(glob.glob(SCORES_GLOB))

    # Unit tests
    for metric in [mir_eval.beat.f_measure,
                   mir_eval.beat.cemgil,
                   mir_eval.beat.goto,
                   mir_eval.beat.p_score,
                   mir_eval.beat.continuity,
                   mir_eval.beat.information_gain]:
        yield (__unit_test_beat_function, metric)
    # Regression tests
    for ref_f, est_f, sco_f in zip(ref_files, est_files, sco_files):
        with open(sco_f, 'r') as f:
            expected_scores = json.load(f)
        # Load in an example beat annotation
        reference_beats = mir_eval.io.load_events(ref_f)
        # Load in an example beat tracker output
        estimated_beats = mir_eval.io.load_events(est_f)
        # Compute scores
        scores = mir_eval.beat.evaluate(reference_beats, estimated_beats)
        # Compare them
        for metric in scores:
            # This is a simple hack to make nosetest's messages more useful
            yield (__check_score, sco_f, metric, scores[metric],
                   expected_scores[metric])
