'''
Unit tests for mir_eval.onset
'''

import numpy as np
import json
import mir_eval
import glob
import warnings
import nose.tools

A_TOL = 1e-12

# Path to the fixture files
REF_GLOB = 'data/onset/ref*.txt'
EST_GLOB = 'data/onset/est*.txt'
SCORES_GLOB = 'data/onset/output*.json'

def __unit_test_onset_function(metric):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        # First, test for a warning on empty onsets
        metric(np.array([]), np.arange(10))
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert str(w[-1].message) == "Reference onsets are empty."
        metric(np.arange(10), np.array([]))
        assert len(w) == 2
        assert issubclass(w[-1].category, UserWarning)
        assert str(w[-1].message) == "Estimated onsets are empty."
        # And that the metric is 0
        assert np.allclose(metric(np.array([]), np.array([])), 0)

    # Now test validation function - onsets must be 1d ndarray
    onsets = np.array([[1., 2.]])
    nose.tools.assert_raises(ValueError, metric, onsets, onsets)
    # onsets must be in seconds (so not huge)
    onsets = np.array([1e10, 1e11])
    nose.tools.assert_raises(ValueError, metric, onsets, onsets)
    # onsets must be sorted
    onsets = np.array([2., 1.])
    nose.tools.assert_raises(ValueError, metric, onsets, onsets)

    # Valid onsets which are the same produce a score of 1 for all metrics
    onsets = np.arange(10, dtype=np.float)
    assert np.allclose(metric(onsets, onsets), 1)


def __regression_test_onset_function(metric, reference_file, estimated_file, score):
    # Load in an example onset annotation
    reference_onsets, _ = mir_eval.io.load_events(reference_file)
    # Load in an example onset tracker output
    estimated_onsets, _ = mir_eval.io.load_events(estimated_file)
    # Ensure that the score is correct
    assert np.allclose(metric(reference_onsets, estimated_onsets), score, atol=A_TOL)


def test_onset_functions():
    # Load in all files in the same order
    ref_files = sorted(glob.glob(REF_GLOB))
    est_files = sorted(glob.glob(EST_GLOB))
    sco_files = sorted(glob.glob(SCORES_GLOB))

    # Unit tests
    for metric in mir_eval.onset.METRICS.values():
        yield (__unit_test_onset_function, metric)
    # Regression tests
    for ref_f, est_f, sco_f in zip(ref_files, est_files, sco_files):
        with open(sco_f, 'r') as f:
            scores = json.load(f)
        for name, metric in mir_eval.onset.METRICS.items():
            yield (__regression_test_onset_function, metric,
                   ref_f, est_f, scores[name])
