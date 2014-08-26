# CREATED: 4/15/14 9:42 AM by Justin Salamon <justin.salamon@nyu.edu>
'''
Unit tests for mir_eval.melody
'''

import numpy as np
import json
import nose.tools
import mir_eval
import glob
import warnings

A_TOL = 1e-12

# Path to the fixture files
REF_GLOB = 'data/melody/ref*.txt'
EST_GLOB = 'data/melody/est*.txt'
SCORES_GLOB = 'data/melody/output*.json'


def test_hz2cents():
    # Unit test some simple values
    hz = np.array([0., 10., 5., 320., 1420.31238974231])
    # Expected cent conversion
    expected_cent = np.array([0., 0., -1200., 6000., 8580.0773605])
    assert np.allclose(mir_eval.melody.hz2cents(hz), expected_cent)


def test_freq_to_voicing():
    # Unit test some simple values
    hz = np.array([0., 100., -132.])
    expected_hz = np.array([0., 100., 132.])
    expected_voicing = np.array([0, 1, 0])
    # Check voicing conversion
    res_hz, res_voicing = mir_eval.melody.freq_to_voicing(hz)
    assert np.all(res_hz == expected_hz)
    assert np.all(res_voicing == expected_voicing)


def test_constant_hop_timebase():
    hop = .1
    end_time = .35
    expected_times = np.array([0, .1, .2, .3])
    res_times = mir_eval.melody.constant_hop_timebase(hop, end_time)
    assert np.allclose(res_times, expected_times)


def test_resample_melody_series():
    # Check for a small example including a zero transition
    times = np.arange(4)/35.0
    cents = np.array([2., 0., -1., 1.])
    voicing = np.array([1, 0, 1, 1])
    times_new = np.linspace(0, .08, 9)
    expected_cents = np.array([2., 2., 2., 0., 0., 0., -.8, -.1, .6])
    expected_voicing = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1])
    (res_cents,
     res_voicing) = mir_eval.melody.resample_melody_series(times, cents,
                                                           voicing, times_new)
    assert np.allclose(res_cents, expected_cents)
    assert np.allclose(res_voicing, expected_voicing)


def test_to_cent_voicing():
    # TODO: Write a simple test for this.  May require pre-computed data.
    pass


def __unit_test_voicing_measures(metric):
    # We need a special test for voicing_measures because it only takes 2 args
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        # First, test for warnings due to empty voicing arrays
        score = metric(np.array([]), np.array([]))
        assert len(w) == 4
        assert np.all([issubclass(wrn.category, UserWarning) for wrn in w])
        assert [str(wrn.message)
                for wrn in w] == ["Reference voicing array is empty.",
                                  "Estimated voicing array is empty.",
                                  "Reference melody has no voiced frames.",
                                  "Estimated melody has no voiced frames."]
        # And that the metric is 0
        assert np.allclose(score, 0)
        # Also test for a warning when the arrays have non-voiced content
        metric(np.ones(10), np.zeros(10))
        assert len(w) == 5
        assert issubclass(w[-1].category, UserWarning)
        assert str(w[-1].message) == "Estimated melody has no voiced frames."

    # Now test validation function - voicing arrays must be the same size
    nose.tools.assert_raises(ValueError, metric, np.ones(10), np.ones(12))
    # Voicing arrays must be bool
    nose.tools.assert_raises(ValueError, metric, np.arange(10), np.ones(10))


def __unit_test_melody_function(metric):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        # First, test for warnings due to empty voicing arrays
        score = metric(np.array([]), np.array([]), np.array([]), np.array([]))
        assert len(w) == 6
        assert np.all([issubclass(wrn.category, UserWarning) for wrn in w])
        assert [str(wrn.message)
                for wrn in w] == ["Reference voicing array is empty.",
                                  "Estimated voicing array is empty.",
                                  "Reference melody has no voiced frames.",
                                  "Estimated melody has no voiced frames.",
                                  "Reference frequency array is empty.",
                                  "Estimated frequency array is empty."]
        # And that the metric is 0
        assert np.allclose(score, 0)
        # Also test for a warning when the arrays have non-voiced content
        metric(np.ones(10), np.arange(10), np.zeros(10), np.arange(10))
        assert len(w) == 7
        assert issubclass(w[-1].category, UserWarning)
        assert str(w[-1].message) == "Estimated melody has no voiced frames."

    # Now test validation function - all inputs must be same length
    nose.tools.assert_raises(ValueError, metric, np.ones(10),
                             np.ones(12), np.ones(10), np.ones(10))
    # Voicing arrays must be bool
    nose.tools.assert_raises(ValueError, metric, np.arange(10),
                             np.arange(10), np.ones(10), np.arange(10))


def __regression_test_voicing_measures(metric, reference_file, estimated_file,
                                       score):
    # Need a separate function because the call structure is different
    # Load in reference melody
    ref_time, ref_freq = mir_eval.io.load_time_series(reference_file)
    # Load in estimated melody
    est_time, est_freq = mir_eval.io.load_time_series(estimated_file)
    # Convert to voicing/cent arrays
    (ref_v, ref_c,
     est_v, est_c) = mir_eval.melody.to_cent_voicing(ref_time, ref_freq,
                                                     est_time, est_freq)
    # Ensure that the score is correct
    assert np.allclose(metric(ref_v, est_v), score, atol=A_TOL)


def __regression_test_melody_function(metric, reference_file, estimated_file,
                                      score):
    # Load in reference melody
    ref_time, ref_freq = mir_eval.io.load_time_series(reference_file)
    # Load in estimated melody
    est_time, est_freq = mir_eval.io.load_time_series(estimated_file)
    # Convert to voicing/cent arrays
    (ref_v, ref_c,
     est_v, est_c) = mir_eval.melody.to_cent_voicing(ref_time, ref_freq,
                                                     est_time, est_freq)
    # Ensure that the score is correct
    assert np.allclose(metric(ref_v, ref_c, est_v, est_c), score, atol=A_TOL)


def test_melody_functions():
    # Load in all files in the same order
    ref_files = sorted(glob.glob(REF_GLOB))
    est_files = sorted(glob.glob(EST_GLOB))
    sco_files = sorted(glob.glob(SCORES_GLOB))

    # Unit tests
    for metric in mir_eval.melody.METRICS.values():
        if metric == mir_eval.melody.voicing_measures:
            yield (__unit_test_voicing_measures, metric)
        else:
            yield (__unit_test_melody_function, metric)
    # Regression tests
    for ref_f, est_f, sco_f in zip(ref_files, est_files, sco_files):
        with open(sco_f, 'r') as f:
            scores = json.load(f)
        for name, metric in mir_eval.melody.METRICS.items():
            if metric == mir_eval.melody.voicing_measures:
                yield (__regression_test_voicing_measures, metric,
                       ref_f, est_f, scores[name])
            else:
                yield (__regression_test_melody_function, metric,
                       ref_f, est_f, scores[name])
