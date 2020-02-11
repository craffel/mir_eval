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


def test_cents2hz():
    # Unit test some simple values
    expected_hz = np.array([0., 5., 320., 1420.31238974231])
    cent = np.array([0., -1200., 6000., 8580.0773605])
    print(mir_eval.melody.cents2hz(cent))
    print(expected_hz)
    assert np.allclose(mir_eval.melody.cents2hz(cent), expected_hz)


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
    cents = np.array([220., 0., -100., 100.])
    voicing = np.array([1, 0, 1, 1])
    times_new = np.linspace(0, .08, 9)
    expected_cents = np.array([220., 220., 220., 0., 0., 0., 100., 100., 100.])
    expected_voicing = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1])
    (res_cents,
     res_voicing) = mir_eval.melody.resample_melody_series(times, cents,
                                                           voicing, times_new)
    print(res_cents)
    print(expected_cents)
    assert np.allclose(res_cents, expected_cents)
    assert np.allclose(res_voicing, expected_voicing)


def test_resample_melody_series_same_times():
    # Check the case where the time bases are identical
    times = np.array([0.0, 0.1, 0.2, 0.3])
    times_new = np.array([0.0, 0.1, 0.2, 0.3])
    cents = np.array([2., 0., -1., 1.])
    voicing = np.array([0, 0, 1, 1])
    expected_cents = np.array([2., 0., -1., 1.])
    expected_voicing = np.array([False, False, True, True])
    (res_cents,
     res_voicing) = mir_eval.melody.resample_melody_series(times, cents,
                                                           voicing, times_new)
    assert np.allclose(res_cents, expected_cents)
    assert np.allclose(res_voicing, expected_voicing)


def test_normalize_inputs():
    # We'll just test a few values from one of the test annotations
    ref_file = sorted(glob.glob(REF_GLOB))[0]
    ref_time, ref_freq = mir_eval.io.load_time_series(ref_file)
    est_file = sorted(glob.glob(EST_GLOB))[0]
    est_time, est_freq = mir_eval.io.load_time_series(est_file)
    ref_v, ref_c, est_v, est_c = mir_eval.melody.normalize_inputs(ref_time,
                                                                 ref_freq,
                                                                 est_time,
                                                                 est_freq)
    # Expected values
    test_range = np.arange(220, 225)
    expected_ref_v = np.array([False, False, False, True, True])
    expected_ref_c = np.array([0., 0., 0., 330.689, 325.321])
    expected_est_v = np.array([False]*5)
    expected_est_c = np.array([220.]*5)
    assert np.allclose(ref_v[test_range], expected_ref_v)
    assert np.allclose(ref_c[test_range], expected_ref_c)
    assert np.allclose(est_v[test_range], expected_est_v)
    assert np.allclose(est_c[test_range], expected_est_c)

    # Test that a 0 is added to the beginning
    for return_item in mir_eval.melody.normalize_inputs(
            np.array([1., 2.]), np.array([440., 442.]), np.array([1., 2.]),
            np.array([441., 443.])):
        assert len(return_item) == 3
        assert return_item[0] == return_item[1]

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


def __check_score(sco_f, metric, score, expected_score):
    assert np.allclose(score, expected_score, atol=A_TOL)


def test_melody_functions():
    # Load in all files in the same order
    ref_files = sorted(glob.glob(REF_GLOB))
    est_files = sorted(glob.glob(EST_GLOB))
    sco_files = sorted(glob.glob(SCORES_GLOB))

    assert len(ref_files) == len(est_files) == len(sco_files) > 0

    # Unit tests
    for metric in [mir_eval.melody.voicing_recall,
                   mir_eval.melody.voicing_false_alarm,
                   mir_eval.melody.raw_pitch_accuracy,
                   mir_eval.melody.raw_chroma_accuracy,
                   mir_eval.melody.overall_accuracy]:
        if (metric == mir_eval.melody.voicing_recall or
                metric == mir_eval.melody.voicing_false_alarm):
            yield (__unit_test_voicing_measures, metric)
        else:
            yield (__unit_test_melody_function, metric)
    # Regression tests
    for ref_f, est_f, sco_f in zip(ref_files, est_files, sco_files):
        with open(sco_f, 'r') as f:
            expected_scores = json.load(f)
        # Load in reference melody
        ref_time, ref_freq = mir_eval.io.load_time_series(ref_f)
        # Load in estimated melody
        est_time, est_freq = mir_eval.io.load_time_series(est_f)
        scores = mir_eval.melody.evaluate(ref_time, ref_freq, est_time,
                                          est_freq)
        for metric in scores:
            # This is a simple hack to make nosetest's messages more useful
            yield (__check_score, sco_f, metric, scores[metric],
                   expected_scores[metric])
