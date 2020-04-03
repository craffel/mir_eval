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

    # Unit test some simple values where voicing is given
    hz = np.array([0., 100., -132., 0, 131.])
    voicing = np.array([0.8, 0.0, 1.0, 0.0, 0.5])
    expected_hz = np.array([0., 100., 132., 0., 131.])
    expected_voicing = np.array([0.0, 0.0, 1.0, 0.0, 0.5])
    # Check voicing conversion
    res_hz, res_voicing = mir_eval.melody.freq_to_voicing(hz, voicing=voicing)
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

    # Check for a small example including a zero transition - nonbinary voicing
    times = np.arange(4)/35.0
    cents = np.array([2., 0., -1., 1.])
    voicing = np.array([0.8, 0.0, 0.2, 1.0])
    times_new = np.linspace(0, .08, 9)
    expected_cents = np.array([2., 2., 2., 0., 0., 0., -.8, -.1, .6])
    expected_voicing = np.array(
        [0.8, 0.52, 0.24, 0.01, 0.08, 0.15, 0.28, 0.56, 0.84]
    )
    (res_cents,
     res_voicing) = mir_eval.melody.resample_melody_series(times, cents,
                                                           voicing, times_new)
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

    # Check the case where the time bases are identical - nonbinary voicing
    times = np.array([0.0, 0.1, 0.2, 0.3])
    times_new = np.array([0.0, 0.1, 0.2, 0.3])
    cents = np.array([2., 0., -1., 1.])
    voicing = np.array([0.5, 0.8, 0.9, 1.0])
    expected_cents = np.array([2., 0., -1., 1.])
    expected_voicing = np.array([0.5, 0.8, 0.9, 1.0])
    (res_cents,
     res_voicing) = mir_eval.melody.resample_melody_series(times, cents,
                                                           voicing, times_new)
    assert np.allclose(res_cents, expected_cents)
    assert np.allclose(res_voicing, expected_voicing)


def test_to_cent_voicing():
    # We'll just test a few values from one of the test annotations
    ref_file = sorted(glob.glob(REF_GLOB))[0]
    ref_time, ref_freq = mir_eval.io.load_time_series(ref_file)
    est_file = sorted(glob.glob(EST_GLOB))[0]
    est_time, est_freq = mir_eval.io.load_time_series(est_file)
    ref_v, ref_c, est_v, est_c = mir_eval.melody.to_cent_voicing(ref_time,
                                                                 ref_freq,
                                                                 est_time,
                                                                 est_freq)
    # Expected values
    test_range = np.arange(220, 225)
    expected_ref_v = np.array([False, False, False, True, True])
    expected_ref_c = np.array([0., 0., 0., 6056.8837818916609,
                               6028.5504583021921])
    expected_est_v = np.array([False]*5)
    expected_est_c = np.array([5351.3179423647571]*5)
    assert np.allclose(ref_v[test_range], expected_ref_v)
    assert np.allclose(ref_c[test_range], expected_ref_c)
    assert np.allclose(est_v[test_range], expected_est_v)
    assert np.allclose(est_c[test_range], expected_est_c)

    # Test that a 0 is added to the beginning
    for return_item in mir_eval.melody.to_cent_voicing(
            np.array([1., 2.]), np.array([440., 442.]), np.array([1., 2.]),
            np.array([441., 443.])):
        assert len(return_item) == 3
        assert return_item[0] == return_item[1]

    # Test custom voicings
    ref_time, ref_freq = mir_eval.io.load_time_series(ref_file)
    _, ref_reward = mir_eval.io.load_time_series("data/melody/reward00.txt")
    _, est_voicing = mir_eval.io.load_time_series(
        "data/melody/voicingest00.txt"
    )
    (ref_v, ref_c,
     est_v, est_c) = mir_eval.melody.to_cent_voicing(ref_time,
                                                     ref_freq,
                                                     est_time,
                                                     est_freq,
                                                     est_voicing=est_voicing,
                                                     ref_reward=ref_reward)
    # Expected values
    test_range = np.arange(220, 225)
    expected_ref_v = np.array([0., 0., 0., 1., 0.3])
    expected_ref_c = np.array([0., 0., 0., 6056.8837818916609,
                               6028.5504583021921])
    expected_est_v = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    expected_est_c = np.array([5351.3179423647571]*5)
    assert np.allclose(ref_v[test_range], expected_ref_v)
    assert np.allclose(ref_c[test_range], expected_ref_c)
    assert np.allclose(est_v[test_range], expected_est_v)
    assert np.allclose(est_c[test_range], expected_est_c)


def test_continuous_voicing_metrics():
    ref_time = np.array([0.0, 0.1, 0.2, 0.3])
    ref_freq = np.array([440.0, 0.0, 220.0, 220.0])

    est_time = np.array([0.0, 0.1, 0.2, 0.3])
    est_freq = np.array([440.1, 330.0, 440.0, 330.0])

    # test different estimate voicings
    all_est_voicing = [
        np.array([1.0, 0.0, 1.0, 1.0]),  # perfect
        np.array([0.0, 1.0, 0.0, 0.0]),  # all wrong
        np.array([0.5, 0.5, 0.5, 0.5]),  # all 0.5
        np.array([0.8, 0.2, 0.8, 0.8]),  # almost right
        np.array([0.2, 0.8, 0.2, 0.2]),  # almost wrong
    ]

    all_expected = [
        # perfect
        {
            'Voicing Recall': 1.0,
            'Voicing False Alarm': 0.0,
            'Raw Pitch Accuracy': 1. / 3.,
            'Raw Chroma Accuracy': 2. / 3.,
            'Overall Accuracy': 0.5,
        },
        # all wrong
        {
            'Voicing Recall': 0.0,
            'Voicing False Alarm': 1.0,
            'Raw Pitch Accuracy': 1. / 3.,
            'Raw Chroma Accuracy': 2. / 3.,
            'Overall Accuracy': 0.0,
        },
        # all 0.5
        {
            'Voicing Recall': 0.5,
            'Voicing False Alarm': 0.5,
            'Raw Pitch Accuracy': 1. / 3.,
            'Raw Chroma Accuracy': 2. / 3.,
            'Overall Accuracy': 0.25,
        },
        # almost right
        {
            'Voicing Recall': 0.8,
            'Voicing False Alarm': 0.2,
            'Raw Pitch Accuracy': 1. / 3.,
            'Raw Chroma Accuracy': 2. / 3.,
            'Overall Accuracy': 0.4,
        },
        # almost wrong
        {
            'Voicing Recall': 0.2,
            'Voicing False Alarm': 0.8,
            'Raw Pitch Accuracy': 1. / 3.,
            'Raw Chroma Accuracy': 2. / 3.,
            'Overall Accuracy': 0.1,
        },
    ]

    for est_voicing, expected_scores in zip(all_est_voicing, all_expected):
        actual_scores = mir_eval.melody.evaluate(ref_time, ref_freq, est_time,
                                                 est_freq,
                                                 est_voicing=est_voicing)
        for metric in actual_scores:
            assert np.isclose(actual_scores[metric], expected_scores[metric])

    # test different rewards
    all_rewards = [
        np.array([0.5, 0.5, 0.5, 0.5]),  # uniform
        np.array([0.3, 0.3, 0.3, 0.3]),  # uniform - different number
        np.array([0.0, 0.0, 0.0, 0.0]),  # all zero
        np.array([1.0, 0.0, 0.0, 0.0]),  # one weight
        np.array([1.0, 0.0, 1.0, 0.0]),  # two weights
        np.array([1.0, 0.0, 0.5, 0.5]),  # slightly generous
        np.array([0.1, 0.0, 0.1, 0.8]),  # big penalty
    ]
    est_voicing = np.array([1.0, 0.0, 1.0, 1.0])

    all_expected = [
        # uniform
        {
            'Voicing Recall': 1.0,
            'Voicing False Alarm': 0.0,
            'Raw Pitch Accuracy': 1. / 3.,
            'Raw Chroma Accuracy': 2. / 3.,
            'Overall Accuracy': 0.5,
        },
        # uniform - different number
        {
            'Voicing Recall': 1.0,
            'Voicing False Alarm': 0.0,
            'Raw Pitch Accuracy': 1. / 3.,
            'Raw Chroma Accuracy': 2. / 3.,
            'Overall Accuracy': 0.5,
        },
        # all zero
        {
            'Voicing Recall': 1.0,
            'Voicing False Alarm': 0.75,
            'Raw Pitch Accuracy': 0.0,
            'Raw Chroma Accuracy': 0.0,
            'Overall Accuracy': 0.25,
        },
        # one weight
        {
            'Voicing Recall': 1.0,
            'Voicing False Alarm': 2. / 3.,
            'Raw Pitch Accuracy': 1.0,
            'Raw Chroma Accuracy': 1.0,
            'Overall Accuracy': 0.5,
        },
        # two weights
        {
            'Voicing Recall': 1.0,
            'Voicing False Alarm': 0.5,
            'Raw Pitch Accuracy': 0.5,
            'Raw Chroma Accuracy': 1.0,
            'Overall Accuracy': 0.5,
        },
        # slightly generous
        {
            'Voicing Recall': 1.0,
            'Voicing False Alarm': 0.0,
            'Raw Pitch Accuracy': 0.5,
            'Raw Chroma Accuracy': 0.75,
            'Overall Accuracy': 0.625,
        },
        # big penalty
        {
            'Voicing Recall': 1.0,
            'Voicing False Alarm': 0.0,
            'Raw Pitch Accuracy': 0.1,
            'Raw Chroma Accuracy': 0.2,
            'Overall Accuracy': 0.325,
        },
    ]

    for ref_reward, expected_scores in zip(all_rewards, all_expected):
        actual_scores = mir_eval.melody.evaluate(ref_time, ref_freq, est_time,
                                                 est_freq,
                                                 est_voicing=est_voicing,
                                                 ref_reward=ref_reward)
        for metric in actual_scores:
            assert np.isclose(actual_scores[metric], expected_scores[metric])


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
    for metric in [mir_eval.melody.voicing_measures,
                   mir_eval.melody.raw_pitch_accuracy,
                   mir_eval.melody.raw_chroma_accuracy,
                   mir_eval.melody.overall_accuracy]:
        if metric == mir_eval.melody.voicing_measures:
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


def test_melody_functions_continuous_voicing_equivalence():
    # Load in all files in the same order
    ref_files = sorted(glob.glob(REF_GLOB))
    est_files = sorted(glob.glob(EST_GLOB))
    sco_files = sorted(glob.glob(SCORES_GLOB))

    assert len(ref_files) == len(est_files) == len(sco_files) > 0

    # Unit tests
    for metric in [mir_eval.melody.voicing_measures,
                   mir_eval.melody.raw_pitch_accuracy,
                   mir_eval.melody.raw_chroma_accuracy,
                   mir_eval.melody.overall_accuracy]:
        if metric == mir_eval.melody.voicing_measures:
            yield (__unit_test_voicing_measures, metric)
        else:
            yield (__unit_test_melody_function, metric)
    # Regression tests
    for ref_f, est_f, sco_f in zip(ref_files, est_files, sco_files):
        with open(sco_f, 'r') as f:
            expected_scores = json.load(f)
        # Load in reference melody
        ref_time, ref_freq = mir_eval.io.load_time_series(ref_f)
        ref_reward = np.ones(ref_time.shape)  # uniform reward
        # Load in estimated melody
        est_time, est_freq = mir_eval.io.load_time_series(est_f)
        # voicing equivalent from frequency
        est_voicing = (est_freq >= 0).astype('float')
        scores = mir_eval.melody.evaluate(ref_time, ref_freq, est_time,
                                          est_freq, est_voicing=est_voicing,
                                          ref_reward=ref_reward)
        for metric in scores:
            # This is a simple hack to make nosetest's messages more useful
            yield (__check_score, sco_f, metric, scores[metric],
                   expected_scores[metric])
