"""
Unit tests for mir_eval.multipitch
"""

import numpy as np
import json
import mir_eval
import glob
import warnings
import nose.tools

A_TOL = 1e-12

# Path to the fixture files
REF_GLOB = 'tests/data/multipitch/ref*.txt'
EST_GLOB = 'tests/data/multipitch/est*.txt'
SCORES_GLOB = 'tests/data/multipitch/output*.json'


def __frequencies_equal(freqs_a, freqs_b):
    if len(freqs_a) != len(freqs_b):
        return False
    else:
        equal = True
        for freq_a, freq_b in zip(freqs_a, freqs_b):
            if freq_a.size != freq_b.size:
                return False
            equal = equal and np.allclose(freq_a, freq_b, atol=A_TOL)
        return equal


def __scores_equal(score_a, score_b):
    keys_a = set(list(score_a.keys()))
    keys_b = set(list(score_b.keys()))
    if keys_a != keys_b:
        return False
    else:
        equal = True
        for k in keys_a:
            value_a = score_a[k]
            value_b = score_b[k]
            equal = equal and np.allclose(value_a, value_b, atol=A_TOL)
    return equal


def test_resample_multif0():
    times = np.array([0.00, 0.01, 0.02, 0.03])
    empty_times = np.array([])
    freqs = [
        np.array([200.]),
        np.array([]),
        np.array([300., 400., 500.]),
        np.array([300., 500.])
    ]
    empty_freqs = []
    target_times1 = times
    target_times2 = np.array([0.001, 0.002, 0.01, 0.029, 0.05])
    target_times3 = empty_times

    expected_freqs1 = freqs
    expected_freqs2 = [
        np.array([200.]),
        np.array([200.]),
        np.array([]),
        np.array([300., 500.]),
        np.array([])
    ]
    expected_freqs3 = empty_freqs
    expected_freqs4 = [np.array([])]*4

    actual_freqs1 = mir_eval.multipitch.resample_multipitch(
        times, freqs, target_times1)
    actual_freqs2 = mir_eval.multipitch.resample_multipitch(
        times, freqs, target_times2)
    actual_freqs3 = mir_eval.multipitch.resample_multipitch(
        times, freqs, target_times3)
    actual_freqs4 = mir_eval.multipitch.resample_multipitch(
        empty_times, empty_freqs, target_times1)

    assert __frequencies_equal(actual_freqs1, expected_freqs1)
    assert __frequencies_equal(actual_freqs2, expected_freqs2)
    assert __frequencies_equal(actual_freqs3, expected_freqs3)
    assert __frequencies_equal(actual_freqs4, expected_freqs4)


def test_frequencies_to_logscale():
    frequencies = [
        np.array([256.]),
        np.array([]),
        np.array([362.03867196751236, 128., 512.]),
        np.array([300., 512.])
    ]
    expected = [
        np.array([96.]),
        np.array([]),
        np.array([102., 84., 108.]),
        np.array([98.745824285950576, 108.])
    ]
    actual = mir_eval.multipitch.frequencies_to_logscale(frequencies)
    assert __frequencies_equal(actual, expected)


def test_logscale_to_single_octave():
    logscale_frequencies = [
        np.array([96.]),
        np.array([]),
        np.array([102., 84., 108.]),
        np.array([98.745824285950576, 108.])
    ]
    expected = [
        np.array([0.]),
        np.array([]),
        np.array([6., 0., 0.]),
        np.array([2.745824285950576, 0.])
    ]
    actual = mir_eval.multipitch.logscale_to_single_octave(
        logscale_frequencies)
    assert __frequencies_equal(actual, expected)


def test_compute_num_freqs():
    frequencies = [
        np.array([256.]),
        np.array([]),
        np.array([362.03867196751236, 128., 512.]),
        np.array([300., 512.])
    ]
    expected = np.array([1, 0, 3, 2])
    actual = mir_eval.multipitch.compute_num_freqs(frequencies)
    assert np.allclose(actual, expected, atol=A_TOL)


def test_compute_num_true_positives():
    ref_freqs = [
        np.array([96., 100.]),
        np.array([]),
        np.array([81.]),
        np.array([102., 84., 108.]),
        np.array([98.745824285950576, 108.])
    ]
    est_freqs = [
        np.array([96.]),
        np.array([]),
        np.array([200., 82.]),
        np.array([102., 84., 108.]),
        np.array([99., 108.])
    ]
    expected = np.array([1, 0, 0, 3, 2])
    actual = mir_eval.multipitch.compute_num_true_positives(
        ref_freqs, est_freqs)
    assert np.allclose(actual, expected, atol=A_TOL)


def test_accuracy_metrics():
    true_positives = np.array([1, 0, 0, 3, 2])
    n_ref = np.array([2, 0, 1, 3, 2])
    n_est = np.array([1, 0, 2, 3, 2])

    expected_precision = 0.75
    expected_recall = 0.75
    expected_accuracy = 0.6

    (actual_precision,
     actual_recall,
     actual_accuarcy) = mir_eval.multipitch.accuracy(
         true_positives, n_ref, n_est)

    assert np.allclose(actual_precision, expected_precision, atol=A_TOL)
    assert np.allclose(actual_recall, expected_recall, atol=A_TOL)
    assert np.allclose(actual_accuarcy, expected_accuracy, atol=A_TOL)


def test_error_score_metrics():
    true_positives = np.array([1, 0, 0, 3, 2])
    n_ref = np.array([2, 0, 1, 3, 2])
    n_est = np.array([1, 0, 2, 3, 2])

    expected_esub = 0.125
    expected_emiss = 0.125
    expected_efa = 0.125
    expected_etot = 0.375

    (actual_esub,
     actual_emiss,
     actual_efa,
     actual_etot) = mir_eval.multipitch.error_score(
         true_positives, n_ref, n_est)

    assert np.allclose(actual_esub, expected_esub, atol=A_TOL)
    assert np.allclose(actual_emiss, expected_emiss, atol=A_TOL)
    assert np.allclose(actual_efa, expected_efa, atol=A_TOL)
    assert np.allclose(actual_etot, expected_etot, atol=A_TOL)


def unit_test_evaluate():
    empty_array = np.array([])
    ref_time = np.array([0.0, 0.1])
    ref_freqs = [np.array([201.]), np.array([])]
    est_time = np.array([0.0, 0.1])
    est_freqs = [np.array([200.]), np.array([])]

    # ref sizes unequal
    nose.tools.assert_raises(
        ValueError, mir_eval.multipitch.evaluate,
        np.array([0.0]), ref_freqs, est_time, est_freqs)

    # est sizes unequal
    nose.tools.assert_raises(
        ValueError, mir_eval.multipitch.evaluate,
        ref_time, ref_freqs, np.array([0.0]), est_freqs)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        # Test for warnings on empty values
        actual_score = mir_eval.multipitch.evaluate(
            ref_time, [empty_array, empty_array],
            est_time, [empty_array, empty_array])
        assert len(w) == 6
        assert issubclass(w[-1].category, UserWarning)
        assert str(w[-1].message) == "Reference frequencies are all empty."

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        # Test for warnings on empty values
        # test all inputs empty
        mir_eval.multipitch.evaluate(empty_array, [], empty_array, [])
        assert len(w) == 10
        assert issubclass(w[-1].category, UserWarning)
        assert str(w[-1].message) == "Reference frequencies are all empty."

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        # reference empty
        mir_eval.multipitch.evaluate(empty_array, [], est_time, est_freqs)
        assert len(w) == 9
        assert issubclass(w[-1].category, UserWarning)
        assert str(w[-1].message) == "Reference frequencies are all empty."

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        # estimate empty
        mir_eval.multipitch.evaluate(ref_time, ref_freqs, empty_array, [])
        assert len(w) == 5
        assert issubclass(w[-1].category, UserWarning)
        assert str(w[-1].message) == "Estimate frequencies are all empty."

        expected_score = {
            'Precision': 0.0,
            'Recall': 0.0,
            'Accuracy': 0.0,
            'Chroma Precision': 0.0,
            'Chroma Recall': 0.0,
            'Chroma Accuracy': 0.0,
            'Substitution Error': 0.0,
            'Miss Error': 0.0,
            'False Alarm Error': 0.0,
            'Total Error': 0.0,
            'Chroma Substitution Error': 0.0,
            'Chroma Miss Error': 0.0,
            'Chroma False Alarm Error': 0.0,
            'Chroma Total Error': 0.0
        }
        assert __scores_equal(actual_score, expected_score)

    # test perfect estimate
    ref_time = np.array([0.0, 0.1, 0.2])
    ref_freqs = [np.array([201.]), np.array([]), np.array([300.5, 87.1])]
    actual_score = mir_eval.multipitch.evaluate(
        ref_time, ref_freqs, ref_time, ref_freqs)
    expected_score = {
        'Precision': 1.0,
        'Recall': 1.0,
        'Accuracy': 1.0,
        'Chroma Precision': 1.0,
        'Chroma Recall': 1.0,
        'Chroma Accuracy': 1.0,
        'Substitution Error': 0.0,
        'Miss Error': 0.0,
        'False Alarm Error': 0.0,
        'Total Error': 0.0,
        'Chroma Substitution Error': 0.0,
        'Chroma Miss Error': 0.0,
        'Chroma False Alarm Error': 0.0,
        'Chroma Total Error': 0.0
    }
    assert __scores_equal(actual_score, expected_score)


def regression_test_evaluate():
    ref_files = sorted(glob.glob(REF_GLOB))
    est_files = sorted(glob.glob(EST_GLOB))
    score_files = sorted(glob.glob(SCORES_GLOB))

    assert len(ref_files) == len(est_files) == len(score_files) > 0

    for ref_f, est_f, score_f in zip(ref_files, est_files, score_files):
        with open(score_f, 'r') as f_handle:
            expected_score = json.load(f_handle)

        ref_times, ref_freqs = mir_eval.io.load_ragged_time_series(ref_f)
        est_times, est_freqs = mir_eval.io.load_ragged_time_series(est_f)

        actual_score = mir_eval.multipitch.evaluate(
            ref_times, ref_freqs, est_times, est_freqs)

        assert __scores_equal(actual_score, expected_score)
