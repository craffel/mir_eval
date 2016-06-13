'''
Unit tests for mir_eval.segment
'''

import numpy as np
import json
import mir_eval
import glob
import warnings
import nose.tools

A_TOL = 1e-12

# Path to the fixture files
REF_GLOB = 'tests/data/segment/ref*.lab'
EST_GLOB = 'tests/data/segment/est*.lab'
SCORES_GLOB = 'tests/data/segment/output*.json'


def __unit_test_boundary_function(metric):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        # Test for warning when empty intervals with no trimming
        metric(np.zeros((0, 2)), np.array([[1, 2], [2, 3]]), trim=False)
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert str(w[-1].message) == "Reference intervals are empty."
        # Now test when 1 interval with trimming
        metric(np.array([[1, 2], [2, 3]]), np.array([[1, 2]]), trim=True)
        assert len(w) == 2
        assert issubclass(w[-1].category, UserWarning)
        assert str(w[-1].message) == "Estimated intervals are empty."
        # Check for correct behavior in empty intervals
        empty_intervals = np.zeros((0, 2))
        if metric == mir_eval.segment.detection:
            assert np.allclose(metric(empty_intervals, empty_intervals), 0)
        else:
            assert np.all(np.isnan(metric(empty_intervals, empty_intervals)))

    # Now test validation function - intervals must be n by 2
    intervals = np.array([1, 2, 3, 4])
    nose.tools.assert_raises(ValueError, metric, intervals, intervals)
    # Interval boundaries must be positive
    intervals = np.array([[-1, 2], [2, 3]])
    nose.tools.assert_raises(ValueError, metric, intervals, intervals)
    # Positive interval durations
    intervals = np.array([[2, 1], [2, 3]])
    nose.tools.assert_raises(ValueError, metric, intervals, intervals)
    # Check for correct behavior when intervals are the same
    correct_intervals = np.array([[0, 1], [1, 2]])
    if metric == mir_eval.segment.detection:
        assert np.allclose(metric(correct_intervals, correct_intervals), 1)
    else:
        assert np.allclose(metric(correct_intervals, correct_intervals), 0)


def __unit_test_structure_function(metric):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        # Test for warning when empty intervals
        score = metric(np.zeros((0, 2)), [], np.zeros((0, 2)), [])
        assert len(w) == 2
        assert issubclass(w[0].category, UserWarning)
        assert issubclass(w[1].category, UserWarning)
        assert str(w[0].message) == "Reference intervals are empty."
        assert str(w[1].message) == "Estimated intervals are empty."
        # And that the metric is 0
        assert np.allclose(score, 0)

    # Now test validation function - intervals must be n by 2
    intervals = np.arange(4)
    labels = ['a', 'b', 'c', 'd']
    nose.tools.assert_raises(ValueError, metric, intervals, labels, intervals,
                             labels)
    # Interval boundaries must be positive
    intervals = np.array([[-1, 2], [2, 3]])
    nose.tools.assert_raises(ValueError, metric, intervals, labels, intervals,
                             labels)
    # Positive interval durations
    intervals = np.array([[2, 1], [2, 3]])
    labels = ['a', 'b']
    nose.tools.assert_raises(ValueError, metric, intervals, labels, intervals,
                             labels)
    # Number of intervals must match number of labels
    labels = ['a']
    nose.tools.assert_raises(ValueError, metric, intervals, labels, intervals,
                             labels)
    # Intervals must start at 0
    intervals = np.array([[1, 2], [2, 3]])
    labels = ['a', 'b']
    nose.tools.assert_raises(ValueError, metric, intervals, labels, intervals,
                             labels)
    # End times must match
    reference_intervals = np.array([[0, 1], [1, 2]])
    estimated_intervals = np.array([[0, 1], [1, 3]])
    nose.tools.assert_raises(ValueError, metric, reference_intervals, labels,
                             estimated_intervals, labels)
    # Check for correct output when input is the same
    estimated_intervals = reference_intervals
    if metric == mir_eval.segment.mutual_information:
        assert np.allclose(metric(reference_intervals, labels,
                                  estimated_intervals, labels),
                           [np.log(2), 1, 1])
    else:
        assert np.allclose(metric(reference_intervals, labels,
                                  estimated_intervals, labels), 1)


def __check_score(sco_f, metric, score, expected_score):
    assert np.allclose(score, expected_score, atol=A_TOL)


def __unit_test_permuted_segments(sco_f, ref_int, ref_lab,
                                  est_int, est_lab, scores):
    # Test for issue #202

    # Generate a random permutation of the reference segments
    idx = np.random.permutation(np.arange(len(ref_int)))

    perm_int = ref_int[idx]
    perm_lab = [ref_lab[_] for _ in idx]

    perm_scores = mir_eval.segment.evaluate(perm_int, perm_lab,
                                            est_int, est_lab)

    for metric in scores:
        __check_score(sco_f, metric, perm_scores[metric], scores[metric])


def test_segment_functions():
    # Load in all files in the same order
    ref_files = sorted(glob.glob(REF_GLOB))
    est_files = sorted(glob.glob(EST_GLOB))
    sco_files = sorted(glob.glob(SCORES_GLOB))

    assert len(ref_files) == len(est_files) == len(sco_files) > 0

    # Unit tests for boundary
    for metric in [mir_eval.segment.detection,
                   mir_eval.segment.deviation]:
        yield (__unit_test_boundary_function, metric)
    # And structure
    for metric in [mir_eval.segment.pairwise,
                   mir_eval.segment.rand_index,
                   mir_eval.segment.ari,
                   mir_eval.segment.mutual_information,
                   mir_eval.segment.nce]:
        yield (__unit_test_structure_function, metric)
    # Regression tests
    for ref_f, est_f, sco_f in zip(ref_files, est_files, sco_files):
        with open(sco_f, 'r') as f:
            expected_scores = json.load(f)
        # Load in an example segmentation annotation
        ref_intervals, ref_labels = mir_eval.io.load_labeled_intervals(ref_f)
        # Load in an example segmentation tracker output
        est_intervals, est_labels = mir_eval.io.load_labeled_intervals(est_f)

        # Compute scores
        scores = mir_eval.segment.evaluate(ref_intervals, ref_labels,
                                           est_intervals, est_labels)
        # Compare them
        for metric in scores:
            # This is a simple hack to make nosetest's messages more useful
            yield (__check_score, sco_f, metric, scores[metric],
                   expected_scores[metric])

        yield (__unit_test_permuted_segments, sco_f,
               ref_intervals, ref_labels,
               est_intervals, est_labels, scores)
