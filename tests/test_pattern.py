"""
Some unit tests for the pattern discovery task.
"""

import numpy as np
import json
import mir_eval
import glob
import warnings
import nose.tools

A_TOL = 1e-12

# Path to the fixture files
REF_GLOB = 'tests/data/pattern/ref*.txt'
EST_GLOB = 'tests/data/pattern/est*.txt'
SCORES_GLOB = 'tests/data/pattern/output*.json'


def __unit_test_pattern_function(metric):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        # First, test for a warning on empty pattern
        metric([[[]]], [[[(100, 20)]]])
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert str(w[-1].message) == 'Reference patterns are empty.'
        metric([[[(100, 20)]]], [[[]]])
        assert len(w) == 2
        assert issubclass(w[-1].category, UserWarning)
        assert str(w[-1].message) == "Estimated patterns are empty."
        # And that the metric is 0
        assert np.allclose(metric([[[]]], [[[]]]), 0)

    # Now test validation function - patterns must contain at least 1 occ
    patterns = [[[(100, 20)]], []]
    nose.tools.assert_raises(ValueError, metric, patterns, patterns)
    # The (onset, midi) tuple must contain 2 elements
    patterns = [[[(100, 20, 3)]]]
    nose.tools.assert_raises(ValueError, metric, patterns, patterns)

    # Valid patterns which are the same produce a score of 1 for all metrics
    patterns = [[[(100, 20), (200, 30)]]]
    assert np.allclose(metric(patterns, patterns), 1)


def __check_score(sco_f, metric, score, expected_score):
    assert np.allclose(score, expected_score, atol=A_TOL)


def test_pattern_functions():
    # Load in all files in the same order
    ref_files = sorted(glob.glob(REF_GLOB))
    est_files = sorted(glob.glob(EST_GLOB))
    sco_files = sorted(glob.glob(SCORES_GLOB))

    assert len(ref_files) == len(est_files) == len(sco_files) > 0

    # Unit tests
    for metric in [mir_eval.pattern.standard_FPR,
                   mir_eval.pattern.establishment_FPR,
                   mir_eval.pattern.occurrence_FPR,
                   mir_eval.pattern.three_layer_FPR,
                   mir_eval.pattern.first_n_three_layer_P,
                   mir_eval.pattern.first_n_target_proportion_R]:
        yield (__unit_test_pattern_function, metric)
    # Regression tests
    for ref_f, est_f, sco_f in zip(ref_files, est_files, sco_files):
        with open(sco_f, 'r') as f:
            expected_scores = json.load(f)
        # Load in reference and estimated patterns
        reference_patterns = mir_eval.io.load_patterns(ref_f)
        estimated_patterns = mir_eval.io.load_patterns(est_f)
        # Compute scores
        scores = mir_eval.pattern.evaluate(reference_patterns,
                                           estimated_patterns)
        # Compare them
        for metric in scores:
            # This is a simple hack to make nosetest's messages more useful
            yield (__check_score, sco_f, metric, scores[metric],
                   expected_scores[metric])
