'''
Tests for mir_eval.key
'''

import mir_eval
import nose.tools
import glob
import json
import numpy as np

A_TOL = 1e-12

# Path to the fixture files
REF_GLOB = 'data/key/ref*.txt'
EST_GLOB = 'data/key/est*.txt'
SCORES_GLOB = 'data/key/output*.json'


def __unit_test_key_function(metric):

    good_keys = ['C major', 'c major', 'C# major', 'Bb minor', 'db minor']
    # All of these are invalid key strings
    bad_keys = ['C maj', 'Cb major', 'C', 'K major', 'F## minor']

    for good_key in good_keys:
        for bad_key in bad_keys:
            # Should raise an error whether we pass a bad key as ref or est
            nose.tools.assert_raises(ValueError, metric, good_key, bad_key)
            nose.tools.assert_raises(ValueError, metric, bad_key, good_key)

    for good_key in good_keys:
        # When the same key is passed for est and ref, score should be 1
        assert mir_eval.key.weighted_score(good_key, good_key) == 1.


def __check_score(sco_f, metric, score, expected_score):
    assert np.allclose(score, expected_score, atol=A_TOL)


def test_key_functions():
    # Load in all files in the same order
    ref_files = sorted(glob.glob(REF_GLOB))
    est_files = sorted(glob.glob(EST_GLOB))
    sco_files = sorted(glob.glob(SCORES_GLOB))

    assert len(ref_files) == len(est_files) == len(sco_files) > 0

    # Unit test all metrics (one for now)
    for metric in [mir_eval.key.weighted_score]:
        yield __unit_test_key_function, metric

    # Regression tests
    for ref_f, est_f, sco_f in zip(ref_files, est_files, sco_files):
        with open(sco_f, 'r') as f:
            expected_scores = json.load(f)
        # Load in an example key annotation
        reference_key = mir_eval.key.load(ref_f)
        # Load in an example key detector output
        estimated_key = mir_eval.key.load(est_f)
        # Compute scores
        scores = mir_eval.key.evaluate(reference_key, estimated_key)
        # Compare them
        for metric in scores:
            # This is a simple hack to make nosetest's messages more useful
            yield (__check_score, sco_f, metric, scores[metric],
                   expected_scores[metric])
