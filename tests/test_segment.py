'''
Unit tests for mir_eval.segment
'''

import numpy as np
import json
import mir_eval
import glob

A_TOL = 1e-12

# Path to the fixture files
REF_GLOB    = 'data/segment/ref*.lab'
EST_GLOB    = 'data/segment/est*.lab'
SCORES_GLOB  = 'data/segment/output*.json'


def __check_score(sco_f, metric, score, expected_score):
    assert np.allclose(score, expected_score, atol=A_TOL)


def test_segment_functions():
    # Load in all files in the same order
    ref_files = sorted(glob.glob(REF_GLOB))
    est_files = sorted(glob.glob(EST_GLOB))
    sco_files = sorted(glob.glob(SCORES_GLOB))

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
