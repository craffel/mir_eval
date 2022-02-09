"""
Unit tests for mir_eval.alignment
"""

import glob
import json

import nose.tools
import numpy as np

import mir_eval

A_TOL = 1e-12

# Path to the fixture files
REF_GLOB = "data/alignment/ref*.txt"
EST_GLOB = "data/alignment/est*.txt"
SCORES_GLOB = "data/alignment/output*.json"


def __unit_test_alignment_function(metric):
    # Now test validation function
    # alignments must be 1d ndarray
    alignments = np.array([[1.0, 2.0]])
    nose.tools.assert_raises(ValueError, metric, alignments, alignments)
    # alignments must be in seconds, and therefore not negative
    alignments = np.array([-1.0, 2.0])
    nose.tools.assert_raises(ValueError, metric, alignments, alignments)
    # alignments must be sorted
    alignments = np.array([2.0, 1.0])
    nose.tools.assert_raises(ValueError, metric, alignments, alignments)
    # predicted and estimated alignments must have same length
    pred_alignments = np.array([[1.0, 2.0]])
    est_alignments = np.array([[1.0]])
    nose.tools.assert_raises(
        ValueError, metric, est_alignments, pred_alignments
    )


def __check_score(sco_f, metric, score, expected_score):
    assert np.allclose(score, expected_score, atol=A_TOL)


def test_alignment_functions():
    # Load in all files in the same order
    ref_files = sorted(glob.glob(REF_GLOB))
    est_files = sorted(glob.glob(EST_GLOB))
    sco_files = sorted(glob.glob(SCORES_GLOB))

    assert len(ref_files) == len(est_files) == len(sco_files) > 0

    # Unit tests
    for metric in [mir_eval.alignment.ae, mir_eval.alignment.pc]:
        yield (__unit_test_alignment_function, metric)
    # Regression tests
    for ref_f, est_f, sco_f in zip(ref_files, est_files, sco_files):
        with open(sco_f, "r") as f:
            expected_scores = json.load(f)
        # Load in an example alignment annotation
        reference_alignments = mir_eval.io.load_events(ref_f)
        # Load in an example alignment tracker output
        estimated_alignments = mir_eval.io.load_events(est_f)
        # Compute scores
        # Setup some total duration
        if ref_f.__contains__("_mirex"):
            # MIREX code test case has this specific duration and computes PCS based on token
            # segments
            duration = 11.911836
        else:
            duration = (
                max(np.max(reference_alignments), np.max(estimated_alignments))
                + 10
            )
        scores = mir_eval.alignment.evaluate(
            reference_alignments, estimated_alignments, duration
        )
        # Compare them
        for metric in scores:
            # This is a simple hack to make nosetest's messages more useful
            yield (
                __check_score,
                sco_f,
                metric,
                scores[metric],
                expected_scores[metric],
            )
