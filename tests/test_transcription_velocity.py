import mir_eval
import numpy as np
import glob
import json
from nose.tools import raises

A_TOL = 1e-12

# Path to the fixture files
REF_GLOB = 'tests/data/transcription_velocity/ref*.txt'
EST_GLOB = 'tests/data/transcription_velocity/est*.txt'
SCORES_GLOB = 'tests/data/transcription_velocity/output*.json'


def test_negative_velocity():

    good_i, good_p, good_v = np.array([[0, 1]]), np.array([100]), np.array([1])
    bad_i, bad_p, bad_v = np.array([[0, 1]]), np.array([100]), np.array([-1])

    yield (raises(ValueError)(mir_eval.transcription_velocity.validate),
           bad_i, bad_p, bad_v, good_i, good_p, good_v)
    yield (raises(ValueError)(mir_eval.transcription_velocity.validate),
           good_i, good_p, good_v, bad_i, bad_p, bad_v)


def test_wrong_shape_velocity():

    good_i, good_p, good_v = np.array([[0, 1]]), np.array([100]), np.array([1])
    bad_i, bad_p, bad_v = np.array([[0, 1]]), np.array([100]), np.array([1, 2])

    yield (raises(ValueError)(mir_eval.transcription_velocity.validate),
           bad_i, bad_p, bad_v, good_i, good_p, good_v)
    yield (raises(ValueError)(mir_eval.transcription_velocity.validate),
           good_i, good_p, good_v, bad_i, bad_p, bad_v)


def test_precision_recall_f1_overlap():
    # Simple unit test
    ref_i = np.array([[0, 1], [.5, .7], [1, 2]])
    ref_p = np.array([100, 110, 80])
    ref_v = np.array([10, 90, 110])
    est_i = np.array([[0, 1], [.5, .7], [1, 2]])
    est_p = np.array([100, 110, 80])
    est_v = np.array([10, 70, 110])
    p, r, f, o = mir_eval.transcription_velocity.precision_recall_f1_overlap(
        ref_i, ref_p, ref_v, est_i, est_p, est_v)
    assert np.allclose((p, r, f, o), (2/3., 2/3., 2/3., 1.))
    p, r, f, o = mir_eval.transcription_velocity.precision_recall_f1_overlap(
        ref_i, ref_p, ref_v, est_i, est_p, est_v, velocity_tolerance=0.3)
    assert np.allclose((p, r, f, o), (1., 1., 1., 1.))


def test_precision_recall_f1_overlap_empty():
    good_i, good_p, good_v = np.array([[0, 1]]), np.array([100]), np.array([1])
    bad_i, bad_p, bad_v = np.empty((0, 2)), np.array([]), np.array([])
    p, r, f, o = mir_eval.transcription_velocity.precision_recall_f1_overlap(
        good_i, good_p, good_v, bad_i, bad_p, bad_v)
    assert (p, r, f, o) == (0., 0., 0., 0.)
    p, r, f, o = mir_eval.transcription_velocity.precision_recall_f1_overlap(
        bad_i, bad_p, bad_v, good_i, good_p, good_v)
    assert (p, r, f, o) == (0., 0., 0., 0.)


def test_precision_recall_f1_overlap_no_overlap():
    p, r, f, o = mir_eval.transcription_velocity.precision_recall_f1_overlap(
        np.array([[1, 2]]), np.array([1]), np.array([1]),
        np.array([[3, 4]]), np.array([1]), np.array([1]))
    assert (p, r, f, o) == (0., 0., 0., 0.)


def __check_score(score, expected_score):
    assert np.allclose(score, expected_score, atol=A_TOL)


def test_regression():

    def _load_transcription_velocity(filename):
        """Loader for data in the format start, end, pitch, velocity."""
        starts, ends, pitches, velocities = mir_eval.io.load_delimited(
            filename, [float, float, int, int])
        # Stack into an interval matrix
        intervals = np.array([starts, ends]).T
        # return pitches and velocities as np.ndarray
        pitches = np.array(pitches)
        velocities = np.array(velocities)
        return intervals, pitches, velocities

    # Regression tests
    ref_files = sorted(glob.glob(REF_GLOB))
    est_files = sorted(glob.glob(EST_GLOB))
    sco_files = sorted(glob.glob(SCORES_GLOB))

    for ref_f, est_f, sco_f in zip(ref_files, est_files, sco_files):
        with open(sco_f, 'r') as f:
            expected_scores = json.load(f)
        # Load in reference transcription
        ref_int, ref_pitch, ref_vel = _load_transcription_velocity(ref_f)
        # Load in estimated transcription
        est_int, est_pitch, est_vel = _load_transcription_velocity(est_f)
        scores = mir_eval.transcription_velocity.evaluate(
            ref_int, ref_pitch, ref_vel, est_int, est_pitch, est_vel)
        for metric in scores:
            # This is a simple hack to make nosetest's messages more useful
            yield (__check_score, scores[metric], expected_scores[metric])
