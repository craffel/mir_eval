"""
Unit tests for mir_eval.chord
"""

import mir_eval
import numpy as np
import nose.tools
import warnings
import glob
import json

A_TOL = 1e-12

# Path to the fixture files
REF_GLOB = 'data/chord/ref*.lab'
EST_GLOB = 'data/chord/est*.lab'
SCORES_GLOB = 'data/chord/output*.json'


def __check_valid(function, parameters, result):
    ''' Helper function for checking the output of a function '''
    assert function(*parameters) == result


def __check_exception(function, parameters, exception):
    ''' Makes sure the provided function throws the provided
    exception given the provided input '''
    nose.tools.assert_raises(exception, function, *parameters)


def test_pitch_class_to_semitone():
    valid_classes = ['Gbb', 'G', 'G#', 'Cb', 'B#']
    valid_semitones = [5, 7, 8, 11, 0]

    for pitch_class, semitone in zip(valid_classes, valid_semitones):
        yield (__check_valid, mir_eval.chord.pitch_class_to_semitone,
               (pitch_class,), semitone)

    invalid_classes = ['Cab', '#C', 'bG']

    for pitch_class in invalid_classes:
        yield (__check_exception, mir_eval.chord.pitch_class_to_semitone,
               (pitch_class,), mir_eval.chord.InvalidChordException)


def test_scale_degree_to_semitone():
    valid_degrees = ['b7', '#3', '1', 'b1', '#7', 'bb5']
    valid_semitones = [10, 5, 0, -1, 12, 5]

    for scale_degree, semitone in zip(valid_degrees, valid_semitones):
        yield (__check_valid, mir_eval.chord.scale_degree_to_semitone,
               (scale_degree,), semitone)

    invalid_degrees = ['7b', '4#', '77']

    for scale_degree in invalid_degrees:
        yield (__check_exception, mir_eval.chord.scale_degree_to_semitone,
               (scale_degree,), mir_eval.chord.InvalidChordException)


def test_validate_chord_label():
    valid_labels = ['C', 'Eb:min/5', 'A#:dim7', 'B:maj(*1,*5)/3', 'A#:sus4']
    # For valid labels, calling the function without an error = pass
    for chord_label in valid_labels:
        yield (mir_eval.chord.validate_chord_label, chord_label)

    invalid_labels = ["C::maj", "C//5", "C((4)", "C5))",
                      "C:maj(*3/3", "Cmaj*3/3)"]

    for chord_label in invalid_labels:
        yield (__check_exception, mir_eval.chord.validate_chord_label,
               (chord_label,), mir_eval.chord.InvalidChordException)


def test_split():
    labels = ['C', 'B:maj(*1,*3)/5', 'Ab:min/b3', 'N', 'G:(3)']
    splits = [['C', 'maj', set(), '1'],
              ['B', 'maj', set(['*1', '*3']), '5'],
              ['Ab', 'min', set(), 'b3'],
              ['N', '', set(), ''],
              ['G', '', set(['3']), '1']]

    for chord_label, split_chord in zip(labels, splits):
        yield (__check_valid, mir_eval.chord.split,
               (chord_label,), split_chord)


def test_join():
    # Arguments are root, quality, extensions, bass
    splits = [('F#', '', None, ''),
              ('F#', 'hdim7', None, ''),
              ('F#', '', {'*b3', '4'}, ''),
              ('F#', '', None, 'b7'),
              ('F#', '', {'*b3', '4'}, 'b7'),
              ('F#', 'hdim7', None, 'b7'),
              ('F#', 'hdim7', {'*b3', '4'}, 'b7')]
    labels = ['F#', 'F#:hdim7', 'F#:(*b3,4)', 'F#/b7',
              'F#:(*b3,4)/b7', 'F#:hdim7/b7', 'F#:hdim7(*b3,4)/b7']

    for split_chord, chord_label in zip(splits, labels):
        yield (__check_valid, mir_eval.chord.join,
               split_chord, chord_label)


def test_rotate_bitmaps_to_roots():
    def __check_bitmaps(bitmaps, roots, expected_bitmaps):
        ''' Helper function for checking bitmaps_to_roots '''
        ans = mir_eval.chord.rotate_bitmaps_to_roots(bitmaps, roots)
        assert np.all(ans == expected_bitmaps)

    bitmaps = [
        [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]]
    roots = [0, 5, 11]
    expected_bitmaps = [
        [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]]

    # The function can operate on many bitmaps/roots at a time
    # but we should only test them one at a time.
    for bitmap, root, expected_bitmap in zip(bitmaps, roots, expected_bitmaps):
        yield (__check_bitmaps, [bitmap], [root], [expected_bitmap])


def test_encode():
    def __check_encode(label, expected_root, expected_intervals,
                       expected_bass):
        ''' Helper function for checking encode '''
        root, intervals, bass = mir_eval.chord.encode(label)
        assert root == expected_root
        assert np.all(intervals == expected_intervals)
        assert bass == expected_bass

    labels = ['B:maj(*1,*3)/5', 'G:dim', 'C:(3)/3']
    expected_roots = [11, 7, 0]
    expected_intervals = [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                          [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]
    expected_bass = [7, 0, 4]

    for label, e_root, e_interval, e_bass in zip(labels,
                                                 expected_roots,
                                                 expected_intervals,
                                                 expected_bass):
        yield (__check_encode, label, e_root, e_interval, e_bass)

    # Non-chord bass notes *must* be explicitly named as extensions when
    #   STRICT_BASS_INTERVALS == True
    mir_eval.chord.STRICT_BASS_INTERVALS = True
    yield (__check_exception, mir_eval.chord.encode,
           ('G:dim(4)/6',), mir_eval.chord.InvalidChordException)
    # Otherwise, we can cut a little slack.
    mir_eval.chord.STRICT_BASS_INTERVALS = False
    yield (__check_encode, 'G:dim(4)/6', 7,
                           [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0], 9)


def test_encode_many():
    def __check_encode_many(labels, expected_roots, expected_intervals,
                            expected_basses):
        ''' Does all of the logic for checking encode_many '''
        roots, intervals, basses = mir_eval.chord.encode_many(labels)
        assert np.all(roots == expected_roots)
        assert np.all(intervals == expected_intervals)
        assert np.all(basses == expected_basses)

    labels = ['B:maj(*1,*3)/5',
              'B:maj(*1,*3)/5',
              'N',
              'C:min',
              'C:min']
    expected_roots = [11, 11, -1, 0, 0]
    expected_intervals = [
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]]
    expected_basses = [7, 7, -1, 0, 0]

    yield (__check_encode_many, labels, expected_roots, expected_intervals,
           expected_basses)


def __check_one_metric(metric, ref_label, est_label, score):
    ''' Checks that a metric function produces score given ref_label and
    est_label '''
    # We provide a dummy interval.  We're just checking one pair
    # of labels at a time.
    assert metric([ref_label], [est_label], np.array([[0, 1]])) == score


def __check_not_comparable(metric, ref_label, est_label):
    ''' Checks that ref_label is not comparable to est_label by metric '''
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        # Try to produce the warning
        score = metric([ref_label], [est_label], np.array([[0, 1]]))
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert str(w[-1].message) == ("No reference chords were comparable "
                                      "to estimated chords, returning 0.")
        # And confirm that the metric is 0
        assert np.allclose(score, 0)


def __regression_test_one_file(metric, reference_file, estimated_file, score):
    # Load in an example chord
    ref_intervals, ref_labels = mir_eval.io.load_intervals(reference_file)
    # Load in an example estimated chord labeling
    est_intervals, est_labels = mir_eval.io.load_intervals(estimated_file)
    # Intersect boundaries
    t_min = max([ref_intervals.min(), est_intervals.min()])
    t_max = min([ref_intervals.max(), est_intervals.max()])
    ref_intervals, ref_labels = mir_eval.util.adjust_intervals(
        ref_intervals, ref_labels, t_min, t_max,
        mir_eval.chord.NO_CHORD, mir_eval.chord.NO_CHORD)
    est_intervals, est_labels = mir_eval.util.adjust_intervals(
        est_intervals, est_labels, t_min, t_max,
        mir_eval.chord.NO_CHORD, mir_eval.chord.NO_CHORD)
    # Merge the time-intervals
    intervals, ref_labels, est_labels = mir_eval.util.merge_labeled_intervals(
        ref_intervals, ref_labels, est_intervals, est_labels)
    # Ensure that the score is correct
    assert np.allclose(metric(ref_labels, est_labels, intervals), score,
                       atol=A_TOL)


def __regression_test_chord_function(metric_name):
    # Load in all files in the same order
    ref_files = sorted(glob.glob(REF_GLOB))
    est_files = sorted(glob.glob(EST_GLOB))
    sco_files = sorted(glob.glob(SCORES_GLOB))
    metric = mir_eval.chord.METRICS[metric_name]
    # Regression tests
    for ref_f, est_f, sco_f in zip(ref_files, est_files, sco_files):
        with open(sco_f, 'r') as f:
            scores = json.load(f)
        yield (__regression_test_one_file, metric,
               ref_f, est_f, scores[metric_name])


def test_thirds():
    ref_labels = ['N', 'C:maj', 'C:maj', 'C:maj', 'C:min',
                  'C:maj', 'G:min', 'C:maj', 'C:min', 'C:min',
                  'C:maj', 'F:maj', 'C:maj', 'A:maj', 'A:maj']
    est_labels = ['N', 'N', 'C:aug', 'C:dim', 'C:dim',
                  'C:sus4', 'G:sus2', 'G:maj', 'C:hdim7', 'C:min7',
                  'C:maj6', 'F:min6', 'C:minmaj7', 'A:7', 'A:9']
    scores = [1.0, 0.0, 1.0, 0.0, 1.0,
              1.0, 0.0, 0.0, 1.0, 1.0,
              1.0, 0.0, 0.0, 1.0, 1.0]

    for ref_label, est_label, score in zip(ref_labels, est_labels, scores):
        yield (__check_one_metric, mir_eval.chord.thirds,
               ref_label, est_label, score)

    for test in __regression_test_chord_function('thirds'):
        yield test


def test_thirds_inv():
    ref_labels = ['C:maj/5',  'G:min',    'C:maj',   'C:min/b3',   'C:min']
    est_labels = ['C:sus4/5', 'G:min/b3', 'C:maj/5', 'C:hdim7/b3', 'C:dim']
    scores = [1.0, 0.0, 0.0, 1.0, 1.0]

    for ref_label, est_label, score in zip(ref_labels, est_labels, scores):
        yield (__check_one_metric, mir_eval.chord.thirds_inv,
               ref_label, est_label, score)

    for test in __regression_test_chord_function('thirds-inv'):
        yield test


def test_triads():
    ref_labels = ['C:min',  'C:maj', 'C:maj', 'C:min', 'C:maj',
                  'C:maj',  'G:min', 'C:maj', 'C:min', 'C:min']
    est_labels = ['C:min7', 'C:7',   'C:aug', 'C:dim', 'C:sus2',
                  'C:sus4', 'G:minmaj7', 'G:maj', 'C:hdim7', 'C:min6']
    scores = [1.0, 1.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0, 1.0]

    for ref_label, est_label, score in zip(ref_labels, est_labels, scores):
        yield (__check_one_metric, mir_eval.chord.triads,
               ref_label, est_label, score)

    for test in __regression_test_chord_function('triads'):
        yield test


def test_triads_inv():
    ref_labels = ['C:maj/5',  'G:min',    'C:maj', 'C:min/b3',  'C:min/b3']
    est_labels = ['C:maj7/5', 'G:min7/5', 'C:7/5', 'C:min6/b3', 'C:dim/b3']
    scores = [1.0, 0.0, 0.0, 1.0, 0.0]

    for ref_label, est_label, score in zip(ref_labels, est_labels, scores):
        yield (__check_one_metric, mir_eval.chord.triads_inv,
               ref_label, est_label, score)

    for test in __regression_test_chord_function('triads-inv'):
        yield test


def test_tetrads():
    ref_labels = ['C:min', 'C:maj', 'C:7', 'C:maj7', 'C:sus2',
                  'C:7/3', 'G:min', 'C:maj', 'C:min', 'C:min']
    est_labels = ['C:min7', 'C:maj6', 'C:9', 'C:maj7/5', 'C:sus2/2',
                  'C:11/b7', 'G:sus2', 'G:maj', 'C:hdim7', 'C:minmaj7']
    scores = [0.0, 0.0, 1.0, 1.0, 1.0,
              1.0, 0.0, 0.0, 0.0, 0.0]

    for ref_label, est_label, score in zip(ref_labels, est_labels, scores):
        yield (__check_one_metric, mir_eval.chord.tetrads,
               ref_label, est_label, score)

    for test in __regression_test_chord_function('tetrads'):
        yield test


def test_tetrads_inv():
    ref_labels = ['C:maj7/5', 'G:min', 'C:7/5', 'C:min/b3', 'C:min9']
    est_labels = ['C:maj7/3', 'G:min/b3', 'C:13/5', 'C:hdim7/b3', 'C:min7']
    scores = [0.0, 0.0, 1.0, 0.0, 1.0]

    for ref_label, est_label, score in zip(ref_labels, est_labels, scores):
        yield (__check_one_metric, mir_eval.chord.tetrads_inv,
               ref_label, est_label, score)

    for test in __regression_test_chord_function('triads-inv'):
        yield test


def test_majmin():
    ref_labels = ['N', 'C:maj', 'C:maj', 'C:min', 'G:maj7']
    est_labels = ['N', 'N', 'C:aug', 'C:dim', 'G']
    scores = [1.0,  0.0, 0.0, 0.0, 1.0]

    for ref_label, est_label, score in zip(ref_labels, est_labels, scores):
        yield (__check_one_metric, mir_eval.chord.majmin,
               ref_label, est_label, score)

    yield (__check_not_comparable, mir_eval.chord.majmin, 'C:aug', 'C:maj')

    for test in __regression_test_chord_function('majmin'):
        yield test


def test_majmin_inv():
    ref_labels = ['C:maj/5',  'G:min',    'C:maj/5', 'C:min7',
                  'G:min/b3', 'C:maj7/5', 'C:7']
    est_labels = ['C:sus4/5', 'G:min/b3', 'C:maj/5', 'C:min',
                  'G:min/b3', 'C:maj/5', 'C:maj']
    scores = [0.0, 0.0, 1.0, 1.0,
              1.0, 1.0, 1.0]

    for ref_label, est_label, score in zip(ref_labels, est_labels, scores):
        yield (__check_one_metric, mir_eval.chord.majmin_inv,
               ref_label, est_label, score)

    ref_not_comparable = ['C:hdim7/b3', 'C:maj/4', 'C:maj/2']
    est_not_comparable = ['C:min/b3', 'C:maj/4', 'C:sus2/2']

    for ref_label, est_label in zip(ref_not_comparable, est_not_comparable):
        yield (__check_not_comparable, mir_eval.chord.majmin_inv,
               ref_label, est_label)

    for test in __regression_test_chord_function('majmin-inv'):
        yield test


def test_sevenths():
    ref_labels = ['C:min',  'C:maj',  'C:7', 'C:maj7',
                  'C:7/3',   'G:min',  'C:maj', 'C:7']
    est_labels = ['C:min7', 'C:maj6', 'C:9', 'C:maj7/5',
                  'C:11/b7', 'G:sus2', 'G:maj', 'C:maj7']
    scores = [0.0, 0.0, 1.0, 1.0,
              1.0, 0.0, 0.0, 0.0]

    for ref_label, est_label, score in zip(ref_labels, est_labels, scores):
        yield (__check_one_metric, mir_eval.chord.sevenths,
               ref_label, est_label, score)

    ref_not_comparable = ['C:sus2', 'C:hdim7']
    est_not_comparable = ['C:sus2/2', 'C:hdim7']
    for ref_label, est_label in zip(ref_not_comparable, est_not_comparable):
        yield (__check_not_comparable, mir_eval.chord.sevenths,
               ref_label, est_label)

    for test in __regression_test_chord_function('sevenths'):
        yield test


def test_sevenths_inv():
    ref_labels = ['C:maj7/5', 'G:min',    'C:7/5', 'C:min7/b7']
    est_labels = ['C:maj7/3', 'G:min/b3', 'C:13/5', 'C:min7/b7']
    scores = [0.0, 0.0, 1.0, 1.0]

    for ref_label, est_label, score in zip(ref_labels, est_labels, scores):
        yield (__check_one_metric, mir_eval.chord.sevenths_inv,
               ref_label, est_label, score)

    yield (__check_not_comparable, mir_eval.chord.sevenths_inv, 'C:dim7/b3',
           'C:dim7/b3')

    for test in __regression_test_chord_function('sevenths-inv'):
        yield test
