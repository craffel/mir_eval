"""
Unit tests for mir_eval.chord
"""

import mir_eval
import numpy as np
import pytest
import nose.tools
import warnings
import glob
import json

A_TOL = 1e-12

# Path to the fixture files
REF_GLOB = 'data/chord/ref*.lab'
EST_GLOB = 'data/chord/est*.lab'
SCORES_GLOB = 'data/chord/output*.json'

ref_files = sorted(glob.glob(REF_GLOB))
est_files = sorted(glob.glob(EST_GLOB))
sco_files = sorted(glob.glob(SCORES_GLOB))

assert len(ref_files) == len(est_files) == len(sco_files) > 0

file_sets = list(zip(ref_files, est_files, sco_files))

@pytest.fixture
def chord_data(request):
    ref_f, est_f, sco_f = request.param
    with open(sco_f, "r") as f:
        expected_scores = json.load(f)
    # Load in reference melody
    ref_intervals, ref_labels = mir_eval.io.load_labeled_intervals(ref_f)
    # Load in estimated melody
    est_intervals, est_labels = mir_eval.io.load_labeled_intervals(est_f)
    return ref_intervals, ref_labels, est_intervals, est_labels, expected_scores


def __check_valid(function, parameters, result):
    ''' Helper function for checking the output of a function '''
    assert function(*parameters) == result


def __check_exception(function, parameters, exception):
    ''' Makes sure the provided function throws the provided
    exception given the provided input '''
    nose.tools.assert_raises(exception, function, *parameters)


@pytest.mark.parametrize('pitch, semitone', [
    ('Gbb', 5),
    ('G', 7),
    ('G#', 8),
    ('Cb', 11),
    ('B#', 0)
])
def test_pitch_class_to_semitone_valid(pitch, semitone):
    assert mir_eval.chord.pitch_class_to_semitone(pitch) == semitone


@pytest.mark.parametrize('pitch', ['Cab', '#C', 'bG'])
@pytest.mark.xfail(raises=mir_eval.chord.InvalidChordException)
def test_pitch_class_to_semitone_fail(pitch):
    mir_eval.chord.pitch_class_to_semitone(pitch)


@pytest.mark.parametrize('degree, semitone',
        [('b7', 10), ('#3', 5), ('1', 0), ('b1', -1), ('#7', 12), ('bb5', 5), ('11', 17), ('#13', 22)])
def test_scale_degree_to_semitone(degree, semitone):
    assert mir_eval.chord.scale_degree_to_semitone(degree) == semitone


@pytest.mark.parametrize('degree', ['7b', '4#', '77', '15'])
@pytest.mark.xfail(raises=mir_eval.chord.InvalidChordException)
def test_scale_degree_to_semitone(degree):
    mir_eval.chord.scale_degree_to_semitone(degree)


def test_scale_degree_to_bitmap():

    def __check_bitmaps(function, parameters, result):
        actual = function(*parameters)
        assert np.all(actual == result), (actual, result)

    valid_degrees = ['3', '*3', 'b1', '9']
    valid_bitmaps = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    for scale_degree, bitmap in zip(valid_degrees, valid_bitmaps):
        yield (__check_bitmaps, mir_eval.chord.scale_degree_to_bitmap,
               (scale_degree, True, 12), np.array(bitmap))

    yield (__check_bitmaps, mir_eval.chord.scale_degree_to_bitmap,
           ('9', False, 12), np.array([0] * 12))

    yield (__check_bitmaps, mir_eval.chord.scale_degree_to_bitmap,
           ('9', False, 15),
           np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))


@pytest.mark.parametrize('label', ['C', 'Eb:min/5', 'A#:dim7', 'B:maj(*1,*5)/3',
                                    'A#:sus4', 'A:(9,11)'])
def test_validate_chord_label(label):
    # For valid labels, calling the function without an error = pass
    mir_eval.chord.validate_chord_label(label)


@pytest.mark.parametrize('label', ["C::maj", "C//5", "C((4)", "C5))",
                      "C:maj(*3/3", "Cmaj*3/3)", 'asdf'])
@pytest.mark.xfail(raises=mir_eval.chord.InvalidChordException)
def test_validate_bad_chord_label(label):
    mir_eval.chord.validate_chord_label(label)


@pytest.mark.parametrize('label, split',
        [('C', ['C', 'maj', set(), '1']),
 ('B:maj(*1,*3)/5', ['B', 'maj', {'*1', '*3'}, '5']),
 ('Ab:min/b3', ['Ab', 'min', set(), 'b3']),
 ('N', ['N', '', set(), '']),
 ('G:(3)', ['G', '', {'3'}, '1'])]
        )
def test_split(label, split):
    assert mir_eval.chord.split(label) == split

@pytest.mark.parametrize('label, split', [('C', ['C', 'maj', set(), '1']), ('C:minmaj7', ['C', 'min', {'7'}, '1'])])
def test_split_extended(label, split):
    # Test with reducing extended chords
    mir_eval.chord.split(label, reduce_extended_chords=True) == split


@pytest.mark.xfail(raises=mir_eval.chord.InvalidChordException)
def test_split_fail():
    # Test that an exception is raised when a chord with an omission but no
    # quality is supplied
    mir_eval.chord.split('C(*5)')


# Arguments are root, quality, extensions, bass
@pytest.mark.parametrize('label, split', [('F#', ('F#', '', None, '')),
 ('F#:hdim7', ('F#', 'hdim7', None, '')),
 ('F#:(*b3,4)', ('F#', '', ['*b3', '4'], '')),
 ('F#/b7', ('F#', '', None, 'b7')),
 ('F#:(*b3,4)/b7', ('F#', '', ['*b3', '4'], 'b7')),
 ('F#:hdim7/b7', ('F#', 'hdim7', None, 'b7')),
 ('F#:hdim7(*b3,4)/b7', ('F#', 'hdim7', ['*b3', '4'], 'b7'))])
def test_join(label, split):
    # Test is relying on implicit parameter ordering here: root, quality, extensions, bass
    assert mir_eval.chord.join(*split) == label


@pytest.mark.parametrize('bitmap, root, expected_bitmap',
[([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
  0,
  [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]),
 ([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
  5,
  [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]),
 ([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
  11,
  [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1])])
def test_rotate_bitmaps_to_roots(bitmap, root, expected_bitmap):
    ans = mir_eval.chord.rotate_bitmaps_to_roots([bitmap], [root])
    assert np.all(ans == [expected_bitmap])


def test_encode():
    def __check_encode(label, expected_root, expected_intervals,
                       expected_bass, reduce_extended_chords,
                       strict_bass_intervals):
        ''' Helper function for checking encode '''
        root, intervals, bass = mir_eval.chord.encode(
            label, reduce_extended_chords=reduce_extended_chords,
            strict_bass_intervals=strict_bass_intervals)
        assert root == expected_root, (root, expected_root)
        assert np.all(intervals == expected_intervals), (intervals,
                                                         expected_intervals)
        assert bass == expected_bass, (bass, expected_bass)

    labels = ['B:maj(*1,*3)/5', 'G:dim', 'C:(3)/3', 'A:9/b3']
    expected_roots = [11, 7, 0, 9]
    expected_intervals = [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                          [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                          # Note that extended scale degrees are dropped.
                          [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0]]
    expected_bass = [7, 0, 4, 3]

    args = list(zip(labels, expected_roots, expected_intervals, expected_bass))
    for label, e_root, e_interval, e_bass in args:
        yield (__check_encode, label, e_root, e_interval, e_bass, False, False)


@pytest.mark.parametrize('label, e_root, e_interval, e_bass, reduce, strict',
        [('B:maj(*1,*3)/5', 11, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 7, False, False),
 ('G:dim', 7, [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0], 0, False, False),
 ('C:(3)/3', 0, [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 4, False, False),
 ('A:9/b3', 9, [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0], 3, False, False),
 ('G:dim(4)/6', 7, [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0], 9, False, False),
 ('A:9', 9, [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0], 0, True, False)]
        )
def test_chord_encode(label, e_root, e_interval, e_bass, reduce, strict):
    root, intervals, bass = mir_eval.chord.encode(label,
        reduce_extended_chords=reduce,
        strict_bass_intervals=strict)
    assert root == e_root, (root, e_root)
    assert np.all(intervals == e_interval), (intervals, e_interval)
    assert bass == e_bass, (bass, e_bass)


@pytest.mark.xfail(raises=mir_eval.chord.InvalidChordException)
def test_chord_encode_fail():
    # Non-chord bass notes *must* be explicitly named as extensions when
    #   strict_bass_intervals == True
    mir_eval.chord.encode('G:dim(4)/6', reduce_extended_chords=False, strict_bass_intervals=True)


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
    assert metric([ref_label], [est_label]) == score


def __check_not_comparable(metric, ref_label, est_label):
    ''' Checks that ref_label is not comparable to est_label by metric '''
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        # Try to produce the warning
        score = mir_eval.chord.weighted_accuracy(metric([ref_label],
                                                        [est_label]),
                                                 np.array([1]))
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert str(w[-1].message) == ("No reference chords were comparable "
                                      "to estimated chords, returning 0.")
        # And confirm that the metric is 0
        assert np.allclose(score, 0)

# TODO(ejhumphrey): Comparison functions lacking unit tests.
# test_root()


def test_mirex():
    ref_labels = ['N', 'C:maj', 'C:maj', 'C:maj', 'C:min', 'C:maj',
                  'C:maj',  'G:min',  'C:maj', 'C:min',   'C:min',
                  'C:maj',  'F:maj',  'C:maj7',    'A:maj', 'A:maj']
    est_labels = ['N', 'N',     'C:aug', 'C:dim', 'C:dim', 'C:5',
                  'C:sus4', 'G:sus2', 'G:maj', 'C:hdim7', 'C:min7',
                  'C:maj6', 'F:min6', 'C:minmaj7', 'A:7',   'A:9']
    scores = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 1.0,
              1.0, 0.0, 1.0, 1.0, 1.0]

    for ref_label, est_label, score in zip(ref_labels, est_labels, scores):
        yield (__check_one_metric, mir_eval.chord.mirex,
               ref_label, est_label, score)

    ref_not_comparable = ['C:5', 'X']
    est_not_comparable = ['C:maj', 'N']

    for ref_label, est_label in zip(ref_not_comparable, est_not_comparable):
        yield (__check_not_comparable, mir_eval.chord.mirex,
               ref_label, est_label)


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

    yield (__check_not_comparable, mir_eval.chord.thirds, 'X', 'N')


def test_thirds_inv():
    ref_labels = ['C:maj/5',  'G:min',    'C:maj',   'C:min/b3',   'C:min']
    est_labels = ['C:sus4/5', 'G:min/b3', 'C:maj/5', 'C:hdim7/b3', 'C:dim']
    scores = [1.0, 0.0, 0.0, 1.0, 1.0]

    for ref_label, est_label, score in zip(ref_labels, est_labels, scores):
        yield (__check_one_metric, mir_eval.chord.thirds_inv,
               ref_label, est_label, score)

    yield (__check_not_comparable, mir_eval.chord.thirds_inv, 'X', 'N')


def test_triads():
    ref_labels = ['C:min',  'C:maj', 'C:maj', 'C:min', 'C:maj',
                  'C:maj',  'G:min',     'C:maj', 'C:min',   'C:min']
    est_labels = ['C:min7', 'C:7',   'C:aug', 'C:dim', 'C:sus2',
                  'C:sus4', 'G:minmaj7', 'G:maj', 'C:hdim7', 'C:min6']
    scores = [1.0, 1.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0, 1.0]

    for ref_label, est_label, score in zip(ref_labels, est_labels, scores):
        yield (__check_one_metric, mir_eval.chord.triads,
               ref_label, est_label, score)

    yield (__check_not_comparable, mir_eval.chord.triads, 'X', 'N')


def test_triads_inv():
    ref_labels = ['C:maj/5',  'G:min',    'C:maj', 'C:min/b3',  'C:min/b3']
    est_labels = ['C:maj7/5', 'G:min7/5', 'C:7/5', 'C:min6/b3', 'C:dim/b3']
    scores = [1.0, 0.0, 0.0, 1.0, 0.0]

    for ref_label, est_label, score in zip(ref_labels, est_labels, scores):
        yield (__check_one_metric, mir_eval.chord.triads_inv,
               ref_label, est_label, score)

    yield (__check_not_comparable, mir_eval.chord.triads_inv, 'X', 'N')


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

    yield (__check_not_comparable, mir_eval.chord.tetrads, 'X', 'N')


def test_tetrads_inv():
    ref_labels = ['C:maj7/5', 'G:min', 'C:7/5', 'C:min/b3', 'C:min9']
    est_labels = ['C:maj7/3', 'G:min/b3', 'C:13/5', 'C:hdim7/b3', 'C:min7']
    scores = [0.0, 0.0, 1.0, 0.0, 1.0]

    for ref_label, est_label, score in zip(ref_labels, est_labels, scores):
        yield (__check_one_metric, mir_eval.chord.tetrads_inv,
               ref_label, est_label, score)

    yield (__check_not_comparable, mir_eval.chord.tetrads_inv, 'X', 'N')


def test_majmin():
    ref_labels = ['N', 'C:maj', 'C:maj', 'C:min', 'G:maj7']
    est_labels = ['N', 'N',     'C:aug', 'C:dim', 'G']
    scores = [1.0,  0.0, 0.0, 0.0, 1.0]

    for ref_label, est_label, score in zip(ref_labels, est_labels, scores):
        yield (__check_one_metric, mir_eval.chord.majmin,
               ref_label, est_label, score)

    ref_not_comparable = ['C:aug', 'X']
    est_not_comparable = ['C:maj', 'N']

    for ref_label, est_label in zip(ref_not_comparable, est_not_comparable):
        yield (__check_not_comparable, mir_eval.chord.majmin,
               ref_label, est_label)


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

    ref_not_comparable = ['C:hdim7/b3', 'C:maj/4', 'C:maj/2', 'X']
    est_not_comparable = ['C:min/b3', 'C:maj/4', 'C:sus2/2',  'N']

    for ref_label, est_label in zip(ref_not_comparable, est_not_comparable):
        yield (__check_not_comparable, mir_eval.chord.majmin_inv,
               ref_label, est_label)


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

    ref_not_comparable = ['C:sus2',   'C:hdim7', 'X']
    est_not_comparable = ['C:sus2/2', 'C:hdim7', 'N']
    for ref_label, est_label in zip(ref_not_comparable, est_not_comparable):
        yield (__check_not_comparable, mir_eval.chord.sevenths,
               ref_label, est_label)


def test_sevenths_inv():
    ref_labels = ['C:maj7/5', 'G:min',    'C:7/5',  'C:min7/b7']
    est_labels = ['C:maj7/3', 'G:min/b3', 'C:13/5', 'C:min7/b7']
    scores = [0.0, 0.0, 1.0, 1.0]

    for ref_label, est_label, score in zip(ref_labels, est_labels, scores):
        yield (__check_one_metric, mir_eval.chord.sevenths_inv,
               ref_label, est_label, score)

    ref_not_comparable = ['C:dim7/b3', 'X']
    est_not_comparable = ['C:dim7/b3', 'N']
    for ref_label, est_label in zip(ref_not_comparable, est_not_comparable):
        yield (__check_not_comparable, mir_eval.chord.sevenths_inv,
               ref_label, est_label)


def test_directional_hamming_distance():
    ref_ivs = np.array([[0., 1.], [1., 2.], [2., 3.]])
    est_ivs = np.array([[0., 0.9], [0.9, 1.8], [1.8, 2.5]])
    dhd_ref_to_est = (0.1 + 0.2 + 0.5) / 3.
    dhd_est_to_ref = (0.0 + 0.1 + 0.2) / 2.5

    dhd = mir_eval.chord.directional_hamming_distance
    assert np.allclose(dhd_ref_to_est, dhd(ref_ivs, est_ivs))
    assert np.allclose(dhd_est_to_ref, dhd(est_ivs, ref_ivs))
    assert np.allclose(0, dhd(ref_ivs, ref_ivs))
    assert np.allclose(0, dhd(est_ivs, est_ivs))

    ivs_overlap_all = np.array([[0., 1.], [0.9, 2.]])
    ivs_overlap_one = np.array([[0., 1.], [0.9, 2.], [2., 3.]])
    with pytest.raises(ValueError):
        dhd(ivs_overlap_all, est_ivs)
    with pytest.raises(ValueError):
        dhd(ivs_overlap_one, est_ivs)


def test_segmentation_functions():
    ref_ivs = np.array([[0., 2.], [2., 2.5], [2.5, 3.2]])
    est_ivs = np.array([[0., 3.], [3., 3.5]])
    true_oseg = 1. - 0.2 / 3.2
    true_useg = 1. - (1. + 0.2) / 3.5
    true_seg = min(true_oseg, true_useg)
    assert np.allclose(true_oseg, mir_eval.chord.overseg(ref_ivs, est_ivs))
    assert np.allclose(true_useg, mir_eval.chord.underseg(ref_ivs, est_ivs))
    assert np.allclose(true_seg, mir_eval.chord.seg(ref_ivs, est_ivs))

    ref_ivs = np.array([[0., 2.], [2., 2.5], [2.5, 3.2]])
    est_ivs = np.array([[0., 2.], [2., 2.5], [2.5, 3.2]])
    true_oseg = 1.0
    true_useg = 1.0
    true_seg = 1.0
    assert np.allclose(true_oseg, mir_eval.chord.overseg(ref_ivs, est_ivs))
    assert np.allclose(true_useg, mir_eval.chord.underseg(ref_ivs, est_ivs))
    assert np.allclose(true_seg, mir_eval.chord.seg(ref_ivs, est_ivs))

    ref_ivs = np.array([[0., 2.], [2., 2.5], [2.5, 3.2]])
    est_ivs = np.array([[0., 3.2]])
    true_oseg = 1.0
    true_useg = 1 - 1.2 / 3.2
    true_seg = min(true_oseg, true_useg)
    assert np.allclose(true_oseg, mir_eval.chord.overseg(ref_ivs, est_ivs))
    assert np.allclose(true_useg, mir_eval.chord.underseg(ref_ivs, est_ivs))
    assert np.allclose(true_seg, mir_eval.chord.seg(ref_ivs, est_ivs))

    ref_ivs = np.array([[0., 2.], [2., 2.5], [2.5, 3.2]])
    est_ivs = np.array([[3.2, 3.5]])
    true_oseg = 1.0
    true_useg = 1.0
    true_seg = 1.0
    assert np.allclose(true_oseg, mir_eval.chord.overseg(ref_ivs, est_ivs))
    assert np.allclose(true_useg, mir_eval.chord.underseg(ref_ivs, est_ivs))
    assert np.allclose(true_seg, mir_eval.chord.seg(ref_ivs, est_ivs))


def test_merge_chord_intervals():
    intervals = np.array([[0., 1.], [1., 2.], [2., 3], [3., 4.], [4., 5.]])
    labels = ['C:maj', 'C:(1,3,5)', 'A:maj', 'A:maj7', 'A:maj7/3']
    assert np.allclose(np.array([[0., 2.], [2., 3], [3., 4.], [4., 5.]]),
                       mir_eval.chord.merge_chord_intervals(intervals, labels))


def test_weighted_accuracy():
    # First, test for a warning on empty beats
    with pytest.warns(UserWarning, match="No nonzero weights, returning 0"):
        score = mir_eval.chord.weighted_accuracy(np.array([1, 0, 1]),
                                                 np.array([0, 0, 0]))
        # And that the metric is 0
        assert np.allclose(score, 0)

    # len(comparisons) must equal len(weights)
    comparisons = np.array([1, 0, 1])
    weights = np.array([1, 1])

    with pytest.raises(ValueError):
        mir_eval.chord.weighted_accuracy(comparisons, weights)

    # Weights must all be positive
    comparisons = np.array([1, 1])
    weights = np.array([-1, -1])
    with pytest.raises(ValueError):
        mir_eval.chord.weighted_accuracy(comparisons, weights)

    # Make sure accuracy = 1 and 0 when all comparisons are True and False resp
    comparisons = np.array([1, 1, 1])
    weights = np.array([1, 1, 1])
    score = mir_eval.chord.weighted_accuracy(comparisons, weights)
    assert np.allclose(score, 1)
    comparisons = np.array([0, 0, 0])
    score = mir_eval.chord.weighted_accuracy(comparisons, weights)
    assert np.allclose(score, 0)


@pytest.mark.parametrize("chord_data", file_sets, indirect=True)
def test_chord_functions(chord_data):
    ref_intervals, ref_labels, est_intervals, est_labels, expected_scores = chord_data

    # Compute scores
    scores = mir_eval.chord.evaluate(ref_intervals, ref_labels,
                                     est_intervals, est_labels)
    # Compare them
    assert scores.keys() == expected_scores.keys()
    for metric in scores:
        assert np.allclose(scores[metric], expected_scores[metric], atol=A_TOL)


def test_quality_to_bitmap():

    # Test simple case
    assert np.all(mir_eval.chord.quality_to_bitmap('maj') == np.array(
        [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]))


@pytest.mark.xfail(raises=mir_eval.chord.InvalidChordException)
@pytest.mark.parametrize('quality', ['maj5', '2', '#7'])
def test_quality_to_bitmap_fail(quality):
    # Check exceptions for qualities not in the QUALITIES list
    mir_eval.chord.quality_to_bitmap(quality)


def test_validate():
    # Test that the validate function raises the appropriate errors and
    # warnings
    with pytest.warns() as w:
        # First, test for warnings on empty labels
        mir_eval.chord.validate([], [])
    assert len(w) == 2
    assert issubclass(w[-1].category, UserWarning)
    assert str(w[-1].message) == "Estimated labels are empty"
    assert issubclass(w[-2].category, UserWarning)
    assert str(w[-2].message) == "Reference labels are empty"

    # Test that error is thrown on different-length labels
    with pytest.raises(ValueError):
        mir_eval.chord.validate([], ['C'])
