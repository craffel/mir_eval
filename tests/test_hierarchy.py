'''
Unit tests for mir_eval.hierarchy
'''

from glob import glob
import re

import warnings
import json

import numpy as np
import scipy.sparse
import mir_eval

from nose.tools import raises


A_TOL = 1e-12


def test_tmeasure_pass():
    # The estimate here gets none of the structure correct.
    ref = [[[0, 30]], [[0, 15], [15, 30]]]
    # convert to arrays
    ref = [np.asarray(_) for _ in ref]

    est = ref[:1]

    def __test(window, frame_size):
        # The estimate should get 0 score here
        scores = mir_eval.hierarchy.tmeasure(ref, est,
                                             window=window,
                                             frame_size=frame_size)

        for k in scores:
            assert k == 0.0

        # The reference should get a perfect score here
        scores = mir_eval.hierarchy.tmeasure(ref, ref,
                                             window=window,
                                             frame_size=frame_size)

        for k in scores:
            assert k == 1.0

    for window in [5, 10, 15, 30, 90, None]:
        for frame_size in [0.1, 0.5, 1.0]:
            yield __test, window, frame_size


def test_tmeasure_warning():

    # Warn if there are missing boundaries from one layer to the next
    ref = [[[0, 5],
            [5, 10]],
           [[0, 10]]]

    ref = [np.asarray(_) for _ in ref]

    warnings.resetwarnings()
    warnings.simplefilter('always')
    with warnings.catch_warnings(record=True) as out:
        mir_eval.hierarchy.tmeasure(ref, ref)

        assert len(out) > 0
        assert out[0].category is UserWarning
        assert ('Segment hierarchy is inconsistent at level 1'
                in str(out[0].message))


def test_tmeasure_fail_span():

    # Does not start at 0
    ref = [[[1, 10]],
           [[1, 5],
            [5, 10]]]

    ref = [np.asarray(_) for _ in ref]

    yield raises(ValueError)(mir_eval.hierarchy.tmeasure), ref, ref

    # Does not end at the right time
    ref = [[[0, 5]],
           [[0, 5],
            [5, 6]]]
    ref = [np.asarray(_) for _ in ref]

    yield raises(ValueError)(mir_eval.hierarchy.tmeasure), ref, ref

    # Two annotaions of different shape
    ref = [[[0, 10]],
           [[0, 5],
            [5, 10]]]
    ref = [np.asarray(_) for _ in ref]

    est = [[[0, 15]],
           [[0, 5],
            [5, 15]]]
    est = [np.asarray(_) for _ in est]

    yield raises(ValueError)(mir_eval.hierarchy.tmeasure), ref, est


def test_tmeasure_fail_frame_size():
    ref = [[[0, 60]],
           [[0, 30],
            [30, 60]]]
    ref = [np.asarray(_) for _ in ref]

    @raises(ValueError)
    def __test(window, frame_size):
        mir_eval.hierarchy.tmeasure(ref, ref,
                                    window=window,
                                    frame_size=frame_size)

    for window in [None, 15, 30]:
        for frame_size in [-1, 0]:
            yield __test, window, frame_size
        if window is not None:
            yield __test, window, 2 * window


def test_lmeasure_pass():

    # The estimate here gets none of the structure correct.
    ref = [[[0, 30]], [[0, 15], [15, 30]]]
    ref_lab = [['A'], ['a', 'b']]

    # convert to arrays
    ref = [np.asarray(_) for _ in ref]

    est = ref[:1]
    est_lab = ref_lab[:1]

    def __test(frame_size):
        # The estimate should get 0 score here
        scores = mir_eval.hierarchy.lmeasure(ref, ref_lab, est, est_lab,
                                             frame_size=frame_size)

        for k in scores:
            assert k == 0.0

        # The reference should get a perfect score here
        scores = mir_eval.hierarchy.lmeasure(ref, ref_lab, ref, ref_lab,
                                             frame_size=frame_size)

        for k in scores:
            assert k == 1.0

    for frame_size in [0.1, 0.5, 1.0]:
        yield __test, frame_size


def test_lmeasure_warning():

    # Warn if there are missing boundaries from one layer to the next
    ref = [[[0, 5],
            [5, 10]],
           [[0, 10]]]

    ref = [np.asarray(_) for _ in ref]
    ref_lab = [['a', 'b'], ['A']]

    warnings.resetwarnings()
    warnings.simplefilter('always')
    with warnings.catch_warnings(record=True) as out:
        mir_eval.hierarchy.lmeasure(ref, ref_lab, ref, ref_lab)

        assert len(out) > 0
        assert out[0].category is UserWarning
        assert ('Segment hierarchy is inconsistent at level 1'
                in str(out[0].message))


def test_lmeasure_fail_span():

    # Does not start at 0
    ref = [[[1, 10]],
           [[1, 5],
            [5, 10]]]

    ref_lab = [['A'], ['a', 'b']]

    ref = [np.asarray(_) for _ in ref]

    yield (raises(ValueError)(mir_eval.hierarchy.lmeasure),
           ref, ref_lab, ref, ref_lab)

    # Does not end at the right time
    ref = [[[0, 5]],
           [[0, 5],
            [5, 6]]]
    ref = [np.asarray(_) for _ in ref]

    yield (raises(ValueError)(mir_eval.hierarchy.lmeasure),
           ref, ref_lab, ref, ref_lab)

    # Two annotations of different shape
    ref = [[[0, 10]],
           [[0, 5],
            [5, 10]]]
    ref = [np.asarray(_) for _ in ref]

    est = [[[0, 15]],
           [[0, 5],
            [5, 15]]]
    est = [np.asarray(_) for _ in est]

    yield (raises(ValueError)(mir_eval.hierarchy.lmeasure),
           ref, ref_lab, est, ref_lab)


def test_lmeasure_fail_frame_size():
    ref = [[[0, 60]],
           [[0, 30],
            [30, 60]]]
    ref = [np.asarray(_) for _ in ref]
    ref_lab = [['A'], ['a', 'b']]

    @raises(ValueError)
    def __test(frame_size):
        mir_eval.hierarchy.lmeasure(ref, ref_lab, ref, ref_lab,
                                    frame_size=frame_size)

    for frame_size in [-1, 0]:
        yield __test, frame_size


def test_hierarchy_regression():

    ref_files = sorted(glob('data/hierarchy/ref*.lab'))
    est_files = sorted(glob('data/hierarchy/est*.lab'))
    out_files = sorted(glob('data/hierarchy/output*.json'))

    ref_hier = [mir_eval.io.load_labeled_intervals(_) for _ in ref_files]
    est_hier = [mir_eval.io.load_labeled_intervals(_) for _ in est_files]

    ref_ints = [seg[0] for seg in ref_hier]
    ref_labs = [seg[1] for seg in ref_hier]
    est_ints = [seg[0] for seg in est_hier]
    est_labs = [seg[1] for seg in est_hier]

    def __test(w, ref_i, ref_l, est_i, est_l, target):
        outputs = mir_eval.hierarchy.evaluate(ref_i, ref_l,
                                              est_i, est_l,
                                              window=w)

        for key in target:
            assert np.allclose(target[key], outputs[key], atol=A_TOL)

    for out in out_files:
        with open(out, 'r') as fdesc:
            target = json.load(fdesc)

        # Extract the window parameter
        window = float(re.match('.*output_w=(\d+).json$', out).groups()[0])
        yield __test, window, ref_ints, ref_labs, est_ints, est_labs, target


def test_count_inversions():

    # inversion count = |{(i, j) : a[i] >= b[j]}|
    a = [2, 4, 6]
    b = [1, 2, 3, 4]

    # All inversions (a, b) are:
    # (2, 1), (2, 2)
    # (4, 1), (4, 2), (4, 3), (4, 4)
    # (6, 1), (6, 2), (6, 3), (6, 4)

    assert mir_eval.hierarchy._count_inversions(a, b) == 10

    # All inversions (b, a) are:
    # (2, 2)
    # (3, 2)
    # (4, 2), (4, 4)

    assert mir_eval.hierarchy._count_inversions(b, a) == 4

    # And test with repetitions
    a = [2, 2, 4]
    b = [1, 2, 4, 4]
    # counts: (a, b)
    # (2, 1), (2, 2)
    # (2, 1), (2, 2)
    # (4, 1), (4, 2), (4, 4), (4, 4)

    assert mir_eval.hierarchy._count_inversions(a, b) == 8

    # count: (b, a)
    # (2, 2), (2, 2)
    # (4, 2), (4, 2), (4, 4)
    # (4, 2), (4, 2), (4, 4)

    assert mir_eval.hierarchy._count_inversions(b, a) == 8


def test_meet():

    frame_size = 1
    int_hier = [np.array([[0, 10]]),
                np.array([[0, 6], [6, 10]]),
                np.array([[0, 2], [2, 4], [4, 6], [6, 8], [8, 10]])]

    lab_hier = [['X'],
                ['A', 'B'],
                ['a', 'b', 'a', 'c', 'b']]

    # Target output
    meet_truth = np.asarray([
        [3, 3, 2, 2, 3, 3, 1, 1, 1, 1],   # (XAa)
        [3, 3, 2, 2, 3, 3, 1, 1, 1, 1],   # (XAa)
        [2, 2, 3, 3, 2, 2, 1, 1, 3, 3],   # (XAb)
        [2, 2, 3, 3, 2, 2, 1, 1, 3, 3],   # (XAb)
        [3, 3, 2, 2, 3, 3, 1, 1, 1, 1],   # (XAa)
        [3, 3, 2, 2, 3, 3, 1, 1, 1, 1],   # (XAa)
        [1, 1, 1, 1, 1, 1, 3, 3, 2, 2],   # (XBc)
        [1, 1, 1, 1, 1, 1, 3, 3, 2, 2],   # (XBc)
        [1, 1, 3, 3, 1, 1, 2, 2, 3, 3],   # (XBb)
        [1, 1, 3, 3, 1, 1, 2, 2, 3, 3],   # (XBb)
    ])
    meet = mir_eval.hierarchy._meet(int_hier, lab_hier, frame_size)

    # Is it the right type?
    assert isinstance(meet, scipy.sparse.csr_matrix)
    meet = meet.todense()

    # Does it have the right shape?
    assert meet.shape == (10, 10)

    # Does it have the right value?
    assert np.all(meet == meet_truth)
