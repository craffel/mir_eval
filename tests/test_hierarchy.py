'''
Unit tests for mir_eval.hierarchy
'''

import numpy as np
import mir_eval
from nose.tools import raises
from glob import glob
import re

import warnings
import json

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

    for window in [15, 30]:
        for frame_size in [-1, 0, 2 * window]:
            yield __test, window, frame_size


def test_tmeasure_regression():

    ref_files = sorted(glob('tests/data/hierarchy/ref*.lab'))
    est_files = sorted(glob('tests/data/hierarchy/est*.lab'))
    out_files = sorted(glob('tests/data/hierarchy/output*.json'))

    ref_hier = [mir_eval.io.load_labeled_intervals(_)[0] for _ in ref_files]
    est_hier = [mir_eval.io.load_labeled_intervals(_)[0] for _ in est_files]

    def __test(w, ref, est, target):

        outputs = mir_eval.hierarchy.evaluate(ref, est, window=w)

        for key in target:
            assert np.allclose(target[key], outputs[key], atol=A_TOL)

    for out in out_files:
        with open(out, 'r') as fdesc:
            target = json.load(fdesc)

        # Extract the window parameter
        window = float(re.match('.*output_w=(\d+).json$', out).groups()[0])
        yield __test, window, ref_hier, est_hier, target
