'''
Unit tests for mir_eval.hierarchy
'''

import numpy as np
import mir_eval
from nose.tools import raises

def test_tmeasure_pass():

    # The estimate here gets none of the structure correct.
    ref = [ [ [0, 60] ], [ [0, 30], [30, 60] ] ]
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


def test_tmeasure_fail_span():

    # Missing a boundary from the first layer
    ref = [[[0, 30],
            [30, 60]],
           [[0, 60]]]

    ref = [np.asarray(_) for _ in ref]

    yield raises(ValueError)(mir_eval.hierarchy.tmeasure), ref, ref

    # Does not start at 0
    ref = [[[10, 60]],
           [[10, 30],
            [30, 60]]]

    ref = [np.asarray(_) for _ in ref]

    yield raises(ValueError)(mir_eval.hierarchy.tmeasure), ref, ref

    # Does not end at the right time
    ref = [[[0, 60]],
           [[0, 60],
            [60, 70]]]
    ref = [np.asarray(_) for _ in ref]

    yield raises(ValueError)(mir_eval.hierarchy.tmeasure), ref, ref


    # Two annotaions of different shape
    ref = [[[0, 60]],
           [[0, 30],
            [30, 60]]]
    ref = [np.asarray(_) for _ in ref]

    est = [[[0, 70]],
           [[0, 30],
            [30, 70]]]
    est = [np.asarray(_) for _ in est]

    yield raises(ValueError)(mir_eval.hierarchy.tmeasure), ref, est


def test_tmeasure_fail_frame():
    ref = [[[0, 60]],
           [[0, 30],
            [30, 60]]]
    ref = [np.asarray(_) for _ in ref]

    @raises(ValueError)
    def __test(window, frame_size):
        mir_eval.hierarchy.tmeasure(ref, ref, window=window, frame_size=frame_size)

    for window in [15, 30]:
        for frame_size in [-1, 0, 2 * window]:
            yield __test, window, frame_size
