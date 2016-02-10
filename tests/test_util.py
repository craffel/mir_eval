''' Unit tests for utils
'''


import numpy as np
import nose.tools
import mir_eval
from mir_eval import util
import collections


def test_interpolate_intervals():
    """Check that an interval set is interpolated properly, with boundaries
    conditions and out-of-range values.
    """
    labels = list('abc')
    intervals = np.array([(n, n + 1.0) for n in range(len(labels))])
    time_points = [-1.0, 0.1, 0.9, 1.0, 2.3, 4.0]
    expected_ans = ['N', 'a', 'a', 'b', 'c', 'N']
    assert (util.interpolate_intervals(intervals, labels, time_points, 'N') ==
            expected_ans)


def test_intervals_to_samples():
    """Check that an interval set is sampled properly, with boundaries
    conditions and out-of-range values.
    """
    labels = list('abc')
    intervals = np.array([(n, n + 1.0) for n in range(len(labels))])

    expected_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    expected_labels = ['a', 'a', 'b', 'b', 'c', 'c']
    result = util.intervals_to_samples(
        intervals, labels, offset=0, sample_size=0.5, fill_value='N')
    assert result[0] == expected_times
    assert result[1] == expected_labels

    expected_times = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75]
    expected_labels = ['a', 'a', 'b', 'b', 'c', 'c']
    result = util.intervals_to_samples(
        intervals, labels, offset=0.25, sample_size=0.5, fill_value='N')
    assert result[0] == expected_times
    assert result[1] == expected_labels


def test_intersect_files():
    """Check that two non-identical yield correct results.
    """
    flist1 = ['/a/b/abc.lab', '/c/d/123.lab', '/e/f/xyz.lab']
    flist2 = ['/g/h/xyz.npy', '/i/j/123.txt', '/k/l/456.lab']
    sublist1, sublist2 = util.intersect_files(flist1, flist2)
    assert sublist1 == ['/e/f/xyz.lab', '/c/d/123.lab']
    assert sublist2 == ['/g/h/xyz.npy', '/i/j/123.txt']
    sublist1, sublist2 = util.intersect_files(flist1[:1], flist2[:1])
    assert sublist1 == []
    assert sublist2 == []


def test_merge_labeled_intervals():
    """Check that two labeled interval sequences merge correctly.
    """
    x_intvs = np.array([
        [0.0,    0.44],
        [0.44,  2.537],
        [2.537, 4.511],
        [4.511, 6.409]])
    x_labels = ['A', 'B', 'C', 'D']
    y_intvs = np.array([
        [0.0,   0.464],
        [0.464, 2.415],
        [2.415, 4.737],
        [4.737, 6.409]])
    y_labels = [0, 1, 2, 3]
    expected_intvs = [
        [0.0,    0.44],
        [0.44,  0.464],
        [0.464, 2.415],
        [2.415, 2.537],
        [2.537, 4.511],
        [4.511, 4.737],
        [4.737, 6.409]]
    expected_x_labels = ['A', 'B', 'B', 'B', 'C', 'D', 'D']
    expected_y_labels = [0,     0,   1,   2,   2,   2,   3]
    new_intvs, new_x_labels, new_y_labels = util.merge_labeled_intervals(
        x_intvs, x_labels, y_intvs, y_labels)

    assert new_x_labels == expected_x_labels
    assert new_y_labels == expected_y_labels
    assert new_intvs.tolist() == expected_intvs

    # Check that invalid inputs raise a ValueError
    y_intvs[-1, -1] = 10.0
    nose.tools.assert_raises(ValueError, util.merge_labeled_intervals, x_intvs,
                             x_labels, y_intvs, y_labels)


def test_boundaries_to_intervals():
    # Basic tests
    boundaries = np.arange(10)
    correct_intervals = np.array([np.arange(10 - 1), np.arange(1, 10)]).T
    intervals = mir_eval.util.boundaries_to_intervals(boundaries)
    assert np.all(intervals == correct_intervals)


def test_adjust_events():
    # Test appending at the end
    events = np.arange(1, 11)
    labels = [str(n) for n in range(10)]
    new_e, new_l = mir_eval.util.adjust_events(events, labels, 0.0, 11.)
    assert new_e[0] == 0.
    assert new_l[0] == '__T_MIN'
    assert new_e[-1] == 11.
    assert new_l[-1] == '__T_MAX'
    assert np.all(new_e[1:-1] == events)
    assert new_l[1:-1] == labels

    # Test trimming
    new_e, new_l = mir_eval.util.adjust_events(events, labels, 0.0, 9.)
    assert new_e[0] == 0.
    assert new_l[0] == '__T_MIN'
    assert new_e[-1] == 9.
    assert np.all(new_e[1:] == events[:-1])
    assert new_l[1:] == labels[:-1]


def test_bipartite_match():
    # This test constructs a graph as follows:
    #   v9 -- (u0)
    #   v8 -- (u0, u1)
    #   v7 -- (u0, u1, u2)
    #   ...
    #   v0 -- (u0, u1, ..., u9)
    #
    # This structure and ordering of this graph should force Hopcroft-Karp to
    # hit each algorithm/layering phase
    #
    G = collections.defaultdict(list)

    u_set = ['u{:d}'.format(_) for _ in range(10)]
    v_set = ['v{:d}'.format(_) for _ in range(len(u_set)+1)]
    for i, u in enumerate(u_set):
        for v in v_set[:-i-1]:
            G[v].append(u)

    matching = util._bipartite_match(G)

    # Make sure that each u vertex is matched
    nose.tools.eq_(len(matching), len(u_set))

    # Make sure that there are no duplicate keys
    lhs = set([k for k in matching])
    rhs = set([matching[k] for k in matching])

    nose.tools.eq_(len(matching), len(lhs))
    nose.tools.eq_(len(matching), len(rhs))

    # Finally, make sure that all detected edges are present in G
    for k in matching:
        v = matching[k]
        assert v in G[k] or k in G[v]


def test_match_notes():

    ref_int, ref_pitch = mir_eval.io.load_valued_intervals(
        'tests/data/transcription/ref00.txt')
    est_int, est_pitch = mir_eval.io.load_valued_intervals(
        'tests/data/transcription/est00.txt')

    matching = mir_eval.util.match_notes(ref_int, ref_pitch, est_int,
                                         est_pitch)

    assert matching == [(0, 0), (1, 1), (3,3)]

    matching = mir_eval.util.match_notes(ref_int, ref_pitch, est_int,
                                         est_pitch, with_offset=True)

    assert matching == [(0, 0), (3, 4)]
