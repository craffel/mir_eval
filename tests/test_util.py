'''
'''


import unittest
import numpy as np

from mir_eval import util


class UtilTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_interpolate_intervals(self):
        """Check that an interval set is interpolated properly, with boundaries
        conditions and out-of-range values.
        """
        labels = list('abc')
        intervals = np.array([(n, n + 1.0) for n in range(len(labels))])
        time_points = [-1.0, 0.1, 0.9, 1.0, 2.3, 4.0]
        expected_ans = ['N', 'a', 'a', 'b', 'c', 'N']
        self.assertEqual(
            util.interpolate_intervals(intervals, labels, time_points, 'N'),
            expected_ans)

    def test_intervals_to_samples(self):
        """Check that an interval set is sampled properly, with boundaries
        conditions and out-of-range values.
        """
        labels = list('abc')
        intervals = np.array([(n, n + 1.0) for n in range(len(labels))])

        expected_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        expected_labels = ['a', 'a', 'b', 'b', 'c', 'c']
        result = util.intervals_to_samples(
            intervals, labels, offset=0, sample_size=0.5, fill_value='N')
        self.assertEqual(result[0], expected_times)
        self.assertEqual(result[1], expected_labels)

        expected_times = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75]
        expected_labels = ['a', 'a', 'b', 'b', 'c', 'c']
        result = util.intervals_to_samples(
            intervals, labels, offset=0.25, sample_size=0.5, fill_value='N')
        self.assertEqual(result[0], expected_times)
        self.assertEqual(result[1], expected_labels)

    def test_intersect_files(self):
        """Check that two non-identical yield correct results.
        """
        flist1 = ['/a/b/abc.lab', '/c/d/123.lab', '/e/f/xyz.lab']
        flist2 = ['/g/h/xyz.npy', '/i/j/123.txt', '/k/l/456.lab']
        sublist1, sublist2 = util.intersect_files(flist1, flist2)
        self.assertEqual(sublist1, ['/e/f/xyz.lab', '/c/d/123.lab'])
        self.assertEqual(sublist2, ['/g/h/xyz.npy', '/i/j/123.txt'])
        sublist1, sublist2 = util.intersect_files(flist1[:1], flist2[:1])
        self.assertEqual(sublist1, [])
        self.assertEqual(sublist2, [])

    def test_merge_labeled_intervals(self):
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

        self.assertEqual(new_x_labels, expected_x_labels)
        self.assertEqual(new_y_labels, expected_y_labels)
        self.assertEqual(new_intvs.tolist(), expected_intvs)

        # Check that invalid inputs raise a ValueError
        y_intvs[-1, -1] = 10.0
        self.assertRaises(
            ValueError,
            util.merge_labeled_intervals,
            x_intvs, x_labels, y_intvs, y_labels)


if __name__ == "__main__":
    unittest.main()
