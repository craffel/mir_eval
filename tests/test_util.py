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


if __name__ == "__main__":
    unittest.main()
