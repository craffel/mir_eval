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


if __name__ == "__main__":
    unittest.main()
