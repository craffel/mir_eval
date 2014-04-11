"""
"""

import unittest
import numpy as np
from mir_eval import pattern, input_output


class PatternTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_load_pattern(self):
        P = input_output.load_patterns("data/pattern/estimate-mono.txt")
        self.assertEqual(len(P), 2)
        self.assertEqual(len(P[0]), 2)
        self.assertEqual(len(P[0][0]), 15)
        self.assertEqual(len(P[0][1]), 22)
        self.assertEqual(len(P[1]), 2)
        self.assertEqual(len(P[1][0]), 19)
        self.assertEqual(len(P[1][1]), 22)


if __name__ == "__main__":
    unittest.main()
