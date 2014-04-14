"""
Some unit tests for the pattern discovery task.

Written by Oriol Nieto (oriol@nyu.edu), 2014
"""

import unittest
from mir_eval import pattern, input_output


class PatternTests(unittest.TestCase):

    def setUp(self):
        self.ref_P = input_output.load_patterns(
            "data/pattern/reference-poly.txt")
        self.est_P = input_output.load_patterns(
            "data/pattern/estimate-poly.txt")
        self.ref2_P = input_output.load_patterns(
            "data/pattern/reference2-poly.txt")
        self.est2_P = input_output.load_patterns(
            "data/pattern/estimate2-poly.txt")
        self.delta = 1e-4

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

    def test_standard_FPR(self):
        F, P, R = pattern.standard_FPR(self.ref_P, self.est_P)
        self.assertAlmostEqual(F, 0.28571, delta=self.delta)
        self.assertAlmostEqual(P, 0.25, delta=self.delta)
        self.assertAlmostEqual(R, 0.33333, delta=self.delta)

    def test_establishment_FPR(self):
        F, P, R = pattern.establishment_FPR(self.ref_P, self.est_P)
        self.assertAlmostEqual(F, 0.25249, delta=self.delta)
        self.assertAlmostEqual(P, 0.25927, delta=self.delta)
        self.assertAlmostEqual(R, 0.24606, delta=self.delta)

    def test_occurrence_FPR(self):
        F, P, R = pattern.occurrence_FPR(self.ref2_P, self.est2_P, thres=.5)
        self.assertAlmostEqual(F, 0.44120, delta=self.delta)
        self.assertAlmostEqual(P, 0.51023, delta=self.delta)
        self.assertAlmostEqual(R, 0.38863, delta=self.delta)
        F, P, R = pattern.occurrence_FPR(self.ref2_P, self.est2_P, thres=.75)
        self.assertAlmostEqual(F, 0.47733, delta=self.delta)
        self.assertAlmostEqual(P, 0.56456, delta=self.delta)
        self.assertAlmostEqual(R, 0.41344, delta=self.delta)

    def test_three_layer_FPR(self):
        F, P, R = pattern.three_layer_FPR(self.ref_P, self.est_P)
        self.assertAlmostEqual(F, 0.10211, delta=self.delta)
        self.assertAlmostEqual(P, 0.09465, delta=self.delta)
        self.assertAlmostEqual(R, 0.11085, delta=self.delta)

if __name__ == "__main__":
    unittest.main()
