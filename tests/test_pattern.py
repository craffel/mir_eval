"""
Some unit tests for the pattern discovery task.

The pickle file with the results ("data/pattern/results.pk") has the
following structure:
    - dictionary with keys:
        -- "mono"   : Results between reference-mono.txt and estimate-mono.txt
        -- "poly"   : Results between reference-poly.txt and estimate-poly.txt
        -- "2-poly" : Results between reference2-poly.txt and
                        estimate2-poly.txt
    - Inside each dictionary element there's a numpy array with the results
        of 21 different metrics as described below:

           idx  |   description
           -------------------------------------
            0   :   nP (number of patterns found)
            1   :   nQ (number of occurrences found)
            2   :   P_est (Establishment precision)
            3   :   R_est (Establishment recall)
            4   :   F_est (Establishment f-measure)
            5   :   P_occ (Occurrence precision at thres=.75)
            6   :   R_occ (Occurrence recall at thres=.75)
            7   :   F_occ (Occurrence f_measure at thres=.75)
            8   :   P_3 (Three-layer precision)
            9   :   R_3 (Three-layer recall)
            10  :   F_3 (Three-layer f-measure)
            11  :   runtime (set to 0, not measured in mir_eval)
            12  :   Fifth Return Time (total time to retrieve the first five
                :        patterns (set to 0, not measured in mir_eval)
            13  :   FFTP_est (First Fife Target Proportion establis. recall)
            14  :   FFP (First Five Precision) three-layer precision
                :        computed over the first five patterns only
            15  :   P_occ (Occurrence precision at thres=.5)
            16  :   R_occ (Occurrence recall at thres=.5)
            17  :   F_occ (Occurrence f_measure at thres=.5)
            18  :   P (Standard precision)
            19  :   R (Standard recall)
            20  :   F (Standard precision)


Written by Oriol Nieto (oriol@nyu.edu), 2014
"""

import pickle
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
        self.results = pickle.load(open("data/pattern/results.pk"))

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
        self.assertAlmostEqual(F, self.results["poly"][20], delta=self.delta)
        self.assertAlmostEqual(P, self.results["poly"][18], delta=self.delta)
        self.assertAlmostEqual(R, self.results["poly"][19], delta=self.delta)

    def test_establishment_FPR(self):
        F, P, R = pattern.establishment_FPR(self.ref_P, self.est_P)
        self.assertAlmostEqual(F, self.results["poly"][4], delta=self.delta)
        self.assertAlmostEqual(P, self.results["poly"][2], delta=self.delta)
        self.assertAlmostEqual(R, self.results["poly"][3], delta=self.delta)

    def test_occurrence_FPR(self):
        F, P, R = pattern.occurrence_FPR(self.ref2_P, self.est2_P, thres=.5)
        self.assertAlmostEqual(F, self.results["2-poly"][17], delta=self.delta)
        self.assertAlmostEqual(P, self.results["2-poly"][15], delta=self.delta)
        self.assertAlmostEqual(R, self.results["2-poly"][16], delta=self.delta)
        F, P, R = pattern.occurrence_FPR(self.ref2_P, self.est2_P, thres=.75)
        self.assertAlmostEqual(F, self.results["2-poly"][7], delta=self.delta)
        self.assertAlmostEqual(P, self.results["2-poly"][5], delta=self.delta)
        self.assertAlmostEqual(R, self.results["2-poly"][6], delta=self.delta)

    def test_three_layer_FPR(self):
        F, P, R = pattern.three_layer_FPR(self.ref_P, self.est_P)
        self.assertAlmostEqual(F, self.results["poly"][10], delta=self.delta)
        self.assertAlmostEqual(P, self.results["poly"][8], delta=self.delta)
        self.assertAlmostEqual(R, self.results["poly"][9], delta=self.delta)

    def test_first_five_three_layer_P(self):
        P = pattern.first_n_three_layer_P(self.ref_P, self.est_P, n=5)
        self.assertAlmostEqual(P, self.results["poly"][14], delta=self.delta)
        P = pattern.first_n_three_layer_P(self.ref2_P, self.est2_P, n=5)
        self.assertAlmostEqual(P, self.results["2-poly"][14], delta=self.delta)

    def test_first_five_target_proportion(self):
        R = pattern.first_n_target_proportion_R(self.ref_P, self.est_P, n=5)
        self.assertAlmostEqual(R, self.results["poly"][13], delta=self.delta)
        R = pattern.first_n_target_proportion_R(self.ref2_P, self.est2_P, n=5)
        self.assertAlmostEqual(R, self.results["2-poly"][13], delta=self.delta)

if __name__ == "__main__":
    unittest.main()
