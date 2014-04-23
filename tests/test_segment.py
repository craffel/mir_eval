'''
Unit tests for mir_eval.structure and mir_eval.boundary
'''

import numpy as np
import json
import mir_eval
import glob

# JSON encoding loses some precision, we'll keep it to 1e-5 precision
A_TOL = 1e-5

# Path to the fixture files
REF_GLOB    = 'data/segment/ref*.lab'
EST_GLOB    = 'data/segment/est*.lab'
SCORES_GLOB  = 'data/segment/output*.json'

def generate_data():

    ref_files = sorted(glob.glob(REF_GLOB))
    est_files = sorted(glob.glob(EST_GLOB))
    sco_files = sorted(glob.glob(SCORES_GLOB))

    for ref_f, est_f, sco_f in zip(ref_files, est_files, sco_files):
        ref_t, ref_l = mir_eval.io.load_intervals(ref_f)
        est_t, est_l = mir_eval.io.load_intervals(est_f)

        with open(sco_f, 'r') as f:
            scores = json.load(f)

        yield ref_t, ref_l, est_t, est_l, scores

def test_boundary_detection():

    def __test_detection(_window, _ref_t, _est_t):
        _ref_t = mir_eval.util.adjust_intervals(_ref_t, t_min=0.0)[0]
        _est_t = mir_eval.util.adjust_intervals(_est_t, t_min=0.0, t_max=_ref_t.max())[0]
        precision, recall, fmeasure = mir_eval.boundary.detection(_ref_t, _est_t, window=_window)

        print precision, recall, fmeasure
        print scores['P@%0.1f'%_window], scores['R@%0.1f'%_window], scores['F@%0.1f'%_window]

        assert np.allclose(precision,   scores['P@%0.1f'%_window], atol=A_TOL)
        assert np.allclose(recall,      scores['R@%0.1f'%_window], atol=A_TOL)
        assert np.allclose(fmeasure,    scores['F@%0.1f'%_window], atol=A_TOL)

    # Iterate over fixtures
    for ref_t, ref_l, est_t, est_l, scores in generate_data():

        # Test boundary detection at each window size
        for window in [0.5, 3.0]:
            yield (__test_detection, window, ref_t, est_t)

    # Done
    pass

def test_boundary_deviation():
    def __test_deviation(_ref_t, _est_t):
        _ref_t = mir_eval.util.adjust_intervals(_ref_t, t_min=0.0)[0]
        _est_t = mir_eval.util.adjust_intervals(_est_t, t_min=0.0, t_max=_ref_t.max())[0]
        t_to_p, p_to_t = mir_eval.boundary.deviation(_ref_t, _est_t)

        print t_to_p, p_to_t
        print scores['True-to-Pred'], scores['Pred-to-True']

        assert np.allclose(t_to_p,  scores['True-to-Pred'], atol=A_TOL)
        assert np.allclose(p_to_t,  scores['Pred-to-True'], atol=A_TOL)

    # Iterate over fixtures
    for ref_t, ref_l, est_t, est_l, scores in generate_data():
        # Test boundary deviation
        yield (__test_deviation, ref_t, est_t)

    # Done
    pass

def test_structure_pairwise():

    def __test_pairwise(_ref_t, _ref_l, _est_t, _est_l):
        _ref_t, _ref_l = mir_eval.util.adjust_intervals(_ref_t, labels=_ref_l, t_min=0.0)
        _est_t, _est_l = mir_eval.util.adjust_intervals(_est_t, labels=_est_l, t_min=0.0, t_max=_ref_t.max())

        precision, recall, fmeasure = mir_eval.structure.pairwise(_ref_t, _ref_l, _est_t, _est_l)

        print precision, recall, fmeasure
        print scores['Pair-P'], scores['Pair-R'], scores['Pair-F']

        assert np.allclose(precision,   scores['Pair-P'], atol=A_TOL)
        assert np.allclose(recall,      scores['Pair-R'], atol=A_TOL)
        assert np.allclose(fmeasure,    scores['Pair-F'], atol=A_TOL)

    # Iterate over fixtures
    for ref_t, ref_l, est_t, est_l, scores in generate_data():

        yield (__test_pairwise, ref_t, ref_l, est_t, est_l)

    # Done
    pass

def test_structure_rand():
    def __test_rand(_ref_t, _ref_l, _est_t, _est_l):
        _ref_t, _ref_l = mir_eval.util.adjust_intervals(_ref_t, labels=_ref_l, t_min=0.0)
        _est_t, _est_l = mir_eval.util.adjust_intervals(_est_t, labels=_est_l, t_min=0.0, t_max=_ref_t.max())

        ari = mir_eval.structure.ari(_ref_t, _ref_l, _est_t, _est_l)

        print ari
        print scores['ARI']

        assert np.allclose(ari,  scores['ARI'], atol=A_TOL)

    # Iterate over fixtures
    for ref_t, ref_l, est_t, est_l, scores in generate_data():
        yield (__test_rand, ref_t, ref_l, est_t, est_l)

    # Done
    pass

def test_structure_mutual_information():
    def __test_mutual_information(_ref_t, _ref_l, _est_t, _est_l):
        _ref_t, _ref_l = mir_eval.util.adjust_intervals(_ref_t, labels=_ref_l, t_min=0.0)
        _est_t, _est_l = mir_eval.util.adjust_intervals(_est_t, labels=_est_l, t_min=0.0, t_max=_ref_t.max())

        mi, ami, nmi = mir_eval.structure.mutual_information(_ref_t, _ref_l, _est_t, _est_l)

        print mi, ami, nmi
        print scores['MI'], scores['AMI'], scores['NMI']

        assert np.allclose(mi,  scores['MI'], atol=A_TOL)
        assert np.allclose(ami, scores['AMI'], atol=A_TOL)
        assert np.allclose(nmi, scores['NMI'], atol=A_TOL)

    # Iterate over fixtures
    for ref_t, ref_l, est_t, est_l, scores in generate_data():
        yield (__test_mutual_information, ref_t, ref_l, est_t, est_l)

    # Done
    pass


def test_structure_entropy():
    def __test_entropy(_ref_t, _ref_l, _est_t, _est_l):
        _ref_t, _ref_l = mir_eval.util.adjust_intervals(_ref_t, labels=_ref_l, t_min=0.0)
        _est_t, _est_l = mir_eval.util.adjust_intervals(_est_t, labels=_est_l, t_min=0.0, t_max=_ref_t.max())

        s_over, s_under, s_f = mir_eval.structure.nce(_ref_t, _ref_l, _est_t, _est_l)

        print s_over, s_under, s_f
        print scores['S_Over'], scores['S_Under'], scores['S_F']

        assert np.allclose(s_over,  scores['S_Over'], atol=A_TOL)
        assert np.allclose(s_under, scores['S_Under'], atol=A_TOL)
        assert np.allclose(s_f, scores['S_F'], atol=A_TOL)

    # Iterate over fixtures
    for ref_t, ref_l, est_t, est_l, scores in generate_data():
        yield (__test_entropy, ref_t, ref_l, est_t, est_l)

    # Done
    pass
