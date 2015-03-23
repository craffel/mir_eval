'''
unit tests for mir_eval.separation

load randomly generated source and estimated source signals and
the output from BSS_eval MATLAB implementation, make sure the results
from mir_eval numerically match.
'''

import numpy as np
import mir_eval
import glob
import nose.tools
import json
import os
import warnings

A_TOL = 1e-12

REF_GLOB = 'tests/data/separation/ref*'
EST_GLOB = 'tests/data/separation/est*'
SCORES_GLOB = 'tests/data/separation/output*.json'


def __load_and_stack_wavs(directory):
    ''' Load all wavs in a directory and stack them vertically into a matrix
    '''
    stacked_audio_data = []
    global_fs = None
    for f in sorted(glob.glob(os.path.join(directory, '*.wav'))):
        audio_data, fs = mir_eval.io.load_wav(f)
        assert (global_fs is None or fs == global_fs)
        global_fs = fs
        stacked_audio_data.append(audio_data)
    return np.vstack(stacked_audio_data)


def __unit_test_separation_function(metric):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        # First, test for a warning on empty audio data
        metric(np.array([]), np.array([]))
        assert len(w) == 2
        assert issubclass(w[-1].category, UserWarning)
        assert str(w[-1].message) == ("estimated_sources is empty, "
                                      "should be of size (nsrc, nsample).  "
                                      "sdr, sir, sar, and perm will all be "
                                      "empty np.ndarrays")
        # And that the metric returns empty arrays
        assert np.allclose(metric(np.array([]), np.array([])), np.array([]))

    # Test for error when there is a silent reference/estimated source
    ref_sources = np.vstack((np.zeros(100),
                             np.random.random_sample((2, 100))))
    est_sources = np.vstack((np.zeros(100),
                             np.random.random_sample((2, 100))))
    nose.tools.assert_raises(ValueError, metric, ref_sources[:2],
                             est_sources[1:])
    nose.tools.assert_raises(ValueError, metric, ref_sources[1:],
                             est_sources[:2])

    # Test for error when shape is different
    ref_sources = np.random.random_sample((4, 100))
    est_sources = np.random.random_sample((3, 100))
    nose.tools.assert_raises(ValueError, metric, ref_sources, est_sources)

    # Test for error when too many sources are provided
    sources = np.random.random_sample((mir_eval.separation.MAX_SOURCES*2, 400))
    nose.tools.assert_raises(ValueError, metric, sources, sources)


def __check_score(sco_f, metric, score, expected_score):
    assert np.allclose(score, expected_score, atol=A_TOL)


def test_separation_functions():
    # Load in all files in the same order
    ref_files = sorted(glob.glob(REF_GLOB))
    est_files = sorted(glob.glob(EST_GLOB))
    sco_files = sorted(glob.glob(SCORES_GLOB))

    assert len(ref_files) == len(est_files) == len(sco_files) > 0

    # Unit tests
    for metric in [mir_eval.separation.bss_eval_sources]:
        yield (__unit_test_separation_function, metric)
    # Regression tests
    for ref_f, est_f, sco_f in zip(ref_files, est_files, sco_files):
        with open(sco_f, 'r') as f:
            expected_scores = json.load(f)
        # Load in example source separation data
        ref_sources = __load_and_stack_wavs(ref_f)
        est_sources = __load_and_stack_wavs(est_f)
        # Compute scores
        scores = mir_eval.separation.evaluate(ref_sources, est_sources)
        # Compare them
        for metric in scores:
            # This is a simple hack to make nosetest's messages more useful
            yield (__check_score, sco_f, metric, scores[metric],
                   expected_scores[metric])
