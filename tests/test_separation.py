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


def __unit_test_empty_input(metric):
    if metric == mir_eval.separation.bss_eval_sources:
        args = [np.array([]), np.array([])]
    elif metric == mir_eval.separation.bss_eval_sources_framewise:
        args = [np.array([]), np.array([]), 40, 20]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        # First, test for a warning on empty audio data
        metric(*args)
        assert len(w) == 2
        assert issubclass(w[-1].category, UserWarning)
        assert str(w[-1].message) == ("estimated_sources is empty, "
                                      "should be of size (nsrc, nsample).  "
                                      "sdr, sir, sar, and perm will all be "
                                      "empty np.ndarrays")
        # And that the metric returns empty arrays
        assert np.allclose(metric(*args), np.array([]))


def __unit_test_silent_input(metric):
    # Test for error when there is a silent reference/estimated source
    ref_sources = np.vstack((np.zeros(100),
                             np.random.random_sample((2, 100))))
    est_sources = np.vstack((np.zeros(100),
                             np.random.random_sample((2, 100))))
    if metric == mir_eval.separation.bss_eval_sources:
        nose.tools.assert_raises(ValueError, metric, ref_sources[:2],
                                 est_sources[1:])
        nose.tools.assert_raises(ValueError, metric, ref_sources[1:],
                                 est_sources[:2])
    elif metric == mir_eval.separation.bss_eval_sources_framewise:
        nose.tools.assert_raises(ValueError, metric, ref_sources[:2],
                                 est_sources[1:], 40, 20)
        nose.tools.assert_raises(ValueError, metric, ref_sources[1:],
                                 est_sources[:2], 40, 20)


def __unit_test_incompatible_shapes(metric):
    # Test for error when shape is different
    sources_4 = np.random.random_sample((4, 100))
    sources_3 = np.random.random_sample((3, 100))
    if metric == mir_eval.separation.bss_eval_sources:
        args1 = [sources_3, sources_4]
        args2 = [sources_4, sources_3]
    elif metric == mir_eval.separation.bss_eval_sources_framewise:
        args1 = [sources_3, sources_4, 40, 20]
        args2 = [sources_4, sources_3, 40, 20]
    nose.tools.assert_raises(ValueError, metric, *args1)
    nose.tools.assert_raises(ValueError, metric, *args2)


def __unit_test_too_many_sources(metric):
    # Test for error when too many sources or references are provided
    many_sources = np.random.random_sample((mir_eval.separation.MAX_SOURCES*2,
                                            400))
    if metric == mir_eval.separation.bss_eval_sources:
        nose.tools.assert_raises(ValueError, metric, many_sources,
                                 many_sources)
    elif metric == mir_eval.separation.bss_eval_sources_framewise:
        nose.tools.assert_raises(ValueError, metric, many_sources,
                                 many_sources, 40, 20)


def __unit_test_default_permutation(metric):
    # Test for default permutation matrix when not computing permutation
    ref_sources = np.random.random_sample((4, 100))
    est_sources = np.random.random_sample((4, 100))
    results = metric(ref_sources, est_sources, compute_permutation=False)
    assert np.array_equal(results[-1], np.asarray([0, 1, 2, 3]))


def __unit_test_invalid_window(metric):
    # Test for invalid win/hop parameter detection
    ref_sources = np.random.random_sample((4, 100))
    est_sources = np.random.random_sample((4, 100))
    nose.tools.assert_raises(
        ValueError, metric, ref_sources, est_sources, 120, 20
    )  # test with window larger than source lengths
    nose.tools.assert_raises(
        ValueError, metric, ref_sources, est_sources, 20, 120
    )  # test with hop larger than source length


def __check_score(sco_f, metric, score, expected_score):
    assert np.allclose(score, expected_score, atol=A_TOL)


def test_separation_functions():
    # Load in all files in the same order
    ref_files = sorted(glob.glob(REF_GLOB))
    est_files = sorted(glob.glob(EST_GLOB))
    sco_files = sorted(glob.glob(SCORES_GLOB))

    assert len(ref_files) == len(est_files) == len(sco_files) > 0

    # Unit tests
    for metric in [mir_eval.separation.bss_eval_sources,
                   mir_eval.separation.bss_eval_sources_framewise]:
        yield (__unit_test_empty_input, metric)
        yield (__unit_test_silent_input, metric)
        yield (__unit_test_incompatible_shapes, metric)
        yield (__unit_test_too_many_sources, metric)
    for metric in [mir_eval.separation.bss_eval_sources]:
        yield (__unit_test_default_permutation, metric)
    for metric in [mir_eval.separation.bss_eval_sources_framewise]:
        yield (__unit_test_invalid_window, metric)
    # Regression tests
    for ref_f, est_f, sco_f in zip(ref_files, est_files, sco_files):
        with open(sco_f, 'r') as f:
            expected_results = json.load(f)
            expected_scores = expected_results['Sources']
            expected_frames = expected_results['Framewise']
        # Load in example source separation data
        ref_sources = __load_and_stack_wavs(ref_f)
        est_sources = __load_and_stack_wavs(est_f)
        # Compute scores
        scores = mir_eval.separation.evaluate(ref_sources, est_sources)
        frame_scores = mir_eval.separation.evaluate(
            ref_sources, est_sources, True,
            window=expected_frames['win'], hop=expected_frames['hop']
        )
        # Compare them
        for metric in scores:
            # This is a simple hack to make nosetest's messages more useful
            yield (__check_score, sco_f, metric, scores[metric],
                   expected_scores[metric])
        for metric in frame_scores:
            if metric is not 'win' or metric is not 'hop':
                # This is a simple hack to make nosetest's messages more useful
                yield (__check_score, sco_f, metric,
                       frame_scores[metric], expected_frames[metric])
