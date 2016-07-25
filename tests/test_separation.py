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
        assert global_fs is None or fs == global_fs
        global_fs = fs
        stacked_audio_data.append(audio_data)
    return np.vstack(stacked_audio_data)


def __generate_multichannel(mono_sig, nchan=2, gain=1.0, reverse=False):
    ''' Turn a single channel (ie. mono) audio sample into a multichannel
    (e.g. stereo)
    Note: to achieve channels of silence pass gain=0
    '''
    # add the channels dimension
    input_3d = np.atleast_3d(mono_sig)
    # get the desired number of channels
    stackin = [input_3d]*nchan
    # apply the gain to the new channels
    stackin[1:] = np.multiply(gain, stackin[1:])
    if reverse:
        # reverse the new channels
        stackin[1:] = stackin[1:][:][::-1]
    return np.dstack(stackin)


def __unit_test_empty_input(metric):
    if (metric == mir_eval.separation.bss_eval_sources or
            metric == mir_eval.separation.bss_eval_images):
        args = [np.array([]), np.array([])]
    elif (metric == mir_eval.separation.bss_eval_sources_framewise or
            metric == mir_eval.separation.bss_eval_images_framewise):
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
    if metric == mir_eval.separation.bss_eval_images:
        ref_sources = np.vstack((np.zeros((1, 100, 2)),
                                 np.random.random_sample((2, 100, 2))))
        est_sources = np.vstack((np.zeros((1, 100, 2)),
                                 np.random.random_sample((2, 100, 2))))
    else:
        ref_sources = np.vstack((np.zeros(100),
                                 np.random.random_sample((2, 100))))
        est_sources = np.vstack((np.zeros(100),
                                 np.random.random_sample((2, 100))))
    if (metric == mir_eval.separation.bss_eval_sources or
            metric == mir_eval.separation.bss_eval_images):
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
    if (metric == mir_eval.separation.bss_eval_images or
            metric == mir_eval.separation.bss_eval_images_framewise):
        sources_4 = np.random.random_sample((4, 100, 2))
        sources_3 = np.random.random_sample((3, 100, 2))
        sources_4_chan = np.random.random_sample((4, 100, 3))
    else:
        sources_4 = np.random.random_sample((4, 100))
        sources_3 = np.random.random_sample((3, 100))
    if (metric == mir_eval.separation.bss_eval_sources or
            metric == mir_eval.separation.bss_eval_images):
        args1 = [sources_3, sources_4]
        args2 = [sources_4, sources_3]
    elif (metric == mir_eval.separation.bss_eval_sources_framewise or
            metric == mir_eval.separation.bss_eval_images_framewise):
        args1 = [sources_3, sources_4, 40, 20]
        args2 = [sources_4, sources_3, 40, 20]
    nose.tools.assert_raises(ValueError, metric, *args1)
    nose.tools.assert_raises(ValueError, metric, *args2)
    if metric == mir_eval.separation.bss_eval_images:
        nose.tools.assert_raises(ValueError, metric, sources_4, sources_4_chan)


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


def __unit_test_too_many_dimensions(metric):
    # Test for detection of too high dimensioned images
    ref_sources = np.random.random_sample((4, 100, 2, 3))
    est_sources = np.random.random_sample((4, 100, 2, 3))
    nose.tools.assert_raises(ValueError, metric, ref_sources, est_sources)


def __unit_test_default_permutation(metric):
    # Test for default permutation matrix when not computing permutation
    if metric == mir_eval.separation.bss_eval_sources:
        ref_sources = np.random.random_sample((4, 100))
        est_sources = np.random.random_sample((4, 100))
    elif metric == mir_eval.separation.bss_eval_images:
        ref_sources = np.random.random_sample((4, 100, 2))
        est_sources = np.random.random_sample((4, 100, 2))
    results = metric(ref_sources, est_sources, compute_permutation=False)
    assert np.array_equal(results[-1], np.asarray([0, 1, 2, 3]))


def __unit_test_framewise_small_window(metric):
    # Test for invalid win/hop parameter detection
    if metric == mir_eval.separation.bss_eval_sources_framewise:
        ref_sources = np.random.random_sample((4, 100))
        est_sources = np.random.random_sample((4, 100))
        # Test with window larger than source length
        assert np.allclose(metric(ref_sources, est_sources, window=120, hop=20),
                           mir_eval.separation.bss_eval_sources(ref_sources,
                                                                est_sources,
                                                                False),
                           atol=A_TOL)
        # Test with hop larger than source length
        assert np.allclose(metric(ref_sources, est_sources, window=20, hop=120),
                           mir_eval.separation.bss_eval_sources(ref_sources,
                                                                est_sources,
                                                                False),
                           atol=A_TOL)
    elif metric == mir_eval.separation.bss_eval_images_framewise:
        ref_images = np.random.random_sample((4, 100, 2))
        est_images = np.random.random_sample((4, 100, 2))
        # Test with window larger than source length
        assert np.allclose(metric(ref_images, est_images, window=120, hop=20),
                           mir_eval.separation.bss_eval_images(ref_images,
                                                               est_images,
                                                               False),
                           atol=A_TOL)
        # Test with hop larger than source length
        assert np.allclose(metric(ref_images, est_images, window=20, hop=120),
                           mir_eval.separation.bss_eval_images(ref_images,
                                                               est_images,
                                                               False),
                           atol=A_TOL)


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
                   mir_eval.separation.bss_eval_sources_framewise,
                   mir_eval.separation.bss_eval_images,
                   mir_eval.separation.bss_eval_images_framewise]:
        yield (__unit_test_empty_input, metric)
        yield (__unit_test_silent_input, metric)
        yield (__unit_test_incompatible_shapes, metric)
        yield (__unit_test_too_many_sources, metric)
        yield (__unit_test_too_many_dimensions, metric)
    for metric in [mir_eval.separation.bss_eval_sources,
                   mir_eval.separation.bss_eval_images]:
        yield (__unit_test_default_permutation, metric)
    for metric in [mir_eval.separation.bss_eval_sources_framewise,
                   mir_eval.separation.bss_eval_images_framewise]:
        yield (__unit_test_framewise_small_window, metric)
    # Regression tests
    for ref_f, est_f, sco_f in zip(ref_files, est_files, sco_files):
        with open(sco_f, 'r') as f:
            expected_results = json.load(f)
            expected_sources = expected_results['Sources']
            expected_frames = expected_results['Framewise']
            expected_images = expected_results['Images']
        # Load in example source separation data
        ref_sources = __load_and_stack_wavs(ref_f)
        est_sources = __load_and_stack_wavs(est_f)
        # Test inference for single source passed as single dimensional array
        if ref_sources.shape[0] == 1 and est_sources.shape[0] == 1:
            ref_sources = ref_sources[0]
            est_sources = est_sources[0]

        # Compute scores
        scores = mir_eval.separation.evaluate(
            ref_sources, est_sources,
            window=expected_frames['win'], hop=expected_frames['hop']
        )
        # Compare them
        for metric in scores:
            if 'Sources - ' in metric:
                test_data_name = metric.replace('Sources - ', '')
                # This is a simple hack to make nosetest's messages more useful
                yield (__check_score, sco_f, metric, scores[metric],
                       expected_sources[test_data_name])
            elif 'Sources Frames - ' in metric:
                test_data_name = metric.replace('Sources Frames - ', '')
                # This is a simple hack to make nosetest's messages more useful
                yield (__check_score, sco_f, metric, scores[metric],
                       expected_frames[test_data_name])

        # Compute scores with images
        ref_images = __generate_multichannel(ref_sources,
                                             expected_images['nchan'])
        est_images = __generate_multichannel(est_sources,
                                             expected_images['nchan'],
                                             expected_images['gain'],
                                             expected_images['reverse'])
        image_scores = mir_eval.separation.evaluate(
            ref_images, est_images
        )
        # Compare them
        for metric in image_scores:
            if 'Images - ' in metric:
                test_data_name = metric.replace('Images - ', '')
                # This is a simple hack to make nosetest's messages more useful
                yield (__check_score, sco_f, metric, image_scores[metric],
                       expected_images[test_data_name])
                       
    # Catch a few exceptions in the evaluate function
    image_scores = mir_eval.separation.evaluate(ref_images, est_images)
    # make sure sources is not being evaluated on images
    assert 'Sources - Source to Distortion' not in image_scores
