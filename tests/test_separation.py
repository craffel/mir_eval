'''
unit tests for mir_eval.separation

load randomly generated source and estimated source signals and
the output from BSS_eval MATLAB implementation, make sure the results
from mir_eval numerically match.
'''

import numpy as np
import mir_eval
import scipy.io.wavfile
import glob
import nose.tools
import json
import os
import warnings

REF_GLOB = 'data/separation/ref*'
EST_GLOB = 'data/separation/est*'
SCORES_GLOB = 'data/separation/output*.json'


def __load_wav(path):
    ''' Wrapper around scipy.io.wavfile for reading a wav '''
    fs, audio_data = scipy.io.wavfile.read(path)
    # Make float
    audio_data = audio_data/32768.0
    return audio_data.T, fs


def __load_and_stack_wavs(directory):
    ''' Load all wavs in a directory and stack them vertically into a matrix
    '''
    stacked_audio_data = []
    global_fs = None
    for f in glob.glob(os.path.join(directory, '*.wav')):
        audio_data, fs = __load_wav(f)
        assert (global_fs is None or fs == global_fs)
        global_fs = fs
        stacked_audio_data.append(audio_data)
    return np.vstack(stacked_audio_data)


def __unit_test_separation_function(metric):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        # First, test for a warning on empty beats
        metric(np.array([]), np.array([]))
        assert len(w) == 2
        assert issubclass(w[-1].category, UserWarning)
        assert str(w[-1].message) == ("estimated_sources is empty, "
                                      "should be of size (nsrc, nsample).  "
                                      "sdr, sir, sar, and perm will all be "
                                      "empty np.ndarrays")
        # And that the metric returns empty arrays
        assert np.allclose(metric(np.array([]), np.array([])), np.array([]))

    # Test for error when shape is different
    ref_sources = np.random.random_sample((4, 100))
    est_sources = np.random.random_sample((3, 100))
    nose.tools.assert_raises(ValueError, metric, ref_sources, est_sources)


def __regression_test_separation_function(metric, ref_f, est_f, score):
    ref_sources = __load_and_stack_wavs(ref_f)
    est_sources = __load_and_stack_wavs(est_f)
    assert np.allclose(metric(ref_sources, est_sources), score)


def test_separation_functions():
    # Load in all files in the same order
    ref_files = sorted(glob.glob(REF_GLOB))
    est_files = sorted(glob.glob(EST_GLOB))
    sco_files = sorted(glob.glob(SCORES_GLOB))

    # Unit tests
    for metric in mir_eval.separation.METRICS.values():
        yield (__unit_test_separation_function, metric)
    # Regression tests
    for ref_f, est_f, sco_f in zip(ref_files, est_files, sco_files):
        with open(sco_f, 'r') as f:
            scores = json.load(f)
        for name, metric in mir_eval.separation.METRICS.items():
            yield (__regression_test_separation_function, metric,
                   ref_f, est_f, scores[name])
