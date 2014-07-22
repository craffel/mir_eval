#!/usr/bin/env python
'''
Utility script for computing source separation metrics

Usage:

./separation_eval.py PATH_TO_REFERENCE_WAVS PATH_TO_ESTIMATED_WAVS
'''

import argparse
import sys
import os
from collections import OrderedDict
import scipy.io.wavfile
import glob
import os
import numpy as np

import mir_eval


def load_wav(path):
    ''' Wrapper around scipy.io.wavfile for reading a wav '''
    fs, audio_data = scipy.io.wavfile.read(path)
    # Make float
    audio_data = audio_data/32768.0
    # Only mono is acceptable
    assert audio_data.ndim == 1
    return audio_data, fs


def evaluate(reference_directory, estimated_directory):
    '''Load data and perform the evaluation'''

    reference_data = []
    estimated_data = []
    global_fs = None
    # Load in each reference file in the supplied dir
    for reference_file in glob.glob(os.path.join(reference_directory,
                                                 '*.wav')):
        audio_data, fs = load_wav(reference_file)
        # Make sure fs is the same for all files
        assert (global_fs is None or fs == global_fs)
        global_fs = fs
        reference_data.append(audio_data)

    for estimated_file in glob.glob(os.path.join(estimated_directory,
                                                 '*.wav')):
        audio_data, fs = load_wav(estimated_file)
        assert (global_fs is None or fs == global_fs)
        global_fs = fs
        estimated_data.append(audio_data)

    # Turn list of audio data arrays into nsrc x nsample arrays
    reference_sources = np.vstack(reference_data)
    estimated_sources = np.vstack(estimated_data)

    # Now compute all the metrics
    scores = OrderedDict()

    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(reference_sources,
                                                            estimated_sources)
    scores['SDR'] = sdr
    scores['SIR'] = sir
    scores['SAR'] = sar

    return scores


def print_evaluation(estimated_file, scores):
    # And print them
    print os.path.basename(estimated_file)
    for key, value in scores.iteritems():
        print '\t{} : {}'.format(key, value)

    pass


def process_arguments():
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(description='mir_eval source separation '
                                                 'evaluation')

    parser.add_argument('reference_directory',
                        action='store',
                        help='path to directory containing reference source '
                               '.wav files')

    parser.add_argument('estimated_directory',
                        action='store',
                        help='path to directory containing estimated source '
                             '.wav files')

    return vars(parser.parse_args(sys.argv[1:]))

if __name__ == '__main__':
    # Get the parameters
    parameters = process_arguments()

    # Compute all the scores
    scores = evaluate(**parameters)
    print_evaluation(parameters['estimated_directory'], scores)
