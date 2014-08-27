#!/usr/bin/env python
'''
Utility script for computing source separation metrics

Usage:

./separation_eval.py PATH_TO_REFERENCE_WAVS PATH_TO_ESTIMATED_WAVS
'''

import argparse
import sys
import os
import glob
import os
import numpy as np
import eval_utilities

import mir_eval


def process_arguments():
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(description='mir_eval source separation '
                                                 'evaluation')

    parser.add_argument('-o',
                        dest='output_file',
                        default=None,
                        type=str,
                        action='store',
                        help='Store results in json format')

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

    reference_data = []
    estimated_data = []
    global_fs = None
    reference_glob = os.path.join(parameters['reference_directory'], '*.wav')
    # Load in each reference file in the supplied dir
    for reference_file in glob.glob(reference_glob):
        audio_data, fs = mir_eval.io.load_wav(reference_file)
        # Make sure fs is the same for all files
        assert (global_fs is None or fs == global_fs)
        global_fs = fs
        reference_data.append(audio_data)

    estimated_glob = os.path.join(parameters['estimated_directory'], '*.wav')
    for estimated_file in glob.glob(estimated_glob):
        audio_data, fs = mir_eval.io.load_wav(estimated_file)
        assert (global_fs is None or fs == global_fs)
        global_fs = fs
        estimated_data.append(audio_data)

    # Turn list of audio data arrays into nsrc x nsample arrays
    reference_sources = np.vstack(reference_data)
    estimated_sources = np.vstack(estimated_data)

    # Compute all the scores
    scores = mir_eval.separation.evaluate(reference_sources, estimated_sources)
    print os.path.basename(parameters['estimated_directory'])
    eval_utilities.print_evaluation(scores)

    if parameters['output_file']:
        print 'Saving results to: ', parameters['output_file']
        eval_utilities.save_results(scores, parameters['output_file'])
