#!/usr/bin/env python
'''
Utility script for computing all multipitch metrics.

Usage:

./multipitch_eval.py REFERENCE.TXT ESTIMATED.TXT
'''

from __future__ import print_function
import argparse
import sys
import os
import eval_utilities

import mir_eval


def process_arguments():
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(description='mir_eval multipitch detection '
                                                 'evaluation')

    parser.add_argument('-o',
                        dest='output_file',
                        default=None,
                        type=str,
                        action='store',
                        help='Store results in json format')

    parser.add_argument('reference_file',
                        action='store',
                        help='path to the reference annotation file')

    parser.add_argument('estimated_file',
                        action='store',
                        help='path to the estimated annotation file')

    return vars(parser.parse_args(sys.argv[1:]))


if __name__ == '__main__':
    # Get the parameters
    parameters = process_arguments()

    # Load in data
    ref_times, ref_freqs = mir_eval.io.load_ragged_time_series(
        parameters['reference_file'])
    est_times, est_freqs = mir_eval.io.load_ragged_time_series(
        parameters['estimated_file'])

    # Compute all the scores
    scores = mir_eval.multipitch.evaluate(
        ref_times, ref_freqs, est_times, est_freqs)
    print("{} vs. {}".format(os.path.basename(parameters['reference_file']),
                             os.path.basename(parameters['estimated_file'])))
    eval_utilities.print_evaluation(scores)

    if parameters['output_file']:
        print('Saving results to: ', parameters['output_file'])
        eval_utilities.save_results(scores, parameters['output_file'])
