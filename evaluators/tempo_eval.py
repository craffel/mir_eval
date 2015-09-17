#!/usr/bin/env python
'''
Utility script for computing all tempo metrics.

Usage:

./tempo_eval.py REFERENCE.TXT ESTIMATED.TXT
'''
from __future__ import print_function

import argparse
import sys
import os
import eval_utilities
import numpy as np

import mir_eval


def process_arguments():
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(description='mir_eval tempo detection '
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
    reference_tempi = mir_eval.io.load_delimited(parameters['reference_file'],
                                                 [float]*3)
    estimated_tempi = mir_eval.io.load_delimited(parameters['estimated_file'],
                                                 [float]*3)

    estimated_tempi = np.concatenate(estimated_tempi[:2])
    reference_weight = reference_tempi[-1][0]
    reference_tempi = np.concatenate(reference_tempi[:2])

    # Compute all the scores
    scores = mir_eval.tempo.evaluate(reference_tempi,
                                     reference_weight,
                                     estimated_tempi)
    print("{} vs. {}".format(os.path.basename(parameters['reference_file']),
                             os.path.basename(parameters['estimated_file'])))

    eval_utilities.print_evaluation(scores)

    if parameters['output_file']:
        print('Saving results to: {}'.format(parameters['output_file']))
        eval_utilities.save_results(scores, parameters['output_file'])
