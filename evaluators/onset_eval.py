#!/usr/bin/env python
'''
Utility script for computing all onset metrics.

Usage:

./onset_eval.py REFERENCE.TXT ESTIMATED.TXT
'''

import argparse
import sys
import os
from collections import OrderedDict
import eval_utilities

import mir_eval


def evaluate(reference_file, estimated_file):
    '''Load data and perform the evaluation'''

    # load the data
    reference_onsets = mir_eval.io.load_events(reference_file)
    estimated_onsets = mir_eval.io.load_events(estimated_file)

    # Now compute all the metrics
    scores = OrderedDict()

    f_measure, precision, recall = mir_eval.onset.f_measure(reference_onsets,
                                                            estimated_onsets)
    scores['F-measure'] = f_measure
    scores['Precision'] = precision
    scores['Recall'] = recall

    return scores


def process_arguments():
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(description='mir_eval onset detection '
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

    # Compute all the scores
    scores = evaluate(parameters['reference_file'],
                      parameters['estimated_file'])
    print os.path.basename(parameters['estimated_file'])
    eval_utilities.print_evaluation(scores)

    if parameters['output_file']:
        print 'Saving results to: ', parameters['output_file']
        eval_utilities.save_results(scores, parameters['output_file'])
