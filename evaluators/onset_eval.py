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

import mir_eval


def evaluate(reference_file, estimated_file):
    '''Load data and perform the evaluation'''

    # load the data
    reference_onsets, _ = mir_eval.io.load_events(reference_file)
    estimated_onsets, _ = mir_eval.io.load_events(estimated_file)

    # Now compute all the metrics
    scores = OrderedDict()

    f_measure, precision, recall = mir_eval.onset.f_measure(reference_onsets,
                                                            estimated_onsets)
    scores['F-measure'] = f_measure
    scores['Precision'] = precision
    scores['Recall'] = recall

    return scores


def print_evaluation(estimated_file, scores):
    # And print them
    print os.path.basename(estimated_file)
    for key, value in scores.iteritems():
        print '\t%11s:\t%0.3f' % (key, value)

    pass


def process_arguments():
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(description='mir_eval onset detection '
                                                 'evaluation')

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
    scores = evaluate(**parameters)
    print_evaluation(parameters['estimated_file'], scores)
