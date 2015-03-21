#!/usr/bin/env python
'''
CREATED:2014-01-17 16:30:07 by Brian McFee <brm2132@columbia.edu>

Compute segmentation evaluation metrics

Usage:

./segment_eval.py TRUTH.TXT PREDICTION.TXT
'''

from __future__ import print_function
import argparse
import sys
import os
import eval_utilities

import mir_eval


def process_arguments():
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(description='mir_eval segmentation '
                                                 'evaluation')

    parser.add_argument('-t',
                        '--trim',
                        dest='trim',
                        default=False,
                        action='store_true',
                        help='Trim beginning and end markers from boundary '
                             'evaluation')

    parser.add_argument('-o',
                        dest='output_file',
                        default=None,
                        type=str,
                        action='store',
                        help='Store results in json format')

    parser.add_argument('reference_file',
                        action='store',
                        help='path to the reference annotation')

    parser.add_argument('estimated_file',
                        action='store',
                        help='path to the estimated annotation')

    return vars(parser.parse_args(sys.argv[1:]))


if __name__ == '__main__':
    # Get the parameters
    parameters = process_arguments()

    # load the data
    ref_file = parameters['reference_file']
    est_file = parameters['estimated_file']
    ref_intervals, ref_labels = mir_eval.io.load_labeled_intervals(ref_file)
    est_intervals, est_labels = mir_eval.io.load_labeled_intervals(est_file)

    # Compute all the scores
    scores = mir_eval.segment.evaluate(ref_intervals, ref_labels,
                                       est_intervals, est_labels,
                                       trim=parameters['trim'])
    print("{} vs. {}".format(os.path.basename(parameters['reference_file']),
                             os.path.basename(parameters['estimated_file'])))
    eval_utilities.print_evaluation(scores)

    if parameters['output_file']:
        print('Saving results to: ', parameters['output_file'])
        eval_utilities.save_results(scores, parameters['output_file'])
