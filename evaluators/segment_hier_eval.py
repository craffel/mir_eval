#!/usr/bin/env python
'''
CREATED:2014-01-17 16:30:07 by Brian McFee <brm2132@columbia.edu>

Compute hierarchical segmentation evaluation metrics

Usage:

./segment_hier_eval.py -r TRUTH_LEVEL1.TXT [TRUTH_LEVEL2.TXT ...] \
                       -e PREDICTION_LEVEL1.TXT [PREDICTION_LEVEL2.TXT ...] \
                       [-o output.json] \
                       [-w WINDOW_SIZE]
'''

from __future__ import print_function
import argparse
import sys
import eval_utilities

import mir_eval
from os.path import basename

from mir_eval.io import load_labeled_intervals


def process_arguments():
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(description='mir_eval hierarchical '
                                     'segmentation evaluation')

    parser.add_argument('-w',
                        '--window',
                        dest='window',
                        default="15.0",
                        type=float,
                        help='Window length for t-measures')

    parser.add_argument('-o',
                        dest='output_file',
                        default=None,
                        type=str,
                        action='store',
                        help='Store results in json format')

    parser.add_argument('-r',
                        '--reference',
                        dest='reference_file',
                        nargs='+',
                        type=str,
                        action='store',
                        help='path to the reference annotation(s) in '
                        '.lab format, ordered from top to bottom of '
                        'the hierarchy')

    parser.add_argument('-e',
                        '--estimate',
                        dest='estimated_file',
                        nargs='+',
                        type=str,
                        action='store',
                        help='path to the estimated annotation(s) in '
                        '.lab format, ordered from top to bottom of '
                        'the hierarchy')

    return vars(parser.parse_args(sys.argv[1:]))


if __name__ == '__main__':
    # Get the parameters
    parameters = process_arguments()

    # load the data
    ref_files = parameters['reference_file']
    est_files = parameters['estimated_file']

    ref = [load_labeled_intervals(_) for _ in ref_files]
    est = [load_labeled_intervals(_) for _ in est_files]
    ref_intervals = [seg[0] for seg in ref]
    ref_labels = [seg[1] for seg in ref]
    est_intervals = [seg[0] for seg in est]
    est_labels = [seg[1] for seg in est]

    # Compute all the scores
    scores = mir_eval.hierarchy.evaluate(ref_intervals, ref_labels,
                                         est_intervals, est_labels,
                                         window=parameters['window'])
    print("{} [...] vs. {} [...]".format(
        basename(parameters['reference_file'][0]),
        basename(parameters['estimated_file'][0])))
    eval_utilities.print_evaluation(scores)

    if parameters['output_file']:
        print('Saving results to: ', parameters['output_file'])
        eval_utilities.save_results(scores, parameters['output_file'])
