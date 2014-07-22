#!/usr/bin/env python
'''
CREATED:2014-01-17 16:30:07 by Brian McFee <brm2132@columbia.edu>

Compute segmentation evaluation metrics

Usage:

./segment_eval.py TRUTH.TXT PREDICTION.TXT
'''

import argparse
import sys
import os
import json
from collections import OrderedDict

import mir_eval


def evaluate(ref_file=None, est_file=None, trim=False):
    '''Load data and perform the evaluation'''

    # load the data
    ref_intervals, ref_labels = mir_eval.io.load_intervals(ref_file)
    est_intervals, est_labels = mir_eval.io.load_intervals(est_file)

    # Adjust timespan of estimations relative to ground truth
    ref_intervals, ref_labels = \
        mir_eval.util.adjust_intervals(ref_intervals, labels=ref_labels,
                                       t_min=0.0)

    est_intervals, est_labels = \
        mir_eval.util.adjust_intervals(est_intervals, labels=est_labels,
                                       t_min=0.0, t_max=ref_intervals[-1, -1])

    # Now compute all the metrics
    M = OrderedDict()
    # Boundary detection
    M['P@0.5'], M['R@0.5'], M['F@0.5'] = \
        mir_eval.boundary.detection(ref_intervals, est_intervals, window=0.5,
                                    trim=trim)

    M['P@3.0'], M['R@3.0'], M['F@3.0'] = \
        mir_eval.boundary.detection(ref_intervals, est_intervals, window=3.0,
                                    trim=trim)

    # Boundary deviation
    M['True-to-Pred'], M['Pred-to-True'] = \
        mir_eval.boundary.deviation(ref_intervals, est_intervals, trim=trim)

    # Pairwise clustering
    M['Pair-P'], M['Pair-R'], M['Pair-F'] = \
        mir_eval.structure.pairwise(ref_intervals, ref_labels, est_intervals,
                                    est_labels)

    # Rand index
    M['RI'] = mir_eval.structure.rand_index(ref_intervals, ref_labels,
                                            est_intervals, est_labels)
    # Adjusted rand index
    M['ARI'] = mir_eval.structure.ari(ref_intervals, ref_labels, est_intervals,
                                      est_labels)

    # Mutual information metrics
    M['MI'], M['AMI'], M['NMI'] = \
        mir_eval.structure.mutual_information(ref_intervals, ref_labels,
                                              est_intervals, est_labels)

    # Conditional entropy metrics
    M['S_Over'], M['S_Under'], M['S_F'] = \
        mir_eval.structure.nce(ref_intervals, ref_labels, est_intervals,
                               est_labels)

    return M


def save_results(results, output_file):

    with open(output_file, 'w') as f:
        json.dump(results, f)


def print_evaluation(est_file, M):
    # And print them
    print os.path.basename(est_file)
    for key, value in M.iteritems():
        print '\t%12s:\t%0.3f' % (key, value)

    pass


def process_arguments():
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(description='mir_eval segmentation'
                                                 'evaluation')

    parser.add_argument('-t',
                        '--trim',
                        dest='trim',
                        default=False,
                        action='store_true',
                        help='Trim beginning and end markers from boundary'
                             'evaluation')

    parser.add_argument('-o',
                        dest='output_file',
                        default=None,
                        type=str,
                        action='store',
                        help='Store results in json format')

    parser.add_argument('ref_file',
                        action='store',
                        help='path to the reference annotation')

    parser.add_argument('est_file',
                        action='store',
                        help='path to the estimated annotation')

    return vars(parser.parse_args(sys.argv[1:]))


if __name__ == '__main__':
    # Get the parameters
    parameters = process_arguments()

    # Compute all the scores
    scores = evaluate(ref_file=parameters['ref_file'],
                      est_file=parameters['est_file'],
                      trim=parameters['trim'])
    print_evaluation(parameters['est_file'], scores)

    if parameters['output_file']:
        print 'Saving results to: ', parameters['output_file']
        save_results(scores, parameters['output_file'])
