#!/usr/bin/env python
'''
CREATED:2014-01-24 12:42:43 by Brian McFee <brm2132@columbia.edu>

Compute beat evaluation metrics


Usage:

./beat_eval.py REFERENCE.TXT ESTIMATED.TXT
'''

import argparse
import sys
import os
from collections import OrderedDict
import json

import mir_eval


def evaluate(reference_file=None, estimated_file=None):
    '''Load data and perform the evaluation'''

    # load the data
    reference_beats = mir_eval.io.load_events(reference_file)
    estimated_beats = mir_eval.io.load_events(estimated_file)

    # Now compute all the metrics

    M = OrderedDict()

    # F-Measure
    M['F-measure'] = mir_eval.beat.f_measure(reference_beats, estimated_beats)

    # Cemgil
    M['Cemgil'], M['Cemgil Best Metric Level'] = \
        mir_eval.beat.cemgil(reference_beats, estimated_beats)

    # Goto
    # XXX:2014-01-24 12:46:31 by Brian McFee <brm2132@columbia.edu>
    # This metric is deprecated
    # M['Goto'] = mir_eval.beat.goto(reference_beats, estimated_beats)

    # P-Score
    M['P-score'] = mir_eval.beat.p_score(reference_beats, estimated_beats)

    # Continuity metrics
    (M['Correct Metric Level Continuous'], M['Correct Metric Level Total'],
     M['Any Metric Level Continuous'], M['Any Metric Level Total']) = \
        mir_eval.beat.continuity(reference_beats, estimated_beats)

    # Information gain
    M['Information gain'] = mir_eval.beat.information_gain(reference_beats,
                                                           estimated_beats)

    return M


def save_results(results, output_file):
    '''Save a results dict into a json file'''
    with open(output_file, 'w') as f:
        json.dump(results, f)


def print_evaluation(estimated_file, M):
    '''Print out a results dict prettily'''
    print os.path.basename(estimated_file)
    for key, value in M.iteritems():
        print '\t%31s:\t%0.3f' % (key, value)

    pass


def process_arguments():
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(description='mir_eval beat detection '
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
    print_evaluation(parameters['estimated_file'], scores)

    if parameters['output_file']:
        print 'Saving results to: ', parameters['output_file']
        save_results(scores, parameters['output_file'])
