#!/usr/bin/env python
'''
CREATED:2014-01-24 12:42:43 by Brian McFee <brm2132@columbia.edu>

Compute beat evaluation metrics


Usage:

./beat_eval.py TRUTH.TXT PREDICTION.TXT
'''

import argparse
import sys
import os
from collections import OrderedDict

import mir_eval

def evaluate(truth_file=None, prediction_file=None):
    '''Load data and perform the evaluation'''

    # load the data
    truth_beats, truth_labels    = mir_eval.io.load_events(truth_file)
    pred_beats, pred_labels      = mir_eval.io.load_events(prediction_file)

    # Now compute all the metrics
    
    M = OrderedDict()
    
    # F-Measure
    M['F-meas'] = mir_eval.beat.f_measure(truth_beats, pred_beats)

    # Cemgil
    M['Cemgil'], M['Cemgil-best'] = mir_eval.beat.cemgil(truth_beats, pred_beats)

    # Goto
    # XXX:2014-01-24 12:46:31 by Brian McFee <brm2132@columbia.edu>
    # This metric is deprecated
    # M['Goto'] = mir_eval.beat.goto(truth_beats, pred_beats)

    # P-Score
    M['P-score'] = mir_eval.beat.p_score(truth_beats, pred_beats)

    # Continuity metrics
    M['CMLc'], M['CMLt'], M['AMLc'], M['AMLt'] = mir_eval.beat.continuity(truth_beats, 
                                                                          pred_beats)

    # Information gain
    M['I.gain'] = mir_eval.beat.information_gain(truth_beats, pred_beats)

    return M

def print_evaluation(prediction_file, M):
    # And print them
    print os.path.basename(prediction_file)
    for key, value in M.iteritems():
        print '\t%12s:\t%0.3f' % (key, value)

    pass

def process_arguments():
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(description='mir_eval beat detection evaluation')

    parser.add_argument(    'truth_file',
                            action      =   'store',
                            help        =   'path to the ground truth annotation')

    parser.add_argument(    'prediction_file',
                            action      =   'store',
                            help        =   'path to the prediction file')

    return vars(parser.parse_args(sys.argv[1:]))
   
if __name__ == '__main__':
    # Get the parameters
    parameters = process_arguments()

    # Compute all the scores
    scores = evaluate(**parameters)
    print_evaluation(parameters['prediction_file'], scores)

