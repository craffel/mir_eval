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
from collections import OrderedDict

import mir_eval

def evaluate(ref_file=None, prediction_file=None):
    '''Load data and perform the evaluation'''

    # load the data
    ref_intervals, ref_labels   = mir_eval.io.load_annotation(ref_file)
    est_intervals, est_labels   = mir_eval.io.load_annotation(prediction_file)

    # Now compute all the metrics
    
    M = OrderedDict()
    # Boundary detection
    M['P@0.5'], M['R@0.5'], M['F@0.5']  = mir_eval.segment.boundary_detection(ref_intervals,
                                                                  est_intervals,
                                                                  window=0.5)

    M['P@3.0'], M['R@3.0'], M['F@3.0']  = mir_eval.segment.boundary_detection(ref_intervals,
                                                                 est_intervals,
                                                                 window=3.0)
    # Boundary deviation
    M['True-to-Pred'], M['Pred-to-True'] = mir_eval.segment.boundary_deviation(ref_intervals,
                                                                  est_intervals)

    # Pairwise clustering
#     M['Pair-P'], M['Pair-R'], M['Pair-F'] = mir_eval.segment.frame_clustering_pairwise(reference_boundaries,
#                                                                          estimated_boundaries)

    # Adjusted rand index
#     M['ARI']                     = mir_eval.segment.frame_clustering_ari(reference_boundaries,
#                                                                     estimated_boundaries)

    # Mutual information metrics
#     M['MI'], M['AMI'], M['NMI']            = mir_eval.segment.frame_clustering_mi(reference_boundaries,
#                                                                    estimated_boundaries)

    # Conditional entropy metrics
#     M['S_Over'], M['S_Under'], M['S_F'] = mir_eval.segment.frame_clustering_nce(reference_boundaries,
#                                                                     estimated_boundaries)

    return M

def print_evaluation(prediction_file, M):
    # And print them
    print os.path.basename(prediction_file)
    for key, value in M.iteritems():
        print '\t%12s:\t%0.3f' % (key, value)

    pass

def process_arguments():
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(description='mir_eval segmentation evaluation')

    parser.add_argument(    'ref_file',
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

