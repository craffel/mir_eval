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

def evaluate(truth_file=None, prediction_file=None):
    '''Load data and perform the evaluation'''

    # load the data
    truth_segs, truth_labels    = mir_eval.io.load_annotation(truth_file)
    pred_segs, pred_labels      = mir_eval.io.load_annotation(prediction_file)

    # Convert to boundaries
    annotated_boundaries        = mir_eval.util.segments_to_boundaries(truth_segs)[0]
    predicted_boundaries        = mir_eval.util.segments_to_boundaries(pred_segs)[0]

    # Adjust the predictions to match the annotation time span (0 to end-of-track)

    # The [0] at the end is to just get the boundaries without the labels
    predicted_boundaries        = mir_eval.util.adjust_times(predicted_boundaries, 
                                                        t_min=annotated_boundaries.min(), 
                                                        t_max=annotated_boundaries.max())[0]

    # Now compute all the metrics
    
    M = OrderedDict()
    # Boundary detection
    M['P@0.5'], M['R@0.5'], M['F@0.5']  = mir_eval.segment.boundary_detection(annotated_boundaries,
                                                                  predicted_boundaries,
                                                                  window=0.5)

    M['P@3.0'], M['R@3.0'], M['F@3.0']  = mir_eval.segment.boundary_detection(annotated_boundaries,
                                                                 predicted_boundaries,
                                                                 window=3.0)
    # Boundary deviation
    M['True-to-Pred'], M['Pred-to-True'] = mir_eval.segment.boundary_deviation(annotated_boundaries,
                                                                  predicted_boundaries)

    # Pairwise clustering
    M['Pair-P'], M['Pair-R'], M['Pair-F'] = mir_eval.segment.frame_clustering_pairwise(annotated_boundaries,
                                                                         predicted_boundaries)

    # Adjusted rand index
    M['ARI']                     = mir_eval.segment.frame_clustering_ari(annotated_boundaries,
                                                                    predicted_boundaries)

    # Mutual information metrics
    M['MI'], M['AMI'], M['NMI']            = mir_eval.segment.frame_clustering_mi(annotated_boundaries,
                                                                   predicted_boundaries)

    # Conditional entropy metrics
    M['S_Over'], M['S_Under'], M['S_F'] = mir_eval.segment.frame_clustering_nce(annotated_boundaries,
                                                                    predicted_boundaries)

    # And print them
    print os.path.basename(prediction_file)
    for key, value in M.iteritems():
        print '%20s:\t%0.3f' % (key, value)

    pass

def process_arguments():
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(description='mir_eval segmentation evaluation')

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

    # Run the beat tracker
    evaluate(**parameters)
