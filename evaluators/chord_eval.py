#!/usr/bin/env python
'''
Compute chord evaluation metrics


Usage:

./chord_eval.py REFERENCE.TXT ESTIMATION.TXT
'''

import argparse
import glob
import os
import sys

from collections import OrderedDict

import mir_eval.chord as chord
from mir_eval import io
import mir_eval.util


def evaluate_set(reference_dir, estimation_dir, fext='lab'):
    '''Compute evaluation over sets of files.

    :parameters:
    - reference_dir : str
        Path to a directory of reference files.
    - estimation_dir : str
        Path to a directory of estimation files.
    - fext : str
        File extension for annotations.
    '''
    ref_files = glob(os.path.join(reference_dir, "*.%s" % fext))
    est_files = glob(os.path.join(estimation_dir, "*.%s" % fext))
    ref_files, est_files = mir_eval.util.intersect_files(ref_files, est_files)

    results = []
    for ref_file, est_file in zip(ref_files, est_files):
        results.append(evaluate_pair(ref_file, est_file))

    return results


def evaluate_pair(reference_file, estimation_file, ):
    '''Load data and perform the evaluation'''

    # load the data
    ref_intervals, ref_labels = io.load_annotation(reference_file)
    est_intervals, est_labels = io.load_annotation(estimation_file)

    # Discretize the reference annotation
    time_grid, ref_labels = mir_eval.util.intervals_to_samples(
        ref_intervals, ref_labels, offset=0.005, sample_size=0.01,
        fill_value='N')

    est_labels = mir_eval.util.interpolate_intervals(
        est_intervals, est_labels, time_grid, fill_value='N')

    # Now compute all the metrics
    M = OrderedDict()
    return M


def print_evaluation(prediction_file, M):
    # And print them
    print os.path.basename(prediction_file)
    for key, value in M.iteritems():
        print '\t%12s:\t%0.3f' % (key, value)

    pass


def process_arguments():
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(
        description='mir_eval chord recognition evaluation')

    parser.add_argument(
        'reference_data', action='store',
        help='Path to a reference annotation file or directory.')

    parser.add_argument(
        'estimation_data', action='store',
        help='Path to estimation annotation file or directory.')

    return vars(parser.parse_args(sys.argv[1:]))

if __name__ == '__main__':
    # Get the parameters
    parameters = process_arguments()

    # Compute all the scores
    scores = evaluate_pair(**parameters)
    print_evaluation(parameters['prediction_file'], scores)
