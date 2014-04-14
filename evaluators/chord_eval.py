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


def collect_fileset(reference_dir, estimation_dir, fext='lab'):
    '''Collect the set of files for evaluation.

    :parameters:
    - reference_dir : str
        Path to a directory of reference files.
    - estimation_dir : str
        Path to a directory of estimation files.
    - fext : str
        File extension for annotations.
    '''
    ref_files = glob.glob(os.path.join(reference_dir, "*.%s" % fext))
    est_files = glob.glob(os.path.join(estimation_dir, "*.%s" % fext))
    ref_files, est_files = mir_eval.util.intersect_files(ref_files, est_files)
    return ref_files, est_files
    # results = []
    # for ref_file, est_file in zip(ref_files, est_files):
    #     results.append(evaluate_pair(ref_file, est_file))

    # return results


def evaluate_pair(reference_file, estimation_file, scores=['dyads']):
    '''Load data and perform the evaluation'''

    # load the data
    ref_intervals, ref_labels = io.load_annotation(reference_file)
    est_intervals, est_labels = io.load_annotation(estimation_file)

    # Discretize the reference annotation
    time_grid, ref_labels = mir_eval.util.intervals_to_samples(
        ref_intervals, ref_labels, offset=0.05, sample_size=0.1,
        fill_value='N')

    est_labels = mir_eval.util.interpolate_intervals(
        est_intervals, est_labels, time_grid, fill_value='N')

    # Now compute all the metrics
    result = OrderedDict()
    try:
        for name in scores:
            result[name] = chord.scorers[name](ref_labels, est_labels)
    except chord.InvalidChordException as err:
        basename = os.path.basename(reference_file)
        print "[%s]: Skipping %s\n\t%s" % (err.name, basename, err.message)
        if err.chord_label in ref_labels:
            offending_file = reference_file
        else:
            offending_file = estimation_file
        result['error'] = (err.chord_label, offending_file)
    return result


def print_evaluation(prediction_file, result):
    # And print them
    print os.path.basename(prediction_file)
    for key, value in result.iteritems():
        if key not in chord.scorers:
            continue
        print '\t%12s:\t%0.3f' % (key, value.mean())


def print_summary(results):
    file_errors = []
    chord_errors = set()
    print "'%s'\n%s" % (chord.InvalidChordException().name,
                        '-'*len(chord.InvalidChordException().name))
    for item in results:
        err_pair = item.get("error", None)
        if err_pair:
            print "Chord: %10s\tFile: %s" % err_pair
            chord_errors.add(err_pair[0])
            file_errors.append(err_pair[1])


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

    args = parser.parse_args(sys.argv[1:])
    data = [args.reference_data, args.estimation_data]
    if all([os.path.isdir(a) for a in data]):
        ref_files, est_files = collect_fileset(data[0],
                                               data[1])
    else:
        for a in data:
            if not os.path.exists(a):
                raise ValueError("File does not exist: %s" % a)
        ref_files = [args.reference_data]
        est_files = [args.estimation_data]
    return ref_files, est_files


if __name__ == '__main__':
    # Get the parameters
    ref_files, est_files = process_arguments()

    # Compute all the scores
    results = []
    for r, f in zip(ref_files, est_files):
        results.append(evaluate_pair(r, f))
        print_evaluation(r, results[-1])

    print_summary(results)
