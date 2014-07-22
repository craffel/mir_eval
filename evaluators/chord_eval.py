#!/usr/bin/env python
'''Compute chord evaluation metrics


Usage: To run an evaluation consistent with MIREX2013, run the following:

./chord_eval.py reference_file.lab \
estimation_file.txt \
-v root majmin majmin-inv sevenths sevenths-inv
'''

import argparse
import glob
import os

import mir_eval


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


def print_evaluation(prediction_file, result):
    # And print them
    print os.path.basename(prediction_file)
    for key, value in result.iteritems():
        if not key.startswith("_"):
            print '\t%12s:\t%0.6f' % (key, value)


def print_summary(results):
    '''
    '''
    file_errors = []
    chord_errors = set()
    print "\n%s\n%s" % (mir_eval.chord.InvalidChordException().name,
                        '-'*len(mir_eval.chord.InvalidChordException().name))
    for item in results:
        err_pair = item.get("_error", None)
        if err_pair:
            print "Chord: %10s\tFile: %s" % err_pair
            chord_errors.add(err_pair[0])
            file_errors.append(err_pair[1])


def parse_input_data(reference_data, estimation_data):
    '''
    '''
    if all([os.path.isdir(a) for a in (reference_data, estimation_data)]):
        ref_files, est_files = collect_fileset(reference_data, estimation_data)
    else:
        for a in (reference_data, estimation_data):
            if not os.path.exists(a):
                raise ValueError("File does not exist: %s" % a)
        ref_files = [reference_data]
        est_files = [estimation_data]
    return ref_files, est_files


def main(reference_data, estimation_data, vocabularies):
    '''
    '''
    ref_files, est_files = parse_input_data(reference_data, estimation_data)
    # Compute all the scores
    results = []
    for ref, est in zip(ref_files, est_files):
        results.append(mir_eval.chord.evaluate_file_pair(ref, est,
                                                         vocabularies))
        print_evaluation(ref, results[-1])

    print_summary(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='mir_eval chord recognition evaluation')

    parser.add_argument(
        'reference_data', action='store',
        help='Path to a reference annotation file or directory.')

    parser.add_argument(
        'estimation_data', action='store',
        help='Path to estimation annotation file or directory.')

    parser.add_argument(
        '-v', '--vocabularies', nargs='+', type=str)

    parser.add_argument(
        '-strict_bass', '--strict_bass', type=bool, default=False)

    args = parser.parse_args()
    mir_eval.chord.STRICT_BASS_INTERVALS = args.strict_bass
    main(args.reference_data, args.estimation_data, args.vocabularies)
