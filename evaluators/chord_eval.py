#!/usr/bin/env python
'''
Compute chord evaluation metrics


Usage:

./chord_eval.py ~/audio-chord-estimation/2011/outputs-lab/Ground-truth/ \
~/mirex-tools/audio-chord-estimation/2011/outputs-lab/CB2/ \
-v mirex dyads dyads-inv \
--strict_bass

'''

import argparse
import glob
import os

from collections import OrderedDict

from mir_eval import io
import mir_eval.chord as chord
import mir_eval.util as util


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
    ref_files, est_files = util.intersect_files(ref_files, est_files)
    return ref_files, est_files


def evaluate_pair(reference_file, estimation_file, vocabularies=['dyads']):
    '''Load data and perform the evaluation between a pair of annotations.

    :parameters:
    - reference_file: str
        Path to a reference annotation.

    - estimation_file: str
        Path to an estimated annotation.

    - vocabularies: list of strings
        Comparisons to make between the reference and estimated sequences.

    :returns:
    -result: dict
        Dictionary containing the averaged scores for each vocabulary, along
        with the total duration of the file ('_weight') and any errors
        ('_error') caught in the process.
    '''

    # load the data
    ref_intervals, ref_labels = io.load_annotation(reference_file)
    est_intervals, est_labels = io.load_annotation(estimation_file)

    # Adjust the estimated intervals to the reference
    est_intervals, est_labels = util.adjust_intervals(
        est_intervals,
        est_labels,
        ref_intervals.min(),
        ref_intervals.max(),
        chord.NO_CHORD)

    # Merge the time-intervals
    intervals, ref_labels, est_labels = util.merge_labeled_intervals(
        ref_intervals, ref_labels, est_intervals, est_labels)

    # Now compute all the metrics
    result = OrderedDict(_weight=intervals.max())
    try:
        for vocab in vocabularies:
            result[vocab] = chord.score(
                ref_labels, est_labels, intervals, vocab)
    except chord.InvalidChordException as err:
        basename = os.path.basename(reference_file)
        print "[%s]: Skipping %s\n\t%s" % (err.name, basename, err.message)
        if err.chord_label in ref_labels:
            offending_file = reference_file
        else:
            offending_file = estimation_file
        result['_error'] = (err.chord_label, offending_file)
    return result


def print_evaluation(prediction_file, result):
    # And print them
    print os.path.basename(prediction_file)
    for key, value in result.iteritems():
        if not key.startswith("_"):
            print '\t%12s:\t%0.3f' % (key, value)


def print_summary(results):
    '''
    '''
    file_errors = []
    chord_errors = set()
    print "\n'%s'\n%s" % (chord.InvalidChordException().name,
                          '-'*len(chord.InvalidChordException().name))
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
        results.append(evaluate_pair(ref, est, vocabularies))
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
    chord.STRICT_BASS_INTERVALS = args.strict_bass
    main(args.reference_data, args.estimation_data, args.vocabularies)
