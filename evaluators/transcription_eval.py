#!/usr/bin/env python
'''
CREATED: 2/9/16 2:59 PM by Justin Salamon <justin.salamon@nyu.edu>

Compute note transcription evaluation metrics

Usage:

./transcription_eval.py REFERENCE.TXT ESTIMATED.TXT
'''

from __future__ import print_function
import argparse
import sys
import os

import mir_eval

from . import eval_utilities


def process_arguments():
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(description='mir_eval transcription '
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


def main():
    # Get the parameters
    parameters = process_arguments()

    # Load in data
    ref_intervals, ref_pitches = mir_eval.io.load_valued_intervals(
        parameters['reference_file'])
    est_intervals, est_pitches = mir_eval.io.load_valued_intervals(
        parameters['estimated_file'])
    # Compute all the scores
    scores = mir_eval.transcription.evaluate(ref_intervals, ref_pitches,
                                             est_intervals, est_pitches)
    print("{} vs. {}".format(os.path.basename(parameters['reference_file']),
                             os.path.basename(parameters['estimated_file'])))
    eval_utilities.print_evaluation(scores)

    if parameters['output_file']:
        print('Saving results to: ', parameters['output_file'])
        eval_utilities.save_results(scores, parameters['output_file'])


if __name__ == '__main__':
    main()
