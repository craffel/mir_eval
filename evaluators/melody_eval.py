#!/usr/bin/env python
'''
CREATED:2014-03-18 by Justin Salamon <justin.salamon@nyu.edu>

Compute melody extraction evaluation measures

Usage:

./melody_eval.py TRUTH.TXT PREDICTION.TXT
(CSV files also accepted)

For a detailed explanation of the measures please refer to:

J. Salamon, E. Gomez, D. P. W. Ellis and G. Richard, "Melody Extraction
from Polyphonic Music Signals: Approaches, Applications and Challenges",
IEEE Signal Processing Magazine, 31(2):118-134, Mar. 2014.
'''

import argparse
import sys
import os
import eval_utilities

import mir_eval


def process_arguments():
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(description='mir_eval melody extraction '
                                                 'evaluation')

    parser.add_argument('-o',
                        dest='output_file',
                        default=None,
                        type=str,
                        action='store',
                        help='Store results in json format')

    parser.add_argument('reference_file',
                        action='store',
                        help='path to the ground truth annotation')

    parser.add_argument('estimated_file',
                        action='store',
                        help='path to the estimation file')

    parser.add_argument("--hop",
                        dest='hop',
                        type=float,
                        default=None,
                        help="hop size (in seconds) to use for the evaluation"
                        " (optional)")

    return vars(parser.parse_args(sys.argv[1:]))


if __name__ == '__main__':
    # Get the parameters
    parameters = process_arguments()

    # Load in the data from the provided files
    (ref_time,
     ref_freq) = mir_eval.io.load_time_series(parameters['reference_file'])
    (est_time,
     est_freq) = mir_eval.io.load_time_series(parameters['estimated_file'])

    # Compute all the scores
    scores = mir_eval.melody.evaluate(ref_time, ref_freq, est_time, est_freq,
                                      hop=parameters['hop'])
    print os.path.basename(parameters['estimated_file'])
    eval_utilities.print_evaluation(scores)

    if parameters['output_file']:
        print 'Saving results to: ', parameters['output_file']
        eval_utilities.save_results(scores, parameters['output_file'])
