#!/usr/bin/env python
"""
Compute pattern discovery evaluation metrics.

Usage:
    ./pattern_eval.py REFERENCE.TXT ESTIMATION.TXT

Example:
    ./pattern_eval.py ../tests/data/pattern/reference-mono.txt \
                      ../tests/data/pattern/estimate-mono.txt

Written by Oriol Nieto (oriol@nyu.edu), 2014
"""

import argparse
import os
import sys
import eval_utilities

import mir_eval


def main():
    """Main function to evaluate the pattern discovery task."""
    parser = argparse.ArgumentParser(description="mir_eval pattern discovery "
                                                 "evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o',
                        dest='output_file',
                        default=None,
                        type=str,
                        action='store',
                        help='Store results in json format')
    parser.add_argument("reference_file",
                        action="store",
                        help="Path to the reference file.")
    parser.add_argument("estimated_file",
                        action="store",
                        help="Path to the estimation file.")
    parameters = vars(parser.parse_args(sys.argv[1:]))

    # Load in data
    ref_patterns = mir_eval.io.load_patterns(parameters['reference_file'])
    est_patterns = mir_eval.io.load_patterns(parameters['estimated_file'])

    # Compute all the scores
    scores = mir_eval.pattern.evaluate(ref_patterns, est_patterns)
    print "{} vs. {}".format(os.path.basename(parameters['reference_file']),
                             os.path.basename(parameters['estimated_file']))
    eval_utilities.print_evaluation(scores)

    if parameters['output_file']:
        print 'Saving results to: ', parameters['output_file']
        eval_utilities.save_results(scores, parameters['output_file'])


if __name__ == '__main__':
    main()
