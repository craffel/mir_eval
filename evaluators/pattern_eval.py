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
from collections import OrderedDict

import mir_eval


def evaluate(ref_file, est_file):
    """Load data and perform the evaluation.

    :param ref_file: Path to the reference file.
    :type ref_file: str
    :param est_file: Path to the estimation file.
    :type est_file: str
    :returns:
        - M: dict
            Results contained in an ordered dictionary.
    """
    # load the data
    ref_patterns = mir_eval.io.load_patterns(ref_file)
    est_patterns = mir_eval.io.load_patterns(est_file)

    # Now compute all the metrics
    M = OrderedDict()

    # Standard scores
    M['F'], M['P'], M['R'] = \
        mir_eval.pattern.standard_FPR(ref_patterns, est_patterns)

    # Establishment scores
    M['F_est'], M['P_est'], M['R_est'] = \
        mir_eval.pattern.establishment_FPR(ref_patterns, est_patterns)

    # Occurrence scores
    M['F_occ.5'], M['P_occ.5'], M['R_occ.5'] = \
        mir_eval.pattern.occurrence_FPR(ref_patterns, est_patterns, thres=.5)
    M['F_occ.75'], M['P_occ.75'], M['R_occ.75'] = \
        mir_eval.pattern.occurrence_FPR(ref_patterns, est_patterns, thres=.75)

    # Three-layer scores
    M['F_3'], M['P_3'], M['R_3'] = \
        mir_eval.pattern.three_layer_FPR(ref_patterns, est_patterns)

    # First Five Patterns scores
    M['FFP'] = mir_eval.pattern.first_n_three_layer_P(ref_patterns,
                                                      est_patterns, n=5)
    M['FFTP_est'] = mir_eval.pattern.first_n_target_proportion_R(ref_patterns,
                                                                 est_patterns,
                                                                 n=5)

    return M


def print_evaluation(estimation_file, M):
    # And print them
    print os.path.basename(estimation_file)
    for key, value in M.iteritems():
        print '\t%12s:\t%0.3f' % (key, value)


def main():
    """Main function to evaluate the pattern discovery task."""
    parser = argparse.ArgumentParser(description="mir_eval pattern discovery "
                                                 "evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("ref_file",
                        action="store",
                        help="Path to the reference file.")
    parser.add_argument("est_file",
                        action="store",
                        help="Path to the estimation file.")
    args = parser.parse_args()

    # Run the evaluations
    scores = evaluate(args.ref_file, args.est_file)

    # Print results
    print_evaluation(args.ref_file, scores)


if __name__ == '__main__':
    main()
