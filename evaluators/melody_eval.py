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

import numpy as np
import argparse
import sys
import os
from collections import OrderedDict
import mir_eval

def evaluate(reference_file, estimated_file):
    '''
    Evaluate two melody (predominant f0) transcriptions, where the first is
    treated as the reference (ground truth) and the second as the estimate to
    be evaluated (prediction).

    Input:
        truth_file - path to reference file. File should contain 2 columns:
              column 1 with timestamps and column 2 the corresponding reference
              frequency values in Hz (see *).
        prediction_file - path to estimate file. File should contain with 2 columns:
              column 1 with timestamps and column 2 the corresponding estimate
              frequency values in Hz (see **).
        hop - the desired hop size (in seconds) to compare the reference and
              estimate sequences (see ***)

    Output:
        M - ordered dictionary containing 5 evaluation measures:
        voicing recall rate -       Fraction of voiced frames in ref estimated as voiced in est
        voicing false alarm rate -  Fraction of unvoiced frames in ref estimated as voiced in est
        raw pitch -                 Fraction of voiced frames in ref for which est gives a correct pitch estimate (within 50 cents)
        raw chroma -                Same as raw pitch, but ignores octave errors
        overall accuracy -          Overall performance measure combining pitch and voicing



    *    Unvoiced frames should be indicated by 0 Hz.
    **   Unvoiced frames can be indicated either by 0 Hz or by a negative Hz
         value - negative values represent the algorithm's pitch estimate for
         frames it has determined as unvoiced, in case they are in fact voiced.
    ***  For a frame-by-frame comparison, both sequences are resampled using
         the provided hop size (in seconds), the default being 10 ms. The
         frequency values of the resampled sequences are obtained via linear
         interpolation of the original frequency values converted to a cent
         scale.
    '''

    # load the data
    ref_time, ref_freq = mir_eval.io.load_time_series(reference_file)
    est_time, est_freq = mir_eval.io.load_time_series(estimated_file)
    # Check if missing sample at time 0 and if so add one
    if ref_time[0] > 0:
        ref_time = np.insert(ref_time, 0, 0)
        ref_freq = np.insert(ref_freq, 0, ref_voicing[0])
    if est_time[0] > 0:
        est_time = np.insert(est_time, 0, 0)
        est_freq = np.insert(est_freq, 0, est_voicing[0])
    # Get separated frequency array and voicing boolean array
    ref_freq, ref_voicing = mir_eval.melody.freq_to_voicing(ref_freq)
    est_freq, est_voicing = mir_eval.melody.freq_to_voicing(est_freq)
    # convert both sequences to cents
    ref_cent = mir_eval.melody.hz2cents(ref_freq)
    est_cent = mir_eval.melody.hz2cents(est_freq)
    # Resample to common time base
    ref_time_grid, ref_cent, ref_voicing = mir_eval.melody.resample_melody_series(ref_time,
                                                                                  ref_cent,
                                                                                  ref_voicing)
    est_time_grid, est_cent, est_voicing = mir_eval.melody.resample_melody_series(est_time,
                                                                                  est_cent,
                                                                                  est_voicing)
    # ensure the estimated sequence is the same length as the reference
    len_diff = ref_cent.shape[0] - est_cent.shape[0]
    if len_diff >= 0:
        est_cent = np.append(est_cent, np.zeros(len_diff))
        est_voicing = np.append(est_voicing, np.zeros(len_diff))
    else:
        est_cent = est_cent[:ref_cent.shape[0]]
        est_voicing = est_voicing[ref_voicing.shape[0]]

    # Compute metrics
    M = OrderedDict()

    # F-Measure
    M['vx_recall'], M['vx_false_alarm'] = mir_eval.melody.voicing_measures(ref_voicing, est_voicing)

    M['raw_pitch'] = mir_eval.melody.raw_pitch_accuracy(ref_cent, ref_voicing, est_cent, est_voicing)

    M['raw_chroma'] = mir_eval.melody.raw_chroma_accuracy(ref_cent, ref_voicing, est_cent, est_voicing)

    M['overall_accuracy'] = mir_eval.melody.overall_accuracy(ref_cent, ref_voicing, est_cent, est_voicing)

    return M

def print_evaluation(prediction_file, M):
    '''
    Pretty print the melody extraction evaluation measures
    '''
    keys = M.keys()
    keys.append(os.path.basename(prediction_file))
    print '%s\t%s\t%s\t%s\t%s\t(%s)' % tuple(keys)
    print '%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f' % tuple(M.values())


def process_arguments():
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(description='mir_eval melody extraction evaluation')

    parser.add_argument('reference_file',
                        action = 'store',
                        help = 'path to the ground truth annotation')

    parser.add_argument('estimated_file',
                        action = 'store',
                        help = 'path to the estimation file')

    return vars(parser.parse_args(sys.argv[1:]))


if __name__ == '__main__':
    # Get the parameters
    parameters = process_arguments()

    # Compute all the scores
    scores = evaluate(**parameters)

    # Print the scores
    print_evaluation(parameters['reference_file'], scores)
