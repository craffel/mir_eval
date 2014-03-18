#!/usr/bin/env python
'''
CREATED:2014-03-18 by Justin Salamon <justin.salamon@nyu.edu>

Compute melody extraction evaluation measures

Usage:

./melody_eval.py TRUTH.TXT PREDICTION.TXT

For a detailed explanation of the measures please refer to:

J. Salamon, E. Gomez, D. P. W. Ellis and G. Richard, "Melody Extraction
from Polyphonic Music Signals: Approaches, Applications and Challenges",
IEEE Signal Processing Magazine, 31(2):118-134, Mar. 2014.
'''

import numpy as np
import scipy as sp
import argparse
import sys
import os
from collections import OrderedDict
import mir_eval

def evaluate(truth_file=None, prediction_file=None, hop=0.010):
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

    # # STEP 0
    # # Cast to numpy arrays and run safety checks
    # try:
    #     ref = np.asarray(ref, dtype=np.float64)
    # except ValueError:
    #     print 'Error: ref could not be read, ' \
    #           'are the time and frequency sequences of the same length?'
    #     return None
    # try:
    #     est = np.asarray(est, dtype=np.float64)
    # except ValueError:
    #     print 'Error: est could not be read, ' \
    #           'are the time and frequency sequences of the same length?'
    #     return None
    #
    # if ref.shape[0] != 2:
    #     print 'Error: ref should be of dimension (2,x), but is of dimension',\
    #         ref.shape
    #     return None
    # if est.shape[0] != 2:
    #     print 'Error: est should of dimension (2,x), but is of dimension', \
    #         est.shape
    #     return None
    #
    # if len(ref[0])==0 or len(est[0])==0:
    #     print 'Error: one of the inputs seems to be empty?'
    #     return None


    # STEP 1
    # load the data
    ref_time, ref_freq = mir_eval.io.load_time_series(truth_file)
    est_time, est_freq = mir_eval.io.load_time_series(prediction_file)

    # STEP 2
    # convert both sequences to cents
    ref_cent = hz2cents(ref_freq)
    est_cent = hz2cents(est_freq)

    # STEP 3
    # Check if missing sample at time 0 and if so add one
    if ref_time[0] > 0:
        ref_time = np.insert(ref_time, 0, 0)
        ref_cent = np.insert(ref_cent, 0, ref_cent[0])
    if est_time[0] > 0:
        est_time = np.insert(est_time, 0, 0)
        est_cent = np.insert(est_cent, 0, est_cent[0])

    # STEP 4
    # resample to common hop size using linear interpolation
    ref_time_grid, ref_cent_interp = resample_time_series(ref_time, ref_cent, hop)
    est_time_grid, est_cent_interp = resample_time_series(est_time, est_cent, hop)

    # STEP 5
    # fix interpolated values between non-zero/zero transitions:
    # interpolating these values doesn't make sense, so replace with value of start point.
    fix_zero_transitions(ref_time, ref_cent, ref_time_grid, ref_cent_interp)
    fix_zero_transitions(est_time, est_cent, est_time_grid, est_cent_interp)

    # STEP 6
    # restore original sign to interpolated sequences
    restore_sign_to_resampled(ref_time, ref_freq, ref_time_grid, ref_cent_interp)
    restore_sign_to_resampled(est_time, est_freq, est_time_grid, est_cent_interp)

    # STEP 7
    # ensure the estimated sequence is the same length as the reference
    est_time_grid = ref_time_grid
    len_diff = len(ref_cent_interp) - len(est_cent_interp)
    if len_diff >= 0:
        est_cent_interp = np.append(est_cent_interp, np.zeros(len_diff))
    else:
        est_cent_interp = np.resize(est_cent_interp, len(ref_cent_interp))

    # STEP 8
    # separate into pitch sequence and voicing indicator sequence
    ref_pitch = np.abs(ref_cent_interp)
    ref_voicing = np.sign(ref_cent_interp)
    est_pitch = np.abs(est_cent_interp)
    est_voicing = np.sign(est_cent_interp)

    # STEP 9
    # Compute the evaluation measures
    M = OrderedDict()

    # F-Measure
    M['vx_recall'], M['vx_false_alarm'] = mir_eval.melody.voicing_measures(ref_voicing, est_voicing)

    M['raw_pitch'] = mir_eval.melody.raw_pitch_accuracy(ref_pitch, ref_voicing, est_pitch, est_voicing)

    M['raw_chroma'] = mir_eval.melody.raw_chroma_accuracy(ref_pitch, ref_voicing, est_pitch, est_voicing)

    M['overall_accuracy'] = mir_eval.melody.overall_accuracy(ref_pitch, ref_voicing, est_pitch, est_voicing)

    return M


def restore_sign_to_resampled(times, values, time_grid, values_resampled):
    index_interp = 0
    for index_orig in range(len(values) - 1):
        if values[index_orig] < 0:
            while index_interp < len(time_grid) and time_grid[
                index_interp] < times[index_orig]:
                index_interp += 1
            while index_interp < len(time_grid) and time_grid[
                index_interp] < times[index_orig + 1]:
                values_resampled[index_interp] *= -1
                index_interp += 1


def fix_zero_transitions(times, values, time_grid, values_resampled):
    index_interp = 0
    for index_orig in range(len(values) - 1):
        if np.logical_xor(values[index_orig] > 0,
                          values[index_orig + 1] > 0):
            while index_interp < len(time_grid) and time_grid[
                index_interp] <= times[index_orig]:
                index_interp += 1
            while index_interp < len(time_grid) and time_grid[
                index_interp] < times[index_orig + 1]:
                values_resampled[index_interp] = values[index_orig]
                index_interp += 1


def resample_time_series(times, values, hop):
    interp_func = sp.interpolate.interp1d(times, values)
    time_grid = np.linspace(0, hop * np.floor(times[-1] / hop), np.floor(times[-1] / hop) + 1)
    values_resampled = interp_func(time_grid)

    return time_grid, values_resampled


def hz2cents(freq_hz):
    # Convert an array of frequency values in Hz to cents using 10 Hz as the
    # base frequency (i.e. 10 Hz = 0 cents)
    # NB: 0 Hz values are not converted!
    base_frequency = 10.0
    freq_cent = np.zeros(len(freq_hz))
    freq_nonz_ind = np.nonzero(freq_hz)[0]
    freq_cent[freq_nonz_ind] = 1200 * np.log2(np.abs(freq_hz[freq_nonz_ind]) / base_frequency)

    return freq_cent


def print_evaluation(prediction_file, M):

    keys = M.keys()
    keys.append(os.path.basename(prediction_file))
    print '%s\t%s\t%s\t%s\t%s\t(%s)' % tuple(keys)
    print '%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f' % tuple(M.values())


def process_arguments():
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(description='mir_eval melody extraction evaluation')

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

    # Compute all the scores
    scores = evaluate(**parameters)

    # Print the scores
    print_evaluation(parameters['prediction_file'], scores)
