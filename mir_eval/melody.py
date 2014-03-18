# CREATED:2014-03-07 by Justin Salamon <justin.salamon@nyu.edu>
'''
Melody extraction evaluation, based on the protocols used in MIREX since 2005.

For a detailed explanation of the measures please refer to:
J. Salamon, E. Gomez, D. P. W. Ellis and G. Richard, "Melody Extraction
from Polyphonic Music Signals: Approaches, Applications and Challenges",
IEEE Signal Processing Magazine, 31(2):118-134, Mar. 2014.
'''

import numpy as np
import sys


def voicing_measures(ref_voicing, est_voicing):
    '''
    Compute the voicing recall and false alarm rates given two voicing indicator
    sequences, one as reference (truth) and the other as the estimate (prediction).
    The sequences must be of the same length.

    Input:
    - ref_voicing : np.array or list
    Reference voicing indicator where val>0 indicates voiced, val<=0 indicates unvoiced

    - est_voicing : np.array or list
    Estimate voicing indicator where val>0 indicates voiced, val<=0 indicates unvoiced

    Output:
    - vx_recall: float
    Voicing recall rate, the fraction of voiced frames in ref indicated as voiced in est

    - vx_false_alarm : float
    Voicing false alarm rate, the fraction of unvoiced frames in ref indicated as voiced in est
    '''

    # check for equal length
    if len(ref_voicing) != len(est_voicing):
        print "Error: inputs must be arrays or lists of the same length"
        return None

    # convert to booleans
    v_ref = np.asarray(ref_voicing) > 0
    v_est = np.asarray(est_voicing) > 0

    uv_ref = np.asarray(ref_voicing) <= 0
    uv_est = np.asarray(est_voicing) <= 0

    # How voicing is computed
    #        | v_ref | uv_ref |
    # -------|-------|--------|
    # v_est  |  TP   |   FP   |
    # -------|-------|------- |
    # uv_est |  FN   |   TN   |
    # -------------------------

    TP = sum(v_ref * v_est)
    FP = sum(uv_ref * v_est)
    FN = sum(v_ref * uv_est)
    TN = sum(uv_ref * uv_est)

    # Voicing recall = fraction of voiced frames according the reference that
    # are declared as voiced by the estimate
    vx_recall = TP / float(TP + FN)

    # Voicing false alarm = fraction of unvoiced frames according to the
    # reference that are declared as voiced by the estimate
    vx_false_alm = FP / float(FP + TN + sys.float_info.epsilon)

    return vx_recall, vx_false_alm


def raw_pitch_accuracy(ref_cent, ref_voicing, est_cent, est_voicing):
    '''
    Compute the raw pitch accuracy given two pitch (frequency) sequences in cents
    and matching voicing indicator sequences. The first pitch and voicing arrays
    are treated as the reference (truth), and the second two as the estimate (prediction).
    All 4 sequences must be of the same length.

    Input:
    - ref_cent : np.array
    Reference pitch sequence in cents

    - ref_voicing : np.array or list
    Reference voicing indicator where val>0 indicates voiced, val<=0 indicates unvoiced

    - est_cent : np.array
    Estimate pitch sequence in cents

    - est_voicing : np.array or list
    Estimate voicing indicator where val>0 indicates voiced, val<=0 indicates unvoiced

    Output:
    - raw_pitch: float
    Raw pitch accuracy, the fraction of voiced frames in ref_cent for which est_cent
    provides a correct frequency values (within 50 cents).
    '''

    l1,l2,l3,l4 = len(ref_cent),len(ref_voicing),len(est_cent),len(est_voicing)
    if l1 != l2 or l1 != l3 or l1 != l4:
        print "Error: all 4 sequences must be of the same length"
        return None

    # convert to booleans
    v_ref = np.asarray(ref_voicing) > 0

    # Raw pitch = the number of voiced frames in the reference for which the
    # estimate provides a correct frequency value (within 50 cents).
    # NB: voicing estimation is ignored in this measure
    cent_diff = np.abs(ref_cent - est_cent)
    raw_pitch = sum(cent_diff[v_ref] <= 50) / float(sum(v_ref))

    return raw_pitch


def raw_chroma_accuracy(ref_cent, ref_voicing, est_cent, est_voicing):
    '''
    Compute the raw chroma accuracy given two pitch (frequency) sequences in cents
    and matching voicing indicator sequences. The first pitch and voicing arrays
    are treated as the reference (truth), and the second two as the estimate (prediction).
    All 4 sequences must be of the same length.

    Input:
    - ref_cent : np.array
    Reference pitch sequence in cents

    - ref_voicing : np.array or list
    Reference voicing indicator where val>0 indicates voiced, val<=0 indicates unvoiced

    - est_cent : np.array
    Estimate pitch sequence in cents

    - est_voicing : np.array or list
    Estimate voicing indicator where val>0 indicates voiced, val<=0 indicates unvoiced

    Output:
    - raw_chroma: float
    Raw chroma accuracy, the fraction of voiced frames in ref_cent for which est_cent
    provides a correct frequency values (within 50 cents), ignoring octave errors
    '''

    l1,l2,l3,l4 = len(ref_cent),len(ref_voicing),len(est_cent),len(est_voicing)
    if l1 != l2 or l1 != l3 or l1 != l4:
        print "Error: all 4 sequences must be of the same length"
        return None

    # convert to booleans
    v_ref = np.asarray(ref_voicing) > 0

    # Raw chroma = same as raw pitch except that octave errors are ignored.
    cent_diff = np.abs(ref_cent - est_cent)
    cent_diff_chroma = abs(cent_diff - 1200 * np.floor(cent_diff / 1200.0 + 0.5))
    raw_chroma = sum(cent_diff_chroma[v_ref] <= 50) / float(sum(v_ref))

    return raw_chroma


def overall_accuracy(ref_cent, ref_voicing, est_cent, est_voicing):
    '''
    Compute the overall accuracy given two pitch (frequency) sequences in cents
    and matching voicing indicator sequences. The first pitch and voicing arrays
    are treated as the reference (truth), and the second two as the estimate (prediction).
    All 4 sequences must be of the same length.

    Input:
    - ref_cent : np.array
    Reference pitch sequence in cents

    - ref_voicing : np.array or list
    Reference voicing indicator where val>0 indicates voiced, val<=0 indicates unvoiced

    - est_cent : np.array
    Estimate pitch sequence in cents

    - est_voicing : np.array or list
    Estimate voicing indicator where val>0 indicates voiced, val<=0 indicates unvoiced

    Output:
    - overall_accuracy: float
    Overall accuracy, the total fraction of correctly estimates frames, where
    provides a correct frequency values (within 50 cents).
    '''

    l1,l2,l3,l4 = len(ref_cent),len(ref_voicing),len(est_cent),len(est_voicing)
    if l1 != l2 or l1 != l3 or l1 != l4:
        print "Error: all 4 sequences must be of the same length"
        return None

    # Compute boolean voicing indicators
    v_ref = np.asarray(ref_voicing) > 0
    v_est = np.asarray(est_voicing) > 0
    uv_ref = np.asarray(ref_voicing) <= 0
    uv_est = np.asarray(est_voicing) <= 0

    # True negatives = frames correctly estimates as unvoiced
    TN = sum(uv_ref * uv_est)

    cent_diff = np.abs(ref_cent - est_cent)
    overall_accuracy = (sum(cent_diff[v_ref * v_est] <= 50) + TN) / float(len(ref_cent))

    return overall_accuracy








