# CREATED: 2/9/16 2:27 PM by Justin Salamon <justin.salamon@nyu.edu>

import mir_eval

def test_precision_recall_f1():

    ref_int, ref_pitch = mir_eval.io.load_valued_intervals(
        'tests/data/transcription/ref00.txt')
    est_int, est_pitch = mir_eval.io.load_valued_intervals(
        'tests/data/transcription/est00.txt')

    precision, recall, f_measure = \
        mir_eval.transcription.precision_recall_f1(ref_int, ref_pitch, est_int,
                                                   est_pitch)

    assert precision == 3/5.
    assert recall == 3/4.
    assert f_measure == 2 * precision * recall / (precision + recall)

    precision, recall, f_measure = \
        mir_eval.transcription.precision_recall_f1(ref_int, ref_pitch, est_int,
                                                   est_pitch, with_offset=True)

    assert precision == 2/5.
    assert recall == 2/4.
    assert f_measure == 2 * precision * recall / (precision + recall)