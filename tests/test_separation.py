'''
unit tests for mir_eval.separation

load randomly generated source and estimated source signals and
the output from BSS_eval MATLAB implementation, make sure the results
from mir_eval numerically match.
'''

import numpy as np
import mir_eval
from scipy import io


def test_bss_eval():
    # load the randomly generated sources and estimated sources
    signals_mat = io.loadmat('data/separation/source_signals.mat')
    estimated_sources, sources = signals_mat['se'], signals_mat['s']
    # load the referred output from BSS_eval MATLAB implementation
    # note that vector loaded from .mat file are reshaped to 2-d
    metrics_mat = io.loadmat('data/separation/bss_output_MATLAB.mat')
    r_sdr, r_sir, r_sar, r_perm = metrics_mat['sdr'].ravel(), \
        metrics_mat['sir'].ravel(), metrics_mat['sar'].ravel(), \
        metrics_mat['perm'].ravel()

    sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(
        estimated_sources, sources)
    # make sure they all match
    assert np.allclose(sdr, r_sdr.ravel())
    assert np.allclose(sir, r_sir.ravel())
    assert np.allclose(sar, r_sar.ravel())
    assert np.all(perm == r_perm.ravel())
