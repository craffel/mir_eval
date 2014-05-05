# -*- coding: utf-8 -*-
'''
CREATED: 2013-08-13 12:31:25 by Dawen Liang <dliang@ee.columbia.edu>

Source separation evaluation:
    BSS-EVAL -- SDR (Source-to-Distortion Ratio), SIR (Source-to-Interferences
                Ratio), and SAR (Source-to-Artifacts Ratio)

'''

import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import fftconvolve
import functools
import collections
import itertools
import warnings

def validate(metric):
    '''Decorator which checks that the input data to a metric
    are valid, and throws helpful errors if not.

    :parameters:
        - metric : function
            Evaluation metric function.  First two arguments must be
            reference_sources and estimated_sources.

    :returns:
        - metric_validated : function
            The function with the onset locations validated
    '''
    # Retain docstring, etc
    @functools.wraps(metric)
    def metric_validated(reference_sources, estimated_sources, *args, **kwargs):
        '''
        Metric with input validated
        '''
        # make sure the input is of shape (nsrc, nsampl)
        if estimated_sources.ndim == 1:
            estimated_sources = estimated_sources[np.newaxis, :]
        if reference_sources.ndim == 1:
            reference_sources = reference_sources[np.newaxis, :]

        if reference_sources.shape != estimated_sources.shape:
            raise ValueError('The shape of estimated sources and the true sources '
                             'should match.  reference_sources.shape = {}, '
                             'estimated_sources = {}'.format(reference_sources.shape,
                                                             estimated_sources.shape))

        if reference_sources.size == 0:
            warnings.warn("reference_sources is empty, should be of size (nsrc, nsample).  "
                          "sdr, sir, sar, and perm will all be empty np.ndarrays")
        if estimated_sources.size == 0:
            warnings.warn("estimated_sources is empty, should be of size (nsrc, nsample).  "
                          "sdr, sir, sar, and perm will all be empty np.ndarrays")

        return metric(reference_sources, estimated_sources, *args, **kwargs)
    return metric_validated

@validate
def bss_eval_sources(reference_sources, estimated_sources):
    '''BSS_EVAL_SOURCES
        MATLAB translation of BSS_EVAL Toolbox

        Ordering and measurement of the separation quality for estimated source
        signals in terms of filtered true source, interference and artifacts.

        The decomposition allows a time-invariant filter distortion of length
        512, as described in Section III.B of the reference below.

    :usage:
        >>> # reference_sources[n] should be an ndarray of samples of the n'th reference source
        >>> # estimated_sources[n] should be the same for the n'th estimated source
        >>> sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(reference_sources,
                                                                       estimated_sources)

    :parameters:
        - reference_sources: ndarray
            (nsrc, nsampl) matrix containing true sources
        - estimated_sources: ndarray
            (nsrc, nsampl) matrix containing estimated sources

    :returns:
        - sdr: ndarray
            (nsrc, ) vector of Signal to Distortion Ratios (SDR)
        - sir: ndarray
            (nsrc, ) vector of Source to Interference Ratios (SIR)
        - sar: ndarray
            (nsrc, ) vector of Sources to Artifacts Ratios (SAR)
        - perm: ndarray
            (nsrc, ) vector containing the best ordering of estimated sources in
            the mean SIR sense (estimated source number perm[j] corresponds to
            true source number j)

    Reference:
        Emmanuel Vincent, Rémi Gribonval, and Cédric Févotte, "Performance
        measurement in blind audio source separation," IEEE Trans. on Audio,
        Speech and Language Processing, 14(4):1462-1469, 2006.

    '''

    # If empty matrices were supplied, return empty lists (special case)
    if reference_sources.size == 0 or estimated_sources.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    nsrc = estimated_sources.shape[0]

    # compute criteria for all possible pair matches
    sdr = np.empty((nsrc, nsrc))
    sir = np.empty((nsrc, nsrc))
    sar = np.empty((nsrc, nsrc))
    for jest in xrange(nsrc):
        for jtrue in xrange(nsrc):
            s_true, e_spat, e_interf, e_artif = \
                    _bss_decomp_mtifilt(reference_sources, estimated_sources[jest],
                                        jtrue, 512)
            sdr[jest, jtrue], sir[jest, jtrue], sar[jest, jtrue] = \
                    _bss_source_crit(s_true, e_spat, e_interf, e_artif)

    # select the best ordering
    perms = list(itertools.permutations(xrange(nsrc)))
    mean_sir = np.empty(len(perms))
    dum = np.arange(nsrc)
    for (i, perm) in enumerate(perms):
        mean_sir[i] = np.mean(sir[perm, dum])
    popt = perms[np.argmax(mean_sir)]
    idx = (popt, dum)
    return (sdr[idx], sir[idx], sar[idx], popt)


def _bss_decomp_mtifilt(reference_sources, estimated_source, j, flen):
    '''
    Decomposition of an estimated source image into four components
    representing respectively the true source image, spatial (or filtering)
    distortion, interference and artifacts, derived from the true source
    images using multichannel time-invariant filters.
    '''
    nsampl = estimated_source.size
    ## decomposition ##
    # true source image
    s_true = np.hstack((reference_sources[j], np.zeros(flen - 1)))
    # spatial (or filtering) distortion
    e_spat = _project(reference_sources[j, np.newaxis, :], estimated_source,
                      flen) - s_true
    # interference
    e_interf = _project(reference_sources, estimated_source, flen) - s_true - e_spat
    # artifacts
    e_artif = -s_true - e_spat - e_interf
    e_artif[:nsampl] += estimated_source
    return (s_true, e_spat, e_interf, e_artif)


def _project(reference_sources, estimated_source, flen):
    '''
    Least-squares projection of estimated source on the subspace spanned by
    delayed versions of reference sources, with delays between 0 and flen-1
    '''
    nsrc, nsampl = reference_sources.shape

    ## computing coefficients of least squares problem via FFT ##
    # zero padding and FFT of input data
    reference_sources = np.hstack((reference_sources, np.zeros((nsrc, flen - 1))))
    estimated_source = np.hstack((estimated_source, np.zeros(flen - 1)))
    n_fft = int(2**np.ceil(np.log2(nsampl + flen - 1)))
    sf = np.fft.fft(reference_sources, n=n_fft, axis=1)
    sef = np.fft.fft(estimated_source, n=n_fft)
    # inner products between delayed versions of reference_sources
    G = np.zeros((nsrc * flen, nsrc * flen))
    for i in xrange(nsrc):
        for j in xrange(nsrc):
            ssf = sf[i] * np.conj(sf[j])
            ssf = np.real(np.fft.ifft(ssf))
            ss = toeplitz(np.hstack((ssf[0], ssf[-1:-flen:-1])),
                                       r=ssf[:flen])
            G[i * flen: (i+1) * flen, j * flen: (j+1) * flen] = ss
            G[j * flen: (j+1) * flen, i * flen: (i+1) * flen] = ss.T
    # inner products between estimated_source and delayed versions of reference_sources
    D = np.zeros(nsrc * flen)
    for i in xrange(nsrc):
        ssef = sf[i] * np.conj(sef)
        ssef = np.real(np.fft.ifft(ssef))
        D[i * flen: (i+1) * flen] = np.hstack((ssef[0], ssef[-1:-flen:-1]))

    ## Computing projection ##
    # Distortion filters
    try:
        C = np.linalg.solve(G, D).reshape(flen, nsrc, order='F')
    except np.linalg.linalg.LinAlgError:
        C = np.linalg.lstsq(G, D)[0].reshape(flen, nsrc, order='F')
    # Filtering
    sproj = np.zeros(nsampl + flen - 1)
    for i in xrange(nsrc):
        sproj += fftconvolve(C[:, i], reference_sources[i])[:nsampl + flen - 1]
    return sproj


def _bss_source_crit(s_true, e_spat, e_interf, e_artif):
    '''
    Measurement of the separation quality for a given source in terms of
    filtered true source, interference and artifacts.
    '''
    # energy ratios
    s_filt = s_true + e_spat
    sdr = 10 * np.log10(np.sum(s_filt**2) / np.sum((e_interf + e_artif)**2))
    sir = 10 * np.log10(np.sum(s_filt**2) / np.sum(e_interf**2))
    sar = 10 * np.log10(np.sum((s_filt + e_interf)**2) / np.sum(e_artif**2))
    return (sdr, sir, sar)

# Create a dictionary which maps the name of each metric
# to the function used to compute it
METRICS = collections.OrderedDict()
METRICS['bss_eval_sources'] = bss_eval_sources
