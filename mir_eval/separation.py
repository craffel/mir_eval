# -*- coding: utf-8 -*-
'''
Source separation algorithms attempt to extract recordings of individual
sources from a recording of a mixture of sources.  Evaluation methods for
source separation compare the extracted sources from reference sources and
attempt to measure the perceptual quality of the separation.

See also the bss_eval MATLAB toolbox:
    http://bass-db.gforge.inria.fr/bss_eval/

Conventions
-----------

An audio signal is expected to be in the format of a 2-dimensional array where
the first dimension goes over the samples of the audio signal and the second
dimension goes over the channels (as in stereo left and right).
When providing a group of estimated or reference sources, they should be
provided in a 3-dimensional array, where the first dimension corresponds to the
source number, the second corresponds to the samples and the third to the
channels.

Metrics
-------

* :func:`mir_eval.separation.bss_eval`: Computes the bss_eval metrics from
  bss_eval. First, it matches the estimated sources to the references through
  time-invariant filters. Then, it computes the source to distortion (SDR),
  source to artifacts (SAR), source to interference (SIR) ratios, plus the image
  to spatial ratio (ISR) for multichannel signals.
  These are computed on a frame by frame basis, (with infinite window size
  meaning the whole signal). Metrics correspond to the bsseval_images version,
  but may optionally correspond to the (deprecated) bsseval_sources version.

References
----------
  .. [#vincent2006performance] Emmanuel Vincent, Rémi Gribonval, and Cédric
      Févotte, "Performance measurement in blind audio source separation," IEEE
      Trans. on Audio, Speech and Language Processing, 14(4):1462-1469, 2006.

'''

import numpy as np
import scipy.fftpack
from scipy.linalg import toeplitz
from scipy.signal import fftconvolve
import itertools
import collections
import warnings
from . import util

# The maximum allowable number of sources (prevents insane computational load)
MAX_SOURCES = 100

def validate(reference_sources, estimated_sources):
    """Checks that the input data to a metric are valid, and throws helpful
    errors if not.

    Parameters
    ----------
    reference_sources : np.ndarray, shape=(nsrc, nsampl,nchan)
        matrix containing true sources
    estimated_sources : np.ndarray, shape=(nsrc, nsampl,nchan)
        matrix containing estimated sources

    """
    if reference_sources.shape != estimated_sources.shape:
        raise ValueError('The shape of estimated sources and the true '
                         'sources should match.  reference_sources.shape '
                         '= {}, estimated_sources.shape '
                         '= {}'.format(reference_sources.shape,
                                       estimated_sources.shape))

    if reference_sources.ndim > 3 or estimated_sources.ndim > 3:
        raise ValueError('The number of dimensions is too high (must be less '
                         'than 3). reference_sources.ndim = {}, '
                         'estimated_sources.ndim '
                         '= {}'.format(reference_sources.ndim,
                                       estimated_sources.ndim))

    if reference_sources.size == 0:
        warnings.warn("reference_sources is empty, should be of size "
                      "(nsrc, nsample, nchan). sdr, sir, sar, and perm will all"
                      " be empty np.ndarrays")
    elif _any_source_silent(reference_sources):
        raise ValueError('All the reference sources should be non-silent (not '
                         'all-zeros), but at least one of the reference '
                         'sources is all 0s, which introduces ambiguity to the'
                         ' evaluation. (Otherwise we can add infinitely many '
                         'all-zero sources.)')

    if estimated_sources.size == 0:
        warnings.warn("estimated_sources is empty, should be of size "
                      "(nsrc, nsample, nchan).  sdr, sir, sar, and perm will "
                      "all be empty np.ndarrays")
    elif _any_source_silent(estimated_sources):
        raise ValueError('All the estimated sources should be non-silent (not '
                         'all-zeros), but at least one of the estimated '
                         'sources is all 0s. Since we require each reference '
                         'source to be non-silent, having a silent estimated '
                         'source will result in an underdetermined system.')

    if (estimated_sources.shape[0] > MAX_SOURCES or
            reference_sources.shape[0] > MAX_SOURCES):
        raise ValueError('The supplied matrices should be of shape (nsrc,'
                         ' nsampl, nchan) but reference_sources.shape[0] = {} '
                         'and estimated_sources.shape[0] = {} which is greater '
                         'than mir_eval.separation.MAX_SOURCES = {}.  To '
                         'override this check, set '
                         'mir_eval.separation.MAX_SOURCES to a '
                         'larger value.'.format(reference_sources.shape[0],
                                                estimated_sources.shape[0],
                                                MAX_SOURCES))
def _any_source_silent(sources):
    """Returns true if the parameter sources has any silent first dimensions"""
    return np.any(np.all(np.sum(
        sources, axis=tuple(range(2, sources.ndim))) == 0, axis=1))

def bss_eval(reference_sources, estimated_sources,
                    window = 2*44100, hop = 1.5*44100,
                    compute_permutation = False,
                    flen = 512,
                    bsseval_sources = False):
    """Implementation of the bss_eval function, adapted from the
    BSS_EVAL Matlab toolbox and mir_eval.separation.

    Measurement of the separation quality for estimated source signals
    in terms of filtered true source, interference and artifacts.
    This method also provides the ISR measure for multichannel signals.

    The decomposition allows a time-invariant filter distortion of length
    flen=512, as described in Section III.B of [#vincent2006performance]_, and
    computes the metrics on a windowed basis.

    Passing ``False`` for ``compute_permutation`` will assume the order for
    the estimated sources matches that of the true sources. Otherwise, all
    permutations are tested, yielding a significant computation overhead.

    Examples
    --------
    >>> # reference_sources[n] should be a 2D ndarray, with first dimension the
    >>> # samples and second dimension the channels of the n'th reference source
    >>> # estimated_sources[n] should be the same for the n'th estimated
    >>> # source
    >>> (sdr, isr, sir, sar,
    ...  perm) = mir_eval.separation.bss_eval(reference_sources,
    ...                                               estimated_sources)

    Parameters
    ----------
    reference_sources : np.ndarray, shape=(nsrc, nsampl, nchan)
        matrix containing true sources
    estimated_sources : np.ndarray, shape=(nsrc, nsampl, nchan)
        matrix containing estimated sources
    window : int, optional
        size of each window for time-varying evaluation. Picking np.inf will
        compute metrics on the whole signal.
    hop : int, optional
        hop size between windows
    compute_permutation : bool, optional
        compute permutation of estimate/source combinations (True by default)
    bsseval_sources : bool, optional
        if  ``True``, results correspond to :func:`bss_eval_sources` from
        the BSS Eval. Note however that this is not recommended because this
        evaluation method amounts to modify the references according to the
        estimated sources, leading to potential problems. For instance, zeroing
        some frequencies in the estimates will lead those to also be zeroed in
        the references, and hence not evaluated, artificially boosting results.
        For this reason, SiSEC always uses the :func:`bss_eval_images` version.
    Returns
    -------
    sdr : np.ndarray, shape=(nsrc, nwin)
        matrix of Signal to Distortion Ratios (SDR). One for each source and
        window
    isr : np.ndarray, shape=(nsrc, nwin)
        matrix of source Image to Spatial distortion Ratios (ISR)
    sir : np.ndarray, shape=(nsrc, nwin)
        matrix of Source to Interference Ratios (SIR)
    sar : np.ndarray, shape=(nsrc, nwin)
        matrix of Sources to Artifacts Ratios (SAR)
    perm : np.ndarray, shape=(nsrc, nwin)
        vector containing the best ordering of estimated sources in
        the mean SIR sense (estimated source number ``perm[j]`` corresponds to
        true source number ``j``).  Note: ``perm`` will be ``(1,2,...,nsrc)``
        if ``compute_permutation`` is ``False``.

    References
    ----------
    .. [#] Emmanuel Vincent, Shoko Araki, Fabian J. Theis, Guido Nolte, Pau
        Bofill, Hiroshi Sawada, Alexey Ozerov, B. Vikrham Gowreesunker, Dominik
        Lutter and Ngoc Q.K. Duong, "The Signal Separation Evaluation Campaign
        (2007-2010): Achievements and remaining challenges", Signal Processing,
        92, pp. 1928-1936, 2012.

    """
    # make sure the input has 3 dimensions
    # assuming input is in shape (nsampl) or (nsrc, nsampl)
    estimated_sources = np.atleast_3d(estimated_sources)
    reference_sources = np.atleast_3d(reference_sources)

    #validate input
    validate(reference_sources, estimated_sources)

    # If empty matrices were supplied, return empty lists (special case)
    if reference_sources.size == 0 or estimated_sources.size == 0:
        return (np.array([]), np.array([]), np.array([]), np.array([]),
            np.array([]))

    # determine size parameters
    (nsrc, nsampl, nchan) = estimated_sources.shape

    #First compute the time-invariant distortion filters from all refs
    #to each source
    G,sf = _compute_reference_correlations(reference_sources, flen)
    C  = np.zeros((nsrc,nsrc,nchan,flen,nchan))
    for src in range(nsrc):
        C[src] = _compute_projection_filters(G, sf, estimated_sources[src])

    # defines all the permutations desired by user
    if compute_permutation:
        candidate_permutations = np.array(list(
                    itertools.permutations(range(nsrc))))
    else:
        candidate_permutations = np.array(range(nsrc))[None,:]

    # prepare criteria variable for all possible pair matches and
    # initialize variables
    if window < nsampl:
        nwin = int(
            np.floor((reference_sources.shape[1] - window + hop) / hop)
            )
    else:
        nwin = 1

    (SDR,ISR,SIR,SAR)=range(4)
    s_r = np.empty((4,nsrc,nsrc,nwin))
    done = np.zeros((nsrc,nsrc))
    for jtrue in range(nsrc):
        for (k,jest) in enumerate(candidate_permutations[:,jtrue]):
            if not done[jest,jtrue]:
                #need to compute the scores for this combination
                Cj = _compute_projection_filters(G[jtrue, jtrue], sf[jtrue],
                                        estimated_sources[jest])
                s_true, e_spat, e_interf, e_artif = \
                    _bss_decomp_mtifilt(
                        reference_sources,
                        estimated_sources[jest], jtrue, C[jest], Cj)
                s_r[:,jest, jtrue,:] = _bss_crit(s_true, e_spat, e_interf,
                    e_artif, window, hop, flen,bsseval_sources)
                done[jest,jtrue]=True

    # select the best ordering
    mean_sir = np.empty((len(candidate_permutations),))
    dum = np.arange(nsrc)
    for (i, perm) in enumerate(candidate_permutations):
        mean_sir[i] = np.mean(s_r[SIR,perm, dum,:])
    popt = candidate_permutations[np.argmax(mean_sir)]
    idx = (popt, dum)
    return (*np.squeeze(s_r[:,popt,dum,:]),np.squeeze(popt.T))

def bss_eval_sources(reference_sources, estimated_sources,
                     compute_permutation=True):
    """
    Wrapper to ``bss_eval`` with the right parameters.
    The call to this function is not recommended. See the description for the
    ``bsseval_sources`` parameter of ``bss_eval``.

    """
    (sdr, isr,sir,sar,perm) = \
        bss_eval(reference_sources, estimated_sources,
                    window = np.inf, hop = np.inf,
                    compute_permutation = compute_permutation, flen = 512,
                    bsseval_sources = True)
    return (sdr, sir, sar, perm)

def bss_eval_sources_framewise(reference_sources, estimated_sources,
                               window=30*44100, hop=15*44100,
                               compute_permutation=False):
    """
    Wrapper to ``bss_eval`` with the right parameters.
    The call to this function is not recommended. See the description for the
    ``bsseval_sources`` parameter of ``bss_eval``.

    """
    (sdr, isr,sir,sar,perm) = \
        bss_eval(reference_sources, estimated_sources,
                    window = window, hop = hop,
                    compute_permutation = compute_permutation, flen = 512,
                    bsseval_sources = True)
    return (sdr, sir, sar, perm)

def bss_eval_images(reference_sources, estimated_sources,
                    compute_permutation=True):
    """
    Wrapper to ``bss_eval`` with the right parameters.

    """
    return bss_eval(reference_sources, estimated_sources,
                    window = np.inf, hop = np.inf,
                    compute_permutation = compute_permutation, flen = 512,
                    bsseval_sources = False)

def bss_eval_images_framewise(reference_sources, estimated_sources,
                              window=30*44100, hop=15*44100,
                              compute_permutation=False):
    """
    Framewise computation of bss_eval_images.
    Wrapper to ``bss_eval`` with the right parameters.

    """
    return bss_eval(reference_sources, estimated_sources,
                    window = window, hop = hop,
                    compute_permutation = compute_permutation,  flen = 512,
                    bsseval_sources = False)

def _bss_decomp_mtifilt(reference_sources, estimated_source, j, C, Cj):
    """Decomposition of an estimated source image into four components
    representing respectively the true source image, spatial (or filtering)
    distortion, interference and artifacts, derived from the true source
    images using multichannel time-invariant filters.
    """
    flen = Cj.shape[-2]

    # zero pad
    #s_true = reference_sources[j]
    s_true = _zeropad(reference_sources[j],flen-1,axis=0)
    estimated_source = _zeropad(estimated_source,flen-1,axis=0)

    # compute appropriate projections
    e_spat = _project(reference_sources[j], Cj) - s_true
    e_interf = _project(reference_sources, C) - s_true - e_spat
    e_artif = - s_true - e_spat - e_interf + estimated_source

    return (s_true, e_spat, e_interf, e_artif)

def _zeropad(sig,N,axis=0):
    """pads with N zeros at the end of the signal, along given axis"""
    #ensures concatenation dimension is the first
    sig = np.moveaxis(sig,axis,0)
    #zero pad
    sig = np.pad(sig,[(0,N),*((0,0),)*(len(sig.shape)-1)],
                            mode='constant',constant_values=0)
    #put back axis in place
    sig = np.moveaxis(sig,0,axis)
    return sig

def _reshape_G(G):
    """From a correlation matrix of size
    nsrc X nsrc X nchan X nchan X flen X flen,
    creates a new one of size
    nsrc*nchan*flen X nsrc*nchan*flen"""
    G = np.moveaxis(G,(1,3),(3,4))
    (nsrc,nchan,flen) = G.shape[0:3]
    G = np.reshape(G, (nsrc*nchan*flen,nsrc*nchan*flen),order="F")
    return G

def _compute_reference_correlations(reference_sources,flen):
    """Compute the inner products between delayed versions of reference_sources
    reference is nsrc X nsamp X nchan.
    Returns
    * the gram matrix G : nsrc X nsrc X nchan X nchan X flen X flen
    * the references spectra sf: nsrc X nchan X flen"""

    #reshape references as nsrc X nchan X nsampl
    (nsrc, nsampl, nchan) = reference_sources.shape
    reference_sources = np.moveaxis(reference_sources,(1),(2))

    # zero padding and FFT of references
    reference_sources = _zeropad(reference_sources,flen-1,axis=2)
    n_fft = int(2**np.ceil(np.log2(nsampl + flen - 1.)))
    sf = scipy.fftpack.fft(reference_sources, n=n_fft, axis=2)

    #compute intercorrelation between sources
    G = np.zeros((nsrc,nsrc,nchan,nchan,flen,flen))
    for (i,c1,j,c2) in itertools.product(*(range(nsrc),range(nchan))*2):
        ssf = sf[i,c1] * np.conj(sf[j,c2])
        ssf = np.real(scipy.fftpack.ifft(ssf))
        G[i, j, c1, c2, ...] = toeplitz(np.hstack((ssf[0], ssf[-1:-flen:-1])),
                          r=ssf[:flen])
    return G, sf

def _compute_projection_filters(G,sf,estimated_source):
    """Least-squares projection of estimated source on the subspace spanned by
    delayed versions of reference sources, with delays between 0 and flen-1
    """
    #shapes
    (nsampl,nchan) = estimated_source.shape
    #handles the case where we are calling this with only one source
    #G should be nsrc X nsrc X nchan X nchan X flen X flen
    #and sf should be nsrc X nchan X flen
    if len(G.shape) == 4:
        G = G[None,None,...]
        sf = sf[None,...]
    nsrc = G.shape[0]
    flen = G.shape[-1]

    #zero pad estimates and put chan in first dimension
    estimated_source = _zeropad(estimated_source.T,flen-1,axis=1)

    #compute its FFT
    n_fft = int(2**np.ceil(np.log2(nsampl + flen - 1.)))
    sef = scipy.fftpack.fft(estimated_source, n=n_fft)

    #compute the cross-correlations between sources and estimates
    D = np.zeros((nsrc,nchan,flen,nchan))
    for (j,cj,c) in itertools.product(range(nsrc),range(nchan),range(nchan)):
        ssef = sf[j,cj] * np.conj(sef[c])
        ssef = np.real(scipy.fftpack.ifft(ssef))
        D[j,cj,:,c] = np.hstack((ssef[0], ssef[-1:-flen:-1]))

    #reshape matrices to build the filters
    D = D.reshape(nsrc*nchan*flen, nchan,order='F')
    G = _reshape_G(G)

    # Distortion filters
    try:
        C = np.linalg.solve(G, D).reshape(nsrc,nchan,flen,nchan, order='F')
    except np.linalg.linalg.LinAlgError:
        C = np.linalg.lstsq(G, D)[0].reshape(nsrc,nchan,flen,nchan, order='F')

    # if we asked for one single reference source,
    # return just a nchan X flen matrix
    if nsrc == 1: C = C[0]
    return C

def _project(reference_sources, C):
    """Project images using pre-computed filters C
    reference_sources are nsrc X nsampl X nchan
    C is nsrc X nchan X flen X nchan
    """
    #shapes: ensure that input is 3d (comprising the source index)
    if len(reference_sources.shape)==2:
        reference_sources = reference_sources[None,...]
        C = C[None,...]

    (nsrc,nsampl, nchan) = reference_sources.shape
    flen = C.shape[-2]

    #zero pad
    reference_sources = _zeropad(reference_sources,flen-1,axis = 1)

    sproj = np.zeros((nchan,nsampl+flen-1))
    #sproj = np.zeros((nchan,nsampl))

    for (j,cj,c) in itertools.product(range(nsrc),range(nchan),range(nchan)):
        sproj[c] += fftconvolve(C[j,cj,:,c],
                                reference_sources[j,:,cj])[:nsampl + flen - 1]
    return sproj.T

def _wsum(sig,window,hop,flen,axis=0):
    """ computes a sum on windowed versions of the signal"""
    if window  >= sig.shape[axis] - flen:
        res = np.empty((1,))
        res[0] = np.sum(sig)
        return res

    sig = np.moveaxis(sig,axis,0)
    length = sig.shape[0]
    nwin = int(np.floor((length - flen +1 - window + hop) / hop))
    new_shape = np.array(sig.shape)
    new_shape[0] = nwin
    res = np.empty((nwin,))
    for k in range(nwin):
        if k < nwin:
            win_slice = slice(k * hop, min(length,k * hop + window))
        else:
            win_slice = slice(k * hop, length)
        res[k, ...] = np.sum(sig[win_slice, ...])
    res = np.moveaxis(res, 0, axis)
    return res

def _bss_crit(s_true, e_spat, e_interf, e_artif,
              window, hop, flen, bsseval_sources):
    """Measurement of the separation quality for a given source in terms of
    filtered true source, interference and artifacts.
    windowed version
    """
    # energy ratios
    energy_s_filt = _wsum( (s_true + e_spat)**2, window, hop, flen)
    if bsseval_sources:
        sdr = _safe_db(energy_s_filt,
                   _wsum((e_interf + e_artif)**2, window, hop, flen))
        isr = np.empty(sdr.shape) * np.nan
    else:
        energy_s_true = _wsum( (s_true)**2, window, hop, flen)
        sdr = _safe_db(energy_s_true,
                   _wsum((e_spat + e_interf + e_artif)**2, window, hop, flen))
        isr = _safe_db(energy_s_true, _wsum( e_spat**2, window, hop, flen))

    sir = _safe_db(energy_s_filt, _wsum( e_interf**2, window, hop, flen))
    sar = _safe_db(_wsum((s_true + e_spat + e_interf)**2, window, hop, flen),
                   _wsum( e_artif**2, window, hop, flen))
    return (sdr, isr, sir, sar)

def _safe_db(num, den):
    """Properly handle the potential +Inf db SIR instead of raising a
    RuntimeWarning.
    """
    indices = np.nonzero(den)
    res = np.empty_like(num)
    res[:] = np.inf
    res[indices] = 10 * np.log10(num[indices] / den[indices])
    return res

def evaluate(reference_sources, estimated_sources, **kwargs):
    """Compute all metrics for the given reference and estimated signals.

    NOTE: This will always compute :func:`mir_eval.separation.bss_eval_images`
    for any valid input and will additionally compute
    :func:`mir_eval.separation.bss_eval_sources` for valid input with fewer
    than 3 dimensions.

    Examples
    --------
    >>> # reference_sources[n] should be an ndarray of samples of the
    >>> # n'th reference source
    >>> # estimated_sources[n] should be the same for the n'th estimated source
    >>> scores = mir_eval.separation.evaluate(reference_sources,
    ...                                       estimated_sources)

    Parameters
    ----------
    reference_sources : np.ndarray, shape=(nsrc, nsampl[, nchan])
        matrix containing true sources
    estimated_sources : np.ndarray, shape=(nsrc, nsampl[, nchan])
        matrix containing estimated sources
    kwargs
        Additional keyword arguments which will be passed to the
        appropriate metric or preprocessing functions.

    Returns
    -------
    scores : dict
        Dictionary of scores, where the key is the metric name (str) and
        the value is the (float) score achieved.

    """
    # Compute all the metrics
    scores = collections.OrderedDict()

    sdr, isr, sir, sar, perm = util.filter_kwargs(
        bss_eval_images,
        reference_sources,
        estimated_sources,
        **kwargs
    )
    scores['Images - Source to Distortion'] = sdr.tolist()
    scores['Images - Image to Spatial'] = isr.tolist()
    scores['Images - Source to Interference'] = sir.tolist()
    scores['Images - Source to Artifact'] = sar.tolist()
    scores['Images - Source permutation'] = perm.tolist()

    sdr, isr, sir, sar, perm = util.filter_kwargs(
        bss_eval_images_framewise,
        reference_sources,
        estimated_sources,
        **kwargs
    )
    scores['Images Frames - Source to Distortion'] = sdr.tolist()
    scores['Images Frames - Image to Spatial'] = isr.tolist()
    scores['Images Frames - Source to Interference'] = sir.tolist()
    scores['Images Frames - Source to Artifact'] = sar.tolist()
    scores['Images Frames - Source permutation'] = perm.tolist()

    # Verify we can compute sources on this input
    if reference_sources.ndim < 3 and estimated_sources.ndim < 3:
        sdr, sir, sar, perm = util.filter_kwargs(
            bss_eval_sources_framewise,
            reference_sources,
            estimated_sources,
            **kwargs
        )
        scores['Sources Frames - Source to Distortion'] = sdr.tolist()
        scores['Sources Frames - Source to Interference'] = sir.tolist()
        scores['Sources Frames - Source to Artifact'] = sar.tolist()
        scores['Sources Frames - Source permutation'] = perm.tolist()

        sdr, sir, sar, perm = util.filter_kwargs(
            bss_eval_sources,
            reference_sources,
            estimated_sources,
            **kwargs
        )
        scores['Sources - Source to Distortion'] = sdr.tolist()
        scores['Sources - Source to Interference'] = sir.tolist()
        scores['Sources - Source to Artifact'] = sar.tolist()
        scores['Sources - Source permutation'] = perm.tolist()

    return scores
