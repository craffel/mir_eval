# -*- coding: utf-8 -*-
'''
BSS Eval toolbox, version 4

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

* :func:`mir_eval.separation.bss_eval`: Computes the bss_eval metrics: source
  to distortion (SDR), source to artifacts (SAR), source to interference (SIR)
  ratios, plus the image to spatial ratio (ISR). These are computed on a frame
  by frame basis, (with infinite window size meaning the whole signal).

  Optionally, the distortion filters are time-varying, corresponding to behavior
  of BSS Eval version 3. Furthermore, metrics may optionally correspond to the
  bsseval_sources version, as defined in the BSS Eval version 2.

References
----------
  .. [#liutkus2018bssevalv4] Antoine Liutkus, Fabian-Robert Stöter and Nobutaka
     Ito, "The 2018 Signal Separation Evaluation Campaign," In Proceedings of
     LVA/ICA 2018.
  .. [#vincent2005bssevalv3] Emmanuel Vincent, Rémi Gribonval, and Cédric
      Févotte, "Performance measurement in blind audio source separation," IEEE
      Trans. on Audio, Speech and Language Processing, 14(4):1462-1469, 2006.
  .. [#fevotte2005bssevalv2] Cédric Févotte, Rémi Gribonval and Emmanuel
     Vincent, "BSS_EVAL toolbox user guide - Revision 2.0", Technical Report
     1706, IRISA, April 2005.

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
                         'sources should match. reference_sources.shape '
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
                      "(nsrc, nsample, nchan). sdr, isr sir, sar, and perm will"
                      " all be empty np.ndarrays")
    elif _any_source_silent(reference_sources):
        raise ValueError('All the reference sources should be non-silent (not '
                         'all-zeros), but at least one of the reference '
                         'sources is all 0s, which introduces ambiguity to the'
                         ' evaluation. (Otherwise we can add infinitely many '
                         'all-zero sources.)')

    if estimated_sources.size == 0:
        warnings.warn("estimated_sources is empty, should be of size "
                      "(nsrc, nsample, nchan).  sdr, isr, sir, sar, and perm "
                      "will all be empty np.ndarrays")
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
                         'and estimated_sources.shape[0] = {} which is greater'
                         'than bsseval.MAX_SOURCES = {}.  To '
                         'override this check, set '
                         'bsseval.MAX_SOURCES to a '
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
             filters_len = 512,
             framewise_filters = False,
             bsseval_sources_version = False
):
    """BSS_EVAL version 4.

    Measurement of the separation quality for estimated source signals
    in terms of source to distortion, interference and artifacts ratios,
    (SDR, SIR, SAR) as well as the image to spatial ratio (ISR), as defined
    in [#vincent2005bssevalv3]

    The metrics are computed on a framewise basis, with overlap allowed between
    the windows.

    The key difference between this version 4 and BSS Eval version 3 is the
    possibility of using the same distortion filters for all windows when
    matching the sources to their estimates, instead of estimating the filters
    anew at every frame, as done in BSS Eval v3.

    This implementation is fully compatible with BSS Eval v2 and v3 written
    in MATLAB.

    Examples
    --------
    >>> # reference_sources[n] should be a 2D ndarray, with first dimension the
    >>> # samples and second dimension the channels of the n'th reference
    >>> # source estimated_sources[n] should be the same for the n'th estimated
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
        size of each window for time-varying evaluation. Picking np.inf or any
        integer greater than nsampl will compute metrics on the whole signal.
    hop : int, optional
        hop size between windows
    compute_permutation : bool, optional
        compute all permutations of estimate/source combinations to compute
        the best scores (False by default). Note that picking True will lead
        to a significant computation overhead.
    filters_len : int, optional
        maximum time lag for the computation of the distortion filters. Default
        is filters_len = 512.
    framewise_filters : bool, optional
        Compute a new distortion filter for each frame (False by default). Note
        that picking True as in BSS Eval v2 and v3 leads to a significant
        computation overhead.
    bsseval_sources_version : bool, optional
        if  ``True``, results correspond to the `bss_eval_sources` version from
        the BSS Eval v2 and v3. Note however that this is not recommended
        because this evaluation method modifies the references according to the
        estimated sources, leading to potential problems for the estimation of
        SDR. For instance, zeroing some frequencies in the estimates will lead
        those to also be zeroed in the references, and hence not evaluated,
        artificially boosting results. For this reason, SiSEC always uses
        the `bss_eval_images` version, corresponding to ``False``.

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
        the mean SIR sense (estimated source number ``perm[j,t]`` corresponds to
        true source number ``j`` at window ``t``).
        Note: ``perm`` will be ``(0,2,...,nsrc-1)`` if ``compute_permutation``
        is ``False``.

    References
    ----------
  .. [#liutkus2018bssevalv4] Antoine Liutkus, Fabian-Robert Stöter and Nobutaka
     Ito, "The 2018 Signal Separation Evaluation Campaign," In Proceedings of
     LVA/ICA 2018.
  .. [#vincent2005bssevalv3] Emmanuel Vincent, Rémi Gribonval, and Cédric
      Févotte, "Performance measurement in blind audio source separation," IEEE
      Trans. on Audio, Speech and Language Processing, 14(4):1462-1469, 2006.

    """
    # make sure the input has 3 dimensions
    # assuming input is in shape (nsampl) or (nsrc, nsampl)
    estimated_sources = np.atleast_3d(estimated_sources)
    reference_sources = np.atleast_3d(reference_sources)

    # validate input
    validate(reference_sources, estimated_sources)

    # If empty matrices were supplied, return empty lists (special case)
    if reference_sources.size == 0 or estimated_sources.size == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([])
        )

    # determine shape parameters
    (nsrc, nsampl, nchan) = estimated_sources.shape

    # defines all the permutations desired by user
    if compute_permutation:
        candidate_permutations = np.array(list(
                    itertools.permutations(range(nsrc))))
    else:
        candidate_permutations = np.array(range(nsrc))[None, :]

    # initialize variables
    nwin = Framing.count(window, hop, nsampl)
    #print('\n--------\n','nsrc = %d, nsampl=%d, nchan=%d,nwin=%d'%(nsrc,nsampl,nchan,nwin),'\n---------\n')

    (SDR, ISR, SIR, SAR) = range(4)
    s_r = np.empty((4, nsrc, nsrc, nwin))

    # define helper functions for computing filters on windows of the signals
    def compute_GsfC(win = slice(0, nsampl)):
        # First compute the references correlations
        G, sf = _compute_reference_correlations(reference_sources[:,win], filters_len)
        # compute the interference distortion filters
        C = np.zeros((nsrc, nsrc, nchan, filters_len, nchan))
        for jtrue in range(nsrc):
            C[jtrue] = _compute_projection_filters(G, sf, estimated_sources[jtrue,win])
        return (G,sf,C)
    def compute_Cj(win = slice(0, nsampl)):
        Cj = np.zeros((nsrc, nsrc, 1, nchan, filters_len, nchan))
        for jtrue in range(nsrc):
            for jest in candidate_permutations[:, jtrue]:
                # compute the projection filters for this combination
                Cj[jtrue,jest] = _compute_projection_filters(
                    G[jtrue, jtrue],
                    sf[jtrue],
                    estimated_sources[jest,win]
                )
        return Cj

    if not framewise_filters:
        # compute filters on whole signals if no framewise filters
        (G,sf,C) = compute_GsfC()
        Cj = compute_Cj()

    # loop over all windows
    for (t,win) in enumerate(Framing(window, hop, nsampl)):
        # if we have time-varying distortion filters
        if framewise_filters:
            (G,sf,C) = compute_GsfC(win)
            Cj = compute_Cj(win)

        #loop over all permutations
        done = np.zeros((nsrc,nsrc))
        for jtrue in range(nsrc):
            for (k, jest) in enumerate(candidate_permutations[:, jtrue]):
                if not done[jtrue,jest]:
                    s_true, e_spat, e_interf, e_artif = \
                        _bss_decomp_mtifilt(
                            reference_sources[:,win],
                            estimated_sources[jest,win], jtrue, C[jest], Cj[jtrue,jest,0])
                    s_r[:, jtrue, jest, t] = _bss_crit(
                        s_true, e_spat, e_interf, e_artif, bsseval_sources_version
                    )
                    #print(jtrue,jest,t,np.sum(s_true**2),np.sum(reference_sources[:,win]**2),np.sum(e_interf**2),'nrustn')
                    done[jtrue, jest] = True

    # select the best ordering
    if framewise_filters:
        # if we have framewise filters, output one permutation for each window
        mean_sir = np.empty((len(candidate_permutations),nwin))
        axis_mean = 0
    else:
        # otherwise, output one permutation for the whole signal as the best
        # average one
        mean_sir = np.empty((len(candidate_permutations),1))
        axis_mean = None
    dum = np.arange(nsrc)
    for (i, perm) in enumerate(candidate_permutations):
        mean_sir[i] = np.mean(s_r[SIR, dum, perm, :],axis = axis_mean)
    popt = candidate_permutations[np.argmax(mean_sir,axis=0)].T

    #now prepare the output
    if not framewise_filters:
        return (*(s_r[:, dum, popt[:,0], :]), popt)
    else:
        result = np.empty((4,nsrc,nwin))
        for (m,t) in itertools.product(range(4),range(nwin)):
            result[m,:,t] = s_r[m, dum, popt[:,t], t]
        return (*result, popt)

def bss_eval_sources(reference_sources, estimated_sources,
                     compute_permutation=True):
    """
    BSS Eval v3 bss_eval_sources

    Wrapper to ``bss_eval`` with the right parameters.
    The call to this function is not recommended. See the description for the
    ``bsseval_sources`` parameter of ``bss_eval``.

    """
    (sdr, isr, sir, sar, perm) = \
        bss_eval(
            reference_sources, estimated_sources,
            window=np.inf, hop=np.inf,
            compute_permutation=compute_permutation, filters_len=512,
            framewise_filters = True,
            bsseval_sources_version=True
        )
    return (sdr, sir, sar, perm)


def bss_eval_sources_framewise(reference_sources, estimated_sources,
                               window=30*44100, hop=15*44100,
                               compute_permutation=False):
    """
    BSS Eval v3 bss_eval_sources_framewise

    Wrapper to ``bss_eval`` with the right parameters.
    The call to this function is not recommended. See the description for the
    ``bsseval_sources`` parameter of ``bss_eval``.

    """
    (sdr, isr, sir, sar, perm) = \
        bss_eval(
            reference_sources, estimated_sources,
            window=window, hop=hop,
            compute_permutation=compute_permutation, filters_len=512,
            framewise_filters = True,
            bsseval_sources_version=True)
    return (sdr, sir, sar, perm)


def bss_eval_images(reference_sources, estimated_sources,
                    compute_permutation=True):
    """
    BSS Eval v3 bss_eval_images

    Wrapper to ``bss_eval`` with the right parameters.

    """
    return bss_eval(
        reference_sources, estimated_sources,
        window=np.inf, hop=np.inf,
        compute_permutation=compute_permutation, filters_len=512,
        framewise_filters = True,
        bsseval_sources_version=False)


def bss_eval_images_framewise(reference_sources, estimated_sources,
                              window=30*44100, hop=15*44100,
                              compute_permutation=False):
    """
    BSS Eval v3 bss_eval_images_framewise

    Framewise computation of bss_eval_images.
    Wrapper to ``bss_eval`` with the right parameters.

    """
    return bss_eval(
        reference_sources, estimated_sources,
        window=window, hop=hop,
        compute_permutation=compute_permutation, filters_len=512,
        framewise_filters = True,
        bsseval_sources_version=False
    )


#Helper functions
class Framing:
    """ helper iterator class to do overlapped windowing"""
    def count(window, hop,len):
        if window < len:
            return int(
                np.floor((len - window + hop) / hop)
                )
        else:
            return 1

    def __init__(self, window, hop, len):
        self.current = 0
        self.window = window
        self.hop = hop
        self.len = len
        self.nwin = Framing.count(window, hop, len)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.nwin:
            raise StopIteration
        else:
            start = self.current * self.hop
            if np.isnan(start) or np.isinf(start):
                start = 0
            stop = min(self.current * self.hop + self.window, self.len)
            if np.isnan(stop) or np.isinf(stop): stop = self.len
            result = slice(start,stop)
            self.current += 1
            return result


def _bss_decomp_mtifilt(reference_sources, estimated_source, j, C, Cj):
    """Decomposition of an estimated source image into four components
    representing respectively the true source image, spatial (or filtering)
    distortion, interference and artifacts, derived from the true source
    images using multichannel time-invariant filters.
    """
    filters_len = Cj.shape[-2]

    # zero pad
    s_true = _zeropad(reference_sources[j], filters_len-1, axis=0)
    estimated_source = _zeropad(estimated_source, filters_len-1, axis=0)

    # compute appropriate projections
    e_spat = _project(reference_sources[j], Cj) - s_true
    e_interf = _project(reference_sources, C) - s_true - e_spat
    e_artif = - s_true - e_spat - e_interf + estimated_source

    return (s_true, e_spat, e_interf, e_artif)


def _zeropad(sig, N, axis=0):
    """pads with N zeros at the end of the signal, along given axis"""
    # ensures concatenation dimension is the first
    sig = np.moveaxis(sig, axis, 0)
    # zero pad
    out = np.zeros((sig.shape[0] + N,) + sig.shape[1:])
    out[:sig.shape[0], ...] = sig
    # put back axis in place
    sig = np.moveaxis(out, 0, axis)
    return sig


def _reshape_G(G):
    """From a correlation matrix of size
    nsrc X nsrc X nchan X nchan X filters_len X filters_len,
    creates a new one of size
    nsrc*nchan*filters_len X nsrc*nchan*filters_len"""
    G = np.moveaxis(G, (1, 3), (3, 4))
    (nsrc, nchan, filters_len) = G.shape[0:3]
    G = np.reshape(G, (nsrc*nchan*filters_len, nsrc*nchan*filters_len), order="F")
    return G


def _compute_reference_correlations(reference_sources, filters_len):
    """Compute the inner products between delayed versions of reference_sources
    reference is nsrc X nsamp X nchan.
    Returns
    * the gram matrix G : nsrc X nsrc X nchan X nchan X filters_len X filters_len
    * the references spectra sf: nsrc X nchan X filters_len"""

    # reshape references as nsrc X nchan X nsampl
    (nsrc, nsampl, nchan) = reference_sources.shape
    reference_sources = np.moveaxis(reference_sources, (1), (2))

    # zero padding and FFT of references
    reference_sources = _zeropad(reference_sources, filters_len - 1, axis = 2)
    n_fft = int(2**np.ceil(np.log2(nsampl + filters_len - 1.)))
    sf = scipy.fftpack.fft(reference_sources, n = n_fft, axis  = 2)

    # compute intercorrelation between sources
    G = np.zeros((nsrc, nsrc, nchan, nchan, filters_len, filters_len))
    for ((i, c1), (j, c2)) in itertools.combinations_with_replacement(
                            itertools.product(range(nsrc), range(nchan)), 2):
        ssf = sf[j, c2] * np.conj(sf[i, c1])
        ssf = np.real(scipy.fftpack.ifft(ssf))
        ss = toeplitz(
            np.hstack((ssf[0], ssf[-1:-filters_len:-1])),
            r=ssf[:filters_len]
            )
        G[j, i, c2, c1] = ss
        G[i, j, c1, c2] = ss.T
    return G, sf


def _compute_projection_filters(G, sf, estimated_source):
    """Least-squares projection of estimated source on the subspace spanned by
    delayed versions of reference sources, with delays between 0 and filters_len-1
    """
    # shapes
    (nsampl, nchan) = estimated_source.shape
    # handles the case where we are calling this with only one source
    # G should be nsrc X nsrc X nchan X nchan X filters_len X filters_len
    # and sf should be nsrc X nchan X filters_len
    if len(G.shape) == 4:
        G = G[None, None, ...]
        sf = sf[None, ...]
    nsrc = G.shape[0]
    filters_len = G.shape[-1]

    # zero pad estimates and put chan in first dimension
    estimated_source = _zeropad(estimated_source.T, filters_len - 1, axis = 1)

    # compute its FFT
    n_fft = int(2**np.ceil(np.log2(nsampl + filters_len - 1.)))
    sef = scipy.fftpack.fft(estimated_source, n = n_fft)

    # compute the cross-correlations between sources and estimates
    D = np.zeros((nsrc, nchan, filters_len, nchan))
    for (j, cj, c) in itertools.product(
        range(nsrc), range(nchan), range(nchan)
    ):
        ssef = sf[j, cj] * np.conj(sef[c])
        ssef = np.real(scipy.fftpack.ifft(ssef))
        D[j, cj, :, c] = np.hstack((ssef[0], ssef[-1:-filters_len:-1]))

    # reshape matrices to build the filters
    D = D.reshape(nsrc*nchan*filters_len, nchan, order='F')
    G = _reshape_G(G)

    # Distortion filters
    try:
        C = np.linalg.solve(G, D).reshape(
            nsrc, nchan, filters_len, nchan, order='F'
        )
    except np.linalg.linalg.LinAlgError:
        C = np.linalg.lstsq(G, D)[0].reshape(
            nsrc, nchan, filters_len, nchan, order='F'
        )

    # if we asked for one single reference source,
    # return just a nchan X filters_len matrix
    if nsrc == 1:
        C = C[0]
    return C


def _project(reference_sources, C):
    """Project images using pre-computed filters C
    reference_sources are nsrc X nsampl X nchan
    C is nsrc X nchan X filters_len X nchan
    """
    # shapes: ensure that input is 3d (comprising the source index)
    if len(reference_sources.shape) == 2:
        reference_sources = reference_sources[None, ...]
        C = C[None, ...]

    (nsrc, nsampl, nchan) = reference_sources.shape
    filters_len = C.shape[-2]

    # zero pad
    reference_sources = _zeropad(reference_sources, filters_len-1, axis=1)
    sproj = np.zeros((nchan, nsampl+filters_len-1))

    for (j, cj, c) in itertools.product(
        range(nsrc), range(nchan), range(nchan)
    ):
        sproj[c] += fftconvolve(
            C[j, cj, :, c],
            reference_sources[j, :, cj]
        )[:nsampl + filters_len - 1]
    return sproj.T


def _bss_crit(s_true, e_spat, e_interf, e_artif,bsseval_sources_version):
    """Measurement of the separation quality for a given source in terms of
    filtered true source, interference and artifacts.

    """
    # energy ratios
    if bsseval_sources_version:
        s_filt = s_true + e_spat
        energy_s_filt = np.sum(s_filt**2)
        sdr = _safe_db(energy_s_filt,
                       np.sum((e_interf + e_artif)**2))
        isr = np.empty(sdr.shape) * np.nan
        sir = _safe_db(energy_s_filt, np.sum(e_interf**2))
        sar = _safe_db(np.sum((s_filt + e_interf)**2),
                       np.sum(e_artif**2))
    else:
        energy_s_true = np.sum((s_true)**2)
        sdr = _safe_db(energy_s_true,
                       np.sum((e_spat + e_interf + e_artif)**2))
        isr = _safe_db(energy_s_true, np.sum(e_spat**2))
        sir = _safe_db(np.sum((s_true + e_spat)**2) , np.sum(e_interf**2))
        sar = _safe_db(np.sum((s_true + e_spat + e_interf)**2) ,
            np.sum(e_artif**2))

    return (sdr, isr, sir, sar)


def _safe_db(num, den):
    """Properly handle the potential +Inf db SIR instead of raising a
    RuntimeWarning.
    """
    if den == 0:
        return np.inf
    else:
        return 10 * np.log10(num / den)

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
