"""Utility sub-module for mir-eval"""

import numpy as np

def f_measure(precision, recall, beta=1.0):
    '''Compute the f-measure from precision and recall scores.

    :parameters:
        - precision : float in (0, 1]
            Precision

        - recall : float in (0, 1]
            Recall

        - beta : float > 0
            Weighting factor for f-measure

    :returns:
        - f_measure : float
            The weighted f-measure
    '''

    if precision == 0 and recall == 0:
        return 0.0

    return (1 + beta**2) * precision * recall / ((beta**2) * precision + recall)

def segments_to_boundaries(times, labels=None, label_prefix='__'):
    '''Convert segment start-end times into boundaries.
    :parameters:
      - times : np.ndarray, shape=(n_events, 2)
          Array of segment start and end-times
          
      - labels : None or list of str
          Optional list of strings describing each event
          
    :returns:
      - boundaries : np.ndarray, shape=(n_segments + 1)
          Segment boundary times, including the end of the final segment

      - labels : list of str or None
          Labels for each event
    '''

    boundaries = np.unique(np.ravel(times))

    if labels is None:
        boundary_labels = None
    else:
        boundary_labels = [seg_label for seg_label in labels]
        boundary_labels.append('%sEND' % label_prefix)

    return boundaries, boundary_labels

def boundaries_to_segments(boundaries, labels=None):
    '''Convert event boundaries into segments
    
    :parameters:
      - boundaries : list-like
          List of event times
          
      - labels : None or list of str
          Optional list of strings describing each event
          
    :returns:
      - segments : np.ndarray, shape=(n_segments, 2)
          Start and end time for each segment

      - labels : list of str or None
          Labels for each event.
    '''

    segments = np.asarray(zip(boundaries[:-1], boundaries[1:]))

    if labels is None:
        segment_labels = None
    else:
        segment_labels = labels[:-1]

    return segments, segment_labels

def adjust_times(times, labels=None, t_min=0.0, t_max=None, label_prefix='__'):
    '''Adjust the given list of event times to span the range [t_min, t_max].

    Any event times outside of the specified range will be removed.

    If the times do not span [t_min, t_max], additional events will be inserted.

    :parameters:
        - times : np.array
            Array of event times (seconds)

        - labels : list or None
            Array of labels

        - t_min : float or None
            Minimum valid event time.

        - t_max : float or None
            Maximum valid event time.
        
        - label_prefix : str
            Prefix string to use for synthetic labels

    :returns:
        - new_times : np.array
            Event times corrected to the given range.
    '''
    if t_min is not None:
        first_idx = np.argwhere(times>= t_min)

        if len(first_idx) > 0:
            # We have events below t_min
            # Crop them out
            if labels is not None:
                labels = labels[first_idx[0]:]
            times = times[first_idx[0]:]

        if times[0] > t_min:
            # Lowest boundary is higher than t_min: add a new boundary and label
            times = np.concatenate( ([t_min], times) )
            if labels is not None:
                labels.insert(0, '%sT_MIN' % label_prefix)

    if t_max is not None:
        last_idx = np.argwhere(times > t_max)

        if len(last_idx) > 0:
            # We have boundaries above t_max.
            # Trim to only boundaries <= t_max
            if labels is not None:
                labels = labels[:last_idx[0]]
            times = times[:last_idx[0]]

        if times[-1] < t_max:
            # Last boundary is below t_max: add a new boundary and label
            times = np.concatenate( (times, [t_max]))
            if labels is not None:
                labels.append('%sT_MAX' % label_prefix)

    return times, labels

def nextpow2(x):
    '''Compute the smallest n such that 2^n >= x
    '''
    return np.ceil(np.log2(x))

def fftfilt(b, x, n_fft=None):
    '''Raw translation of MATLAB fftfilt function (for only 1-d signal).
        Filter the 1-d signal x with the FIR filter described by the
        coefficients in b using the overlap-add method. If the FFT
        length n_fft is not specified, it and the overlap-add block length
        are selected so as to minimize the computational cost of
        the filtering operation.

    :parameters:
      - b : np.array
          FIR filter coefficients

      - x : np.array
          Input signal to be filtered

      - n_fft : float or None
          FFT size

    :returns:
      - y : np.array
          The filtered output signal

    :raises:
      - ValueError
          If the FIR filter is too long (> 2^20)
    '''
    nx = len(x)
    nb = len(b)
    if nb > 2**20:
        raise ValueError('Filters of length greater than 2^20 are not supported.')

    # Determine the FFT and block lengths to use:
    if n_fft is None:
        if nb >= nx:
            n_fft = 2**nextpow2(nb + nx - 1)
            L = nx
        else:
            fft_flops = np.array([18, 59, 138, 303, 660, 1441, 3150, 6875,
                                  14952, 32373, 69762, 149647, 319644, 680105,
                                  1441974, 3047619, 6422736, 13500637,
                                  28311786, 59244791])
            n = 2**np.arange(1, 21)
            dum = (n > (nb - 1))
            n, fft_flops = n[dum], fft_flops[dum]
            L = n - nb + 1
            idx = np.argmin(np.ceil(float(nx) / L) * fft_flops)
            n_fft, L = n[idx], L[idx]
    else:
        if n_fft < nb:
            n_fft = nb
        n_fft = 2**nextpow2(n_fft)
        L = n_fft - nb + 1

    n_fft = int(n_fft)
    B = np.fft.fft(b, n=n_fft)
    y = np.zeros_like(x, dtype='complex')

    # overlap and add
    istart = 0
    while istart < nx:
        iend = min(istart + L, nx)
        if iend == istart + 1:
            # need to fft a scalar
            X = x[istart] * np.ones((n_fft, 1))
        else:
            X = np.fft.fft(x[istart: iend], n=n_fft)
        Y = np.fft.ifft(X * B)
        yend = min(nx, istart + n_fft)
        y[istart: yend] += Y[:(yend - istart)]
        istart += L
    if not (np.any(np.imag(x)) or np.any(np.imag(b))):
        y = np.real(y)
    return y
