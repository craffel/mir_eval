"""Utility sub-module for mir-eval"""

import numpy as np

def multiline(filename, sep='\t'):
    '''Iterate over rows from a ragged, multi-line file.
        This is primarily useful for tasks which may contain multiple variable-length
        annotations per track: eg, beat tracking, onset detection, or structural
        segmentation.

    :example:
        beat_annotations.txt:
            0.5, 1.0, 1.5
            0.25, 0.5, 0.75, 1.0, 1.25, 1.5

        multiline('beat_annotations.txt')

        will iterate over the lines of the file, yielding an np.array for each line.

    :parameters:
        - filename : string
            Path to the input file

        - sep : string
            Field delimiter

    :yields:
        - row : np.array
            Contents of each line of the file
    '''

    with open(filename, 'r') as f:
        for line in f:
            yield np.fromstring(line, sep=sep)
    pass

def adjust_boundaries(boundaries, labels=None, t_min=0.0, t_max=None, prefix='__'):
    '''Adjust the given list of boundary times to span the range [t_min, t_max].

    Any boundaries outside of the specified range will be removed.

    If the boundaries do not span [t_min, t_max], additional boundaries will be added.

    :parameters:
        - boundaries : np.array
            Array of boundary times (seconds)

        - labels : list or None
            Array of labels

        - t_min : float or None
            Minimum valid boundary time.

        - t_max : float or None
            Maximum valid boundary time.

    :returns:
        - new_boundaries : np.array
            Boundary times corrected to the given range.
    '''
    if t_min is not None:
        first_idx = np.argwhere(boundaries >= t_min)

        if len(first_idx) > 0:
            # We have boundaries below t_min
            # Crop them out
            if labels is not None:
                labels = labels[first_idx[0]:]
            boundaries = boundaries[first_idx[0]:]

        if boundaries[0] > t_min:
            # Lowest boundary is higher than t_min: add a new boundary and label
            boundaries = np.concatenate( ([t_min], boundaries) )
            if labels is not None:
                labels.insert(0, '%sT_MIN' % prefix)

    if t_max is not None:
        last_idx = np.argwhere(boundaries > t_max)

        if len(last_idx) > 0:
            # We have boundaries above t_max.
            # Trim to only boundaries <= t_max
            if labels is not None:
                labels = labels[:last_idx[0]]
            boundaries = boundaries[:last_idx[0]]

        if boundaries[-1] < t_max:
            # Last boundary is below t_max: add a new boundary and label
            boundaries = np.concatenate( (boundaries, [t_max]))
            if labels is not None:
                labels.append('%sT_MAX' % prefix)

    return boundaries, labels

def import_segments(filename, sep='\t', t_min=0.0, t_max=None, prefix='__', converter=float):
    '''Import segments from an annotation file.

    :parameters:
      - filename : str
          Path to the annotation file
      - sep : str
          Separator character
      - t_min : float >=0 or None
          Trim or pad to this minimum time
      - t_max : float >=0 or None
          Trim or pad to this maximum time
      - prefix : str
          String to append to any synthetically generated labels
      - converter : function
          Function to convert time-stamp data into numerics. Defaults to float().

    :returns:
      - seg_times : np.ndarray
          List of segment boundaries
      - seg_labels : list of str
          Labels for each segment boundary

    :raises:
      - ValueError
          If the input data is improperly formed.
    '''

    starts = []
    ends   = []
    labels = []


    def interpret_data(data, HAS_ENDS, HAS_LABELS):

        # If we've already interpreted the data, skip out
        if HAS_ENDS is not None:
            return HAS_ENDS, HAS_LABELS

        if len(data) == 1:
            # Only have start times
            HAS_ENDS   = False
            HAS_LABELS = False
        elif len(data) == 3:
            # Start times, end times, and labels
            HAS_ENDS   = True
            HAS_LABELS = True
        else:
            # If the converter throws a ValueError on the last column,
            # treat it as a label.
            try:
                converter(data[-1])
                HAS_ENDS = True
                HAS_LABELS = False
            except ValueError:
                HAS_LABELS = True
                HAS_ENDS = False

        return HAS_ENDS, HAS_LABELS

    HAS_ENDS   = None
    HAS_LABELS = None

    with open(filename, 'r') as f:
        for row, data in enumerate(f):

            # Split the data, filter out empty columns
            data = filter(lambda x: len(x) > 0, data.strip().split(sep, 3))

            HAS_ENDS, HAS_LABELS = interpret_data(data, HAS_ENDS, HAS_LABELS)

            if HAS_LABELS:
                labels.append(data[-1])
            else:
                labels.append('%s%03d' % (prefix, row))

            if HAS_ENDS:
                ends.append(converter(data[1]))

            starts.append(converter(data[0]))

    if HAS_ENDS:
        # We need to make an extra label for that last boundary time
        labels.append('%sEND' % prefix)
    else:
        # We need to generate the ends vector, and shuffle starts
        ends    = starts[1:]
        starts  = starts[:-1]

    # Verify that everything is in proper order
    starts      = np.asarray(starts)
    ends        = np.asarray(ends)
    bad_segs    = np.argwhere(starts > ends).flatten()
    if len(bad_segs) > 0:
        raise ValueError('Segment end precedes start at %s:%d start=%0.3f, end=%0.3f' % (
                            filename, 1+row, starts[bad_segs[0]], ends[bad_segs[0]]))

    seg_times = np.concatenate([starts, ends[-1:]])

    return adjust_boundaries(seg_times, labels=labels, t_min=t_min, t_max=t_max, prefix=prefix)


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
