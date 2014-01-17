"""Annotation input/output functions"""

import numpy as np

from . import util

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

    return util.adjust_boundaries(seg_times, labels=labels, t_min=t_min, t_max=t_max, prefix=prefix)
