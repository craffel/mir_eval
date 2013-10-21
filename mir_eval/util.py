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

def import_segment_boundaries(filename, cols=[0,1], sep=None, t_max=None):
    '''Import segment boundaries from an annotation file.  
    In typical MIREX fashion, annotations are formatted as:

    START_1     END_1   LABEL_1
    START_2     END_2   LABEL_2
    ...

    where START_* and END_* are given in seconds.
    
    We assume that END_i = START_i+1.

    :parameters:
        - filename : string
            Path to the annotation file

        - cols : list of ints
            Which columns of the file specify the boundaries

        - sep : string or None
            Field delimiter. See `numpy.loadtxt()` for details.

        - t_max : float or None
            If provided, the boundaries will be truncated or expanded 
            to the given time. This is useful for correcting incomplete
            annotations against a ground-truth reference.

    :returns:
        - boundaries : np.array
            List of all segment boundaries
    '''

    # Load the input
    values = np.loadtxt(filename, usecols=cols, delimiter=sep)

    # Flatten out the boundaries, remove dupes
    values = np.unique(values.flatten())

    # Does it start at time 0?
    if values[0] > 0:
        values = np.concatenate( ([0.0], values))

    if t_max is not None:
        # Make sure we're not past the end of the track
        values = values[values <= t_max]

        # Pad out with a silence segment if undercomplete
        if values[-1] < t_max:
            values = np.concatenate((values, [t_max]))

    return values
