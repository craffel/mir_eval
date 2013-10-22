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

def adjust_segment_boundaries(boundaries, t_min=0.0, t_max=None):
    '''Adjust the given list of boundary times to span the range [t_min, t_max].

    Any boundaries outside of the specified range will be removed.

    If the boundaries do not span [t_min, t_max], additional boundaries will be added.

    :parameters:
        - boundaries : np.array
            Array of boundary times (seconds)

        - t_min : float or None
            Minimum valid boundary time.

        - t_max : float or None
            Maximum valid boundary time.

    :returns:
        - new_boundaries : np.array
            Boundary times corrected to the given range.
    '''
    if t_min is not None:
        boundaries = np.concatenate( ([t_min], boundaries) )
        boundaries = boundaries[t_min <= boundaries]

    if t_max is not None:
        boundaries = np.concatenate( (boundaries, [t_max]) )
        boundaries = boundaries[boundaries <= t_max]

    return np.unique(boundaries)

def import_segment_boundaries(filename, cols=[0,1], sep=None, t_min=0.0, t_max=None):
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

        - t_min : float
            Minimum valid boundary time.

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

    return adjust_segment_boundaries(values, t_min=t_min, t_max=t_max)
