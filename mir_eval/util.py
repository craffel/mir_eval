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
