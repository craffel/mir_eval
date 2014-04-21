"""Annotation input/output functions"""

import numpy as np
import re
import os


def load_events(filename, delimiter=r'\s+', converter=None, label_prefix='__'):
    r'''Import time-stamp events from an annotation file.  This is primarily useful for
        processing events which lack duration, such as beats or onsets.

        The annotation file may be either of two formats:
        - Single-column.  Each line contains a single value, corresponding to the time of
        the annotated event.

        - Double-column.  Each line contains two values, separated by ``delimiter``: the
        first contains the time of the annotated event, and the second contains its
        label.

        :parameters:
        - filename : str
        Path to the annotation file

        - delimiter : str
        Separator regular expression.
        By default, lines will be split by any amount of whitespace ('\s+')

        - converter : function
        Function to convert time-stamp data into numerics. Defaults to float().

        - label_prefix : str
        String to append to any synthetically generated labels

        :returns:
        - event_times : np.ndarray
        array of event times (float)
        - event_labels : list of str
        list of corresponding event labels
        '''

    if converter is None:
        converter = float

    times   = []
    labels  = []

    splitter = re.compile(delimiter)

    # Note: we do io manually here for two reasons.
    #   1. The csv module has difficulties with unicode, which may lead
    #      to failures on certain annotation strings
    #
    #   2. numpy's text loader does not handle non-numeric data
    #
    with open(filename, 'r') as input_file:
        for row, line in enumerate(input_file, 1):
            data = splitter.split(line.strip(), 1)

            if len(data) == 1:
                times.append(converter(data[0]))
                labels.append('%s%d' % (label_prefix, row))

            elif len(data) == 2:
                times.append(converter(data[0]))
                labels.append(data[1])

            else:
                raise ValueError('Parse error on %s:%d:\n\t%s' % (filename, row, line))

    times = np.asarray(times)

    return times, labels


def load_intervals(filename, delimiter=r'\s+', converter=None, label_prefix='__'):
    r'''Import labeled intervals from an annotation file.  This is primarily useful for
        processing events which span a duration, such as segmentation, chords, or instrument
        activation.

        The annotation file may be either of two formats:
        - Double-column.  Each line contains two values, separated by ``delimiter``,
        corresponding to the start and end time annotated event.

        - Triple-column.  Each line contains three values, separated by ``delimiter``.
        The first two values specify the start and end times, the last value specifies
        the label for the event (e.g. "Verse" or "A:min").

        :parameters:
        - filename : str
        Path to the annotation file

        - delimiter : str
        Separator regular expression.
        By default, lines will be split by any amount of whitespace ('\s+')

        - converter : function
        Function to convert time-stamp data into numerics. Defaults to float().

        - label_prefix : str
        String to append to any synthetically generated labels

        :returns:
        - event_times : np.ndarray, shape=(n_events, 2)
        array of event start and end times

        - event_labels : list of str
        list of corresponding event labels
        '''

    if converter is None:
        converter = float

    times   = []
    labels  = []

    splitter = re.compile(delimiter)

    with open(filename, 'r') as input_file:
        for row, line in enumerate(input_file, 1):
            data = splitter.split(line.strip(), 2)

            if len(data) == 2:
                times.append([converter(data[0]), converter(data[1])])
                labels.append('%s%d' % (label_prefix, row))

            elif len(data) == 3:
                times.append([converter(data[0]), converter(data[1])])
                labels.append(data[2])

            else:
                raise ValueError('parse error %s:%d:\n%s' % (filename, row, line))

    times = np.asarray(times)

    return times, labels


def load_time_series(filename, delimiter=None):
    r'''Import a time series from an annotation file.  This is primarily useful for
        processing dense time series with timestamps and corresponding numeric values

        The annotation file must be of the following format:
        - Double-column.  Each line contains two values, separated by ``delimiter``: the
        first contains the timestamp, and the second contains its corresponding
        numeric value.

        :parameters:
        - filename : str
        Path to the annotation file

        - delimiter : str
        Column separator. By default, lines will be split by any amount of
        whitespace, unless the file ending is .csv, in which case a comma ','
        is used as the delimiter.

        :returns:
        - times : np.ndarray
        array of timestamps (float)
        - values : np.ndarray
        array of corresponding numeric values (float)
        '''

    # Note: unlike load_events, here we expect float data in both columns,
    # so we can just use numpy's text load (np.loadtxt)

    if os.path.splitext(filename)[1] == '.csv':
        delimiter = ','

    try:
        data = np.loadtxt(filename,'float','#',delimiter)
    except ValueError:
        raise ValueError('Error: could no load %s, please check if it is in the correct 2 column format' % os.path.basename(filename))

    data = data.T

    # we do however want to make sure the data is in the right format!
    if data.shape[0] != 2:
        raise ValueError('Error: %s should be of dimension (2,x), but is of dimension %s' % (os.path.basename(filename),data.shape))

    times = data[0]
    values = data[1]

    return times, values


def load_patterns(filename):
    """Loads the patters contained in the filename and puts them into a list
    of patterns, each pattern being a list of occurrence, and each
    occurrence being a list of (onset, midi) pairs.

    The input file must be formatted as described in MIREX 2013:
        http://www.music-ir.org/mirex/wiki/2013:Discovery_of_Repeated_Themes_\
            %26_Sections

    :param filename: The input file path containing the patterns of a given
        given piece using the MIREX 2013 format.
    :type filename: str.
    :returns: list -- the list of patterns, containing all their occurrences,
        using the following format:
            P = [pattern1, ..., patternN]
            pattern = [occurrence1, ..., occurrenceM]
            occurrence = [onset_midi1, ..., onset_midiO]
            onset_midi = (onset_time, midi_number)
        E.g.:
            P = [[[(77.0, 67.0), (77.5, 77.0), ... ]]]
    """

    def append_if_necessary(list, element):
        """Appends element to a list only if the element is not an empty
        list."""
        if element != []:
            list.append(element)

    with open(filename, "r") as input_file:
        P = []              # List with all the patterns
        pattern = []        # Current pattern, which will contain all occs
        occurrence = []     # Current occurrence, containing (onset, midi)
        for line in input_file.readlines():
            if "pattern" in line:
                append_if_necessary(pattern, occurrence)
                append_if_necessary(P, pattern)
                occurrence = []
                pattern = []
                continue
            if "occurrence" in line:
                append_if_necessary(pattern, occurrence)
                occurrence = []
                continue
            string_values = line.split(",")
            onset_midi = (float(string_values[0]), float(string_values[1]))
            occurrence.append(onset_midi)

        # Add last occurrence and pattern to P
        append_if_necessary(pattern, occurrence)
        append_if_necessary(P, pattern)

    return P
