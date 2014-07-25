"""Annotation input/output functions"""

import numpy as np
import re
import os
import warnings

from . import util


def load_delimited(filename, converters, delimiter=r'\s+'):
    '''
    Utility function for loading in data from an annotation file where columns
    are delimited.  The number of columns is inferred from the length of
    the provided converters list.

    :usage:
        >>> # Load in a one-column list of event times (floats)
        >>> load_delimited('events.tsv', [float])
        >>> # Load in a list of labeled events, separated by commas
        >>> load_delimited('labeled_events.csv', [float, str], ',')

    :parameters:
         - filename : str
            Path to the annotation file
         - converters : list of functions
            Each entry in column n of the file will be cast by the function
            converters[n].
         - delimiter : str
            Separator regular expression.
            By default, lines will be split by any amount of whitespace ('\s+')

    :returns:
        - columns : tuple of lists
            Each list in this tuple corresponds to values in one of the columns
            in the file.
    '''
    # Initialize list of empty lists
    n_columns = len(converters)
    columns = tuple(list() for _ in xrange(n_columns))

    # Create re object for splitting lines
    splitter = re.compile(delimiter)

    # Note: we do io manually here for two reasons.
    #   1. The csv module has difficulties with unicode, which may lead
    #      to failures on certain annotation strings
    #
    #   2. numpy's text loader does not handle non-numeric data
    #
    with open(filename, 'r') as input_file:
        for row, line in enumerate(input_file, 1):
            # Split each line using the supplied delimiter
            data = splitter.split(line.strip(), n_columns - 1)

            # Throw a helpful error if we got an unexpected # of columns
            if n_columns != len(data):
                raise ValueError('Expected {} columns, got {} at '
                                 '{}:{:d}:\n\t{}'.format(n_columns, len(data),
                                                         filename, row, line))

            for value, column, converter in zip(data, columns, converters):
                # Try converting the value, throw a helpful error on failure
                try:
                    converted_value = converter(value)
                except:
                    raise ValueError("Couldn't convert value {} using {} "
                                     "found at {}:{:d}:\n\t{}".format(
                                         value, converter.__name__, filename,
                                         row, line))
                column.append(converted_value)

    # Sane output
    if n_columns == 1:
        return columns[0]
    else:
        return columns


def load_events(filename, delimiter=r'\s+'):
    r'''
    Import time-stamp events from an annotation file.  The file should
    consist of a single column of numeric values corresponding to the event
    times. This is primarily useful for processing events which lack duration,
    such as beats or onsets.

    :parameters:
        - filename : str
            Path to the annotation file

        - delimiter : str
            Separator regular expression.
            By default, lines will be split by any amount of whitespace ('\s+')

    :returns:
        - event_times : np.ndarray
            array of event times (float)

    '''
    # Use our universal function to load in the events
    events = load_delimited(filename, [float], delimiter)
    events = np.array(events)
    # Validate them, but throw a warning in place of an error
    try:
        util.validate_events(events)
    except ValueError as error:
        warnings.warn(error.args[0])

    return events


def load_labeled_events(filename, delimiter=r'\s+'):
    r'''
    Import labeled time-stamp events from an annotation file.  The file should
    consist of two columns; the first having numeric values corresponding to
    the event times and the second having string labels for each event.  This
    is primarily useful for processing labeled events which lack duration, such
    as beats with metric beat number or onsets with an instrument label.

    :parameters:
        - filename : str
            Path to the annotation file

        - delimiter : str
            Separator regular expression.
            By default, lines will be split by any amount of whitespace ('\s+')

    :returns:
        - event_times : np.ndarray
            array of event times (float)

        - labels : list of str
            list of labels

    '''
    # Use our universal function to load in the events
    events, labels = load_delimited(filename, [float, str], delimiter)
    events = np.array(events)
    # Validate them, but throw a warning in place of an error
    try:
        util.validate_events(events)
    except ValueError as error:
        warnings.warn(error.args[0])

    return events, labels


def load_intervals(filename, delimiter=r'\s+',
                   converter=None, label_prefix='__'):
    r'''Import labeled intervals from an annotation file.  This is primarily
        useful for processing events which span a duration, such as
        segmentation, chords, or instrument activation.

        The annotation file may be either of two formats:
         - Double-column.  Each line contains two values, separated by
           ``delimiter``, corresponding to the start and end time annotated
           event.

         - Triple-column.  Each line contains three values, separated by
           ``delimiter``.  The first two values specify the start and end
           times, the last value specifies the label for the event (e.g.
           "Verse" or "A:min").

        :parameters:
          - filename : str
              Path to the annotation file

          - delimiter : str
              Separator regular expression.
              By default, lines will be split by any amount of whitespace
              ('\s+')

          - converter : function
              Function to convert time-stamp data into numerics. Defaults to
              float().

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

    times = []
    labels = []

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
                raise ValueError('parse error %s:%d:\n%s' %
                                 (filename, row, line))

    times = np.asarray(times)

    try:
        util.validate_intervals(times)
    except ValueError as error:
        warnings.warn(error.args[0])

    return times, labels


def load_time_series(filename, delimiter=None):
    r'''Import a time series from an annotation file.  This is primarily useful
        for processing dense time series with timestamps and corresponding
        numeric values

        The annotation file must be of the following format:
          - Double-column.  Each line contains two values, separated by
            ``delimiter``: the first contains the timestamp, and the second
            contains its corresponding numeric value.

        :parameters:
          - filename : str
              Path to the annotation file

          - delimiter : str
              Column separator. By default, lines will be split by any amount
              of whitespace, unless the file ending is .csv, in which case a
              comma ',' is used as the delimiter.

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
        data = np.loadtxt(filename, 'float', '#', delimiter)
    except ValueError:
        raise ValueError('Error: could no load %s, please check if it is '
                         'in the correct 2 column format'
                         % os.path.basename(filename))

    data = data.T

    # we do however want to make sure the data is in the right format!
    if data.shape[0] != 2:
        raise ValueError('Error: %s should be of dimension (2,x), but is '
                         'of dimension %s'
                         % (os.path.basename(filename), data.shape))

    times = data[0]
    values = data[1]

    return times, values


def load_patterns(filename):
    """Loads the patters contained in the filename and puts them into a list
    of patterns, each pattern being a list of occurrence, and each
    occurrence being a list of (onset, midi) pairs.

    The input file must be formatted as described in MIREX 2013:
        http://www.music-ir.org/mirex/wiki/2013:Discovery_of_Repeated_Themes_%26_Sections

    :params:
      - filename : str
          The input file path containing the patterns of a given
          given piece using the MIREX 2013 format.

    :returns:
       - pattern_list : list
           the list of patterns, containing all their occurrences,
           using the following format::

             pattern_list = [pattern1, ..., patternN]
             pattern = [occurrence1, ..., occurrenceM]
             occurrence = [onset_midi1, ..., onset_midiO]
             onset_midi = (onset_time, midi_number)

             E.g.:
             P = [[[(77.0, 67.0), (77.5, 77.0), ... ]]]
    """

    with open(filename, "r") as input_file:
        # List with all the patterns
        pattern_list = []
        # Current pattern, which will contain all occs
        pattern = []
        # Current occurrence, containing (onset, midi)
        occurrence = []
        for line in input_file.readlines():
            if "pattern" in line:
                if occurrence != []:
                    pattern.append(occurrence)
                if pattern != []:
                    pattern_list.append(pattern)
                occurrence = []
                pattern = []
                continue
            if "occurrence" in line:
                if occurrence != []:
                    pattern.append(occurrence)
                occurrence = []
                continue
            string_values = line.split(",")
            onset_midi = (float(string_values[0]), float(string_values[1]))
            occurrence.append(onset_midi)

        # Add last occurrence and pattern to pattern_list
        if occurrence != []:
            pattern.append(occurrence)
        if pattern != []:
            pattern_list.append(pattern)

    return pattern_list
