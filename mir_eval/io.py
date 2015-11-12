"""
Functions for loading in annotations from files in different formats.
"""

import numpy as np
import re
import warnings
import scipy.io.wavfile
import six

from . import util


def load_delimited(filename, converters, delimiter=r'\s+'):
    r"""Utility function for loading in data from an annotation file where columns
    are delimited.  The number of columns is inferred from the length of
    the provided converters list.

    Examples
    --------
    >>> # Load in a one-column list of event times (floats)
    >>> load_delimited('events.txt', [float])
    >>> # Load in a list of labeled events, separated by commas
    >>> load_delimited('labeled_events.csv', [float, str], ',')

    Parameters
    ----------
    filename : str
        Path to the annotation file
    converters : list of functions
        Each entry in column ``n`` of the file will be cast by the function
        ``converters[n]``.
    delimiter : str
        Separator regular expression.
        By default, lines will be split by any amount of whitespace.

    Returns
    -------
    columns : tuple of lists
        Each list in this tuple corresponds to values in one of the columns
        in the file.

    """
    # Initialize list of empty lists
    n_columns = len(converters)
    columns = tuple(list() for _ in range(n_columns))

    # Create re object for splitting lines
    splitter = re.compile(delimiter)

    # Keep track of whether we create our own file handle
    own_fh = False
    # If the filename input is a string, need to open it
    if isinstance(filename, six.string_types):
        # Remember that we need to close it later
        own_fh = True
        # Open the file for reading
        input_file = open(filename, 'r')
    # If the provided has a read attribute, we can use it as a file handle
    elif hasattr(filename, 'read'):
        input_file = filename
    # Raise error otherwise
    else:
        raise ValueError('filename must be a string or file handle')

    # Note: we do io manually here for two reasons.
    #   1. The csv module has difficulties with unicode, which may lead
    #      to failures on certain annotation strings
    #
    #   2. numpy's text loader does not handle non-numeric data
    #
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
                                     value, converter.__name__, filename, row,
                                     line))
            column.append(converted_value)

    # Close the file handle if we opened it
    if own_fh:
        input_file.close()

    # Sane output
    if n_columns == 1:
        return columns[0]
    else:
        return columns


def load_events(filename, delimiter=r'\s+'):
    r"""Import time-stamp events from an annotation file.  The file should
    consist of a single column of numeric values corresponding to the event
    times. This is primarily useful for processing events which lack duration,
    such as beats or onsets.

    Parameters
    ----------
    filename : str
        Path to the annotation file
    delimiter : str
        Separator regular expression.
        By default, lines will be split by any amount of whitespace.

    Returns
    -------
    event_times : np.ndarray
        array of event times (float)

    """
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
    r"""Import labeled time-stamp events from an annotation file.  The file should
    consist of two columns; the first having numeric values corresponding to
    the event times and the second having string labels for each event.  This
    is primarily useful for processing labeled events which lack duration, such
    as beats with metric beat number or onsets with an instrument label.

    Parameters
    ----------
    filename : str
        Path to the annotation file
    delimiter : str
        Separator regular expression.
        By default, lines will be split by any amount of whitespace.

    Returns
    -------
    event_times : np.ndarray
        array of event times (float)
    labels : list of str
        list of labels

    """
    # Use our universal function to load in the events
    events, labels = load_delimited(filename, [float, str], delimiter)
    events = np.array(events)
    # Validate them, but throw a warning in place of an error
    try:
        util.validate_events(events)
    except ValueError as error:
        warnings.warn(error.args[0])

    return events, labels


def load_intervals(filename, delimiter=r'\s+'):
    r"""Import intervals from an annotation file.  The file should consist of two
    columns of numeric values corresponding to start and end time of each
    interval.  This is primarily useful for processing events which span a
    duration, such as segmentation, chords, or instrument activation.

    Parameters
    ----------
    filename : str
        Path to the annotation file
    delimiter : str
        Separator regular expression.
        By default, lines will be split by any amount of whitespace.

    Returns
    -------
    intervals : np.ndarray, shape=(n_events, 2)
        array of event start and end times

    """
    # Use our universal function to load in the events
    starts, ends = load_delimited(filename, [float, float], delimiter)
    # Stack into an interval matrix
    intervals = np.array([starts, ends]).T
    # Validate them, but throw a warning in place of an error
    try:
        util.validate_intervals(intervals)
    except ValueError as error:
        warnings.warn(error.args[0])

    return intervals


def load_labeled_intervals(filename, delimiter=r'\s+'):
    r"""Import labeled intervals from an annotation file.  The file should consist
    of three columns: Two consisting of numeric values corresponding to start
    and end time of each interval and a third corresponding to the label of
    each interval.  This is primarily useful for processing events which span a
    duration, such as segmentation, chords, or instrument activation.

    Parameters
    ----------
    filename : str
        Path to the annotation file
    delimiter : str
        Separator regular expression.
        By default, lines will be split by any amount of whitespace.

    Returns
    -------
    intervals : np.ndarray, shape=(n_events, 2)
        array of event start and end time
    labels : list of str
        list of labels

    """
    # Use our universal function to load in the events
    starts, ends, labels = load_delimited(filename, [float, float, str],
                                          delimiter)
    # Stack into an interval matrix
    intervals = np.array([starts, ends]).T
    # Validate them, but throw a warning in place of an error
    try:
        util.validate_intervals(intervals)
    except ValueError as error:
        warnings.warn(error.args[0])

    return intervals, labels


def load_time_series(filename, delimiter=r'\s+'):
    r"""Import a time series from an annotation file.  The file should consist of
    two columns of numeric values corresponding to the time and value of each
    sample of the time series.

    Parameters
    ----------
    filename : str
        Path to the annotation file
    delimiter : str
        Separator regular expression.
        By default, lines will be split by any amount of whitespace.

    Returns
    -------
    times : np.ndarray
        array of timestamps (float)
    values : np.ndarray
        array of corresponding numeric values (float)

    """
    # Use our universal function to load in the events
    times, values = load_delimited(filename, [float, float], delimiter)
    times = np.array(times)
    values = np.array(values)

    return times, values


def load_patterns(filename):
    """Loads the patters contained in the filename and puts them into a list
    of patterns, each pattern being a list of occurrence, and each
    occurrence being a list of (onset, midi) pairs.

    The input file must be formatted as described in MIREX 2013:
    http://www.music-ir.org/mirex/wiki/2013:Discovery_of_Repeated_Themes_%26_Sections

    Parameters
    ----------
    filename : str
        The input file path containing the patterns of a given piece using the
        MIREX 2013 format.

    Returns
    -------
    pattern_list : list
        The list of patterns, containing all their occurrences,
        using the following format::

            onset_midi = (onset_time, midi_number)
            occurrence = [onset_midi1, ..., onset_midiO]
            pattern = [occurrence1, ..., occurrenceM]
            pattern_list = [pattern1, ..., patternN]

        where `N` is the number of patterns, `M[i]` is the number of
        occurrences of the `i`'th pattern, and `O[j]` is the number of onsets
        in the `j`'th occurrence.  E.g.::

            occ1 = [(0.5, 67.0), (1.0, 67.0), (1.5, 67.0), (2.0, 64.0)]
            occ2 = [(4.5, 65.0), (5.0, 65.0), (5.5, 65.0), (6.0, 62.0)]
            pattern1 = [occ1, occ2]

            occ1 = [(10.5, 67.0), (11.0, 67.0), (11.5, 67.0), (12.0, 64.0),
                    (12.5, 69.0), (13.0, 69.0), (13.5, 69.0), (14.0, 67.0),
                    (14.5, 76.0), (15.0, 76.0), (15.5, 76.0), (16.0, 72.0)]
            occ2 = [(18.5, 67.0), (19.0, 67.0), (19.5, 67.0), (20.0, 62.0),
                    (20.5, 69.0), (21.0, 69.0), (21.5, 69.0), (22.0, 67.0),
                    (22.5, 77.0), (23.0, 77.0), (23.5, 77.0), (24.0, 74.0)]
            pattern2 = [occ1, occ2]

            pattern_list = [pattern1, pattern2]

    """

    # Keep track of whether we create our own file handle
    own_fh = False
    # If the filename input is a string, need to open it
    if isinstance(filename, six.string_types):
        # Remember that we need to close it later
        own_fh = True
        # Open the file for reading
        input_file = open(filename, 'r')
    # If the provided has a read attribute, we can use it as a file handle
    elif hasattr(filename, 'read'):
        input_file = filename
    # Raise error otherwise
    else:
        raise ValueError('filename must be a string or file handle')

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

    # If we opened an input file, we need to close it
    if own_fh:
        input_file.close()

    return pattern_list


def load_wav(path, mono=True):
    """Loads a .wav file as a numpy array using scipy.io.wavfile.

    Parameters
    ----------
    path : str
        Path to a .wav file
    mono : bool
        If the provided .wav has more than one channel, it will be
        converted to mono if ``mono=True``. (Default value = True)

    Returns
    -------
    audio_data : np.ndarray
        Array of audio samples, normalized to the range [-1., 1.]
    fs : int
        Sampling rate of the audio data

    """

    fs, audio_data = scipy.io.wavfile.read(path)
    # Make float in range [-1, 1]
    if audio_data.dtype == 'int8':
        audio_data = audio_data/float(2**8)
    elif audio_data.dtype == 'int16':
        audio_data = audio_data/float(2**16)
    elif audio_data.dtype == 'int32':
        audio_data = audio_data/float(2**24)
    else:
        raise ValueError('Got unexpected .wav data type '
                         '{}'.format(audio_data.dtype))
    # Optionally convert to mono
    if mono and audio_data.ndim != 1:
        audio_data = audio_data.mean(axis=1)
    return audio_data, fs
