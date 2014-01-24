"""Annotation input/output functions"""

import numpy as np
import re

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

def load_annotation(filename, delimiter=r'\s+', converter=None, label_prefix='__'):
    r'''Import annotation events from an annotation file.  This is primarily useful for
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

