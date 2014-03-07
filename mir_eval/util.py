"""Utility sub-module for mir-eval"""

import numpy as np

def index_labels(labels):
    '''Convert a list of string identifiers into numerical indices.

    :parameters:
        - labels : list, shape=(n,)
          A list of annotations, e.g., segment or chord labels from an annotation file.
          ``labels[i]`` can be any hashable type (such as `str` or `int`)

    :returns:
        - indices : list, shape=(n,)
          Numerical representation of `labels`

        - index_to_label : dict
          Mapping to convert numerical indices back to labels.
          `labels[i] == index_to_label[indices[i]]``
    '''

    label_to_index = {}
    index_to_label = {}

    # First, build the unique label mapping
    for index, s in enumerate(sorted(set(labels))):
        label_to_index[s]       = index
        index_to_label[index]   = s

    # Remap the labels to indices
    indices = [label_to_index[s] for s in labels]

    # Return the converted labels, and the inverse mapping
    return indices, index_to_label

def intervals_to_samples(intervals, labels, sample_size=0.1):
    '''Convert an array of labeled time intervals to annotated samples.
    
    :parameters:
        - intervals : np.ndarray, shape=(n, d)
            An array of time intervals, as returned by
            ``mir_eval.io.load_annotation``.
            The `i`th interval spans time ``intervals[i, 0]`` to ``intervals[i, 1]``.

        - labels : list, shape=(n,)
            The annotation for each interval

        - sample_size : float > 0
            duration of each sample to be generated (in seconds)

    :returns:
        - sample_labels : list
            array of segment labels for each generated sample

    ..note::
        Segment intervals will be rounded down to the nearest multiple 
        of ``frame_size``.
    '''

    # Round intervals to the sample size
    intervals = np.round(intervals / sample_size)

    # Build the frame label array
    y = []
    for (i, (start, end)) in enumerate(zip(intervals[:, 0], intervals[:, 1])):
        y.extend([labels[i]] * int( (end - start) ))

    return y

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

def intervals_to_boundaries(intervals, labels=None, label_prefix='__'):
    '''Convert segment interval times into boundaries.
    :parameters:
      - intervals : np.ndarray, shape=(n_events, 2)
          Array of segment start and end-times

      - labels : None or list of str
          Optional list of strings describing each event

    :returns:
      - boundaries : np.ndarray, shape=(n_segments + 1)
          Segment boundary times, including the end of the final segment

      - labels : list of str or None
          Labels for each event
    '''

    boundaries = np.unique(np.ravel(intervals))

    if labels is None:
        boundary_labels = None
    else:
        boundary_labels = [seg_label for seg_label in labels]
        boundary_labels.append('%sEND' % label_prefix)

    return boundaries, boundary_labels

def boundaries_to_intervals(boundaries, labels=None):
    '''Convert an array of event times into intervals

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

    intervals = np.asarray(zip(boundaries[:-1], boundaries[1:]))

    if labels is None:
        interval_labels = None
    else:
        interval_labels = labels[:-1]

    return intervals, interval_labels

def adjust_intervals(intervals, labels=None, t_min=0.0, t_max=None, label_prefix='__'):
    '''Adjust a list of time intervals to span the range [t_min, t_max].

    Any intervals lying completely outside the specified range will be removed.

    Any intervals lying partially outside the specified range will be truncated.

    If the specified range exceeds the span of the provided data in either direction,
    additional intervals will be appended.

    '''

    if t_min is not None:
        # Find the intervals that end at or after t_min
        first_idx = np.argwhere(intervals[:, 1] >= t_min)
        
        if len(first_idx) > 0:
            # If we have events below t_min, crop them out
            if labels is not None:
                labels = labels[int(first_idx[0]):]
            # Clip to the range (t_min, +inf)
            intervals = intervals[int(first_idx[0]):]
        intervals = np.maximum(t_min, intervals)

        if intervals[0, 0] > t_min:
            # Lowest boundary is higher than t_min: add a new boundary and label
            intervals = np.vstack( ([t_min, intervals[0, 0]], intervals) )
            if labels is not None:
                labels.insert(0, '%sT_MIN' % label_prefix)

    if t_max is not None:
        # Find the intervals that begin after t_max
        last_idx = np.argwhere(intervals[:, 0] > t_max)

        if len(last_idx) > 0:
            # We have boundaries above t_max.
            # Trim to only boundaries <= t_max
            if labels is not None:
                labels = labels[:int(last_idx[0])]
            # Clip to the range (-inf, t_max)
            intervals = intervals[:int(last_idx[0])]

        intervals = np.minimum(t_max, intervals)

        if intervals[-1, -1] < t_max:
            # Last boundary is below t_max: add a new boundary and label
            intervals = np.vstack( (intervals, [intervals[-1, -1], t_max]) )
            if labels is not None:
                labels.append('%sT_MAX' % label_prefix)

    return intervals, labels

def adjust_events(events, labels=None, t_min=0.0, t_max=None, label_prefix='__'):
    '''Adjust the given list of event times to span the range [t_min, t_max].

    Any event times outside of the specified range will be removed.

    If the times do not span [t_min, t_max], additional events will be inserted.

    :parameters:
        - events : np.array
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
        first_idx = np.argwhere(events >= t_min)

        if len(first_idx) > 0:
            # We have events below t_min
            # Crop them out
            if labels is not None:
                labels = labels[int(first_idx[0]):]
            events = events[int(first_idx[0]):]

        if events[0] > t_min:
            # Lowest boundary is higher than t_min: add a new boundary and label
            events = np.concatenate( ([t_min], events) )
            if labels is not None:
                labels.insert(0, '%sT_MIN' % label_prefix)

    if t_max is not None:
        last_idx = np.argwhere(events> t_max)

        if len(last_idx) > 0:
            # We have boundaries above t_max.
            # Trim to only boundaries <= t_max
            if labels is not None:
                labels = labels[:int(last_idx[0])]
            events = events[:int(last_idx[0])]

        if events[-1] < t_max:
            # Last boundary is below t_max: add a new boundary and label
            events= np.concatenate( (events, [t_max]) )
            if labels is not None:
                labels.append('%sT_MAX' % label_prefix)

    return events, labels
