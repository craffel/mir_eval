# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

'''
A variety of evaluation techniques for determining a beat tracker's accuracy
Based on the methods described in
    Matthew E. P. Davies,  Norberto Degara, and Mark D. Plumbley.
    "Evaluation Methods for Musical Audio Beat Tracking Algorithms",
    Queen Mary University of London Technical Report C4DM-TR-09-06
    London, United Kingdom, 8 October 2009.
See also the Beat Evaluation Toolbox:
    https://code.soundsoftware.ac.uk/projects/beat-evaluation/
'''

# <codecell>

import numpy as np
import functools
import collections
from . import util

# <codecell>

def trim_beats(beats, min_beat_time=5.):
    '''Removes beats before min_beat_time.  A common preprocessing step.

    :parameters:
        - beats : ndarray
            Array of beat times in seconds.
        - min_beat_time : float
            Minimum beat time to allow, default 5
    
    :returns:
        - beats_trimmed : ndarray
            Trimmed beat array.
    '''
    # Remove beats before min_beat_time
    return beats[beats > min_beat_time]

# <codecell>

def validate(metric):
    '''Decorator which checks that the input annotations to a metric
    look like valid beat time arrays, and throws helpful errors if not.

    :parameters:
        - metric : function
            Evaluation metric function.  First two arguments must be
            reference_beats and estimated_beats.

    :returns:
        - metric_validated : function
            The function with the beat times validated
    '''
    # Retain docstring, etc
    @functools.wraps(metric)
    def metric_validated(reference_beats, estimated_beats, *args, **kwargs):
        '''
        Metric with input beat annotations validated
        '''
        for beats in [reference_beats, estimated_beats]:
            # Make sure beat locations are 1-d np ndarrays
            if beats.ndim != 1:
                raise ValueError('Beat locations should be 1-d numpy ndarray')
            # Make sure no beat times are huge
            if (beats > 30000).any():
                raise ValueError('A beat at time {}'.format(beats.max()) + \
                                 ' was found; should be in seconds.')
            # Make sure no beat times are negative
            if (beats < 0).any():
                raise ValueError('Negative beat locations found')
            # Make sure beat times are increasing
            if (np.diff(beats) < 0).any():
                raise ValueError('Beats should be in increasing order.')
        return metric(reference_beats, estimated_beats, *args, **kwargs)
    return metric_validated

# <codecell>

def _get_reference_beat_variations(reference_beats):
    '''
    Return metric variations of the reference beats

    :parameters:
        - reference_beats : np.ndarray
            beat locations in seconds

    :returns:
        - reference_beats : np.ndarray
            Original beat locations
        - off_beat : np.ndarray
            180 degrees out of phase from the original beat locations
        - double : np.ndarray
            Beats at 1/2 the original tempo
        - half_odd : np.ndarray
            Half tempo, odd beats
        - half_even : np.ndarray
            Half tempo, even beats
    '''

    # Create annotations at twice the metric level
    interpolated_indices = np.arange(0, reference_beats.shape[0]-.5, .5)
    original_indices = np.arange(0, reference_beats.shape[0])
    double_reference_beats = np.interp(interpolated_indices,
                                       original_indices,
                                       reference_beats)
    # Return metric variations:
    #True, off-beat, double tempo, half tempo odd, and half tempo even
    return (reference_beats,
           double_reference_beats[1::2],
           double_reference_beats,
           reference_beats[::2],
           reference_beats[1::2])

# <codecell>

@validate
def f_measure(reference_beats,
              estimated_beats,
              f_measure_threshod=0.07):
    '''
    Compute the F-measure of correct vs incorrectly predicted beats.
    "Corectness" is determined over a small window.
    
    :usage:
        >>> reference_beats = mir_eval.beat.trim_beats(mir_eval.io.load_events('reference.txt'))
        >>> estimated_beats = mir_eval.beat.trim_beats(mir_eval.io.load_events('estimated.txt'))
        >>> f_measure = mir_eval.beat.f_measure(reference_beats, estimated_beats)

    :parameters:
        - reference_beats : np.ndarray
            reference beat times, in seconds
        - estimated_beats : np.ndarray 
            estimated beat times, in seconds
        - f_measure_threshold : float
            Window size, in seconds, default 0.07
    
    :returns:
        - f_score : float
            The computed F-measure score
    '''
    # When estimated beats are empty, no beats are correct; metric is 0
    if estimated_beats.size == 0:
        return 0
    # Values for calculating F measure
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    for beat in reference_beats:
        # Calculate window edges
        window_min = beat - f_measure_threshod
        window_max = beat + f_measure_threshod
        # Find the (indeces of the) beats in the window
        correct_beats = np.logical_and(estimated_beats >= window_min,
                                       estimated_beats <= window_max)
        beats_in_window = np.flatnonzero(correct_beats)
        # Remove beats in this window so that they are only counted once
        estimated_beats = np.delete(estimated_beats, beats_in_window)
        # No beats found in window - add a false negative
        if beats_in_window.shape[0] == 0:
            false_negatives += 1
        # One or more beats in the window
        elif beats_in_window.shape[0] >= 1:
            # Add a hit and false positives for each spurious beat
            true_positives += 1
            if beats_in_window.shape[0] > 1:
                false_positives += beats_in_window.shape[0] - 1
    # Add in all remaining beats to false positives
    false_positives += false_positives + estimated_beats.shape[0]
    # Compute precision and recall
    precision = true_positives/float(true_positives + false_positives)
    recall = true_positives/float(true_positives + false_negatives)
    return util.f_measure(precision, recall)

# <codecell>

@validate
def cemgil(reference_beats,
           estimated_beats,
           cemgil_sigma=0.04):
    '''
    Cemgil's score, computes a gaussian error of each estimated beat.
    Compares against the original beat times and all metrical variations.

    :usage:
        >>> reference_beats = mir_eval.beat.trim_beats(mir_eval.io.load_events('reference.txt'))
        >>> estimated_beats = mir_eval.beat.trim_beats(mir_eval.io.load_events('estimated.txt'))
        >>> cemgil_score, cemgil_max = mir_eval.beat.cemgil(reference_beats, estimated_beats)

    :parameters:
        - reference_beats : np.ndarray 
            reference beat times, in seconds
        - estimated_beats : np.ndarray
            query beat times, in seconds
        - cemgil_sigma : float
            Sigma parameter of gaussian error windows, default 0.04
    
    :returns:
        - cemgil_score : float
            Cemgil's score for the original reference beats
        - cemgil_max :
            The best Cemgil score for all metrical variations
    '''
    # When estimated beats are empty, no beats are correct; metric is 0
    if estimated_beats.size == 0:
        return 0
    # We'll compute Cemgil's accuracy for each variation
    accuracies = []
    for reference_beats in _get_reference_beat_variations(reference_beats):
        accuracy = 0
        # Cycle through beats
        for beat in reference_beats:
            # Find the error for the closest beat to the reference beat
            beat_diff = np.min(np.abs(beat - estimated_beats))
            # Add gaussian error into the accuracy
            accuracy += np.exp(-(beat_diff**2)/(2.0*cemgil_sigma**2))
        # Normalize the accuracy
        accuracy /= .5*(estimated_beats.shape[0] + reference_beats.shape[0])
        # Add it to our list of accuracy scores
        accuracies.append(accuracy)
    # Return raw accuracy with non-varied annotations
    # and maximal accuracy across all variations
    return accuracies[0], np.max(accuracies)

# <codecell>

@validate
def goto(reference_beats,
         estimated_beats,
         goto_threshold=0.2,
         goto_mu=0.2,
         goto_sigma=0.2):
    '''
    Calculate Goto's score, a binary 1 or 0 depending on some specific
    heuristic criteria
    
    :usage:
        >>> reference_beats = mir_eval.beat.trim_beats(mir_eval.io.load_events('reference.txt'))
        >>> estimated_beats = mir_eval.beat.trim_beats(mir_eval.io.load_events('estimated.txt'))
        >>> goto_score = mir_eval.beat.goto(reference_beats, estimated_beats)

    :parameters:
        - reference_beats : np.ndarray
            reference beat times, in seconds
        - estimated_beats : np.ndarray
            query beat times, in seconds
        - goto_threshold : float
            Threshold of beat error for a beat to be "correct", default 0.2
        - goto_mu : float
            The mean of the beat errors in the continuously correct
            track must be less than this, default 0.2
        - goto_sigma : float
            The std of the beat errors in the continuously
            correct track must be less than this, default 0.2
    
    :returns:
        - goto_score : float
            Either 1.0 or 0.0 if some specific criteria are met
    '''
    # When estimated beats are empty, no beats are correct; metric is 0
    if estimated_beats.size == 0:
        return 0
    # Error for each beat
    beat_error = np.ones(reference_beats.shape[0])
    # Flag for whether the reference and estimated beats are paired
    paired = np.zeros(reference_beats.shape[0])
    # Keep track of Goto's three criteria
    goto_criteria = 0
    for n in xrange(1, reference_beats.shape[0]-1):
        # Get previous inner-reference-beat-interval
        previous_interval = 0.5*(reference_beats[n] - reference_beats[n-1])
        # Window start - in the middle of the current beat and the previous
        window_min = reference_beats[n] - previous_interval
        # Next inter-reference-beat-interval
        next_interval = 0.5*(reference_beats[n+1] - reference_beats[n])
        # Window end - in the middle of the current beat and the next
        window_max = reference_beats[n] + next_interval
        # Get estimated beats in the window
        beats_in_window = np.logical_and((estimated_beats >= window_min),
                                         (estimated_beats <= window_max))
        # False negative/positive
        if beats_in_window.sum() == 0 or beats_in_window.sum() > 1:
            paired[n] = 0
            beat_error[n] = 1
        else:
            # Single beat is paired!
            paired[n] = 1
            # Get offset of the estimated beat and the reference beat
            offset = estimated_beats[beats_in_window] - reference_beats[n]
            # Scale by previous or next interval
            if offset < 0:
                beat_error[n] = offset/previous_interval
            else:
                beat_error[n] = offset/next_interval
    # Get indices of incorrect beats
    correct_beats = np.flatnonzero(np.abs(beat_error) > goto_threshold)
    # All beats are correct (first and last will be 0 so always correct)
    if correct_beats.shape[0] < 3:
        # Get the track of correct beats
        track = beat_error[correct_beats[0] + 1:correct_beats[-1] - 1]
        goto_criteria = 1
    else:
        # Get the track of maximal length
        track_length = np.max(np.diff(correct_beats))
        track_start = np.nonzero(np.diff(correct_beats) == track_length)[0][0]
        # Is the track length at least 25% of the song?
        if track_length - 1 > .25*(reference_beats.shape[0] - 2):
            goto_criteria = 1
            start_beat = correct_beats[track_start]
            end_beat = correct_beats[track_start + 1]
            track = beat_error[start_beat:end_beat]
    # If we have a track
    if goto_criteria:
        # Are mean and std of the track less than the required thresholds?
        if np.mean(track) < goto_mu and np.std(track) < goto_sigma:
            goto_criteria = 3
    # If all criteria are met, score is 100%!
    return 1.0*(goto_criteria == 3)

# <codecell>

@validate
def p_score(reference_beats,
            estimated_beats,
            p_score_threshold=0.2):
    '''
    Get McKinney's P-score.
    Based on the autocorrelation of the reference and estimated beats
    
    :usage:
        >>> reference_beats = mir_eval.beat.trim_beats(mir_eval.io.load_events('reference.txt'))
        >>> estimated_beats = mir_eval.beat.trim_beats(mir_eval.io.load_events('estimated.txt'))
        >>> p_score = mir_eval.beat.p_score(reference_beats, estimated_beats)

    :parameters:
        - reference_beats : np.ndarray 
            reference beat times, in seconds
        - estimated_beats : np.ndarray
            query beat times, in seconds
        - p_score_threshold : float
            Window size will be p_score_threshold*median(inter_annotation_intervals), default 0.2
            
    :returns:
        - correlation : float
            McKinney's P-score
    '''
    # When estimated beats are empty, no beats are correct; metric is 0
    if estimated_beats.size == 0:
        return 0
    # Quantize beats to 10ms
    sampling_rate = int(1.0/0.010)
    # Get the largest time index
    end_point = np.int(np.ceil(np.max([np.max(estimated_beats),
                                       np.max(reference_beats)])))
    # Make impulse trains with impulses at beat locations
    annotations_train = np.zeros(end_point*sampling_rate + 1)
    beat_indices = np.ceil(reference_beats*sampling_rate).astype(np.int)
    annotations_train[beat_indices] = 1.0
    estimated_train = np.zeros(end_point*sampling_rate + 1)
    beat_indices = np.ceil(estimated_beats*sampling_rate).astype(np.int)
    estimated_train[beat_indices] = 1.0
    # Window size to take the correlation over
    # defined as .2*median(inter-annotation-intervals)
    annotation_intervals = np.diff(np.flatnonzero(annotations_train))
    win_size = int(np.round(p_score_threshold*np.median(annotation_intervals)))
    # Get full correlation
    train_correlation = np.correlate(annotations_train, estimated_train, 'full')
    # Get the middle element - note we are rounding down on purpose here
    middle_lag = train_correlation.shape[0]/2
    # Truncate to only valid lags (those corresponding to the window)
    start = middle_lag - win_size
    end = middle_lag + win_size + 1
    train_correlation = train_correlation[start:end]
    # Compute and return the P-score
    n_beats = np.max([estimated_beats.shape[0], reference_beats.shape[0]])
    return np.sum(train_correlation)/n_beats

# <codecell>

@validate
def continuity(reference_beats,
               estimated_beats,
               continuity_phase_threshold=0.175,
               continuity_period_threshold=0.175):
    '''
    Get metrics based on how much of the estimated beat sequence is
    continually correct.
    
    :usage:
        >>> reference_beats = mir_eval.beat.trim_beats(mir_eval.io.load_events('reference.txt'))
        >>> estimated_beats = mir_eval.beat.trim_beats(mir_eval.io.load_events('estimated.txt'))
        >>> CMLc, CMLt, AMLc, AMLt = mir_eval.beat.continuity(reference_beats, estimated_beats)

    :parameters:
        - reference_beats : np.ndarray
            reference beat times, in seconds
        - estimated_beats : np.ndarray
            query beat times, in seconds
        - continuity_phase_threshold : float
            Allowable ratio of how far is the estimated beat 
            can be from the reference beat, default 0.175
        - continuity_period_threshold : float
            Allowable distance between the inter-beat-interval
            and the inter-annotation-interval, default 0.175
    
    :returns:
        - CMLc : float
            Correct metric level, continuous accuracy
        - CMLt : float
            Correct metric level, total accuracy (continuity not required)
        - AMLc : float
            Any metric level, continuous accuracy
        - AMLt : float
            Any metric level, total accuracy (continuity not required)
    '''
    # When estimated beats are empty, no beats are correct; metric is 0
    if estimated_beats.size == 0:
        return 0
    # Accuracies for each variation
    continuous_accuracies = []
    total_accuracies = []
    # Get accuracy for each variation
    for reference_beats in _get_reference_beat_variations(reference_beats):
        # Annotations that have been used
        n_annotations = np.max([reference_beats.shape[0],
                               estimated_beats.shape[0]])
        used_annotations = np.zeros(n_annotations)
        # Whether or not we are continuous at any given point
        beat_successes = np.zeros(n_annotations)
        # Is this beat correct?
        beat_success = 0
        for m in xrange(estimated_beats.shape[0]):
            beat_success = 0
            # Get differences for this beat
            beat_differences = np.abs(estimated_beats[m] - reference_beats)
            # Get nearest annotation index
            nearest = np.argmin(beat_differences)
            min_difference = beat_differences[nearest]
            # Have we already used this annotation?
            if used_annotations[nearest] == 0:
                # Is this the first beat or first annotation?
                # If so, look forward.
                if ((m == 0 or nearest == 0) and
                    (m + 1 < estimated_beats.shape[0])):
                    # How far is the estimated beat from the reference beat,
                    # relative to the inter-annotation-interval?
                    reference_interval = reference_beats[nearest + 1] - \
                                         reference_beats[nearest]
                    phase = np.abs(min_difference/reference_interval)
                    # How close is the inter-beat-interval
                    # to the inter-annotation-interval?
                    estimated_interval = estimated_beats[m + 1] - \
                                         estimated_beats[m]
                    period = np.abs(1 - estimated_interval/reference_interval)
                    if (phase < continuity_phase_threshold and
                        period < continuity_period_threshold):
                        # Set this annotation as used
                        used_annotations[nearest] = 1
                        # This beat is matched
                        beat_success = 1
                # This beat/annotation is not the first
                else:
                    # How far is the estimated beat from the reference beat,
                    # relative to the inter-annotation-interval?
                    reference_interval = reference_beats[nearest] - \
                                         reference_beats[nearest - 1]
                    phase = np.abs(min_difference/reference_interval)
                    # How close is the inter-beat-interval
                    # to the inter-annotation-interval?
                    estimated_interval = estimated_beats[m] - \
                                         estimated_beats[m - 1]
                    reference_interval = reference_beats[nearest] - \
                                         reference_beats[nearest - 1]
                    period = np.abs(1 - estimated_interval/reference_interval)
                    if (phase < continuity_phase_threshold and
                        period < continuity_period_threshold):
                        # Set this annotation as used
                        used_annotations[nearest] = 1
                        # This beat is matched
                        beat_success = 1
            # Set whether this beat is matched or not
            beat_successes[m] = beat_success
        # Add 0s at the begnning and end
        # so that we at least find the beginning/end of the estimated beats
        beat_successes = np.append(np.append(0, beat_successes), 0)
        # Where is the beat not a match?
        beat_failures = np.nonzero(beat_successes == 0)[0]
        # Take out those zeros we added
        beat_successes = beat_successes[1:-1]
        # Get the continuous accuracy as the longest track of successful beats
        longest_track = np.max(np.diff(beat_failures)) - 1
        continuous_accuracy = longest_track/(1.0*beat_successes.shape[0])
        continuous_accuracies.append(continuous_accuracy)
        # Get the total accuracy - all sequences
        total_accuracy = np.sum(beat_successes)/(1.0*beat_successes.shape[0])
        total_accuracies.append(total_accuracy)
    # Grab accuracy scores
    return (continuous_accuracies[0],
            total_accuracies[0],
            np.max(continuous_accuracies),
            np.max(total_accuracies))

# <codecell>

@validate
def information_gain(reference_beats,
                     estimated_beats,
                     bins=41):
    '''
    Get the information gain - K-L divergence of the beat error histogram
    to a uniform histogram
    
    :usage:
        >>> reference_beats = mir_eval.beat.trim_beats(mir_eval.io.load_events('reference.txt'))
        >>> estimated_beats = mir_eval.beat.trim_beats(mir_eval.io.load_events('estimated.txt'))
        >>> information_gain = mir_eval.beat.information_gain(reference_beats, estimated_beats)
    
    :parameters:
        - reference_beats : np.ndarray
            reference beat times, in seconds
        - estimated_beats : np.ndarray
            query beat times, in seconds
        - bins : int
            Number of bins in the beat error histogram, default 41
    
    :returns:
        - information_gain_score : float
            Entropy of beat error histogram
    '''
    # When estimated beats are empty, no beats are correct; metric is 0
    if estimated_beats.size == 0:
        return 0
    # To match beat evaluation toolbox
    bins -= 1
    # Get entropy for reference beats->estimated beats
    # and estimated beats->reference beats
    forward_entropy = _get_entropy(reference_beats, estimated_beats, bins)
    backward_entropy = _get_entropy(estimated_beats, reference_beats, bins)
    # Pick the larger of the entropies
    norm = np.log2(bins)
    if forward_entropy > backward_entropy:
        # Note that the beat evaluation toolbox does not normalize
        information_gain_score = (norm - forward_entropy)/norm
    else:
        information_gain_score = (norm - backward_entropy)/norm
    return information_gain_score

def _get_entropy(reference_beats, estimated_beats, bins):
    '''
    Helper function for information gain
    (needs to be run twice - once backwards, once forwards)

    :parameters:
        - reference_beats : np.ndarray
            reference beat times, in seconds
        - estimated_beats : np.ndarray
            query beat times, in seconds
        - bins : int
            Number of bins in the beat error histogram
    
    :returns:
        - entropy : float
            Entropy of beat error histogram
    '''
    beat_error = np.zeros(estimated_beats.shape[0])
    for n in xrange(estimated_beats.shape[0]):
        # Get index of closest annotation to this beat
        beat_distances = estimated_beats[n] - reference_beats
        closest_beat = np.argmin(np.abs(beat_distances))
        absolute_error = beat_distances[closest_beat]
        # If the first annotation is closest...
        if closest_beat == 0:
            # Inter-annotation interval - space between first two beats
            interval = .5*(reference_beats[1] - reference_beats[0])
        # If last annotation is closest...
        if closest_beat == (reference_beats.shape[0] - 1):
            interval = .5*(reference_beats[-1] - reference_beats[-2])
        else:
            if absolute_error < 0:
                # Closest annotation is the one before the current beat
                # so look at previous inner-annotation-interval
                start = reference_beats[closest_beat]
                end = reference_beats[closest_beat - 1]
                interval = .5*(start - end)
            else:
                # Closest annotation is the one after the current beat
                # so look at next inner-annotation-interval
                start = reference_beats[closest_beat + 1]
                end = reference_beats[closest_beat]
                interval = .5*(start - end)
        # The actual error of this beat
        beat_error[n] = .5*absolute_error/interval
    # Trick to deal with bin boundaries
    beat_error = np.round(10000*beat_error)/10000.0
    # Put beat errors in range (-.5, .5)
    beat_error = np.mod(beat_error + .5, -1) + .5
    # These are set so that np.hist gives the same result as histogram
    # in the beat evaluation toolbox.  They are not correct.
    bin_step = 1.0/(bins - 1.0)
    histogram_bins = np.arange(-.5 + bin_step, .5, bin_step)
    histogram_bins = np.concatenate(([-.5, -.5 + bin_step/4],
                                     histogram_bins,
                                     [.5 - bin_step/4, .5]))
    # Get the histogram
    raw_bin_values = np.histogram(beat_error, histogram_bins)[0]
    # Add the last bin height to the first bin
    raw_bin_values[0] += raw_bin_values[-1]
    raw_bin_values = np.delete(raw_bin_values, -1)
    # Turn into a proper probability distribution
    raw_bin_values = raw_bin_values/(1.0*np.sum(raw_bin_values))
    # Set zero-valued bins to 1 to make the entropy calculation well-behaved
    raw_bin_values[raw_bin_values == 0] = 1
    # Calculate entropy
    return -np.sum(raw_bin_values * np.log2(raw_bin_values))

# <codecell>

# Create a dictionary which maps the name of each metric 
# to the function used to compute it
metrics = collections.OrderedDict()
metrics['F-measure'] = f_measure
metrics['Cemgil'] = cemgil
metrics['P-score'] = p_score
metrics['Continuity'] = continuity
metrics['Information Gain'] = information_gain

