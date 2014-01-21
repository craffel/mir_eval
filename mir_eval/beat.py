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

# <codecell>

def _clean_beats(annotated_beats, generated_beats, min_beat_time=5.0):
    '''
    Utility function to "clean up" the beats.
    Validates and sorts beat times and removes beats before min_beat_time

    Input:
        annotated_beats - np.ndarray of reference beat times, in seconds
        generated_beats - np.ndarray of query beat times, in seconds
        min_beat_time - Ignore all beats before this time, in seconds
    '''

    # Make sure beat locations are 1-d np ndarrays
    if len(annotated_beats.shape) != 1 or len(generated_beats.shape) != 1:
        raise ValueError('Beat locations should be 1-d numpy ndarray')
    # Make sure some beats fall before min_beat_time
    if not (annotated_beats > min_beat_time).any():
        error = 'No annotated beats found before {}s'.format(min_beat_time)
        raise ValueError(error)
    # Make sure no beat times are huge
    if (annotated_beats > 30000).any() or (generated_beats > 30000).any():
        error = 'Very large beat times found - they should be in seconds.'
        raise ValueError(error)
    # Make sure no beat times are negative
    if (annotated_beats < 0).any() or (generated_beats < 0).any():
        raise ValueError('Beat locations should not be negative')

    # Make sure beats are sorted
    annotated_beats = np.sort(annotated_beats)
    generated_beats = np.sort(generated_beats)
    # Ignore beats up to min_beat_time
    annotated_beats = annotated_beats[annotated_beats > min_beat_time]
    generated_beats = generated_beats[generated_beats > min_beat_time]
    return annotated_beats, generated_beats

# <codecell>

def _get_annotated_beat_variations(annotated_beats):
    '''
    Return metric variations of the annotated beats

    Input:
        annotated_beats - np.ndarry of beat locations in seconds
    Output:
        annotated_beats - Original beat locations
        off_beat - 180 degrees out of phase from the original beat locations
        double - Annotated beats at 1/2 the original tempo
        half_odd - Half tempo, odd beats
        half_even - Half tempo, even beats
    '''

    # Create annotations at twice the metric level
    interpolated_indices = np.arange(0, annotated_beats.shape[0]-.5, .5)
    original_indices = np.arange(0, annotated_beats.shape[0])
    double_annotated_beats = np.interp(interpolated_indices,
                                       original_indices,
                                       annotated_beats)
    # Return metric variations:
    #True, off-beat, double tempo, half tempo odd, and half tempo even
    return (annotated_beats,
           double_annotated_beats[1::2],
           double_annotated_beats,
           annotated_beats[::2],
           annotated_beats[1::2])

# <codecell>

def f_measure(annotated_beats,
              generated_beats,
              min_beat_time=5.0,
              f_measure_threshod=0.07):
    '''
    Compute the F-measure of correct vs incorrectly predicted beats.
    "Corectness" is determined over a small window.

    Input:
        annotated_beats - np.ndarray of reference beat times, in seconds
        generated_beats - np.ndarray of query beat times, in seconds
        min_beat_time - Ignore all beats before this time, in seconds
        f_measure_threshold - Window size, in seconds
    Output:
        f_score - The computed F-measure score
    '''
    # Validate and clean up beat times
    annotated_beats, generated_beats = _clean_beats(annotated_beats,
                                                    generated_beats,
                                                    min_beat_time)
    # Special case when annotated beats are empty
    if generated_beats.shape == (0,):
        return 0
    # Values for calculating F measure
    false_positives = 0.0
    false_negatives = 0.0
    true_positives = 0.0
    for beat in annotated_beats:
        # Calculate window edges
        window_min = beat - f_measure_threshod
        window_max = beat + f_measure_threshod
        # Find the (indeces of the) beats in the window
        correct_beats = np.logical_and(generated_beats >= window_min,
                                       generated_beats <= window_max)
        beats_in_window = np.flatnonzero(correct_beats)
        # Remove beats in this window so that they are only counted once
        generated_beats = np.delete(generated_beats, beats_in_window)
        # No beats found in window - add a false negative
        if beats_in_window.shape[0] == 0:
            false_negatives += 1.0
        # One or more beats in the window
        elif beats_in_window.shape[0] >= 1:
            # Add a hit and false positives for each spurious beat
            true_positives += 1.0
            if beats_in_window.shape[0] > 1:
                false_positives += beats_in_window.shape[0] - 1
    # Add in all remaining beats to false positives
    false_positives += false_positives + generated_beats.shape[0]
    # Calculate F-measure ensuring that we don't divide by 0
    if 2.0*true_positives + false_positives + false_negatives > 0:
        f_score = 2.0*true_positives/(2.0*true_positives
                                      + false_positives
                                      + false_negatives)
    else:
        f_score = 0
    return f_score

# <codecell>

def cemgil(annotated_beats,
           generated_beats,
           min_beat_time=5.0,
           cemgil_sigma=0.04):
    '''
    Cemgil's score, computes a gaussian error of each generated beat.

    Input:
        annotated_beats - np.ndarray of reference beat times, in seconds
        generated_beats - np.ndarray of query beat times, in seconds
        min_beat_time - Ignore all beats before this time, in seconds
        cemgil_sigma - Sigma parameter of gaussian error windows
    Output:
        cemgil_score - Cemgil's score for the original annotated beats
        cemgil_max - The best Cemgil score for all metrical variations
    '''
    # Validate and clean up beat times
    annotated_beats, generated_beats = _clean_beats(annotated_beats,
                                                    generated_beats,
                                                    min_beat_time)
    # Special case when annotated beats are empty
    if generated_beats.shape == (0,):
        return 0
    # We'll compute Cemgil's accuracy for each variation
    accuracies = []
    for annotated_beats in _get_annotated_beat_variations(annotated_beats):
        accuracy = 0
        # Cycle through beats
        for beat in annotated_beats:
            # Find the error for the closest beat to the annotated beat
            beat_diff = np.min(np.abs(beat - generated_beats))
            # Add gaussian error into the accuracy
            accuracy += np.exp(-(beat_diff**2)/(2.0*cemgil_sigma**2))
        # Normalize the accuracy
        accuracy /= .5*(generated_beats.shape[0] + annotated_beats.shape[0])
        # Add it to our list of accuracy scores
        accuracies.append(accuracy)
    # Return raw accuracy with non-varied annotations
    # and maximal accuracy across all variations
    return accuracies[0], np.max(accuracies)

# <codecell>

def goto(annotated_beats,
         generated_beats,
         min_beat_time=5.0,
         goto_threshold=0.2,
         goto_mu=0.2,
         goto_sigma=0.2):
    '''
    Calculate Goto's score, a binary 1 or 0 depending on some specific
    heuristic criteria

    Input:
        annotated_beats - np.ndarray of reference beat times, in seconds
        generated_beats - np.ndarray of query beat times, in seconds
        min_beat_time - Ignore all beats before this time, in seconds
        goto_threshold - Threshold of beat error for a beat to be "correct"
        goto_mu - The mean of the beat errors in the continuously correct
            track must be less than this
        goto_sigma - The std of the beat errors in the continuously
            correct track must be less than this
    Output:
        goto_score - Binary 1 or 0 if some specific criteria are met
    '''
    # Validate and clean up beat times
    annotated_beats, generated_beats = _clean_beats(annotated_beats,
                                                    generated_beats,
                                                    min_beat_time)
    # Special case when annotated beats are empty
    if generated_beats.shape == (0,):
        return 0
    # Error for each beat
    beat_error = np.ones(annotated_beats.shape[0])
    # Flag for whether the annotated and generated beats are paired
    paired = np.zeros(annotated_beats.shape[0])
    # Keep track of Goto's three criteria
    goto_criteria = 0
    for n in xrange(1, annotated_beats.shape[0]-1):
        # Get previous inner-annotated-beat-interval
        previous_interval = 0.5*(annotated_beats[n] - annotated_beats[n-1])
        # Window start - in the middle of the current beat and the previous
        window_min = annotated_beats[n] - previous_interval
        # Next inter-annotated-beat-interval
        next_interval = 0.5*(annotated_beats[n+1] - annotated_beats[n])
        # Window end - in the middle of the current beat and the next
        window_max = annotated_beats[n] + next_interval
        # Get generated beats in the window
        beats_in_window = np.logical_and((generated_beats >= window_min),
                                         (generated_beats <= window_max))
        # False negative/positive
        if beats_in_window.sum() == 0 or beats_in_window.sum() > 1:
            paired[n] = 0
            beat_error[n] = 1
        else:
            # Single beat is paired!
            paired[n] = 1
            # Get offset of the generated beat and the annotated beat
            offset = generated_beats[beats_in_window] - annotated_beats[n]
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
        if track_length - 1 > .25*(annotated_beats.shape[0] - 2):
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

def p_score(annotated_beats,
            generated_beats,
            min_beat_time=5.0,
            p_score_threshold=0.2):
    '''
    Get McKinney's P-score.
    Based on the autocorrelation of the annotated and generated beats

    Input:
        annotated_beats - np.ndarray of reference beat times, in seconds
        generated_beats - np.ndarray of query beat times, in seconds
        min_beat_time - Ignore all beats before this time, in seconds
        p_score_threshold - Window size will be
            p_score_threshold*median(inter_annotation_intervals)
    Output:
        correlation - McKinney's P-score
    '''
    # Validate and clean up beat times
    annotated_beats, generated_beats = _clean_beats(annotated_beats,
                                                    generated_beats,
                                                    min_beat_time)
    # Special case when annotated beats are empty
    if generated_beats.shape == (0,):
        return 0
    # Quantize beats to 10ms
    sampling_rate = int(1.0/0.010)
    # Get the largest time index
    end_point = np.int(np.ceil(np.max([np.max(generated_beats),
                                       np.max(annotated_beats)])))
    # Make impulse trains with impulses at beat locations
    annotations_train = np.zeros(end_point*sampling_rate + 1)
    beat_indices = np.ceil(annotated_beats*sampling_rate).astype(np.int)
    annotations_train[beat_indices] = 1.0
    generated_train = np.zeros(end_point*sampling_rate + 1)
    beat_indices = np.ceil(generated_beats*sampling_rate).astype(np.int)
    generated_train[beat_indices] = 1.0
    # Window size to take the correlation over
    # defined as .2*median(inter-annotation-intervals)
    annotation_intervals = np.diff(np.flatnonzero(annotations_train))
    win_size = int(np.round(p_score_threshold*np.median(annotation_intervals)))
    # Get full correlation
    train_correlation = np.correlate(annotations_train, generated_train, 'full')
    # Get the middle element - note we are rounding down on purpose here
    middle_lag = train_correlation.shape[0]/2
    # Truncate to only valid lags (those corresponding to the window)
    start = middle_lag - win_size
    end = middle_lag + win_size + 1
    train_correlation = train_correlation[start:end]
    # Compute and return the P-score
    n_beats = np.max([generated_beats.shape[0], annotated_beats.shape[0]])
    return np.sum(train_correlation)/n_beats

# <codecell>

def continuity(annotated_beats,
               generated_beats,
               min_beat_time=5.0,
               continuity_phase_threshold=0.175,
               continuity_period_threshold=0.175):
    '''
    Get metrics based on how much of the generated beat sequence is
    continually correct.

    Input:
        annotated_beats - np.ndarray of reference beat times, in seconds
        generated_beats - np.ndarray of query beat times, in seconds
        min_beat_time - Ignore all beats before this time, in seconds
        continuity_phase_threshold - Allowable ratio of how far is the
            generated beat can be from the annotated beat
        continuity_period_threshold - Allowable distance between the
            inter-beat-interval and the inter-annotation-interval
    Output:
        CMLc - Correct metric level, continuous accuracy
        CMLt - Correct metric level, total accuracy (continuity not required)
        AMLc - Any metric level, continuous accuracy
        AMLt - Any metric level, total accuracy (continuity not required)
    '''
    # Validate and clean up beat times
    annotated_beats, generated_beats = _clean_beats(annotated_beats,
                                                    generated_beats,
                                                    min_beat_time)
    # Special case when annotated beats are empty
    if generated_beats.shape == (0,):
        return 0
    # Accuracies for each variation
    continuous_accuracies = []
    total_accuracies = []
    # Get accuracy for each variation
    for annotated_beats in _get_annotated_beat_variations(annotated_beats):
        # Annotations that have been used
        n_annotations = np.max([annotated_beats.shape[0],
                               generated_beats.shape[0]])
        used_annotations = np.zeros(n_annotations)
        # Whether or not we are continuous at any given point
        beat_successes = np.zeros(n_annotations)
        # Is this beat correct?
        beat_success = 0
        for m in xrange(generated_beats.shape[0]):
            beat_success = 0
            # Get differences for this beat
            beat_differences = np.abs(generated_beats[m] - annotated_beats)
            # Get nearest annotation index
            nearest = np.argmin(beat_differences)
            min_difference = beat_differences[nearest]
            # Have we already used this annotation?
            if used_annotations[nearest] == 0:
                # Is this the first beat or first annotation?
                # If so, look forward.
                if ((m == 0 or nearest == 0) and
                    (m + 1 < generated_beats.shape[0])):
                    # How far is the generated beat from the annotated beat,
                    # relative to the inter-annotation-interval?
                    annotated_interval = annotated_beats[nearest + 1] - \
                                         annotated_beats[nearest]
                    phase = np.abs(min_difference/annotated_interval)
                    # How close is the inter-beat-interval
                    # to the inter-annotation-interval?
                    generated_interval = generated_beats[m + 1] - \
                                         generated_beats[m]
                    period = np.abs(1 - generated_interval/annotated_interval)
                    if (phase < continuity_phase_threshold and
                        period < continuity_period_threshold):
                        # Set this annotation as used
                        used_annotations[nearest] = 1
                        # This beat is matched
                        beat_success = 1
                # This beat/annotation is not the first
                else:
                    # How far is the generated beat from the annotated beat,
                    # relative to the inter-annotation-interval?
                    annotated_interval = annotated_beats[nearest] - \
                                         annotated_beats[nearest - 1]
                    phase = np.abs(min_difference/annotated_interval)
                    # How close is the inter-beat-interval
                    # to the inter-annotation-interval?
                    generated_interval = generated_beats[m] - \
                                         generated_beats[m - 1]
                    annotated_interval = annotated_beats[nearest] - \
                                         annotated_beats[nearest - 1]
                    period = np.abs(1 - generated_interval/annotated_interval)
                    if (phase < continuity_phase_threshold and
                        period < continuity_period_threshold):
                        # Set this annotation as used
                        used_annotations[nearest] = 1
                        # This beat is matched
                        beat_success = 1
            # Set whether this beat is matched or not
            beat_successes[m] = beat_success
        # Add 0s at the begnning and end
        # so that we at least find the beginning/end of the generated beats
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

def information_gain(annotated_beats,
                     generated_beats,
                     min_beat_time=5.0,
                     bins=41):
    '''
    Get the information gain - K-L divergence of the beat error histogram
    to a uniform histogram

    Input:
        annotated_beats - np.ndarray of reference beat times, in seconds
        generated_beats - np.ndarray of query beat times, in seconds
        min_beat_time - Ignore all beats before this time, in seconds
        bins - Number of bins in the beat error histogram
    Output:
        information_gain_score - Entropy of beat error histogram
    '''
    # To match beat evaluation toolbox
    bins -= 1
    # Validate and clean up beat times
    annotated_beats, generated_beats = _clean_beats(annotated_beats,
                                                    generated_beats,
                                                    min_beat_time)
    # Special case when annotated beats are empty
    if generated_beats.shape == (0,):
        return 0
    # Get entropy for annotated beats->generated beats
    # and generated beats->annotated beats
    forward_entropy = _get_entropy(annotated_beats, generated_beats, bins)
    backward_entropy = _get_entropy(generated_beats, annotated_beats, bins)
    # Pick the larger of the entropies
    norm = np.log2(bins)
    if forward_entropy > backward_entropy:
        # Note that the beat evaluation toolbox does not normalize
        information_gain_score = (norm - forward_entropy)/norm
    else:
        information_gain_score = (norm - backward_entropy)/norm
    return information_gain_score

def _get_entropy(annotated_beats, generated_beats, bins):
    '''
    Helper function for information gain
    (needs to be run twice - once backwards, once forwards)

    Input:
        annotated_beats - np.ndarray of reference beat times, in seconds
        generated_beats - np.ndarray of query beat times, in seconds
        bins - Number of bins in the beat error histogram
    Output:
        entropy - Entropy of beat error histogram
    '''
    beat_error = np.zeros(generated_beats.shape[0])
    for n in xrange(generated_beats.shape[0]):
        # Get index of closest annotation to this beat
        beat_distances = generated_beats[n] - annotated_beats
        closest_beat = np.argmin(np.abs(beat_distances))
        absolute_error = beat_distances[closest_beat]
        # If the first annotation is closest...
        if closest_beat == 0:
            # Inter-annotation interval - space between first two beats
            interval = .5*(annotated_beats[1] - annotated_beats[0])
        # If last annotation is closest...
        if closest_beat == (annotated_beats.shape[0] - 1):
            interval = .5*(annotated_beats[-1] - annotated_beats[-2])
        else:
            if absolute_error < 0:
                # Closest annotation is the one before the current beat
                # so look at previous inner-annotation-interval
                start = annotated_beats[closest_beat]
                end = annotated_beats[closest_beat - 1]
                interval = .5*(start - end)
            else:
                # Closest annotation is the one after the current beat
                # so look at next inner-annotation-interval
                start = annotated_beats[closest_beat + 1]
                end = annotated_beats[closest_beat]
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

