'''
Methods which sonify annotations for "evaluation by ear".
All functions return a raw signal at the specified sampling rate.
'''

import numpy as np


def clicks(times, fs, click=None, length=None):
    '''
    Returns a signal with the signal 'click' placed at each specified time

    :inputs:
        - times : np.ndarray
            times to place clicks, in seconds
        - fs : int
            desired sampling rate of the output signal
        - click : np.ndarray
            click signal, defaults to a 1 kHz blip
        - length : int
            desired number of samples in the output signal,
            defaults to times.max()*fs + click.shape[0] + 1
    :outputs:
        - click_signal : np.ndarray
            Synthesized click signal
    '''
    # Create default click signal
    if click is None:
        # 1 kHz tone, 100ms
        click = np.sin(2*np.pi*np.arange(fs*.1)*1000/(1.*fs))
        # Exponential decay
        click *= np.exp(-np.arange(fs*.1)/(fs*.01))
    # Set default length
    if length is None:
        length = times.max()*fs + click.shape[0] + 1
    # Pre-allocate click signal
    click_signal = np.zeros(length)
    # Place clicks
    for time in times:
        # Compute the boundaries of the click
        start = int(time*fs)
        end = start + click.shape[0]
        # Make sure we don't try to output past the end of the signal
        if start >= length:
            break
        if end >= length:
            click_signal[start:] = click[:length - start]
            break
        # Normally, just add a click here
        click_signal[start:end] = click
    return click_signal


def time_frequency(gram, frequencies, times, fs, function=np.sin, length=None):
    '''
    Reverse synthesis of a time-frequency representation of a signal

    :inputs:
        - gram : np.ndarray
            gram[n, m] is the magnitude of frequencies[n]
            from times[n] to times[n + 1]
        - frequencies : np.ndarray
            array of size gram.shape[0] denoting the frequency of
            each row of gram
        - times : np.ndarray
            array of size gram.shape[1] denoting the start time of each
            column of gram
        - fs : int
            desired sampling rate of the output signal
        - function : function
            function to use to synthesize notes, should be 2pi-periodic
        - length : int
            desired number of samples in the output signal,
            defaults to times[-1]*fs
    :outputs:
        - output : np.ndarray
            synthetized version of the piano roll
    '''
    # Default value for length
    if length is None:
        length = int(times[-1]*fs)

    def _fast_synthesize(frequency):
        ''' A faster (approximate) way to synthesize a signal
            synthesize a few periods then repeat that signal '''
        # Generate ten periods at this frequency
        ten_periods = int(10*fs*(1./frequency))
        short_signal = function(2*np.pi*np.arange(ten_periods)*frequency/fs)
        # Repeat the signal until it's of the desired length
        n_repeats = int(np.ceil(length/float(short_signal.shape[0])))
        return np.tile(short_signal, n_repeats)[:length]

    # Pre-allocate output signal
    output = np.zeros(length)
    for n, frequency in enumerate(frequencies):
        # Get a waveform of length samples at this frequency
        wave = _fast_synthesize(frequency)
        # Zero out up to first time
        wave[:int(times[0]*fs)] = 0
        # Scale each time interval by the piano roll magnitude
        for m, (start, end) in enumerate(zip(times[:-1], times[1:])):
            wave[int(start*fs):int(end*fs)] *= gram[n, m]
        # Sume into the aggregate output waveform
        output += wave
    # Normalize
    output /= np.abs(output).max()
    return output


def chroma(chromagram, times, fs):
    '''
    Reverse synthesis of a chromagram (semitone matrix)

    :parameters:
        - chromagram : np.ndarray, shape=(12, times.shape[0])
            Chromagram matrix, where each row represents a semitone [C->Bb]
            i.e., chromagram[3, j] is the magnitude of D# from times[j] to
            times[j + 1]
        - times : np.ndarray
            The start time of each column in the chromagram
        - fs : int
            Sampling rate to synthesize audio data at

    :returns:
        - output : np.ndarray
            Synthesized chromagram
    '''
    # We'll just use time_frequency with a Shepard tone-gram
    # To create the Shepard tone-gram, we copy the chromagram across 7 octaves
    n_octaves = 7
    # starting from C2
    base_note = 24
    # and weight each octave by a normal distribution
    # The normal distribution has mean 72 (one octave above middle C)
    # and std 6 (one half octave)
    mean = 72
    std = 6
    notes = np.arange(12*n_octaves) + base_note
    shepard_weight = np.exp(-(notes - mean)**2./(2.*std**2.))
    # Copy the chromagram matrix vertically n_octaves times
    gram = np.tile(chromagram.T, n_octaves).T
    # Apply Sheppard weighting
    gram *= shepard_weight.reshape(-1, 1)
    # Compute frequencies
    frequencies = 440.0*(2.0**((notes - 69)/12.0))
    return time_frequency(gram, frequencies, times, fs)
