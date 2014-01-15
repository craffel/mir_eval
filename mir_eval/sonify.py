# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

'''
Methods which sonify annotations for "evaluation by ear".  All functions return a raw signal at the specified sampling rate.
'''

# <codecell>

import numpy as np

# <codecell>

def clicks(times, fs, click=None, length=None):
    '''
    Returns a signal with the signal 'click' placed at each time specified in times
    
    Input:
        times - np.ndarray of times to place clicks, in seconds
        fs - desired sampling rate of the output signal
        click - click signal, defaults to an 1 kHz blip
        length - desired number of samples in the output signal, defaults to times.max()*fs + click.shape[0] + 1
    Output:
        click_signal - Synthesized click signal
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

# <codecell>

def piano_roll(piano_roll, frequencies, times, fs, function=np.sin, length=None):
    '''
    Synthesize a semigram using sinusoids
    
    Input:
        piano_roll - np.ndarray where piano_roll[n, m] is the magnitude of frequencies[n] from times[n] to times[n + 1]
        frequencies - np.ndarray of size piano_roll.shape[0] denoting the frequency of each row of piano_roll
        times - np.ndarray of size piano_roll.shape[1] denoting the start time of each column of piano_roll
        fs - desired sampling rate of the output signal
        function - function to use to synthesize notes, should be 2\pi-periodic
        length - desired number of samples in the output signal, defaults to times[-1]*fs
    Output:
        output - synthetized version of the piano roll
    '''
    # Default value for length
    if length is None:
        length = times[-1]*fs
    def _fast_synthesize(frequency):
        ''' A faster (approximte) way to synthesize a signal - synthesize a few periods then repeat that signal '''
        # Generate ten periods at this frequency
        ten_periods = int(10*fs*(1./frequency))
        short_signal = function(2*np.pi*np.arange(ten_periods)*frequency/fs)
        # Repeat the signal until it's of the desired length
        return np.tile(short_signal, np.ceil(length/short_signal.shape[0]))[:length]
    # Pre-allocate output signal
    output = np.zeros(length)
    for n, frequency in enumerate(frequencies):
        # Get a waveform of length samples at this frequency
        wave = _fast_synthesize(frequency)
        # Scale each time interval by the piano roll magnitude
        for m, (start, end) in enumerate(zip(times[:-1], times[1:])):
            wave[int(start*fs):int(end*fs)] *= piano_roll[n, m]
        # Sume into the aggregate output waveform
        output += wave
    # Normalize
    output /= np.abs(output).max()
    return output

