""" Unit tests for sonification methods """

import mir_eval
import numpy as np


def test_clicks():
    # Test output length for a variety of parameter settings
    for times in [np.array([1.]), np.arange(10)*1.]:
        for fs in [8000, 44100]:
            click_signal = mir_eval.sonify.clicks(times, fs)
            assert len(click_signal) == times.max()*fs + int(fs*.1) + 1
            click_signal = mir_eval.sonify.clicks(times, fs, length=1000)
            assert len(click_signal) == 1000
            click_signal = mir_eval.sonify.clicks(
                times, fs, click=np.zeros(1000))
            assert len(click_signal) == times.max()*fs + 1000 + 1


def test_time_frequency():
    # Test length for different inputs
    for fs in [8000, 44100]:
        signal = mir_eval.sonify.time_frequency(
            np.random.standard_normal((100, 1000)), np.arange(1, 101),
            np.linspace(0, 10, 1000), fs)
        assert len(signal) == 10*fs
        signal = mir_eval.sonify.time_frequency(
            np.random.standard_normal((100, 1000)), np.arange(1, 101),
            np.linspace(0, 10, 1000), fs, length=fs*11)
        assert len(signal) == 11*fs


def test_chroma():
    for fs in [8000, 44100]:
        signal = mir_eval.sonify.chroma(
            np.random.standard_normal((12, 1000)),
            np.linspace(0, 10, 1000), fs)
        assert len(signal) == 10*fs
        signal = mir_eval.sonify.chroma(
            np.random.standard_normal((12, 1000)),
            np.linspace(0, 10, 1000), fs, length=fs*11)
        assert len(signal) == 11*fs


def test_chords():
    for fs in [8000, 44100]:
        intervals = np.array([np.arange(10), np.arange(1, 11)]).T
        signal = mir_eval.sonify.chords(
            ['C', 'C:maj', 'D:min7', 'E:min', 'C#', 'C', 'C', 'C', 'C', 'C'],
            intervals, fs)
        assert len(signal) == 10*fs
        signal = mir_eval.sonify.chords(
            ['C', 'C:maj', 'D:min7', 'E:min', 'C#', 'C', 'C', 'C', 'C', 'C'],
            intervals, fs, length=fs*11)
        assert len(signal) == 11*fs
