""" Unit tests for input/output functions """

import numpy as np
import json
import mir_eval
import warnings
import nose.tools
import tempfile


def test_load_delimited():
    # Test for ValueError when a non-string or file handle is passed
    nose.tools.assert_raises(
        IOError, mir_eval.io.load_delimited, None, [int])
    # Test for a value error when the wrong number of columns is passed
    with tempfile.TemporaryFile('r+') as f:
        f.write('10 20')
        f.seek(0)
        nose.tools.assert_raises(
            ValueError, mir_eval.io.load_delimited, f, [int, int, int])

    # Test for a value error on conversion failure
    with tempfile.TemporaryFile('r+') as f:
        f.write('10 a 30')
        f.seek(0)
        nose.tools.assert_raises(
            ValueError, mir_eval.io.load_delimited, f, [int, int, int])


def test_load_events():
    # Test for a warning when invalid events are supplied
    with tempfile.TemporaryFile('r+') as f:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            # Non-increasing is invalid
            f.write('10\n9')
            f.seek(0)
            events = mir_eval.io.load_events(f)
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert (str(w[-1].message) ==
                    'Events should be in increasing order.')
            # Make sure events were read in correctly
            assert np.all(events == [10, 9])


def test_load_labeled_events():
    # Test for a value error when invalid labeled events are supplied
    with tempfile.TemporaryFile('r+') as f:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            # Non-increasing is invalid
            f.write('10 a\n9 b')
            f.seek(0)
            events, labels = mir_eval.io.load_labeled_events(f)
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert (str(w[-1].message) ==
                    'Events should be in increasing order.')
            # Make sure events were read in correctly
            assert np.all(events == [10, 9])
            # Make sure labels were read in correctly
            assert labels == ['a', 'b']


def test_load_intervals():
    # Test for a value error when invalid labeled events are supplied
    with tempfile.TemporaryFile('r+') as f:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            # Non-increasing is invalid
            f.write('10 9\n9 10')
            f.seek(0)
            intervals = mir_eval.io.load_intervals(f)
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert (str(w[-1].message) ==
                    'All interval durations must be strictly positive')
            # Make sure intervals were read in correctly
            assert np.all(intervals == [[10, 9], [9, 10]])


def test_load_labeled_intervals():
    # Test for a value error when invalid labeled events are supplied
    with tempfile.TemporaryFile('r+') as f:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            # Non-increasing is invalid
            f.write('10 9 a\n9 10 b')
            f.seek(0)
            intervals, labels = mir_eval.io.load_labeled_intervals(f)
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert (str(w[-1].message) ==
                    'All interval durations must be strictly positive')
            # Make sure intervals were read in correctly
            assert np.all(intervals == [[10, 9], [9, 10]])
            assert labels == ['a', 'b']


def test_load_valued_intervals():
    # Test for a value error when invalid valued events are supplied
    with tempfile.TemporaryFile('r+') as f:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            # Non-increasing is invalid
            f.write('10 9 5\n9 10 6')
            f.seek(0)
            intervals, values = mir_eval.io.load_valued_intervals(f)
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert (str(w[-1].message) ==
                    'All interval durations must be strictly positive')
            # Make sure intervals were read in correctly
            assert np.all(intervals == [[10, 9], [9, 10]])
            assert np.all(values == [5, 6])


def test_load_ragged_time_series():
    # Test for ValueError when a non-string or file handle is passed
    nose.tools.assert_raises(
        IOError, mir_eval.io.load_ragged_time_series, None, float,
        header=False)
    # Test for a value error on conversion failure
    with tempfile.TemporaryFile('r+') as f:
        f.write('10 a 30')
        f.seek(0)
        nose.tools.assert_raises(
            ValueError, mir_eval.io.load_ragged_time_series, f, float,
            header=False)
    # Test for a value error on invalid time stamp
    with tempfile.TemporaryFile('r+') as f:
        f.write('a 10 30')
        f.seek(0)
        nose.tools.assert_raises(
            ValueError, mir_eval.io.load_ragged_time_series, f, int,
            header=False)
    # Test for a value error on invalid time stamp with header
    with tempfile.TemporaryFile('r+') as f:
        f.write('x y z\na 10 30')
        f.seek(0)
        nose.tools.assert_raises(
            ValueError, mir_eval.io.load_ragged_time_series, f, int,
            header=True)
