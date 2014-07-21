r'''Functions and other supporting code for wrangling and comparing chords for
evaluation purposes.


Conventions:
-------------------------------
- Pitch class counting starts at C, e.g. C: 0, D:2, E:4, F:5, etc.


Defintions:
-------------------------------
- chord label: String representation of a chord name, e.g. "G:maj(4)/5"
- scale degree: String representation of a diatonic interval, relative to the
    root note, e.g. 'b6', '#5', or '7'
- bass interval: String representation of the bass note's scale degree.
- bitmap: Positional binary vector indicating active pitch classes; may be
    absolute or relative depending on context in the code.


A Note on Comparison Functions:
-------------------------------
There are, for better or worse, a variety of ways that chords can be compared
in a musically justifiable manner. While some rules may have tradition, and
therefore inertia in their favor, there is no single 'right' way to compare
two sequences of chord labels. Embracing this reality, several comparison
rules are provided in the hope that this may allow for more nuanced insight
into the performance and, ultimately, the behaviour of a computational system.

- 'mirex'*
    A estimated chord is considered correct if it shares *at least* three pitch
    classes in common.

- 'mirex-augdim'*
    Same as above, with the difference being an estimation only needs 2 pitch
    classes in common with the reference to be considered 'correct' for
    augmented or diminished chords, instead of the normal 3 pitches.

- 'near-exact'*
    Chords are correct only if they are nearly identical to the level of
    extensions, e.g. score('F#:maj(6)/5', 'F#:maj/5') = 1.0; this includes
    enharmonic spellings, e.g. score('F#:maj', 'Gb:maj') = 0.0.

- 'pitch_class'*
    Chords are compared at the level of pitch classes. This means that
    enharmonics and inversions are considered equal, e.g. score('F#:maj',
    'Gb:maj') = 1.0 and score('C:maj6'=[C,E,G,A], 'A:min7'=[A,C,E,G]) = 1.0.

- 'thirds'
    Chords are compared at the level of major or minor thirds (root and third),
    For example, both score('A:7', 'A:maj') and score('A:min', 'A:dim') equal
    1.0, as the third is major and minor in quality, respectively.

- 'thirds-inv'
    Same as above, but sensitive to inversions.

- 'triads'
    Chords are considered at the level of triads (major, minor, augmented,
    diminished, suspended), meaning that, in addition to the root, the quality
    is only considered through #5th scale degree (for augmented chords). For
    example, score('A:7', 'A:maj') = 1.0, while score('A:min', 'A:dim') and
    score('A:aug', 'A:maj') = 0.0.

- 'triads-inv'
    Same as above, but sensitive to inversions.

- 'tetrads'
    Chords are considered at the level of the entire quality in closed voicing,
    i.e. spanning only a single octave; extended chords (9's, 11's and 13's)
    are rolled into a single octave with any upper voices included as
    extensions. For example, score('A:7', 'A:9') = 1.0 and score('A:7',
    'A:maj7') = 0.0.

- 'tetrads-inv'
    Same as above, but sensitive to inversions.

- 'pitch_class-recall'*
    Recall on pitch classes in ref/est, based on the best-guess spelling of the
    chord given all information at hand, including extensions. For example,
    both score("A:min(\*5)"=[A, C], "C:maj6(\*3)"=[C, G, A]) and
    score("C:maj(\*3, \*5)"=[C], "C:min(\*b3, \*5)"=[C]) = 1.0.

- 'pitch_class-precision'*
    Precision on pitch classes in ref/est, using the same rules described in
    pitch class recall.

- 'pitch_class-f'*
    The harmonic mean (f-measure) is computed with the above recall and
    precision measures.

'''

import numpy as np
import decorator
import warnings
import collections

from mir_eval import input_output as io
from mir_eval import util

BITMAP_LENGTH = 12
NO_CHORD = "N"
NO_CHORD_ENCODED = -1, np.array([0]*BITMAP_LENGTH), -1
# See Line 445
STRICT_BASS_INTERVALS = False


class InvalidChordException(Exception):
    r'''Exception class for suspect / invalid chord labels.'''

    def __init__(self, message='', chord_label=None):
        self.message = message
        self.chord_label = chord_label
        self.name = self.__class__.__name__
        super(InvalidChordException, self).__init__(message)


# --- Chord Primitives ---
def _pitch_classes():
    r'''Map from pitch class (str) to semitone (int).'''
    pitch_classes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    semitones = [0, 2, 4, 5, 7, 9, 11]
    return dict([(c, s) for c, s in zip(pitch_classes, semitones)])


def _scale_degrees():
    r'''Mapping from scale degrees (str) to semitones (int).'''
    degrees = ['1', '2', '3', '4', '5', '6', '7', '9', '10', '11', '12', '13']
    semitones = [0, 2, 4, 5, 7, 9, 11, 14, 16, 17, 19, 21]
    return dict([(d, s) for d, s in zip(degrees, semitones)])


# Maps pitch classes (strings) to semitone indexes (ints).
PITCH_CLASSES = _pitch_classes()


def pitch_class_to_semitone(pitch_class):
    r'''Convert a pitch class to semitone.

    :parameters:
     - pitch_class : str
        Spelling of a given pitch class, e.g. 'C#', 'Gbb'

    :returns:
     - semitone : int
        Semitone value of the pitch class.

    :raises:
     - InvalidChordException
    '''
    semitone = 0
    for idx, char in enumerate(pitch_class):
        if char == '#' and idx > 0:
            semitone += 1
        elif char == 'b' and idx > 0:
            semitone -= 1
        elif idx == 0:
            semitone = PITCH_CLASSES.get(char)
        else:
            raise InvalidChordException(
                "Pitch class improperly formed: %s" % pitch_class)
    return semitone % 12


# Maps scale degrees (strings) to semitone indexes (ints).
SCALE_DEGREES = _scale_degrees()


def scale_degree_to_semitone(scale_degree):
    r'''Convert a scale degree to semitone.

    :parameters:
     - scale degree : str
        Spelling of a relative scale degree, e.g. 'b3', '7', '#5'

    :returns:
     - semitone : int
        Relative semitone of the scale degree, wrapped to a single octave

    :raises:
     - InvalidChordException
    '''
    semitone = 0
    offset = 0
    if scale_degree.startswith("#"):
        offset = scale_degree.count("#")
        scale_degree = scale_degree.strip("#")
    elif scale_degree.startswith('b'):
        offset = -1 * scale_degree.count("b")
        scale_degree = scale_degree.strip("b")

    semitone = SCALE_DEGREES.get(scale_degree, None)
    if semitone is None:
        raise InvalidChordException(
            "Scale degree improperly formed: %s" % scale_degree)
    return semitone + offset


def scale_degree_to_bitmap(scale_degree):
    '''Create a bitmap representation of a scale degree.

    Note that values in the bitmap may be negative, indicating that the
    semitone is to be removed.

    :parameters:
     - scale_degree : str
        Spelling of a relative scale degree, e.g. 'b3', '7', '#5'

    :returns:
     - bitmap : np.ndarray, in [-1, 0, 1]
        Bitmap representation of this scale degree (12-dim).
    '''
    sign = 1
    if scale_degree.startswith("*"):
        sign = -1
        scale_degree = scale_degree.strip("*")
    edit_map = [0] * BITMAP_LENGTH
    sd_idx = scale_degree_to_semitone(scale_degree)
    if sd_idx < BITMAP_LENGTH:
        edit_map[sd_idx % BITMAP_LENGTH] = sign
    return np.array(edit_map)


# Maps quality strings to bitmaps, corresponding to relative pitch class
# semitones, i.e. vector[0] is the tonic.
QUALITIES = {
    #           1     2     3     4  5     6     7
    'maj':     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'min':     [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    'aug':     [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    'dim':     [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    'sus4':    [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    'sus2':    [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    '7':       [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj7':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min7':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    'minmaj7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    'maj6':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    'min6':    [1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
    'dim7':    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    'hdim7':   [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    'maj9':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min9':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    '9':       [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'b9':      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    '#9':      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'min11':   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    '11':      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    '#11':     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj13':   [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min13':   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    '13':      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'b13':     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    '':        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}


def quality_to_bitmap(quality):
    '''Return the bitmap for a given quality.

    :parameters:
      - quality : str
          Chord quality name.

    :returns:
      - bitmap : np.ndarray, in [0, 1]
          Bitmap representation of this quality (12-dim).

    :raises:
      - InvalidChordException
    '''
    if quality not in QUALITIES:
        raise InvalidChordException(
            "Unsupported chord quality shorthand: '%s' "
            "Did you mean to reduce extended chords?" % quality)
    return np.array(QUALITIES[quality])


# Maps extended chord qualities to the subset above, translating additional
# voicings to extensions as a set of scale degrees (strings).
# TODO(ejhumphrey): Revisit how minmaj7's are mapped. This is how TMC did it,
#   but MMV handles it like a separate quality (rather than an add7).
EXTENDED_QUALITY_REDUX = {
    'minmaj7': ('min', set(['7'])),
    'maj9':    ('maj7', set(['9'])),
    'min9':    ('min7', set(['9'])),
    '9':       ('7', set(['9'])),
    'b9':      ('7', set(['b9'])),
    '#9':      ('7', set(['#9'])),
    '11':      ('7', set(['9', '11'])),
    '#11':     ('7', set(['9', '#11'])),
    '13':      ('7', set(['9', '11', '13'])),
    'b13':     ('7', set(['9', '11', 'b13'])),
    'min11':   ('min7', set(['9', '11'])),
    'maj13':   ('maj7', set(['9', '11', '13'])),
    'min13':   ('min7', set(['9', '11', '13']))}


def reduce_extended_quality(quality):
    '''Map an extended chord quality to a simpler one, moving upper voices to
    a set of scale degree extensions.

    :parameters:
     - quality : str
        Extended chord quality to reduce.

    :returns:
     - base_quality : str
        New chord quality.
     - extensions : set
        Scale degrees extensions for the quality.
    '''
    return EXTENDED_QUALITY_REDUX.get(quality, (quality, set()))


# --- Chord Label Parsing ---
def validate_chord_label(chord_label):
    '''Test for well-formedness of a chord label.

    :parameters:
     - chord : str
        Chord label to validate.

    :raises:
     - InvalidFormatException
    '''
    # Test for single special characters
    for one_char in [':', '/', '(', ')']:
        if chord_label.count(one_char) > 1:
            raise InvalidChordException(
                "Chord label may only contain one '%s'. "
                "Received: '%s'" % (one_char, chord_label))

    # Test for closed parens
    parens = [paren in chord_label for paren in ['(', ')']]
    if any(parens) and not all(parens):
        raise InvalidChordException(
            "Chord label must have closed parentheses. "
            "Received: '%s'" % chord_label)


def split(chord_label, reduce_extended_chords=False):
    '''Parse a chord label into its four constituent parts:
        - root
        - quality shorthand
        - scale degrees
        - bass

    Note: Chords lacking quality AND interval information are major.
      - If a quality is specified, it is returned.
      - If an interval is specified WITHOUT a quality, the quality field is
        empty.

    Some examples::

        'C' -> ['C', 'maj', {}, '1']
        'G#:min(*b3,*5)/5' -> ['G#', 'min', {'*b3', '*5'}, '5']
        'A:(3)/6' -> ['A', '', {'3'}, '6']


    :parameters:
     - chord_label : str
        A chord label.

    :returns:
     - chord_parts : list
        Split version of the chord label.
    '''
    chord_label = str(chord_label)
    validate_chord_label(chord_label)
    if chord_label == NO_CHORD:
        return [chord_label, '', set(), '']

    bass = '1'
    if "/" in chord_label:
        chord_label, bass = chord_label.split("/")

    scale_degrees = set()
    omission = False
    if "(" in chord_label:
        chord_label, scale_degrees = chord_label.split("(")
        omission = "*" in scale_degrees
        scale_degrees = scale_degrees.strip(")")
        scale_degrees = set([i.strip() for i in scale_degrees.split(",")])

    # Note: Chords lacking quality AND added interval information are major.
    #   If a quality shorthand is specified, it is returned.
    #   If an interval is specified WITHOUT a quality, the quality field is
    #     empty.
    #   Intervals specifying omissions MUST have a quality.
    if omission and ":" not in chord_label:
        raise InvalidChordException(
            "Intervals specifying omissions MUST have a quality.")
    quality = '' if scale_degrees else 'maj'
    if ":" in chord_label:
        chord_root, quality_name = chord_label.split(":")
        # Extended chords (with ":"s) may not explicitly have Major qualities,
        # so only overwrite the default if the string is not empty.
        if quality_name:
            quality = quality_name.lower()
    else:
        chord_root = chord_label

    if reduce_extended_chords:
        quality, addl_scale_degrees = reduce_extended_quality(quality)
        scale_degrees.update(addl_scale_degrees)

    return [chord_root, quality, scale_degrees, bass]


def join(chord_root, quality='', extensions=None, bass=''):
    r'''Join the parts of a chord into a complete chord label.

    :parameters:
     - chord_root : str
        Root pitch class of the chord, e.g. 'C', 'Eb'
     - quality : str
        Quality of the chord, e.g. 'maj', 'hdim7'
     - extensions : list
        Any added or absent scaled degrees for this chord, e.g. ['4', '\*3']
     - bass : str
        Scale degree of the bass note, e.g. '5'.

    :returns:
     - chord_label : str
        A complete chord label.

    :raises:
     - InvalidChordException
         Thrown if the provided args yield a garbage chord label.
    '''
    chord_label = chord_root
    if quality or extensions:
        chord_label += ":%s" % quality
    if extensions:
        chord_label += "(%s)" % ",".join(extensions)
    if bass:
        chord_label += "/%s" % bass
    validate_chord_label(chord_label)
    return chord_label


# --- Chords to Numerical Representations ---
def encode(chord_label, reduce_extended_chords=False):
    """Translate a chord label to numerical representations for evaluation.

    :parameters:
     - chord_label : str
        Chord label to encode.

     - reduce_extended_chords : bool, default=False
        Map the upper voicings of extended chords (9's, 11's, 13's) to semitone
        extensions.

    :returns:
     - root_number : int
        Absolute semitone of the chord's root.

     - semitone_bitmap : np.ndarray of ints, in [0, 1]
        12-dim vector of relative semitones in the chord spelling.

     - bass_number : int
        Relative semitone of the chord's bass note, e.g. 0=root, 7=fifth, etc.

    :raises:
     - InvalidChordException
         Thrown if the given bass note is not explicitly
         named as an extension.
    """

    if chord_label == NO_CHORD:
        return NO_CHORD_ENCODED
    chord_root, quality, scale_degrees, bass = split(
        chord_label, reduce_extended_chords=reduce_extended_chords)

    root_number = pitch_class_to_semitone(chord_root)
    bass_number = scale_degree_to_semitone(bass) % 12

    semitone_bitmap = quality_to_bitmap(quality)
    semitone_bitmap[0] = 1

    for scale_degree in scale_degrees:
        semitone_bitmap += scale_degree_to_bitmap(scale_degree)

    semitone_bitmap = (semitone_bitmap > 0).astype(np.int)
    if not semitone_bitmap[bass_number] and STRICT_BASS_INTERVALS:
        raise InvalidChordException(
            "Given bass scale degree is absent from this chord: "
            "%s" % chord_label, chord_label)
    else:
        semitone_bitmap[bass_number] = 1
    return root_number, semitone_bitmap, bass_number


def encode_many(chord_labels, reduce_extended_chords=False):
    """Translate a set of chord labels to numerical representations for sane
    evaluation.

    :parameters:
     - chord_labels : list
        Set of chord labels to encode.

     - reduce_extended_chords : bool, default=True
        Map the upper voicings of extended chords (9's, 11's, 13's) to semitone
        extensions.

    :returns:
     - root_number : np.ndarray of ints
        Absolute semitone of the chord's root.

     - interval_bitmap : np.ndarray of ints, in [0, 1]
        12-dim vector of relative semitones in the given chord quality.

     - bass_number : np.ndarray
        Relative semitones of the chord's bass notes.

    :raises:
     - InvalidChordException
        Thrown if the given bass note is not explicitly
        named as an extension.
    """
    num_items = len(chord_labels)
    roots, basses = np.zeros([2, num_items], dtype=np.int)
    semitones = np.zeros([num_items, 12], dtype=np.int)
    local_cache = dict()
    for i, label in enumerate(chord_labels):
        result = local_cache.get(label, None)
        if result is None:
            result = encode(label, reduce_extended_chords)
            local_cache[label] = result
        roots[i], semitones[i], basses[i] = result
    return roots, semitones, basses


def rotate_bitmap_to_root(bitmap, chord_root):
    '''Circularly shift a relative bitmap to its asbolute pitch classes.

    For clarity, the best explanation is an example. Given 'G:Maj', the root
    and quality map are as follows::

        root=5
        quality=[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]  # Relative chord shape

    After rotating to the root, the resulting bitmap becomes::

        abs_quality = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]  # G, B, and D

    :parameters:
        - bitmap : np.ndarray, shape=(12,)
            Bitmap of active notes, relative to the given root.

        - chord_root : int
            Absolute pitch class number.

    :returns:
        - bitmap : np.ndarray, shape=(12,)
            Absolute bitmap of active pitch classes.
    '''
    bitmap = np.asarray(bitmap)
    assert bitmap.ndim == 1, "Currently only 1D bitmaps are supported."
    idxs = list(np.nonzero(bitmap))
    idxs[-1] = (idxs[-1] + chord_root) % 12
    abs_bitmap = np.zeros_like(bitmap)
    abs_bitmap[idxs] = 1
    return abs_bitmap


def rotate_bitmaps_to_roots(bitmaps, roots):
    '''Circularly shift a relative bitmaps to asbolute pitch classes.

    See rotate_bitmap_to_root for more information.

    :parameters:
        - bitmap : np.ndarray, shape=(N, 12)
            Bitmap of active notes, relative to the given root.

        - root : np.ndarray, shape=(N,)
            Absolute pitch class number.

    :returns:
        - bitmap : np.ndarray, shape=(N, 12)
            Absolute bitmaps of active pitch classes.
    '''
    abs_bitmaps = []
    for bitmap, chord_root in zip(bitmaps, roots):
        abs_bitmaps.append(rotate_bitmap_to_root(bitmap, chord_root))
    return np.asarray(abs_bitmaps)


def rotate_bass_to_root(bass, chord_root):
    '''Rotate a relative bass interval to its asbolute pitch class.

    :parameters:
        - bass : int
            Relative bass interval.
        - chord_root : int
            Absolute root pitch class.

    :returns:
        - bass : int
            Pitch class of the bass intervalself.
    '''
    return (bass + chord_root) % 12


# --- Comparison Routines ---
@decorator.decorator
def validate(comparison):
    '''Decorator which checks that the input annotations to a comparison
    function look like valid chord labels.

    :parameters:
        - comparison : function
            Chord label comparison function.  The two arguments must be
            reference_labels and estimated_labels.

    :returns:
        - comparison_validated : function
            The function with the labels validated.
    '''
    def comparison_validated(reference_labels, estimated_labels, intervals):
        '''Comparison with labels validated.'''
        N = len(reference_labels)
        M = len(estimated_labels)
        if N != M:
            raise ValueError(
                "Chord comparison received different length lists: "
                "len(reference)=%d\tlen(estimates)=%d" % (N, M))
        for labels in [reference_labels, estimated_labels]:
            for chord_label in labels:
                validate_chord_label(chord_label)
        # When either label list is empty, warn the user
        if len(reference_labels) == 0:
            warnings.warn('Reference labels are empty')
        if len(estimated_labels) == 0:
            warnings.warn('Estimated labels are empty')

        # Intervals should be (n, 2) array
        if intervals.ndim != 2 or intervals.shape[1] != 2:
            raise ValueError('intervals should be an ndarray'
                             ' of size (n, 2)')
        # There should be as many intervals as labels
        if intervals.shape[0] != N:
            raise ValueError('intervals contains {} entries but '
                             'len(reference_labels) = len(estimated_labels)'
                             ' = {}'.format(intervals.shape[0], N))
        if 0 in np.diff(np.array(intervals), axis=1):
            warnings.warn('Zero-duration interval')

        return comparison(reference_labels, estimated_labels, intervals)

    return comparison_validated


@decorator.decorator
def score(comparator):
    '''
    Decorator to convert a comparator into a metric function.
    '''
    def metric(reference_labels, estimated_labels, intervals):
        '''
        Score wrapper for a comparator.
        '''
        # Return 0 when no labels are given
        if len(reference_labels) == 0 or len(estimated_labels) == 0:
            return 0
        # Compute comparison scores, in [0, 1] or -1
        comparison_scores = comparator(reference_labels, estimated_labels)
        # Find all comparison scores which are valid
        valid_idx = (comparison_scores >= 0)
        # If no comparable chords were provided, warn and return 0
        if valid_idx.sum() == 0:
            warnings.warn("No reference chords were comparable "
                          "to estimated chords, returning 0.")
            return 0
        # Convert intervals to durations
        durations = np.abs(np.diff(intervals, axis=-1)).flatten()
        # Remove any uncomparable labels
        comparison_scores = comparison_scores[valid_idx]
        durations = durations[valid_idx]
        # Get total amount of time
        total_time = float(np.sum(durations))
        # Weight each score by the relative proportion of the total duration
        duration_weights = np.asarray(durations, dtype=float)/total_time
        # Score is the sum of all weighted comparisons
        return np.sum(comparison_scores*duration_weights)
    return metric


@validate
@score
def thirds(reference_labels, estimated_labels):
    '''Compare chords along root & third relationships.

    :usage:
        >>> ref_intervals, ref_labels = mir_eval.io.load_intervals('ref.lab')
        >>> est_intervals, est_labels = mir_eval.io.load_intervals('est.lab')
        >>> est_intervals, est_labels = mir_eval.util.adjust_intervals(
                est_intervals, est_labels, ref_intervals.min(),
                ref_intervals.max(), mir_eval.chord.NO_CHORD,
                mir_eval.chord.NO_CHORD)
        >>> (intervals,
             ref_labels,
             est_labels) = mir_eval.util.merge_labeled_intervals(
                 ref_intervals, ref_labels, est_intervals, est_labels)
        >>> score = mir_eval.chord.thirds(ref_labels, est_labels, intervals)

    :parameters:
        - reference_labels : list, len=n
            Reference chord labels to score against.
        - estimated_labels : list, len=n
            Estimated chord labels to score against.
        - intervals : np.ndarray, shape=(n, 2)
            Start and end time of each chord label

    :returns:
        - comparison_scores : np.ndarray, shape=(n,), dtype=np.float
            Comparison scores, in [0.0, 1.0]
    '''
    ref_roots, ref_semitones = encode_many(reference_labels, False)[:2]
    est_roots, est_semitones = encode_many(estimated_labels, False)[:2]

    eq_roots = ref_roots == est_roots
    eq_thirds = ref_semitones[:, 3] == est_semitones[:, 3]
    return (eq_roots * eq_thirds).astype(np.float)


@validate
@score
def thirds_inv(reference_labels, estimated_labels):
    '''Score chords along root, third, & bass relationships.

    :usage:
        >>> ref_intervals, ref_labels = mir_eval.io.load_intervals('ref.lab')
        >>> est_intervals, est_labels = mir_eval.io.load_intervals('est.lab')
        >>> est_intervals, est_labels = mir_eval.util.adjust_intervals(
                est_intervals, est_labels, ref_intervals.min(),
                ref_intervals.max(), mir_eval.chord.NO_CHORD,
                mir_eval.chord.NO_CHORD)
        >>> (intervals,
             ref_labels,
             est_labels) = mir_eval.util.merge_labeled_intervals(
                 ref_intervals, ref_labels, est_intervals, est_labels)
        >>> score = mir_eval.chord.thirds_inv(ref_labels,
                                              est_labels,
                                              intervals)

    :parameters:
        - reference_labels : list, len=n
            Reference chord labels to score against.
        - estimated_labels : list, len=n
            Estimated chord labels to score against.
        - intervals : np.ndarray, shape=(n, 2)
            Start and end time of each chord label

    :returns:
        - scores : np.ndarray, shape=(n,), dtype=np.float
            Comparison scores, in [0.0, 1.0]
    '''
    ref_roots, ref_semitones, ref_bass = encode_many(reference_labels, False)
    est_roots, est_semitones, est_bass = encode_many(estimated_labels, False)

    eq_root = ref_roots == est_roots
    eq_bass = ref_bass == est_bass
    eq_third = ref_semitones[:, 3] == est_semitones[:, 3]
    return (eq_root * eq_third * eq_bass).astype(np.float)


@validate
@score
def triads(reference_labels, estimated_labels):
    '''Compare chords along triad (root & quality to #5) relationships.

    :usage:
        >>> ref_intervals, ref_labels = mir_eval.io.load_intervals('ref.lab')
        >>> est_intervals, est_labels = mir_eval.io.load_intervals('est.lab')
        >>> est_intervals, est_labels = mir_eval.util.adjust_intervals(
                est_intervals, est_labels, ref_intervals.min(),
                ref_intervals.max(), mir_eval.chord.NO_CHORD,
                mir_eval.chord.NO_CHORD)
        >>> (intervals,
             ref_labels,
             est_labels) = mir_eval.util.merge_labeled_intervals(
                 ref_intervals, ref_labels, est_intervals, est_labels)
        >>> score = mir_eval.chord.triads(ref_labels, est_labels, intervals)

    :parameters:
        - reference_labels : list, len=n
            Reference chord labels to score against.
        - estimated_labels : list, len=n
            Estimated chord labels to score against.
        - intervals : np.ndarray, shape=(n, 2)
            Start and end time of each chord label

    :returns:
        - comparison_scores : np.ndarray, shape=(n,), dtype=np.float
            Comparison scores, in {0.0, 1.0}
    '''
    ref_roots, ref_semitones = encode_many(reference_labels, False)[:2]
    est_roots, est_semitones = encode_many(estimated_labels, False)[:2]

    eq_roots = ref_roots == est_roots
    eq_semitones = np.all(
        np.equal(ref_semitones[:, :8], est_semitones[:, :8]), axis=1)
    return (eq_roots * eq_semitones).astype(np.float)


@validate
@score
def triads_inv(reference_labels, estimated_labels):
    '''Score chords along triad (root, quality to #5, & bass) relationships.

    :usage:
        >>> ref_intervals, ref_labels = mir_eval.io.load_intervals('ref.lab')
        >>> est_intervals, est_labels = mir_eval.io.load_intervals('est.lab')
        >>> est_intervals, est_labels = mir_eval.util.adjust_intervals(
                est_intervals, est_labels, ref_intervals.min(),
                ref_intervals.max(), mir_eval.chord.NO_CHORD,
                mir_eval.chord.NO_CHORD)
        >>> (intervals,
             ref_labels,
             est_labels) = mir_eval.util.merge_labeled_intervals(
                 ref_intervals, ref_labels, est_intervals, est_labels)
        >>> score = mir_eval.chord.triads_inv(ref_labels,
                                              est_labels,
                                              intervals)

    :parameters:
        - reference_labels : list, len=n
            Reference chord labels to score against.
        - estimated_labels : list, len=n
            Estimated chord labels to score against.
        - intervals : np.ndarray, shape=(n, 2)
            Start and end time of each chord label

    :returns:
        - scores : np.ndarray, shape=(n,), dtype=np.float
            Comparison scores, in {0.0, 1.0}
    '''
    ref_roots, ref_semitones, ref_bass = encode_many(reference_labels, False)
    est_roots, est_semitones, est_bass = encode_many(estimated_labels, False)

    eq_roots = ref_roots == est_roots
    eq_basses = ref_bass == est_bass
    eq_semitones = np.all(
        np.equal(ref_semitones[:, :8], est_semitones[:, :8]), axis=1)
    return (eq_roots * eq_semitones * eq_basses).astype(np.float)


@validate
@score
def tetrads(reference_labels, estimated_labels):
    '''Compare chords along tetrad (root & full quality) relationships.

    :usage:
        >>> ref_intervals, ref_labels = mir_eval.io.load_intervals('ref.lab')
        >>> est_intervals, est_labels = mir_eval.io.load_intervals('est.lab')
        >>> est_intervals, est_labels = mir_eval.util.adjust_intervals(
                est_intervals, est_labels, ref_intervals.min(),
                ref_intervals.max(), mir_eval.chord.NO_CHORD,
                mir_eval.chord.NO_CHORD)
        >>> (intervals,
             ref_labels,
             est_labels) = mir_eval.util.merge_labeled_intervals(
                 ref_intervals, ref_labels, est_intervals, est_labels)
        >>> score = mir_eval.chord.tetrads(ref_labels, est_labels, intervals)

    :parameters:
        - reference_labels : list, len=n
            Reference chord labels to score against.
        - estimated_labels : list, len=n
            Estimated chord labels to score against.
        - intervals : np.ndarray, shape=(n, 2)
            Start and end time of each chord label

    :returns:
        - comparison_scores : np.ndarray, shape=(n,), dtype=np.float
            Comparison scores, in {0.0, 1.0}
    '''
    ref_roots, ref_semitones = encode_many(reference_labels, False)[:2]
    est_roots, est_semitones = encode_many(estimated_labels, False)[:2]

    eq_roots = ref_roots == est_roots
    eq_semitones = np.all(np.equal(ref_semitones, est_semitones), axis=1)
    return (eq_roots * eq_semitones).astype(np.float)


@validate
@score
def tetrads_inv(reference_labels, estimated_labels):
    '''Compare chords along seventh (root, quality) relationships.

    :usage:
        >>> ref_intervals, ref_labels = mir_eval.io.load_intervals('ref.lab')
        >>> est_intervals, est_labels = mir_eval.io.load_intervals('est.lab')
        >>> est_intervals, est_labels = mir_eval.util.adjust_intervals(
                est_intervals, est_labels, ref_intervals.min(),
                ref_intervals.max(), mir_eval.chord.NO_CHORD,
                mir_eval.chord.NO_CHORD)
        >>> (intervals,
             ref_labels,
             est_labels) = mir_eval.util.merge_labeled_intervals(
                 ref_intervals, ref_labels, est_intervals, est_labels)
        >>> score = mir_eval.chord.tetrads_inv(ref_labels,
                                               est_labels,
                                               intervals)

    :parameters:
        - reference_labels : list, len=n
            Reference chord labels to score against.
        - estimated_labels : list, len=n
            Estimated chord labels to score against.
        - intervals : np.ndarray, shape=(n, 2)
            Start and end time of each chord label

    :returns:
        - comparison_scores : np.ndarray, shape=(n,), dtype=np.float
            Comparison scores, in {0.0, 1.0}
    '''
    ref_roots, ref_semitones, ref_bass = encode_many(reference_labels, False)
    est_roots, est_semitones, est_bass = encode_many(estimated_labels, False)

    eq_roots = ref_roots == est_roots
    eq_basses = ref_bass == est_bass
    eq_semitones = np.all(np.equal(ref_semitones, est_semitones), axis=1)
    return (eq_roots * eq_semitones * eq_basses).astype(np.float)


@validate
@score
def root(reference_labels, estimated_labels):
    '''Compare chords according to roots.

    :usage:
        >>> ref_intervals, ref_labels = mir_eval.io.load_intervals('ref.lab')
        >>> est_intervals, est_labels = mir_eval.io.load_intervals('est.lab')
        >>> est_intervals, est_labels = mir_eval.util.adjust_intervals(
                est_intervals, est_labels, ref_intervals.min(),
                ref_intervals.max(), mir_eval.chord.NO_CHORD,
                mir_eval.chord.NO_CHORD)
        >>> (intervals,
             ref_labels,
             est_labels) = mir_eval.util.merge_labeled_intervals(
                 ref_intervals, ref_labels, est_intervals, est_labels)
        >>> score = mir_eval.chord.root(ref_labels, est_labels, intervals)

    :parameters:
        - reference_labels : list, len=n
            Reference chord labels to score against.
        - estimated_labels : list, len=n
            Estimated chord labels to score against.
        - intervals : np.ndarray, shape=(n, 2)
            Start and end time of each chord label

    :returns:
        - comparison_scores : np.ndarray, shape=(n,), dtype=np.float
            Comparison scores, in [0.0, 1.0], or -1 if the comparison is out of
            gamut.
    '''

    ref_roots = encode_many(reference_labels, False)[0]
    est_roots = encode_many(estimated_labels, False)[0]
    return (ref_roots == est_roots).astype(np.float)


@validate
@score
def mirex(reference_labels, estimated_labels):
    '''Compare chords along MIREX rules.

    :usage:
        >>> ref_intervals, ref_labels = mir_eval.io.load_intervals('ref.lab')
        >>> est_intervals, est_labels = mir_eval.io.load_intervals('est.lab')
        >>> est_intervals, est_labels = mir_eval.util.adjust_intervals(
                est_intervals, est_labels, ref_intervals.min(),
                ref_intervals.max(), mir_eval.chord.NO_CHORD,
                mir_eval.chord.NO_CHORD)
        >>> (intervals,
             ref_labels,
             est_labels) = mir_eval.util.merge_labeled_intervals(
                 ref_intervals, ref_labels, est_intervals, est_labels)
        >>> score = mir_eval.chord.mirex(ref_labels, est_labels, intervals)

    :parameters:
        - reference_labels : list, len=n
            Reference chord labels to score against.
        - estimated_labels : list, len=n
            Estimated chord labels to score against.
        - intervals : np.ndarray, shape=(n, 2)
            Start and end time of each chord label

    :returns:
        - comparison_scores : np.ndarray, shape=(n,), dtype=np.float
            Comparison scores, in {0.0, 1.0}
    '''
    min_intersection = 3
    ref_data = encode_many(reference_labels, False)
    ref_chroma = rotate_bitmaps_to_roots(ref_data[1], ref_data[0])
    est_data = encode_many(estimated_labels, False)
    est_chroma = rotate_bitmaps_to_roots(est_data[1], est_data[0])

    eq_chroma = (ref_chroma * est_chroma).sum(axis=-1)
    return (eq_chroma >= min_intersection).astype(np.float)


@validate
@score
def majmin(reference_labels, estimated_labels):
    '''Compare chords along major-minor rules. Chords with qualities outside
    Major/minor/no-chord are ignored.

    :usage:
        >>> ref_intervals, ref_labels = mir_eval.io.load_intervals('ref.lab')
        >>> est_intervals, est_labels = mir_eval.io.load_intervals('est.lab')
        >>> est_intervals, est_labels = mir_eval.util.adjust_intervals(
                est_intervals, est_labels, ref_intervals.min(),
                ref_intervals.max(), mir_eval.chord.NO_CHORD,
                mir_eval.chord.NO_CHORD)
        >>> (intervals,
             ref_labels,
             est_labels) = mir_eval.util.merge_labeled_intervals(
                 ref_intervals, ref_labels, est_intervals, est_labels)
        >>> score = mir_eval.chord.majmin(ref_labels, est_labels, intervals)

    :parameters:
        - reference_labels : list, len=n
            Reference chord labels to score against.
        - estimated_labels : list, len=n
            Estimated chord labels to score against.
        - intervals : np.ndarray, shape=(n, 2)
            Start and end time of each chord label

    :returns:
        - comparison_scores : np.ndarray, shape=(n,), dtype=np.float
            Comparison scores, in [0.0, 1.0], or -1 if the comparison is out of
            gamut.
    '''
    maj_semitones = np.array(QUALITIES['maj'][:8])
    min_semitones = np.array(QUALITIES['min'][:8])

    ref_roots, ref_semitones, _ = encode_many(reference_labels, False)
    est_roots, est_semitones, _ = encode_many(estimated_labels, False)

    eq_root = ref_roots == est_roots
    eq_quality = np.all(np.equal(ref_semitones[:, :8],
                                 est_semitones[:, :8]), axis=1)
    comparison_scores = (eq_root * eq_quality).astype(np.float)

    # Test for Major / Minor / No-chord
    is_maj = np.all(np.equal(ref_semitones[:, :8], maj_semitones), axis=1)
    is_min = np.all(np.equal(ref_semitones[:, :8], min_semitones), axis=1)
    is_none = ref_roots < 0

    # Only keep majors, minors, and Nones (NOR)
    comparison_scores[(is_maj + is_min + is_none) == 0] = -1

    # Disable chords that disrupt this quality (apparently)
    # ref_voicing = np.all(np.equal(ref_qualities[:, :8],
    #                               ref_notes[:, :8]), axis=1)
    # comparison_scores[ref_voicing == 0] = -1
    # est_voicing = np.all(np.equal(est_qualities[:, :8],
    #                               est_notes[:, :8]), axis=1)
    # comparison_scores[est_voicing == 0] = -1
    return comparison_scores


@validate
@score
def majmin_inv(reference_labels, estimated_labels):
    '''Compare chords along major-minor rules, with inversions. Chords with
    qualities outside Major/minor/no-chord are ignored, and the bass note must
    exist in the triad (bass in [1, 3, 5]).

    :usage:
        >>> ref_intervals, ref_labels = mir_eval.io.load_intervals('ref.lab')
        >>> est_intervals, est_labels = mir_eval.io.load_intervals('est.lab')
        >>> est_intervals, est_labels = mir_eval.util.adjust_intervals(
                est_intervals, est_labels, ref_intervals.min(),
                ref_intervals.max(), mir_eval.chord.NO_CHORD,
                mir_eval.chord.NO_CHORD)
        >>> (intervals,
             ref_labels,
             est_labels) = mir_eval.util.merge_labeled_intervals(
                 ref_intervals, ref_labels, est_intervals, est_labels)
        >>> score = mir_eval.chord.majmin_inv(ref_labels,
                                              est_labels,
                                              intervals)

    :parameters:
        - reference_labels : list, len=n
            Reference chord labels to score against.
        - estimated_labels : list, len=n
            Estimated chord labels to score against.
        - intervals : np.ndarray, shape=(n, 2)
            Start and end time of each chord label

    :returns:
        - comparison_scores : np.ndarray, shape=(n,), dtype=np.float
            Comparison scores, in [0.0, 1.0], or -1 if the comparison is out of
            gamut.
    '''
    maj_semitones = np.array(QUALITIES['maj'][:8])
    min_semitones = np.array(QUALITIES['min'][:8])

    ref_roots, ref_semitones, ref_bass = encode_many(reference_labels, False)
    est_roots, est_semitones, est_bass = encode_many(estimated_labels, False)

    eq_root_bass = (ref_roots == est_roots) * (ref_bass == est_bass)
    eq_semitones = np.all(np.equal(ref_semitones[:, :8],
                                   est_semitones[:, :8]), axis=1)
    comparison_scores = (eq_root_bass * eq_semitones).astype(np.float)

    # Test for Major / Minor / No-chord
    is_maj = np.all(np.equal(ref_semitones[:, :8], maj_semitones), axis=1)
    is_min = np.all(np.equal(ref_semitones[:, :8], min_semitones), axis=1)
    is_none = ref_roots < 0

    # Only keep majors, minors, and Nones (NOR)
    comparison_scores[(is_maj + is_min + is_none) == 0] = -1

    # Disable inversions that are not part of the quality
    valid_inversion = np.ones(ref_bass.shape, dtype=bool)
    bass_idx = ref_bass >= 0
    valid_inversion[bass_idx] = ref_semitones[bass_idx, ref_bass[bass_idx]]
    comparison_scores[valid_inversion == 0] = -1
    return comparison_scores


@validate
@score
def sevenths(reference_labels, estimated_labels):
    '''Compare chords along MIREX 'sevenths' rules. Chords with qualities
    outside [maj, maj7, 7, min, min7, N] are ignored.

    :usage:
        >>> ref_intervals, ref_labels = mir_eval.io.load_intervals('ref.lab')
        >>> est_intervals, est_labels = mir_eval.io.load_intervals('est.lab')
        >>> est_intervals, est_labels = mir_eval.util.adjust_intervals(
                est_intervals, est_labels, ref_intervals.min(),
                ref_intervals.max(), mir_eval.chord.NO_CHORD,
                mir_eval.chord.NO_CHORD)
        >>> (intervals,
             ref_labels,
             est_labels) = mir_eval.util.merge_labeled_intervals(
                 ref_intervals, ref_labels, est_intervals, est_labels)
        >>> score = mir_eval.chord.sevenths(ref_labels,
                                            est_labels,
                                            intervals)

    :parameters:
        - reference_labels : list, len=n
            Reference chord labels to score against.
        - estimated_labels : list, len=n
            Estimated chord labels to score against.
        - intervals : np.ndarray, shape=(n, 2)
            Start and end time of each chord label

    :returns:
        - comparison_scores : np.ndarray, shape=(n,), dtype=np.float
            Comparison scores, in [0.0, 1.0], or -1 if the comparison is out of
            gamut.
    '''
    seventh_qualities = ['maj', 'min', 'maj7', '7', 'min7', '']
    valid_semitones = np.array([QUALITIES[name] for name in seventh_qualities])

    ref_roots, ref_semitones = encode_many(reference_labels, False)[:2]
    est_roots, est_semitones = encode_many(estimated_labels, False)[:2]

    eq_root = ref_roots == est_roots
    eq_semitones = np.all(np.equal(ref_semitones, est_semitones), axis=1)
    comparison_scores = (eq_root * eq_semitones).astype(np.float)

    # Test for reference chord inclusion
    is_valid = np.array([np.all(np.equal(ref_semitones, semitones), axis=1)
                         for semitones in valid_semitones])
    # Drop if NOR
    comparison_scores[np.sum(is_valid, axis=0) == 0] = -1
    return comparison_scores


@validate
@score
def sevenths_inv(reference_labels, estimated_labels):
    '''Compare chords along MIREX 'sevenths' rules. Chords with qualities
    outside [maj, maj7, 7, min, min7, N] are ignored.

    :usage:
        >>> ref_intervals, ref_labels = mir_eval.io.load_intervals('ref.lab')
        >>> est_intervals, est_labels = mir_eval.io.load_intervals('est.lab')
        >>> est_intervals, est_labels = mir_eval.util.adjust_intervals(
                est_intervals, est_labels, ref_intervals.min(),
                ref_intervals.max(), mir_eval.chord.NO_CHORD,
                mir_eval.chord.NO_CHORD)
        >>> (intervals,
             ref_labels,
             est_labels) = mir_eval.util.merge_labeled_intervals(
                 ref_intervals, ref_labels, est_intervals, est_labels)
        >>> score = mir_eval.chord.sevenths_inv(ref_labels,
                                                est_labels,
                                                intervals)

    :parameters:
        - reference_labels : list, len=n
            Reference chord labels to score against.
        - estimated_labels : list, len=n
            Estimated chord labels to score against.
        - intervals : np.ndarray, shape=(n, 2)
            Start and end time of each chord label

    :returns:
        - comparison_scores : np.ndarray, shape=(n,), dtype=np.float
            Comparison scores, in [0.0, 1.0], or -1 if the comparison is out of
            gamut.
    '''
    seventh_qualities = ['maj', 'min', 'maj7', '7', 'min7', '']
    valid_semitones = np.array([QUALITIES[name] for name in seventh_qualities])

    ref_roots, ref_semitones, ref_basses = encode_many(reference_labels, False)
    est_roots, est_semitones, est_basses = encode_many(estimated_labels, False)

    eq_roots_basses = (ref_roots == est_roots) * (ref_basses == est_basses)
    eq_semitones = np.all(np.equal(ref_semitones, est_semitones), axis=1)
    comparison_scores = (eq_roots_basses * eq_semitones).astype(np.float)

    # Test for Major / Minor / No-chord
    is_valid = np.array([np.all(np.equal(ref_semitones, semitones), axis=1)
                         for semitones in valid_semitones])
    comparison_scores[np.sum(is_valid, axis=0) == 0] = -1

    # Disable inversions that are not part of the quality
    valid_inversion = np.ones(ref_basses.shape, dtype=bool)
    bass_idx = ref_basses >= 0
    valid_inversion[bass_idx] = ref_semitones[bass_idx, ref_basses[bass_idx]]
    comparison_scores[valid_inversion == 0] = -1
    return comparison_scores


# Create an ordered dict mapping metric names to functions
METRICS = collections.OrderedDict()
# MIREX2013 Methods
METRICS['root'] = root
METRICS['majmin'] = majmin
METRICS['majmin-inv'] = majmin_inv
METRICS['sevenths'] = sevenths
METRICS['sevenths-inv'] = sevenths_inv
# Older / Other methods
METRICS['mirex09'] = mirex
METRICS['thirds'] = thirds
METRICS['thirds-inv'] = thirds_inv
METRICS['triads'] = triads
METRICS['triads-inv'] = triads_inv
METRICS['tetrads'] = tetrads
METRICS['tetrads-inv'] = tetrads_inv


def evaluate_file_pair(reference_file, estimation_file,
                       vocabularies=None, boundary_mode='fit-to-ref'):
    '''Load data and perform the evaluation between a pair of annotations.

    :parameters:
     - reference_file : str
        Path to a reference annotation.

     - estimation_file : str
        Path to an estimated annotation.

     - vocabularies : list of strings
        Comparisons to make between the reference and estimated sequences.

     - boundary_mode : str
        Method for resolving sequences of different lengths, one of:
          'intersect':
              Truncate both to the time range
              on which both sequences are defined.
          'fit-to-ref':
              Pad the estimation to match the reference,
              filling missing labels with 'no-chord'.
          'fit-to-est':
              Pad the reference to match the estimation,
              filling missing labels with 'no-chord'.

    :returns:
     - result : dict
        Dictionary containing the averaged scores for each vocabulary, along
        with the total duration of the file ('_weight') and any errors
        ('_error') caught in the process.
    '''

    if vocabularies is None:
        vocabularies = ['minmaj']

    # load the data
    ref_intervals, ref_labels = io.load_intervals(reference_file)
    est_intervals, est_labels = io.load_intervals(estimation_file)

    if boundary_mode == 'intersect':
        t_min = max([ref_intervals.min(), est_intervals.min()])
        t_max = min([ref_intervals.max(), est_intervals.max()])
    elif boundary_mode == 'fit-to-ref':
        t_min = ref_intervals.min()
        t_max = ref_intervals.max()
    elif boundary_mode == 'fit-to-est':
        t_min = est_intervals.min()
        t_max = est_intervals.max()
    else:
        raise ValueError("Unsupported boundary mode: %s" % boundary_mode)

    # Reduce the two annotations to the time intersection of both interval
    #  sequences.
    (ref_intervals,
     ref_labels) = util.filter_labeled_intervals(*util.adjust_intervals(
         ref_intervals, ref_labels, t_min, t_max, NO_CHORD, NO_CHORD))

    (est_intervals,
     est_labels) = util.filter_labeled_intervals(*util.adjust_intervals(
         est_intervals, est_labels, t_min, t_max, NO_CHORD, NO_CHORD))

    # Merge the time-intervals
    try:
        intervals, ref_labels, est_labels = util.merge_labeled_intervals(
            ref_intervals, ref_labels, est_intervals, est_labels)
    except IndexError:
        print est_intervals
        print est_labels
        raise IndexError

    # return ref_labels, est_labels

    # Now compute all the metrics
    result = collections.OrderedDict(_weight=intervals.max())
    try:
        for vocab in vocabularies:
            result[vocab] = METRICS[vocab](ref_labels, est_labels, intervals)
    except InvalidChordException as err:
        if err.chord_label in ref_labels:
            offending_file = reference_file
        else:
            offending_file = estimation_file
        result['_error'] = (err.chord_label, offending_file)
    return result
