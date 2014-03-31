'''Functions and other supporting code for wrangling and comparing chords for
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

'mirex'
- count pitch class overlap. Requires
  an additional switch argument:

'mirex-augdim'
- augdim_switch. Boolean. If True,
  only require 2 pitch classes in common
  with gt to get a point for augmented
  or diminished chords and 3 otherwise.
  Strange, but seems to be what MIREX team
  does.

'exact'
- chords are correct only if they are identical,
  including enharmonics ie 'F#:maj' != 'Gb:maj'

'pitch_class'
- chords are reduced to pitch classes,
  and compared at this level. This means that
  enharmonics and inversions are considered equal
  i.e. score('F#:maj', 'Gb:maj') = 1.0 and
  score('C:maj6' = [C,E,G,A], 'A:min7' = [A,C,E,G]) = 1.0

'dyads'
- chords are mapped to major or minor dyads,
  and compared at this level. For example,
  score_thirds('A:7', 'A:maj') = 1.0, but also
  score_thirds('A:min', 'A:dim') = 1.0 as dim gets
  mapped to min. Probably a bit sketchy,
  but is a common metric

'triads'
- chords are mapped to triad (major, minor,
  augmented, diminished, suspended) and
  compared at this level. For example,
  score('A:7','A:maj') = 1.0,
  score('A:min', 'A:dim') = 0.0

'sevenths'
- chords are mapped to 7th type (7, maj7,
  min7, minmaj7, susb7, dim7) and compared
  at this level. For example:
  score('A:7', 'A:9') = 1.0,
  score('A:7', 'A:maj7') = 0.0

'pitch_class-recall'
- recall on pitch classes in ref/est. Chords are not
  reduced to a simpler alphabet in this evaluation

'pitch_class-precision'
- precision on pitch classes in ref/est. Chords are not
  reduced to a simpler alphabet in this evaluation

'pitch_class-f'
- f-measure on pitch classes. Chords are not
reduced to a simpler alphabet in this evaluation

'''

import numpy as np

NO_CHORD = "N"
NO_CHORD_ENCODED = -1, np.array([0]*12), np.array([0]*12), -1


class InvalidChordException(BaseException):
    '''Hollow class for invalid formatting.'''
    pass


# --- Chord Primitives ---
def _pitch_classes():
    '''Map from pitch class (str) to semitone (int).'''
    pitch_classes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    semitones = [0, 2, 4, 5, 7, 9, 11]
    return dict([(c, s) for c, s in zip(pitch_classes, semitones)])


def _scale_degrees():
    '''Mapping from scale degrees (str) to semitones (int).'''
    degrees = ['1', '2', '3', '4', '5', '6', '7', '9', '10', '11', '12', '13']
    semitones = [0, 2, 4, 5, 7, 9, 11, 2, 4, 5, 7, 9]
    return dict([(d, s) for d, s in zip(degrees, semitones)])


# Maps pitch classes (strings) to semitone indexes (ints).
PITCH_CLASSES = _pitch_classes()


def pitch_class_to_semitone(pitch_class):
    '''Convert a pitch class to semitone.

    :parameters:
    - pitch_class: str
        Spelling of a given pitch class, e.g. 'C#', 'Gbb'

    :returns:
    - semitone: int
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
    '''Convert a scale degree to semitone.

    :parameters:
    - scale degree: str
        Spelling of a relative scale degree, e.g. 'b3', '7', '#5'

    :returns:
    - semitone: int
        Relative semitone value of the scale degree.

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
    return (semitone + offset) % 12


def scale_degree_to_bitmap(scale_degree):
    '''Create a bitmap representation of a scale degree.

    Note that values in the bitmap may be negative, indicating that the
    semitone is to be removed.

    :parameters:
    - scale_degree: str
        Spelling of a relative scale degree, e.g. 'b3', '7', '#5'

    :returns:
    - bitmap: np.ndarray, in [-1, 0, 1]
        Bitmap representation of this scale degree (12-dim).
    '''
    sign = 1
    if scale_degree.startswith("*"):
        sign = -1
        scale_degree = scale_degree.strip("*")
    edit_map = [0] * 12
    edit_map[scale_degree_to_semitone(scale_degree)] = sign
    return np.array(edit_map)


# Maps quality strings to bitmaps, corresponding to relative pitch class
# semitones, i.e. vector[0] is the tonic.
QUALITIES = {
    'maj':   [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'min':   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    'aug':   [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    'dim':   [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    'sus4':  [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    'sus2':  [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    '7':     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj7':  [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min7':  [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    'maj6':  [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    'min6':  [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    'dim7':  [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    'hdim7': [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]}


def quality_to_bitmap(quality):
    '''Return the bitmap for a given quality.

    :parameters:
    quality: str
        Chord quality name.

    :returns:
    bitmap: np.ndarray, in [0, 1]
        Bitmap representation of this quality (12-dim).

    Raises
    ------
    InvalidChordException
    '''
    if not quality in QUALITIES:
        raise InvalidChordException(
            "Unsupported chord quality: '%s' "
            "Did you mean to reduce extended chords?" % quality)
    return np.array(QUALITIES[quality])


# Maps extended chord qualities to the subset above, translating additional
# voicings to extensions as a set of scale degrees (strings).
EXTENDED_QUALITY_REDUX = {
    'minmaj7': ('min',  set(['7'])),
    'maj9':    ('maj7', set(['9'])),
    'min9':    ('min7', set(['9'])),
    '9':       ('7',    set(['9'])),
    'b9':      ('7',    set(['b9'])),
    '#9':      ('7',    set(['#9'])),
    '11':      ('7',    set(['9', '11'])),
    '#11':     ('7',    set(['9', '#11'])),
    '13':      ('7',    set(['9', '11', '13'])),
    'b13':     ('7',    set(['9', '11', 'b13'])),
    'min11':   ('min7', set(['9', '11'])),
    'maj13':   ('maj7', set(['9', '11', '13'])),
    'min13':   ('min7', set(['9', '11', '13']))}


def reduce_extended_quality(quality):
    '''Map an extended chord quality to a simpler one, moving upper voices to
    a set of scale degree extensions.

    :parameters:
    - quality: str
        Extended chord quality to reduce.

    :returns:
    - base_quality: str
        New chord quality.
    - extensions: set
        Scale degrees extensions for the quality.
    '''
    return EXTENDED_QUALITY_REDUX.get(quality, (quality, set()))


# --- Chord Label Parsing ---
def validate_chord_label(chord_label):
    '''Test for well-formedness of a chord label.

    :parameters:
    - chord: str
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
    - quality
    - extensions
    - bass

    Some examples:
        'C' -> ['C', 'maj', {}, '1']
        'G#:min(*b3,*5)/5' -> ['G#', 'min', {'*b3', '*5'}, '5']


    :parameters:
    - chord_label: str
        A chord label.

    :returns:
    - chord_parts: list
        Split version of the chord label.
    '''
    chord_label = str(chord_label)
    validate_chord_label(chord_label)
    if chord_label == NO_CHORD:
        return [chord_label, '', set(), '']

    bass = '1'
    if "/" in chord_label:
        chord_label, bass = chord_label.split("/")

    extensions = set()
    if "(" in chord_label:
        chord_label, extensions = chord_label.split("(")
        extensions = extensions.strip(")")
        extensions = set([e.strip() for e in extensions.split(",")])

    # By default, unspecified qualities are major.
    quality = 'maj'
    if ":" in chord_label:
        root, quality_name = chord_label.split(":")
        # Extended chords (with ":"s) may not explicitly have Major qualities,
        # so only overwrite the default if the string is not empty.
        if quality_name:
            quality = quality_name.lower()
    else:
        root = chord_label

    if reduce_extended_chords:
        quality, addl_extensions = reduce_extended_quality(quality)
        extensions.update(addl_extensions)

    return [root, quality, extensions, bass]


def join(root, quality='', extensions=None, bass=''):
    '''Join the parts of a chord into a complete chord label.

    :parameters:
    root: str
        Root pitch class of the chord, e.g. 'C', 'Eb'
    quality: str
        Quality of the chord, e.g. 'maj', 'hdim7'
    extensions: list
        Any added or absent scaled degrees for this chord, e.g. ['4', '*3']
    bass: str
        Scale degree of the bass note, e.g. '5'.

    :returns:
    chord_label: str
        A complete chord label.

    :raises:
    - InvalidChordException: Thrown if the provided args yield a garbage chord
        label.
    '''
    chord_label = root
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
    - chord_label: str
        Chord label to encode.

    - reduce_extended_chords: bool, default=False
        Map the upper voicings of extended chords (9's, 11's, 13's) to semitone
        extensions.

    :returns:
    -root_number: int
        Absolute semitone of the chord's root.

    - quality_bitmap: np.ndarray of ints, in [0, 1]
        12-dim vector of relative semitones in the chord quality.

    - note_bitmap: np.ndarray of ints, in [0, 1]
        12-dim vector of relative semitones in the chord spelling.

    - bass_number: int
        Relative semitone of the chord's bass note, e.g. 0=root, 7=fifth, etc.

    :raises:
    - InvalidChordException: Thrown if the given bass note is not explicitly
        named as an extension.
    """

    if chord_label == NO_CHORD:
        return NO_CHORD_ENCODED
    root, quality, exts, bass = split(
        chord_label, reduce_extended_chords=reduce_extended_chords)

    root_number = pitch_class_to_semitone(root)
    bass_number = scale_degree_to_semitone(bass)
    quality_bitmap = quality_to_bitmap(quality)

    note_bitmap = np.array(quality_bitmap)
    for sd in list(exts):
        note_bitmap += scale_degree_to_bitmap(sd)

    note_bitmap = (note_bitmap > 0).astype(np.int)
    if not note_bitmap[bass_number]:
        raise InvalidChordException(
            "Given bass scale degree is absent from this chord: "
            "%s" % chord_label)
    return root_number, quality_bitmap, note_bitmap, bass_number


def encode_many(chord_labels, reduce_extended_chords=False):
    """Translate a set of chord labels to numerical representations for sane
    evaluation.

    :parameters:
    - chord_labels: list
        Set of chord labels to encode.

    - reduce_extended_chords: bool, default=True
        Map the upper voicings of extended chords (9's, 11's, 13's) to semitone
        extensions.

    Returns
    -------
    - root_number: np.ndarray of ints
        Absolute semitone of the chord's root.

    - quality_bitmap: np.ndarray of ints, in [0, 1]
        12-dim vector of relative semitones in the given chord quality.

    - note_bitmap: np.ndarray of ints, in [0, 1]
        12-dim vector of relative semitones in the given chord spelling.

    - bass_number: np.ndarray
        Relative semitones of the chord's bass notes.

    :raises:
    - InvalidChordException: Thrown if the given bass note is not explicitly
        named as an extension.
    """
    num_items = len(chord_labels)
    roots, basses = np.zeros([2, num_items], dtype=np.int)
    qualities, notes = np.zeros([2, num_items, 12], dtype=np.int)
    for i, c in enumerate(chord_labels):
        roots[i], qualities[i], notes[i], basses[i] = encode(
            c, reduce_extended_chords)
    return roots, qualities, notes, basses


def rotate_bitmap_to_root(bitmap, root):
    '''Circularly shift a relative bitmap to its asbolute pitch classes.

    For clarity, the best explanation is an example. Given 'G:Maj', the root
    and quality map are as follows:
        root=5
        quality=[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]  # Relative chord shape

    After rotating to the root, the resulting bitmap becomes:
        abs_quality = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]  # G, B, and D

    :parameters:
        - bitmap: np.ndarray, shape=(12,)
            Bitmap of active notes, relative to the given root.

        - root: int
            Absolute pitch class number.

    :returns:
        - bitmap: np.ndarray, shape=(12,)
            Absolute bitmap of active pitch classes.
    '''
    assert bitmap.ndim == 1, "Currently only 1D bitmaps are supported."
    idxs = list(np.nonzero(bitmap))
    idxs[-1] = (idxs[-1] + root) % 12
    abs_bitmap = np.zeros_like(bitmap)
    abs_bitmap[idxs] = 1
    return abs_bitmap


def rotate_bass_to_root(bass, root):
    '''Rotate a relative bass interval to its asbolute pitch class.

    :parameters:
        - bass: int
            Relative bass interval.
        - root: int
            Absolute root pitch class.

    :returns:
        - bass: int
            Pitch class of the bass intervalself.
    '''
    return (bass + root) % 12


# --- Evaluation Routines ---
def validate(comparison):
    '''Decorator which checks that the input annotations to a comparison
    function look like valid chord labels.

    :parameters:
        - comparison : function
            Evaluation comparison function.  First two arguments must be
            reference_labels and estimated_labels.

    :returns:
        - comparison_validated : function
            The function with the labels validated.
    '''
    def comparison_validated(reference_labels, estimated_labels, *args,
                             **kwargs):
        '''
        Comparison with labels validated.
        '''
        N = len(reference_labels)
        M = len(estimated_labels)
        if N != M:
            raise ValueError(
                "Chord comparison received different length lists: "
                "len(reference)=%d\tlen(estimates)=%d" % (N, M))
        for labels in [reference_labels, estimated_labels]:
            for chord_label in labels:
                validate_chord_label(chord_label)

        return comparison(reference_labels, estimated_labels, *args, **kwargs)
    return comparison_validated


@validate
def score_dyads(reference_labels, estimated_labels):
    '''Score chords along dyadic (root and third) relationships.

    :parameters:
        - reference_labels : list, len=n
            Reference chord labels to score against.
        - estimated_labels : list, len=n
            Estimated chord labels to score against.

    :returns:
        - scores : np.ndarray, shape=(n,), dtype=np.float
            Comparison scores, in {0.0, 1.0}
    '''
    ref_roots, ref_qualities = encode_many(reference_labels, True)[:2]
    est_roots, est_qualities = encode_many(estimated_labels, True)[:2]

    correct_root = ref_roots == est_roots
    correct_third = ref_qualities[:, 3] == est_qualities[:, 3]
    return (correct_root * correct_third).astype(np.float)
