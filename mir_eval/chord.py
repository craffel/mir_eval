'''Chord Definitions -- because some things just need to be hard-coded.

Note that pitch class counting starts at C, e.g.
     C: 0, D:2, E:4, F:5, ...
'''

import numpy as np

NO_CHORD = "N"


class InvalidChordException(BaseException):
    '''Hollow class for invalid formatting.'''
    pass


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


def _validate(chord_label):
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


def split(chord_label, extended_chords=False):
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
    _validate(chord_label)
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

    if not extended_chords:
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
    _validate(chord_label)
    return chord_label
