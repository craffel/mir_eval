"""
"""

import unittest
from mir_eval import chord


class ChordTests(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_pitch_class_to_semitone(self):
        self.assertEqual(chord.pitch_class_to_semitone('Gbb'), 5)
        self.assertEqual(chord.pitch_class_to_semitone('G'), 7)
        self.assertEqual(chord.pitch_class_to_semitone('G#'), 8)
        self.assertEqual(chord.pitch_class_to_semitone('Cb'), 11)
        self.assertEqual(chord.pitch_class_to_semitone('B#'), 0)

        self.assertRaises(
            chord.InvalidChordException,
            chord.pitch_class_to_semitone, "Cab")

        self.assertRaises(
            chord.InvalidChordException,
            chord.pitch_class_to_semitone, "#C")

        self.assertRaises(
            chord.InvalidChordException,
            chord.pitch_class_to_semitone, "bG")

    def test_scale_degree_to_semitone(self):
        self.assertEqual(chord.scale_degree_to_semitone('b7'), 10)
        self.assertEqual(chord.scale_degree_to_semitone('#3'), 5)
        self.assertEqual(chord.scale_degree_to_semitone('1'), 0)
        self.assertEqual(chord.scale_degree_to_semitone('b1'), 11)
        self.assertEqual(chord.scale_degree_to_semitone('#7'), 0)
        self.assertEqual(chord.scale_degree_to_semitone('bb5'), 5)

        self.assertRaises(
            chord.InvalidChordException,
            chord.scale_degree_to_semitone, "7b")

        self.assertRaises(
            chord.InvalidChordException,
            chord.scale_degree_to_semitone, "4#")

        self.assertRaises(
            chord.InvalidChordException,
            chord.scale_degree_to_semitone, "77")

    def test_well_formedness(self):
        # Good chords should pass.
        for chord_label in ['C', 'Eb:min/5', 'A#:dim7',
                            'B:maj(*1,*5)/3', 'A#:sus4']:
            chord.validate_chord_label(chord_label)

        # Bad chords should fail.
        self.assertRaises(chord.InvalidChordException,
                          chord.validate_chord_label, "C::maj")
        self.assertRaises(chord.InvalidChordException,
                          chord.validate_chord_label, "C//5")
        self.assertRaises(chord.InvalidChordException,
                          chord.validate_chord_label, "C((4)")
        self.assertRaises(chord.InvalidChordException,
                          chord.validate_chord_label, "C5))")
        self.assertRaises(chord.InvalidChordException,
                          chord.validate_chord_label, "C:maj(*3/3")
        self.assertRaises(chord.InvalidChordException,
                          chord.validate_chord_label, "Cmaj*3/3)")

    def test_split(self):
        self.assertEqual(chord.split('C'), ['C', 'maj', set(), '1'])
        self.assertEqual(chord.split('B:maj(*1,*3)/5'),
                         ['B', 'maj', set(['*1', '*3']), '5'])
        self.assertEqual(chord.split('Ab:min/b3'), ['Ab', 'min', set(), 'b3'])
        self.assertEqual(chord.split('N'), ['N', '', set(), ''])

    def test_join(self):
        self.assertEqual(chord.join('F#'), 'F#')
        self.assertEqual(chord.join('F#', quality='hdim7'), 'F#:hdim7')
        self.assertEqual(
            chord.join('F#', extensions={'*b3', '4'}), 'F#:(*b3,4)')
        self.assertEqual(chord.join('F#', bass='b7'), 'F#/b7')
        self.assertEqual(chord.join('F#', extensions={'*b3', '4'}, bass='b7'),
                         'F#:(*b3,4)/b7')
        self.assertEqual(chord.join('F#', quality='hdim7', bass='b7'),
                         'F#:hdim7/b7')
        self.assertEqual(chord.join('F#', 'hdim7', {'*b3', '4'}, 'b7'),
                         'F#:hdim7(*b3,4)/b7')

    def test_rotate_bitmaps_to_roots(self):
        bitmaps = [
            [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]]
        roots = [0, 5, 11]
        expected_bitmaps = [
            [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]]
        ans = chord.rotate_bitmaps_to_roots(bitmaps, roots).tolist()
        self.assertEqual(ans, expected_bitmaps)

    def test_encode(self):
        root, quality, notes, bass = chord.encode('B:maj(*1,*3)/5')
        self.assertEqual(root, 11)
        self.assertEqual(quality.tolist(),
                         [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        self.assertEqual(notes.tolist(),
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        self.assertEqual(bass, 7)

        root, quality, notes, bass = chord.encode('G:dim')
        self.assertEqual(root, 7)
        self.assertEqual(quality.tolist(),
                         [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0])
        self.assertEqual(notes.tolist(),
                         [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0])
        self.assertEqual(bass, 0)

        # Non-chord bass notes *must* be explicitly named as extensions when
        #   STRICT_BASS_INTERVALS == True
        chord.STRICT_BASS_INTERVALS = True
        self.assertRaises(
            chord.InvalidChordException, chord.encode, 'G:dim(4)/6')
        # Otherwise, we can cut a little slack.
        chord.STRICT_BASS_INTERVALS = False
        root, quality, notes, bass = chord.encode('G:dim(4)/6')
        self.assertEqual(bass, 9)
        self.assertEqual(notes.tolist(),
                         [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0])

    def test_encode_many(self):
        input_list = ['B:maj(*1,*3)/5',
                      'B:maj(*1,*3)/5',
                      'N',
                      'C:min',
                      'C:min']
        expected_roots = [11, 11, -1, 0, 0]
        expected_qualities = [
            [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
        ]
        expected_notes = [
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        ]
        expected_basses = [7, 7, -1, 0, 0]
        roots, qualities, notes, basses = chord.encode_many(input_list)
        self.assertEqual(roots.tolist(), expected_roots)
        self.assertEqual(qualities.tolist(), expected_qualities)
        self.assertEqual(notes.tolist(), expected_notes)
        self.assertEqual(basses.tolist(), expected_basses)

    def test_compare_thirds(self):
        ref = ['N', 'C:maj', 'C:maj', 'C:maj', 'C:min']
        est = ['N', 'N',     'C:aug', 'C:dim', 'C:dim']
        ans = [1.0,  0.0,     1.0,     0.0,     1.0]
        self.assertEqual(chord.compare_thirds(ref, est).tolist(), ans)

        ref = ['C:maj',  'G:min',  'C:maj', 'C:min',   'C:min']
        est = ['C:sus4', 'G:sus2', 'G:maj', 'C:hdim7', 'C:min7']
        ans = [1.0,       0.0,      0.0,     1.0,       1.0]
        self.assertEqual(chord.compare_thirds(ref, est).tolist(), ans)

        ref = ['C:maj',  'F:maj',  'C:maj',     'A:maj', 'A:maj']
        est = ['C:maj6', 'F:min6', 'C:minmaj7', 'A:7',   'A:9']
        ans = [1.0,       0.0,      0.0,         1.0,     1.0]
        self.assertEqual(chord.compare_thirds(ref, est).tolist(), ans)

    def test_compare_thirds_inv(self):
        ref = ['C:maj/5',  'G:min',    'C:maj',   'C:min/b3',   'C:min']
        est = ['C:sus4/5', 'G:min/b3', 'C:maj/5', 'C:hdim7/b3', 'C:dim']
        ans = [1.0,         0.0,        0.0,       1.0,          1.0]
        self.assertEqual(chord.compare_thirds_inv(ref, est).tolist(), ans)

    def test_compare_triads(self):
        ref = ['C:min',  'C:maj', 'C:maj', 'C:min', 'C:maj']
        est = ['C:min7', 'C:7',   'C:aug', 'C:dim', 'C:sus2']
        ans = [1.0,       1.0,     0.0,     0.0,     0.0]
        self.assertEqual(chord.compare_triads(ref, est).tolist(), ans)
        ref = ['C:maj',  'G:min',     'C:maj', 'C:min',   'C:min']
        est = ['C:sus4', 'G:minmaj7', 'G:maj', 'C:hdim7', 'C:min6']
        ans = [0.0,       1.0,         0.0,     0.0,       1.0]
        self.assertEqual(chord.compare_triads(ref, est).tolist(), ans)

    def test_compare_triads_inv(self):
        ref = ['C:maj/5',  'G:min',    'C:maj', 'C:min/b3',  'C:min/b3']
        est = ['C:maj7/5', 'G:min7/5', 'C:7/5', 'C:min6/b3', 'C:dim/b3']
        ans = [1.0,         0.0,        0.0,     1.0,         0.0]
        self.assertEqual(chord.compare_triads_inv(ref, est).tolist(), ans)

    def test_compare_tetrads(self):
        ref = ['C:min',  'C:maj',  'C:7', 'C:maj7',   'C:sus2']
        est = ['C:min7', 'C:maj6', 'C:9', 'C:maj7/5', 'C:sus2/2']
        ans = [0.0,       0.0,      1.0,   1.0,        1.0]
        self.assertEqual(chord.compare_tetrads(ref, est).tolist(), ans)

        # TODO(ejhumphrey): Revisit how minmaj7's are mapped.
        ref = ['C:7/3',   'G:min',  'C:maj', 'C:min',   'C:min']
        est = ['C:11/b7', 'G:sus2', 'G:maj', 'C:hdim7', 'C:minmaj7']  # um..?
        ans = [1.0,        0.0,      0.0,     0.0,       1.0]
        self.assertEqual(chord.compare_tetrads(ref, est).tolist(), ans)

    def test_compare_tetrads_inv(self):
        ref = ['C:maj7/5', 'G:min',    'C:7/5',  'C:min/b3',   'C:min']
        est = ['C:maj7/3', 'G:min/b3', 'C:13/5', 'C:hdim7/b3', 'C:minmaj7/7']
        ans = [0.0,         0.0,        1.0,      0.0,          0.0]
        self.assertEqual(chord.compare_tetrads_inv(ref, est).tolist(), ans)

    def test_compare_majmin(self):
        ref = ['N', 'C:maj', 'C:maj', 'C:aug', 'C:min', 'G:maj7']
        est = ['N', 'N',     'C:aug', 'C:maj', 'C:dim', 'G']
        ans = [1.0,  0.0,     0.0,     -1.0,     0.0,    1.0]
        self.assertEqual(chord.compare_majmin(ref, est).tolist(), ans)

    def test_compare_majmin_inv(self):
        ref = ['C:maj/5',  'G:min',    'C:maj/5', 'C:hdim7/b3', 'C:min7']
        est = ['C:sus4/5', 'G:min/b3', 'C:maj/5', 'C:min/b3',   'C:min']
        ans = [0.0,         0.0,        1.0,       -1.0,         1.0]
        self.assertEqual(chord.compare_majmin_inv(ref, est).tolist(), ans)


if __name__ == "__main__":
    unittest.main()
