Changes
=======

v0.4
----

- `#189`_: expanded transcription metrics
- `#195`_: added pitch contour sonification
- `#196`_: added the `display` submodule
- `#203`_: support unsorted segment intervals
- `#205`_: correction in documentation for `sonify.time_frequency`
- `#208`_: refactored file/buffer loading
- `#210`_: added `io.load_tempo`
- `#212`_: added frame-wise blind-source separation evaluation

.. _#189: https://github.com/craffel/mir_eval/issues/189
.. _#195: https://github.com/craffel/mir_eval/issues/195
.. _#196: https://github.com/craffel/mir_eval/issues/196
.. _#203: https://github.com/craffel/mir_eval/issues/203
.. _#205: https://github.com/craffel/mir_eval/issues/205
.. _#208: https://github.com/craffel/mir_eval/issues/208
.. _#210: https://github.com/craffel/mir_eval/issues/210
.. _#212: https://github.com/craffel/mir_eval/issues/212

v0.3
----
- `#170`_: implemented transcription metrics
- `#173`_: fixed a bug in chord sonification
- `#175`_: filter_kwargs passes through `**kwargs`
- `#181`_: added key detection metrics

.. _#170: https://github.com/craffel/mir_eval/issues/170
.. _#173: https://github.com/craffel/mir_eval/issues/173
.. _#175: https://github.com/craffel/mir_eval/issues/175
.. _#181: https://github.com/craffel/mir_eval/issues/181

v0.2
----

- `#103`_: incomplete files passed to `melody.evaluate` should warn
- `#109`_: `STRICT_BASS_INTERVALS` is now an argument to `chord.encode`
- `#122`_: improved handling of corner cases in beat tracking
- `#136`_: improved test coverage 
- `#138`_: PEP8 compliance
- `#139`_: converted documentation to numpydoc style
- `#147`_: fixed a rounding error in segment intervals
- `#150`_: `sonify.chroma` and `sonify.chords` pass `kwargs` to `time_frequecy`
- `#151`_: removed `labels` support from `util.boundaries_to_intervals`
- `#159`_: fixed documentation error in `chord.tetrads`
- `#160`_: fixed documentation error in `util.intervals_to_samples`

.. _#103: https://github.com/craffel/mir_eval/issues/103
.. _#109: https://github.com/craffel/mir_eval/issues/109
.. _#122: https://github.com/craffel/mir_eval/issues/122
.. _#136: https://github.com/craffel/mir_eval/issues/136
.. _#138: https://github.com/craffel/mir_eval/issues/138
.. _#139: https://github.com/craffel/mir_eval/issues/139
.. _#147: https://github.com/craffel/mir_eval/issues/147
.. _#150: https://github.com/craffel/mir_eval/issues/150
.. _#151: https://github.com/craffel/mir_eval/issues/151
.. _#159: https://github.com/craffel/mir_eval/issues/159
.. _#160: https://github.com/craffel/mir_eval/issues/160


v0.1
----

- Initial public release.
