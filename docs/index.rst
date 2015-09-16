**************************
``mir_eval`` Documentation
**************************

``mir_eval`` is a Python library which provides a transparent, standaridized, and straightforward way to evaluate Music Information Retrieval systems.
It can be used in any of the following ways:

* By importing it and calling it from your Python code (see :ref:`installation` and :ref:`mir_eval_quickstart`)
* Via the included evaluator Python scripts (see :ref:`installation` and :ref:`evaluators`)

If you use ``mir_eval`` in a research project, please cite the following paper:

C. Raffel, B. McFee, E. J. Humphrey, J. Salamon, O. Nieto, D. Liang, and D. P. W. Ellis, `"mir_eval: A Transparent Implementation of Common MIR Metrics" <http://colinraffel.com/publications/ismir2014mir_eval.pdf>`_, Proceedings of the 15th International Conference on Music Information Retrieval, 2014.

.. _installation:

Installing ``mir_eval``
=======================

The simplest way to install ``mir_eval`` is by using ``pip``, which will also install the required dependencies (Scipy and Numpy) if needed.
To install ``mir_eval`` using ``pip``, simply run

``pip install mir_eval``

Alternatively, you can install ``mir_eval`` from source by first installing Scipy/Numpy via ``pip`` or by following the instructions here:
http://www.scipy.org/install.html
and then running

``python setup.py install``

from the source directory.

If you don't use Python and want to get started as quickly as possible, you might consider using `Anaconda <https://store.continuum.io/cshop/anaconda/>`_ which makes it easy to install a Python environment which can run ``mir_eval``.

.. _evaluators:

Quickstart: Using the evaluators
================================

The fastest way to get up and running with ``mir_eval`` is to use the evaluators.
These are scripts which can be run from the command line and utilize ``mir_eval`` to compute metrics according to reference and estimated annotations you provide.
To use the evaluators, you must first install ``mir_eval`` and its dependencies (see :ref:`installation`).
The evaluator Python scripts can be found in the ``mir_eval`` github repository in the ``evaluators`` folder:

http://github.com/craffel/mir_eval/tree/master/evaluators

One evaluator is included for each of the MIR tasks implemented in ``mir_eval``.
By way of example, we'll cover the usage of the beat detection evaluator ``beat_eval``.
To use an evaluator for a different task, simply replace ``beat_eval`` in the following with the name of the evaluator for the task you're interested in.
To get usage help, simply run

``./beat_eval.py --help``

As an example, to evaluate generated beat times stored in the file ``estimated_beats.txt`` against ground-truth beats stored in the file ``reference_beats.txt`` and store the resulting scores in ``results.json``, simply run

``./beat_eval.py -o results.json reference_beats.txt estimated_beats.txt``

The file ``results.json`` will now contain the achieved scores in machine-parsable, human-readable json format.  Nice!


.. _mir_eval_quickstart:

Quickstart: Using ``mir_eval`` in Python code
=============================================

Once you've installed ``mir_eval`` (see :ref:`installation`), you can import it in your Python code as follows:

``import mir_eval``

From here, you will typically either load in data and call the ``evaluate()`` function from the appropriate submodule like so::

  reference_beats = mir_eval.io.load_events('reference_beats.txt')
  estimated_beats = mir_eval.io.load_events('estimated_beats.txt')
  # Scores will be a dict containing scores for all of the metrics
  # implemented in mir_eval.beat.  The keys are metric names
  # and values are the scores achieved
  scores = mir_eval.beat.evaluate(reference_beats, estimated_beats)

or you'll load in the data, do some preprocessing, and call specific metric functions from the appropriate submodule like so::

  reference_beats = mir_eval.io.load_events('reference_beats.txt')
  estimated_beats = mir_eval.io.load_events('estimated_beats.txt')
  # Crop out beats before 5s, a common preprocessing step
  reference_beats = mir_eval.beat.trim_beats(reference_beats)
  estimated_beats = mir_eval.beat.trim_beats(estimated_beats)
  # Compute the F-measure metric and store it in f_measure
  f_measure = mir_eval.beat.f_measure(reference_beats, estimated_beats)

The documentation for each metric function, found in the :ref:`mir_eval` section below, contains further usage information.

.. _mir_eval:

``mir_eval``
============

The structure of the ``mir_eval`` Python module is as follows:
Each MIR task for which evaluation metrics are included in ``mir_eval`` is given its own submodule, and each metric is defined as a separate function in each submodule.
Every metric function includes detailed documentation, example usage, input validation, and references to the original paper which defined the metric (see the subsections below).
The task submodules also all contain a function ``evaluate()``, which takes as input reference and estimated annotations and returns a dictionary of scores for all of the metrics implemented (for casual users, this is the place to start).
Finally, each task submodule also includes functions for common data pre-processing steps.

``mir_eval`` also includes the following additional submodules:

* ``io`` which contains convenience functions for loading in task-specific data from common file formats
* ``util`` which includes miscellaneous functionality shared across the submodules
* ``sonify`` which implements some simple methods for synthesizing annotations of various formats for "evaluation by ear".

The following subsections document each submodule.

:mod:`mir_eval.beat`
--------------------
.. automodule:: mir_eval.beat
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

:mod:`mir_eval.chord`
---------------------
.. automodule:: mir_eval.chord
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

:mod:`mir_eval.melody`
----------------------
.. automodule:: mir_eval.melody
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

:mod:`mir_eval.onset`
---------------------
.. automodule:: mir_eval.onset
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

:mod:`mir_eval.pattern`
-----------------------
.. automodule:: mir_eval.pattern
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

:mod:`mir_eval.segment`
-----------------------
.. automodule:: mir_eval.segment
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

:mod:`mir_eval.separation`
--------------------------
.. automodule:: mir_eval.separation
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

:mod:`mir_eval.util`
--------------------
.. automodule:: mir_eval.util
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

:mod:`mir_eval.io`
------------------
.. automodule:: mir_eval.input_output
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

:mod:`mir_eval.sonify`
----------------------
.. automodule:: mir_eval.sonify
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

