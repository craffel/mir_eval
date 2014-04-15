#!/usr/bin/env python
"""Top-level module for mir-evaluate."""

# Import all submodules (for each task)
from . import beat
from . import chord
from . import input_output as io
from . import onset
from . import boundary
from . import structure
from . import separation
from . import util
from . import sonify
from . import melody
from . import pattern

__version__ = '0.0.1'
