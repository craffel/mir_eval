#!/usr/bin/env python
"""Top-level module for mir-evaluate."""

# Import all submodules (for each task)
from . import beat
from . import chord
from . import input_output as io
from . import onset
from . import segment
from . import separation
from . import util
from . import sonify

__version__ = '0.0.1'
