# `mir_eval` evaluators

## Usage

To run an evaluator script `task_eval.py`, simply run

`./task_eval.py ref_file est_file arguments`

For help,

`./task_eval.py --help`

Usage of the pre-build binaries (available [here](http://labrosa.ee.columbia.edu/mir_eval/mir_eval.tar.gz)) is identical, except you must omit the `.py`, e.g.

`./task_eval ref_file est_file arguments`

## Creating binaries

To create stand-alone binaries, you need [pyinstaller](http://www.pyinstaller.org/).  It's also recommended that you build the binaries in a virtualenv with only `mir_eval`, `scipy`, `numpy` and `pyinstaller` installed to avoid including unneeded dependencies.  Then, just run

`pyinstaller --one_file task_eval.py`

to build the binary.
