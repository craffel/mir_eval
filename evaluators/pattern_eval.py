#!/usr/bin/env python
"""
Compute pattern discovery evaluation metrics.

Usage:
    ./pattern_eval.py REFERENCE.TXT ESTIMATION.TXT

Example:
    ./pattern_eval.py ../tests/data/pattern/reference-mono.txt \
                      ../tests/data/pattern/estimate-mono.txt

Written by Oriol Nieto (oriol@nyu.edu), 2014
"""
import mir_eval

if __name__ == '__main__':
    mir_eval.pattern.main()
