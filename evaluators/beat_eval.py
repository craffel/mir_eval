#!/usr/bin/env python
'''
CREATED:2014-01-24 12:42:43 by Brian McFee <brm2132@columbia.edu>

Compute beat evaluation metrics

Usage:

./beat_eval.py REFERENCE.TXT ESTIMATED.TXT
'''
import mir_eval

if __name__ == '__main__':
    mir_eval.beat.main()
