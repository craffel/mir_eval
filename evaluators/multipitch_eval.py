#!/usr/bin/env python
'''
Utility script for computing all multipitch metrics.

Usage:

./multipitch_eval.py REFERENCE.TXT ESTIMATED.TXT
'''
import mir_eval

if __name__ == '__main__':
    mir_eval.multipitch.main()
