#!/usr/bin/env python
'''
Utility script for computing all onset metrics.

Usage:

./onset_eval.py REFERENCE.TXT ESTIMATED.TXT
'''
import mir_eval

if __name__ == '__main__':
    mir_eval.onset.main()
