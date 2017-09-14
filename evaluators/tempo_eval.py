#!/usr/bin/env python
'''
Utility script for computing all tempo metrics.

Usage:

./tempo_eval.py REFERENCE.TXT ESTIMATED.TXT
'''
import mir_eval

if __name__ == '__main__':
    mir_eval.tempo.main()
