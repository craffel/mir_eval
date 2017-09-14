#!/usr/bin/env python
'''
Compute chord evaluation metrics

Usage:

./chord_eval.py TRUTH.TXT PREDICTION.TXT
'''
import mir_eval

if __name__ == '__main__':
    mir_eval.chord.main()
