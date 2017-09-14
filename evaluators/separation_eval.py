#!/usr/bin/env python
'''
Utility script for computing source separation metrics

Usage:

./separation_eval.py PATH_TO_REFERENCE_WAVS PATH_TO_ESTIMATED_WAVS
'''
import mir_eval

if __name__ == '__main__':
    mir_eval.separation.main()
