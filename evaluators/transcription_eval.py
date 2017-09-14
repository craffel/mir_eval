#!/usr/bin/env python
'''
CREATED: 2/9/16 2:59 PM by Justin Salamon <justin.salamon@nyu.edu>

Compute note transcription evaluation metrics

Usage:

./transcription_eval.py REFERENCE.TXT ESTIMATED.TXT
'''

import mir_eval

if __name__ == '__main__':
    mir_eval.transcription.main()
