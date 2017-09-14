#!/usr/bin/env python
'''
CREATED:2014-01-17 16:30:07 by Brian McFee <brm2132@columbia.edu>

Compute hierarchical segmentation evaluation metrics

Usage:

./segment_hier_eval.py -r TRUTH_LEVEL1.TXT [TRUTH_LEVEL2.TXT ...] \
                       -e PREDICTION_LEVEL1.TXT [PREDICTION_LEVEL2.TXT ...] \
                       [-o output.json] \
                       [-w WINDOW_SIZE]
'''
import mir_eval

if __name__ == '__main__':
    mir_eval.hierarchy.main()
