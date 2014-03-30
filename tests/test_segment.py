'''
Unit tests for mir_eval.beat
'''

import numpy as np
import mir_eval
import pickle
import sys
sys.path.append('../evaluators')
import segment_eval

def test_segment_functions():
    # Compute metrics for an example ground truth/annotation
    metrics = segment_eval.evaluate('data/segment/reference.lab', 'data/segment/estimate.lab')
    # Load a snapshot of the metrics to regression test against
    with open('data/segment/reference_scores.pickle') as f:
        reference_metrics = pickle.load(f)
    # Fail if a different number of metrics are present
    assert len(metrics) == len(reference_metrics)
    for metric, score in metrics.items():
        # Check that the score matches
        assert np.allclose(score, reference_metrics[metric])
