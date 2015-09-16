from __future__ import print_function
from future.utils import iteritems
import json


def save_results(results, output_file):
    '''
    Write a result dictionary out as a .json file.

    :parameters:
        - results : dict
            Results dictionary, where keys are metric names and values are
            the corresponding scores
        - output_file : str
            Path to .json file to write to
    '''
    with open(output_file, 'w') as f:
        json.dump(results, f)


def print_evaluation(results):
    '''
    Print out a results dict.

    :parameters:
        - results : dict
            Results dictionary, where keys are metric names and values are
            the corresponding scores
    '''
    max_len = max([len(key) for key in results])
    for key, value in iteritems(results):
        if type(value) == float:
            print('\t{:>{}} : {:.3f}'.format(key, max_len, value))
        else:
            print('\t{:>{}} : {}'.format(key, max_len, value))
