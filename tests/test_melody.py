# CREATED: 4/15/14 9:42 AM by Justin Salamon <justin.salamon@nyu.edu>
'''
Unit tests for mir_eval.melody
'''

import numpy as np
import os, sys
sys.path.append('../evaluators')
import melody_eval

def test_melody_functions():

    songs = ['daisy1','daisy2','daisy3','daisy4','jazz1','jazz2','jazz3','jazz4','midi1','midi2','midi3','midi4','opera_fem2','opera_fem4','opera_male3','opera_male5','pop1','pop2','pop3','pop4']
    refpath = 'data/melody/mirex2011/adc2004_ref/original/' # Original/official REF files
    # refpath = 'data/melody/mirex2011/adc2004_ref/resampled10ms/' # REF files resampled to 10ms hop size (not sure how though)
    estpath = 'data/melody/mirex2011/adc2004_SG2/'
    resultspath = 'data/melody/mirex2011/adc2004_results/SG2_per_track_results_mapped.csv' # MIREX 2011 official results
    # resultspath = 'data/melody/mirex2011/adc2004_results/SG2_per_track_results_mapped_JS.csv' # Justin's results based on old home-made code

    # create results dictionary
    results = np.loadtxt(resultspath, dtype='string', delimiter=',')
    keys = results[0]
    results_dict = {}
    for i in range(1,len(results)):
        value_dict = {}
        for k in range(1,len(keys)):
            value_dict[keys[k]] = results[i][k]
        results_dict[results[i][0]] = value_dict

    hop = 0.01

    for song in songs:
        print song

        reffile = os.path.join(refpath, song + "REF.txt")
        estfile = os.path.join(estpath, song + "_mel.txt")

        M = melody_eval.evaluate(reffile, estfile)

        # compare results
        for metric in M.keys():
            mirex_result = float(results_dict[song + '.wav'][metric])
            mireval_result = M[metric]
            diff = np.abs(mirex_result - mireval_result)
            if diff > 0.01:
                print "\t%s: %.3f [mx:%.3f me:%.3f]" % (metric, diff, mirex_result, mireval_result)
