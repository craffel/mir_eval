#!/usr/bin/env python
'''
Generate data for regression tests.
This is a pretty specialized file and should probably only be used if you know
what you're doing.
'''

import mir_eval
import glob
import json
import numpy as np
import os
import sys


def load_separation_data(folder):
    '''
    Loads in a stacked matrix of the .wavs in the provided folder.
    We need this because there's no specialized loader in .io for it.
    '''
    data = []
    global_fs = None
    # Load in each reference file in the supplied dir
    for reference_file in glob.glob(os.path.join(folder, '*.wav')):
        audio_data, fs = mir_eval.io.load_wav(reference_file)
        # Make sure fs is the same for all files
        assert (global_fs is None or fs == global_fs)
        global_fs = fs
        data.append(audio_data)
    return np.vstack(data)


if __name__ == '__main__':
    # This dict will contain tuples of (submodule, loader, glob path)
    # The keys are 'beat', 'chord', etc.
    # Whatever is passed in as argv will be grabbed from it and the data for
    # that task will be generated.
    tasks = {}
    tasks['beat'] = (mir_eval.beat, mir_eval.io.load_events,
                     'data/beat/{}*.txt')
    tasks['chord'] = (mir_eval.chord, mir_eval.io.load_labeled_intervals,
                      'data/chord/{}*.lab')
    tasks['melody'] = (mir_eval.melody, mir_eval.io.load_time_series,
                       'data/melody/{}*.txt')
    tasks['onset'] = (mir_eval.onset, mir_eval.io.load_events,
                      'data/onset/{}*.txt')
    tasks['pattern'] = (mir_eval.pattern, mir_eval.io.load_patterns,
                        'data/pattern/{}*.txt')
    tasks['segment'] = (mir_eval.segment, mir_eval.io.load_labeled_intervals,
                        'data/segment/{}*.lab')
    tasks['separation'] = (mir_eval.separation, load_separation_data,
                           'data/separation/{}*')
    for task in sys.argv[1:]:
        print 'Generating data for {}'.format(task)
        submodule, loader, data_glob = tasks[task]
        for ref_file, est_file in zip(glob.glob(data_glob.format('ref')),
                                      glob.glob(data_glob.format('est'))):
            ref_data = loader(ref_file)
            est_data = loader(est_file)
            if type(ref_data) == tuple:
                scores = submodule.evaluate(*(ref_data + est_data))
            else:
                scores = submodule.evaluate(ref_data, est_data)
            output_file = ref_file.replace('ref', 'output')
            output_file = os.path.splitext(output_file)[0] + '.json'
            with open(output_file, 'w') as f:
                json.dump(scores, f)
