def evaluate_chords(GT, P, resolution=0.001, trim_method='min', method='MIREX', augdim_switch=True):

  '''
  Evaluate two chord transcriptions. Each should be a list of tuples:
  GT = [(start_1, end_1, chord_1), ... , (start_n, end_n, chord_n)]
  P  = [(start_1, end_1, chord_1), ... , (start_m, end_m, chord_m)]

  times measured in seconds, chords in Chris Harte's format:

  http://ismir2005.ismir.net/proceedings/1080.pdf

  Inputs: GT, list of tuples          - Ground truth chords
          P, list of tuples           - Predicted chords
          resolution, float           - frame rate to use.
          trim_method, string         - how to deal with predictions and gts of 
                                        different length. One of:
                                        'min' - choose the true length to be minimum of 
                                        gt and p. Everything else is trimmed off
                                        'max' - true length is max. Anything shorter is 
                                        scored as 0.0
                                        'GT'  - always trust the GT length
                                        'P'   - always trust the P length   

          method, string              - scoring method. currently supported:
                                        'MIREX'                 - count pitch class overlap. Requires
                                                                  an additional switch argument:
                                                                - augdim_switch. Boolean. If True,
                                                                  only require 2 pitch classes in common
                                                                  with gt to get a point for augmented
                                                                  or diminished chords and 3 otherwise.
                                                                  Strange, but seems to be what MIREX team 
                                                                  does.

                                        'Correct'               - chords are correct only if they are identical,
                                                                  ie 'A:min7' != 'C:maj6'

                                        'Correct_at_minmaj'     - chords are mapped to major or minor triads,
                                                                  and compared at this level

                                        'Correct_at_triad'      - chords are mapped to triad (major, minor, 
                                                                  augmented, diminished, suspended) and 
                                                                  compared at this level   

                                        'Correct_at_seventh'    - chords are mapped to 7th type (7, maj7, 
                                                                  min7, minmaj7, susb7, dim7) and compared
                                                                  at this level        
                                                                  
                                        'Correct_at_pitchclass' - chords are reduced to pitch classes, 
                                                                  and compared at this level                                 

    Outputs: accuracy, float                                                  
  '''
  import numpy as np


  # 1 - Sample the GT and Prediction
  # --------------------------------

  # How should we trim the GT, P?
  GT_len = float(GT[-1][1].strip())
  P_len = float(P[-1][1].strip())

  maxlen = np.max([GT_len, P_len])
  minlen = np.min([GT_len, P_len])

  if trim_method == 'min':
    t_max = minlen
  elif trim_method == 'max':
  	t_max = maxlen
  elif trim_method == 'GT':
    t_max = GT_len
  elif trim_method == 'P':
    t_max = P_len

  # Initialise the sampled GT and P to be None.
  GT_sample = [None] * int(t_max / resolution)
  P_sample = [None] * int(t_max / resolution)

  # initialise t (time), it (time index), current_p (index in p)
  # current_gt (index in gt)
  t = 0.0
  it = 0
  current_p = 0
  current_gt = 0
  while t < t_max:
  
    # store, if <= GT_max
    if t < GT_len and it < len(GT_sample):
      GT_sample[ it ] = GT[ current_gt ][2]
  
    if t < P_len and it < len(P_sample):
      P_sample [ it ] = P[ current_p ][2]

    # increase counts
    t = t + resolution
    it = it + 1

    # see if out of current boundaries
    if t > float(GT[ current_gt ][1]):
      current_gt = current_gt + 1
      # but don't go over
      if current_gt == len(GT):
        current_gt = current_gt - 1	

    if t > float(P[ current_p ][1]):
      current_p = current_p + 1
      if current_p == len(P):
        current_p = current_p - 1	


  # 2 - precompute the performance of every chord against every other
  # -----------------------------------------------------------------
  all_P = list(set([p[2] for p in P if p[2]]))
  all_GT = list(set([gt[2] for gt in GT]))
  
  # form a dictionary
  scoring_dict = dict()
  for p in all_P:
    for gt in all_GT:
      if p == 'N' or gt == 'N':
        continue
      else:
        score = score_two_chords(p, gt, method=method, augdim_switch=augdim_switch)
        scoring_dict[(p, gt)] = score                     


  # 3 - Score accuracy for every frame
  # ----------------------------------
  accuracy = []
  for igt,(gt,p) in enumerate(zip(GT_sample, P_sample)):

    # first deal with 'N', which has no pitch classes,
    # and only matches itself. Also None, which means
    # that either GT_sample or P_sample is longer than
    # the other
    if p == 'N' and gt == 'N':
      accuracy.append(True)
    elif p == 'N' and gt != 'N':
      accuracy.append(False)
    elif p != 'N' and gt == 'N':
      accuracy.append(False)
    elif p == None or gt == None:
      accuracy.append(False)  
    else:


      # both p and gt are 'real' chords. score them 
      # according to the chosen method
      p = p.strip()
      gt = gt.strip()
      accuracy.append(scoring_dict[(p,gt)])
    
  # 4 - return average
  # ------------------  
  return np.mean(accuracy)

def score_two_chords(p, gt, method='MIREX', augdim_switch=True):

  '''
  Scores two chords against each other

  '''
  # NOTE: 'N' and 'None' are dealt with in
  # evaluate_chords, so  every chord passed to 
  # this function should have a pitch class representation.

  if method == 'MIREX':

    # get pitch classes
    pc_p = reduce_chords([p], 'MIREX')[0]
    pc_gt = reduce_chords([gt], 'MIREX')[0]   
  
    # how many notes are needed for correct overlap?
    min_overlap = 3
    if augdim_switch and ('dim' in gt or 'aug' in gt):
      min_overlap = 2	

    # count overlap
    if len(set(pc_p).intersection(set(pc_gt))) >= min_overlap:
      return 1.0
    else:
      return 0.0

  elif method == 'Correct':
    if p == gt:
      return 1.0
    else:
      return 0.0

  elif method == 'Correct_at_minmaj':
    minmaj_p = reduce_chords([p], 'minmaj')[0]
    minmaj_gt = reduce_chords([gt], 'minmaj')[0]         
    if minmaj_p == minmaj_gt:
      return 1.0
    else:
      return 0.0

  elif method == 'Correct_at_triad':
    triad_p = reduce_chords([p], 'triads')[0]
    triad_gt = reduce_chords([gt], 'triads')[0]
    if triad_p == triad_gt:
      return 1.0
    else:
      return 0.0
  
  elif method == 'Correct_at_seventh':
    seventh_p = reduce_chords([p], 'sevenths')[0]
    seventh_gt = reduce_chords([gt], 'sevenths')[0] 

    if seventh_p == seventh_gt:
      return 1.0
    else:
      return 0.0

  elif method == 'Correct_at_pitchclass':
    # get pitch classes
    pc_p = reduce_chords([p], 'MIREX')[0]
    pc_gt = reduce_chords([gt], 'MIREX')[0]  

    if pc_p == pc_gt:
      return 1.0
    else:
      return 0.0    
    
  else:
    raise NameError('No such scoring method: ' + method)         

def reduce_chords(chords, alphabet):

  import numpy as np

  # Define the maps
  enharmonic_map = {'A':'A', 'Ab':'Ab', 'A#':'Bb',
                    'B':'B', 'Bb':'Bb', 'B#':'C',
                    'C':'C', 'Cb':'B',  'C#':'Db',
                    'D':'D', 'Db':'Db', 'D#':'Eb',
                    'E':'E', 'Eb':'Eb', 'E#':'F',
                    'F':'F', 'Fb':'E',  'F#':'Gb',
                    'G':'G', 'Gb':'Gb', 'G#':'Ab',
                   }

 
  enharmonic_pitch_classes_map = {'A':0, 'Ab':11, 'A#':1,
                                  'B':2, 'Bb':1, 'B#':3,
                                  'C':3, 'Cb':2,  'C#':4,
                                  'D':5, 'Db':4, 'D#':6,
                                  'E':7, 'Eb':6, 'E#':8,
                                  'F':8, 'Fb':7,  'F#':9,
                                  'G':10, 'Gb':9, 'G#':11,
                                 }
  # Alphabets         
  MIREX_map = {'':                 [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
               'maj':              [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
               'min':              [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
               'maj6':             [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
               'min7':             [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
               '7':                [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
               'dim7':             [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
               'maj7':             [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
               'sus4':             [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
               '9':                [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
               'min9':             [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
               'dim':              [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
               'aug':              [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
               'sus2':             [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               '(1':               [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'sus4(2)':          [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
               'maj(11)':          [1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1],
               '/5':               [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
               '/3':               [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
               'min/b7':           [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
               '/9':               [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
               'min/5':            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
               'sus4(2)/2':        [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
               '/7':               [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
               '/b7':              [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
               '/6':               [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
               '/b6':              [1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
               '/2':               [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
               '9/5':              [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
               'maj/9':            [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
               'min7/4':           [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0],
               'maj(9)':           [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
               'maj(#11)':         [1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
               'sus4(b7)':         [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
               'dim/b3':           [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
               'maj6/3':           [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
               'maj6/5':           [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
               'maj6/2':           [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
               'min7(*b3)':        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
               '(1)':              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               '9(11)':            [1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0],
               'min/6':            [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
               'min/b3':           [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
               '7(#9)':            [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0],
               'maj9':             [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
               '9(*3)':            [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
               'min(4)':           [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
               '(5)':              [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               'min/7':            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
               'min/3':            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
               '7(b9)':            [1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
               '7/3':              [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
               'min(6)':           [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
               'min(b6)/5':        [1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
               'min(9)':           [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
               'maj(2)/2':         [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
               '7/b7':             [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
               'aug(9':            [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
               'maj/3':            [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
               'min6':             [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
               '7/b3':             [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0],
               '7/2':              [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
               '7/b2':             [1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
               'hdim7/b7':         [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
               'hdim7':            [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
               'maj9(*7)':         [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
               'sus4/5':           [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
               'min(2)':           [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
               'min7(*5':          [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
               'min(*5)':          [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
               'min(*b3)':         [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               'maj7/5':           [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
               'sus2(b7)':         [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
               'min7/b3':          [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
               'dim/b5':           [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
               'maj(#4)/5':        [1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
               'maj(13)':          [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
               '/4':               [1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
               'maj(*3)':          [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               'min/4':            [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
               'maj6(9)':          [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
               '9(*3':             [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
               'min7/b7':          [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
               'dim7/b3':          [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
               '(b3':              [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
               '/b3':              [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
               '7sus4':            [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
               'min7(4)/5':        [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
               'min7(4)/b7':       [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
               'maj(9)/5':         [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
               'maj(9)/6':         [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
               'maj7/7':           [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
               'maj/2':            [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
               'sus4(9)':          [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
               'maj7(9)':          [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
               'maj(2)':           [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
               'min7(9)':          [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
               'maj(9)/9':         [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
               'maj(b9)':          [1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
               'hdim7/b3':         [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
               'maj7(*5)':         [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
               'min(*b3)/5':       [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               'min(*3)/5':        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               'min7(*5)/b7':      [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
               'min(*5)/b7':       [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],  
               'maj7(*b5)':        [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
               'min/2':            [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
               'maj6/b7':          [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0],
               '(b6)':             [1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
               'maj(4)':           [1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
               '(7)':              [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], 
               '(6)':              [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
               '7/5':              [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
               '/#4':              [1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
               'maj(*1)/#1':       [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # wtf?
               'min(9)/b3':        [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
               'maj(*1)/5':        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
               '(3)':              [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
               'aug/#5':           [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
               'maj/5':            [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
               'min6/b3':          [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
               'min6/5':           [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
               'dim7/b9':          [1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
               'dim7/7':           [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
               'dim7/2':           [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0],
               'dim7/5':           [1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
               'min7(2':           [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
               'minmaj7':          [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
               'sus4/4':           [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
               'maj7/3':           [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
               '(9)':              [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
               'min7/5':           [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
               'min6/6':           [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
               'maj(9)/3':         [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
               'minmaj7/b3':       [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
               'minmaj7/5':        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
               '7(*5':             [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
               '7(13)':            [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0],
               'min7(4)':          [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
               'maj(*5)':          [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
               'aug/3':            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
               'dim/b7':           [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]
             }

  minmaj_map = {'':                'maj',
               'maj':              'maj',
               'min':              'min',
               'maj6':             'maj',
               'min7':             'min',
               '7':                'maj',
               'dim7':             'min',
               'maj7':             'maj',
               'sus4':             'maj',
               '9':                'maj',
               'min9':             'min',
               'dim':              'min',
               'aug':              'maj',
               'sus2':             'maj',
               '(1':               'maj',
               'sus4(2)':          'maj',
               'maj(11)':          'maj',
               '/5':               'maj',
               '/3':               'maj',
               'min/b7':           'min',
               '/9':               'maj',
               'min/5':            'min',
               'sus4(2)/2':        'maj',
               '/7':               'maj',
               '/b7':              'maj',
               '/6':               'maj',
               '/b6':              'maj',
               '/2':               'maj',
               '9/5':              'maj',
               'maj/9':            'maj',
               'min7/4':           'min',
               'maj(9)':           'maj',
               'maj(#11)':         'maj',
               'sus4(b7)':         'maj',
               'dim/b3':           'min',
               'maj6/3':           'maj',
               'maj6/5':           'maj',
               'maj6/2':           'maj',
               'min7(*b3)':        'min',
               '(1)':              'maj',
               '9(11)':            'maj',
               'min/6':            'min',
               'min/b3':           'min',
               '7(#9)':            'maj',
               'maj9':             'maj',
               '9(*3)':            'maj',
               'min(4)':           'min',
               '(5)':              'maj',
               'min/7':            'min',
               'min/3':            'min',
               '7(b9)':            'maj',
               '7/3':              'maj',
               'min(6)':           'min',
               'min(b6)/5':        'min',
               'min(9)':           'min',
               'maj(2)/2':         'maj',
               '7/b7':             'maj',
               'aug(9':            'maj',
               'maj/3':            'maj',
               'min6':             'min',
               '7/b3':             'maj',
               '7/2':              'maj',
               '7/b2':             'maj',
               'hdim7/b7':         'min',
               'hdim7':            'min',
               'maj9(*7)':         'maj',
               'sus4/5':           'maj',
               'min(2)':           'min',
               'min7(*5':          'min',
               'min(*5)':          'min',
               'min(*b3)':         'min',
               'maj7/5':           'maj',
               'sus2(b7)':         'maj',
               'min7/b3':          'min',
               'dim/b5':           'min',
               'maj(#4)/5':        'maj',
               'maj(13)':          'maj',
               '/4':               'maj',
               'maj(*3)':          'maj',
               'min/4':            'min',
               'maj6(9)':          'maj',
               '9(*3':             'maj',
               'min7/b7':          'min',
               'dim7/b3':          'min',
               '(b3':              'min',
               '/b3':              'min',
               '7sus4':            'maj',
               'min7(4)/5':        'min',
               'min7(4)/b7':       'min',
               'maj(9)/5':         'maj',
               'maj(9)/6':         'maj',
               'maj7/7':           'maj',
               'maj/2':            'maj',
               'sus4(9)':          'maj',
               'maj7(9)':          'maj',
               'maj(2)':           'maj',
               'min7(9)':          'min',
               'maj(9)/9':         'maj',
               'maj(b9)':          'maj',
               'hdim7/b3':         'min',
               'maj7(*5)':         'maj',
               'min(*b3)/5':       'min',
               'min(*3)/5':        'min',
               'min7(*5)/b7':      'min',
               'min(*5)/b7':       'min',  
               'maj7(*b5)':        'maj',
               'min/2':            'min',
               'maj6/b7':          'maj',
               '(b6)':             'maj',
               'maj(4)':           'maj',
               '(7)':              'maj', 
               '(6)':              'maj',
               '7/5':              'maj',
               '/#4':              'maj',
               'maj(*1)/#1':       'maj',  
               'min(9)/b3':        'min',
               'maj(*1)/5':        'maj',
               '(3)':              'maj',
               'aug/#5':           'min',
               'maj/5':            'maj',
               'min6/b3':          'min',
               'min6/5':           'min',
               'dim7/b9':          'min',
               'dim7/7':           'min',
               'dim7/2':           'min',
               'dim7/5':           'min',
               'min7(2':           'min',
               'minmaj7':          'min',
               'sus4/4':           'maj',
               'maj7/3':           'maj',
               '(9)':              'maj',
               'min7/5':           'min',
               'min6/6':           'min',
               'maj(9)/3':         'maj',
               'minmaj7/b3':       'min',
               'minmaj7/5':        'min',
               '7(*5':             'maj',
               '7(13)':            'maj',
               'min7(4)':          'min',
               'maj(*5)':          'maj',
               'aug/3':            'maj',
               'dim/b7':           'min'
             }

  triad_map = {'':                 'maj',
               'maj':              'maj',
               'min':              'min',
               'maj6':             'maj',
               'min7':             'min',
               '7':                'maj',
               'dim7':             'dim',
               'maj7':             'maj',
               'sus4':             'sus',
               '9':                'maj',
               'min9':             'min',
               'dim':              'dim',
               'aug':              'aug',
               'sus2':             'sus',
               '(1':               'maj',
               'sus4(2)':          'sus',
               'maj(11)':          'maj',
               '/5':               'maj',
               '/3':               'maj',
               'min/b7':           'min',
               '/9':               'maj',
               'min/5':            'min',
               'sus4(2)/2':        'sus',
               '/7':               'maj',
               '/b7':              'maj',
               '/6':               'maj',
               '/b6':              'maj',
               '/2':               'maj',
               '9/5':              'maj',
               'maj/9':            'maj',
               'min7/4':           'min',
               'maj(9)':           'maj',
               'maj(#11)':         'maj',
               'sus4(b7)':         'sus',
               'dim/b3':           'dim',
               'maj6/3':           'maj',
               'maj6/5':           'maj',
               'maj6/2':           'maj',
               'min7(*b3)':        'min',
               '(1)':              'maj',
               '9(11)':            'maj',
               'min/6':            'min',
               'min/b3':           'min',
               '7(#9)':            'maj',
               'maj9':             'maj',
               '9(*3)':            'maj',
               'min(4)':           'min',
               '(5)':              'maj',
               'min/7':            'min',
               'min/3':            'min',
               '7(b9)':            'maj',
               '7/3':              'maj',
               'min(6)':           'min',
               'min(b6)/5':        'min',
               'min(9)':           'min',
               'maj(2)/2':         'maj',
               '7/b7':             'maj',
               'aug(9':            'aug',
               'maj/3':            'maj',
               'min6':             'min',
               '7/b3':             'maj',
               '7/2':              'maj',
               '7/b2':             'maj',
               'hdim7/b7':         'dim',
               'hdim7':            'dim',
               'maj9(*7)':         'maj',
               'sus4/5':           'sus',
               'min(2)':           'min',
               'min7(*5':          'min',
               'min(*5)':          'min',
               'min(*b3)':         'min',
               'maj7/5':           'maj',
               'sus2(b7)':         'sus',
               'min7/b3':          'min',
               'dim/b5':           'dim',
               'maj(#4)/5':        'maj',
               'maj(13)':          'maj',
               '/4':               'maj',
               'maj(*3)':          'maj',
               'min/4':            'min',
               'maj6(9)':          'maj',
               '9(*3':             'maj',
               'min7/b7':          'min',
               'dim7/b3':          'dim',
               '(b3':              'min',
               '/b3':              'min',
               '7sus4':            'maj',
               'min7(4)/5':        'min',
               'min7(4)/b7':       'min',
               'maj(9)/5':         'maj',
               'maj(9)/6':         'maj',
               'maj7/7':           'maj',
               'maj/2':            'maj',
               'sus4(9)':          'sus',
               'maj7(9)':          'maj',
               'maj(2)':           'maj',
               'min7(9)':          'min',
               'maj(9)/9':         'maj',
               'maj(b9)':          'maj',
               'hdim7/b3':         'dim',
               'maj7(*5)':         'maj',
               'min(*b3)/5':       'min',
               'min(*3)/5':        'min',
               'min7(*5)/b7':      'min',
               'min(*5)/b7':       'min',  
               'maj7(*b5)':        'maj',
               'min/2':            'min',
               'maj6/b7':          'maj',
               '(b6)':             'maj',
               'maj(4)':           'maj',
               '(7)':              'maj', 
               '(6)':              'maj',
               '7/5':              'maj',
               '/#4':              'maj',
               'maj(*1)/#1':       'maj',  
               'min(9)/b3':        'min',
               'maj(*1)/5':        'maj',
               '(3)':              'maj',
               'aug/#5':           'aug',
               'maj/5':            'maj',
               'min6/b3':          'min',
               'min6/5':           'min',
               'dim7/b9':          'dim',
               'dim7/7':           'dim',
               'dim7/2':           'dim',
               'dim7/5':           'dim',
               'min7(2':           'min',
               'minmaj7':          'min',
               'sus4/4':           'sus',
               'maj7/3':           'maj',
               '(9)':              'maj',
               'min7/5':           'min',
               'min6/6':           'min',
               'maj(9)/3':         'maj',
               'minmaj7/b3':       'min',
               'minmaj7/5':        'min',
               '7(*5':             'maj',
               '7(13)':            'maj',
               'min7(4)':          'min',
               'maj(*5)':          'maj',
               'aug/3':            'aug',
               'dim/b7':           'dim'
             }


  seventh_map = {'':               'maj',
               'maj':              'maj',
               'min':              'min',
               'maj6':             'maj',
               'min7':             'min7',
               '7':                '7',
               'dim7':             'dim7',
               'maj7':             'maj7',
               'sus4':             'sus',
               '9':                '7',
               'min9':             'min7',
               'dim':              'dim',
               'aug':              'aug',
               'sus2':             'sus',
               '(1':               'maj',
               'sus4(2)':          'sus',
               'maj(11)':          'maj',
               '/5':               'maj',
               '/3':               'maj',
               'min/b7':           'min7',
               '/9':               '7',
               'min/5':            'min',
               'sus4(2)/2':        'sus',
               '/7':               '7',
               '/b7':              '7',
               '/6':               'maj',
               '/b6':              'maj',
               '/2':               'maj',
               '9/5':              'maj',
               'maj/9':            'maj',
               'min7/4':           'min7',
               'maj(9)':           'maj',
               'maj(#11)':         'maj',
               'sus4(b7)':         'susb7',
               'dim/b3':           'dim',
               'maj6/3':           'maj',
               'maj6/5':           'maj',
               'maj6/2':           'maj',
               'min7(*b3)':        'min7',
               '(1)':              'maj',
               '9(11)':            '7',
               'min/6':            'min',
               'min/b3':           'min',
               '7(#9)':            '7',
               'maj9':             'maj7',
               '9(*3)':            '7',
               'min(4)':           'min',
               '(5)':              'maj',
               'min/7':            'minmaj7',
               'min/3':            'min',
               '7(b9)':            '7',
               '7/3':              '7',
               'min(6)':           'min',
               'min(b6)/5':        'min',
               'min(9)':           'min',
               'maj(2)/2':         'maj',
               '7/b7':             '7',
               'aug(9':            'aug',
               'maj/3':            'maj',
               'min6':             'min',
               '7/b3':             '7',
               '7/2':              '7',
               '7/b2':             '7',
               'hdim7/b7':         'dim7',
               'hdim7':            'dim7',
               'maj9(*7)':         'maj7',
               'sus4/5':           'sus',
               'min(2)':           'min',
               'min7(*5':          'min7',
               'min(*5)':          'min',
               'min(*b3)':         'min',
               'maj7/5':           'maj7',
               'sus2(b7)':         'susb7',
               'min7/b3':          'min7',
               'dim/b5':           'dim',
               'maj(#4)/5':        'maj',
               'maj(13)':          'maj',
               '/4':               'maj',
               'maj(*3)':          'maj',
               'min/4':            'min',
               'maj6(9)':          'maj',
               '9(*3':             '7',
               'min7/b7':          'min7',
               'dim7/b3':          'dim7',
               '(b3':              'min',
               '/b3':              'min',
               '7sus4':            'susb7',
               'min7(4)/5':        'min7',
               'min7(4)/b7':       'min7',
               'maj(9)/5':         'maj',
               'maj(9)/6':         'maj',
               'maj7/7':           'maj7',
               'maj/2':            'maj',
               'sus4(9)':          'sus',
               'maj7(9)':          'maj7',
               'maj(2)':           'maj',
               'min7(9)':          'min7',
               'maj(9)/9':         'maj',
               'maj(b9)':          'maj',
               'hdim7/b3':         'dim7',
               'maj7(*5)':         'maj7',
               'min(*b3)/5':       'min',
               'min(*3)/5':        'min',
               'min7(*5)/b7':      'min7',
               'min(*5)/b7':       'min7',  
               'maj7(*b5)':        'maj7',
               'min/2':            'min',
               'maj6/b7':          '7',
               '(b6)':             'maj',
               'maj(4)':           'maj',
               '(7)':              'maj7', 
               '(6)':              'maj',
               '7/5':              '7',
               '/#4':              'maj',
               'maj(*1)/#1':       'maj',  
               'min(9)/b3':        'min',
               'maj(*1)/5':        'maj',
               '(3)':              'maj',
               'aug/#5':           'aug',
               'maj/5':            'maj',
               'min6/b3':          'min',
               'min6/5':           'min',
               'dim7/b9':          'dim7',
               'dim7/7':           'dim7',
               'dim7/2':           'dim7',
               'dim7/5':           'dim7',
               'min7(2':           'min7',
               'minmaj7':          'minmaj7',
               'sus4/4':           'sus',
               'maj7/3':           'maj7',
               '(9)':              'maj',
               'min7/5':           'min7',
               'min6/6':           'min',
               'maj(9)/3':         'maj',
               'minmaj7/b3':       'minmaj7',
               'minmaj7/5':        'minmaj7',
               '7(*5':             '7',
               '7(13)':            '7',
               'min7(4)':          'min7',
               'maj(*5)':          'maj',
               'aug/3':            'aug',
               'dim/b7':           'dim7'
             }

  # Collect them           
  alphabet_map = {'minmaj': minmaj_map, 'MIREX': MIREX_map, 'triads': triad_map, 'sevenths': seventh_map}

  new_chords = []
  for chord in chords:

    if chord == 'N':

      new_chords.append((start_time, end_time, chord))

    else:
    
      # get rootnote and chord type
      root, chordtype, bass = chord_2_rootchordbass(chord)

      # map the root
      enharmonic_root = enharmonic_map[root]
      
      # map the chord_type according to the alphabet
      reduced_chord = alphabet_map[alphabet][chordtype + bass]
      #reduced_chord = alphabet_map[alphabet][chordtype] 

      # append
      if alphabet == 'MIREX':
        # if doing MIREX evaluation, the 'chord' is a collection of pitch classes

        # Pitch class number
        root_pitchclass = enharmonic_pitch_classes_map[root]

        # collect pitch classes, add root offset, and take mod 12
        pitches = [np.mod(ip + root_pitchclass, 12) for ip,p in enumerate(reduced_chord) if p == 1]
        
        new_chords.append(pitches)

      else:

        # the reduced chord is a string
        new_chords.append(enharmonic_root + ':' + reduced_chord)

  return new_chords  

def chord_2_rootchordbass(chord):

  # split a chord into (rootnote, chordtype, bass)
  if chord == 'N':
    root = ''
    chordtype = 'N'
    bass = ''
  else:
    if '/' in chord:
      preslash, bass_preslash = chord.split('/')	
      bass = '/' + bass_preslash
    else:
      bass = ''
      preslash = chord

    if ':' in preslash:
      root, chordtype = preslash.split(':')
    else:
      root = preslash
      chordtype = ''

  return root, chordtype, bass          	
