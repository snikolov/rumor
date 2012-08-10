import matplotlib.pyplot as plt
import numpy as np
import pickle

from params import *

# Draw ROC curves
def roc(res_path):
  f = open(res_path, 'r')
  results = pickle.load(f)
  f.close()

  paramsets = results[0]
  statsets = results[1]
  
  """
  # Sort just to make sure (TODO: untested)
  sorted_indices = [ i[0] for i in sorted(enumerate(paramsets),
                                          key=lambda x:x[1]) ]
  paramsets = sorted(paramsets)
  stats = [ stats[sorted_indices[i]] for i in range(len(stats)) ]
  """

  plt.ion()
  plt.figure()
  plt.show()
  # Assuming parameter combinations are sorted, go through them sequentially and
  # note the index of the parameter that changes. When that index changes, we
  # make a new plot that show the variation of the variable at the new index.
  for psi in xrange(len(paramsets)):
    curr_params = paramsets[psi]
    # Take the difference between the numerical values.
    if psi > 0:
      prev_params = paramsets[psi - 1]
      print 'prev', prev_params
      print 'curr', curr_params
      delta_params = [ curr_params[i] - prev_params[i]
                       for i in range(2,len(prev_params))
                       if type(curr_params[i]) == type(0) and \
                         type(prev_params[i]) == type(0) ]
      print 'delta', delta_params
      mod_indices = np.where(np.array(delta_params) != 0)
      if len(mod_indices[0]) > 1:
        # We're starting a new experiment.
        print 'New experiment!'
        plt.hold(False)
    for stats in statsets[psi]:
      plt.scatter(stats['fpr'], stats['tpr'])
      plt.xlim([0,1])
      plt.ylim([0,1])
      plt.hold(True)
    raw_input()
      
    
  """
  Parameters of interest.
  0 - threshold
  1 - cmpr_window
  2 - w_smooth
  3 - gamma
  4 - detection_window_hrs
  5 - req_consec_detections
  """
  
  # Mean early (in hours)
  
