import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pprint
import string

from operator import attrgetter
from params import *

np.seterr(all = 'raise')

pp = pprint.PrettyPrinter(indent = 2)
# Draw ROC curves
def roc(res_path):
  f = open(res_path, 'r')
  results = pickle.load(f)
  f.close()

  paramsets = results[0]
  statsets = results[1]

  """
  # Sort just to make sure (TODO: untested) TODO: This will come in handy when
  # getting data from multiple files.
  sorted_indices = [ i[0] for i in sorted(enumerate(paramsets),
                                          key=lambda x:x[1]) ]
  paramsets = sorted(paramsets)
  stats = [ stats[sorted_indices[i]] for i in range(len(stats)) ]
  """
  save_fig = True
  plot = True
  pnt = False
  if plot:
    plt.close('all')
    plt.ion()
    plt.figure()
    plt.show()

  if save_fig:
    if not os.path.exists('fig'):
      os.mkdir('fig')  

  # TODO: Get this from params somehow... though the relevant subset still has
  # to be specified.
  all_attrs = ['gamma', 'cmpr_window', 'threshold', 'w_smooth',
               'detection_window_hrs', 'req_consec_detections']

  # Assuming parameter combinations are sorted, go through them sequentially and
  # note the index of the parameter that changes. When that index changes, we
  # make a new plot that show the variation of the variable at the new index.
  for var_attr in all_attrs:
    if save_fig:
      if not os.path.exists(os.path.join('fig', var_attr)):
        os.mkdir(os.path.join('fig', var_attr))
    if plot:
      plt.hold(False)

    if pnt:
      print 'Varying', var_attr
    const_attrs = all_attrs[:]
    const_attrs.remove(var_attr)
    enum_sorted = sorted(enumerate(paramsets),
      key = lambda x: [ x[1]._asdict()[attr] for attr in const_attrs ])
    paramsets_sorted = [ elt[1] for elt in enum_sorted ]
    indices_sorted = [ elt[0] for elt in enum_sorted ]
    statsets_sorted = [ statsets[indices_sorted[i]] for i in range(len(statsets)) ]

    var_attr_count = 0
    var_attr_values = []
    anything_plotted = False

    for psi in xrange(len(paramsets_sorted)):
      # Take the difference between the numerical values.
      curr_params = paramsets_sorted[psi]

      const_attr_str = string.join(
        [ attr + '=' + str(curr_params._asdict()[attr]) 
          for attr in const_attrs ],
        ',')

      if psi > 0:
        prev_params = paramsets_sorted[psi - 1]
        if pnt:
          print psi
          print prev_params
        # print 'curr', curr_params
        delta_params = [ curr_params[i] - prev_params[i]
                         for i in range(2,len(prev_params))
                         if type(curr_params[i]) is type(0) or \
                           type(curr_params[i]) is type(0.0) ]
        # print 'delta', delta_params
        mod_indices = np.where(np.array(delta_params) != 0)
        if len(mod_indices[0]) > 1:
          # We're starting a new experiment.    
          if pnt:
            print 'New experiment!'

          if plot:
            plt.hold(False)
            if save_fig:
              if anything_plotted:
                plt.savefig(os.path.join('fig', var_attr,
                                         const_attr_str) + '.png')
            else:
              plt.draw()
              raw_input()

          var_attr_count = 0
          var_attr_values = []
          anything_plotted = False
    
      var_attr_values.append(curr_params._asdict()[var_attr])
      if plot:
        fprs = [ stats['fpr']
                 for stats in statsets_sorted[psi]
                 if stats ]
        tprs = [ stats['tpr']
                 for stats in statsets_sorted[psi]
                 if stats ]
        if pnt:
          print fprs, tprs
        if fprs and tprs:
          anything_plotted = True
          mfprs = np.mean(fprs)
          mtprs = np.mean(tprs)
          sfprs = np.std(fprs)
          stprs = np.std(tprs)
          plt.scatter(mfprs, mtprs, s = 20 * (var_attr_count + 0.5))
          plt.hold(True)
          plt.errorbar(mfprs, mtprs, xerr = sfprs, yerr = stprs)
        else:
          # Make an empty plot, so we know there was no data.
          plt.scatter([],[])
        plt.xlim([-0.1,1.1])
        plt.ylim([-0.1,1.1])
        plt.hold(True)
        plt.title(const_attr_str + '\n' + var_attr + '=' + str(var_attr_values),
                  fontsize = 11)

      var_attr_count += 1

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
  
