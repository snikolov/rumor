import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pprint
import string
import matplotlib.patches as patches

from math import exp
from matplotlib.path import Path
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
  save_fig = False
  plot = True
  pnt = False
  if plot:
    plt.close('all')
    plt.ion()
    fig = plt.figure()
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

    # Initialize variables for each ROC curve. These will be reset when a new
    # curve is ready to be drawn.
    var_attr_count = 0
    var_attr_values = []
    mean_fprs = [0,1]
    mean_tprs = [0,1]
    std_fprs = []
    std_tprs = []

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
            if save_fig:
              if len(mean_fprs) > 2:
                # There were Points other than the manually added (0,0) and (1,1).
                plt.savefig(os.path.join('fig', var_attr,
                                         const_attr_str) + '.png')
            else:
              # +-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
              # | PLOT SCATTER  
              # +-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
              plot_scatter = False
              if plot_scatter and len(mean_fprs) > 2:
                plt.scatter(mean_fprs, mean_tprs,
                            s = 20 * (var_attr_count + 0.5), c = 'k')
                plt.hold(True)
                # Don't take the first and last if we've put a dummy 0 and 1 at
                # each end of the means lists.
                plt.errorbar(mean_fprs[1:-1], mean_tprs[1:-1], xerr = std_fprs,
                             yerr = std_tprs, color = 'k', linestyle = 'None')
                plt.title(const_attr_str + '\n' + var_attr + '=' + \
                            str(var_attr_values),
                          fontsize = 11)
                plt.xlim([-0.1,1.1])
                plt.ylim([-0.1,1.1])
                plt.draw()
                #raw_input()

              #plt.hold(True)
              # Sort points from left to right.

              # +-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
              # | PLOT LINES  
              # +-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
              plot_curves = True
              if plot_curves:
                mean_fprs_ltor_enum = sorted(enumerate(mean_fprs),
                                             key = lambda x:x[1])
                mean_fprs_ltor = [ mean_fprs[i] 
                                   for (i,v) in mean_fprs_ltor_enum ]
                mean_tprs_ltor = [ mean_tprs[i]
                                   for (i,v) in mean_fprs_ltor_enum ]
                num_unique = len(set([ (mean_fprs_ltor[i], mean_tprs_ltor[i]) 
                                       for i in range(len(mean_fprs_ltor)) ]))
                if num_unique > 3:
                  # Plot bezier curves.
                  plot_bezier = False
                  if plot_bezier:
                    verts = [ (mean_fprs_ltor[i], mean_tprs_ltor[i])
                              for i in range(len(mean_fprs_ltor)) ]              
                    codes = [ Path.CURVE4 ] * (len(verts) - 1)
                    codes.insert(0, Path.MOVETO)
                    path = Path(verts, codes)
                    ax = plt.gca()
                    patch = patches.PathPatch(path, facecolor='none', lw=2)
                    # Manually clear axes. This isn't a plotting command, so hold
                    # = False has no effect.
                    if not plt.ishold():
                      plt.cla()
                    ax.add_patch(patch)
                  plot_lines = True
                  if plot_lines:
                    # Plot lines.
                    plt.plot(mean_fprs_ltor, mean_tprs_ltor, color = 'k',
                             linewidth = 1)
                    plt.draw()
                    plt.draw()
                  plt.xlim([-0.1,1.1])
                  plt.ylim([-0.1,1.1])
                  plt.hold(True)
              
          # Reset variables for next ROC curve.
          mean_fprs = [1,0]
          mean_tprs = [1,0]
          std_fprs = []
          std_tprs = []
          var_attr_count = 0
          var_attr_values = []
    
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
          mfprs = np.mean(fprs)
          mtprs = np.mean(tprs)
          sfprs = np.std(fprs)
          stprs = np.std(tprs)
          
          # Record these so we plot them before moving on to the next ROC curve.
          mean_fprs.insert(1, mfprs)
          mean_tprs.insert(1, mtprs)
          std_fprs.append(sfprs)
          std_tprs.append(stprs)

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
  
