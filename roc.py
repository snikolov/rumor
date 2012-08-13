import itertools
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

def early(res_path):
  f = open(res_path, 'r')
  results = pickle.load(f)
  f.close()

  paramsets = results[0]
  statsets = results[1]

  pass

# Draw ROC curves
def roc(res_paths):
  paramsets = []
  statsets = []
  for res_path in res_paths:
    f = open(res_path, 'r')
    results = pickle.load(f)
    f.close()
    for paramset in results[0]:
      paramsets.append(paramset)
    for statset in results[1]:
      statsets.append(statset)

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
      plt.subplot(121)
      plt.hold(False)
      plt.subplot(122)
      plt.hold(False)
      print 'Press any key'
      raw_input()
    if pnt:
      print 'Varying', var_attr

    delta_fprs = []
    delta_tprs = []

    const_attrs = all_attrs[:]
    const_attrs.remove(var_attr)
    # Sort by variable parameter.
    enum_sorted = sorted(enumerate(paramsets),
      key = lambda x: x[1]._asdict()[var_attr])
    # Sort by constant parameters.
    enum_sorted = sorted(enum_sorted,
      key = lambda x: [ x[1]._asdict()[attr] for attr in const_attrs ])
    paramsets_sorted = [ elt[1] for elt in enum_sorted ]
    indices_sorted = [ elt[0] for elt in enum_sorted ]
    statsets_sorted = [ statsets[indices_sorted[i]] for i in range(len(statsets)) ]

    # Initialize variables for each ROC curve. These will be reset when a new
    # curve is ready to be drawn.
    var_attr_count = 0
    var_attr_values = []
    mean_fprs = []
    mean_tprs = []
    std_fprs = []
    std_tprs = []
    all_fprs = []
    all_tprs = []
    
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
        if pnt:
          print 'delta', delta_params
        mod_indices = np.where(np.array(delta_params) != 0)
        if len(mod_indices[0]) > 1:
          # We're starting a new experiment.    
          if pnt:
            print 'New experiment!'

          # Record values for plotting 2d histograms of ROC curve deltas for
          # var_attr.

          delta_fprs.extend([ (mean_fprs[i] - mean_fprs[0]) / \
                                (var_attr_values[i] - var_attr_values[0])
                              for i in range(1, len(mean_fprs)) ])
          delta_tprs.extend([ (mean_tprs[i] - mean_tprs[0]) / \
                                (var_attr_values[i] - var_attr_values[0])
                              for i in range(1, len(mean_tprs)) ])

          # Alternate method for computing delta_fprs and delta_tprs. Compute
          # across all combinations of ROC for different trials.
          """
          # zip on so much data is expensive!
          all_tprs_prod = itertools.product(*all_tprs)
          all_fprs_prod = itertools.product(*all_fprs)
          for fprs_combo, tprs_combo in zip(all_fprs_prod, all_tprs_prod):
            delta_fprs.extend([ fprs_combo[i] - fprs_combo[0]
                                for i in range(1, len(fprs_combo)) ])
            delta_tprs.extend([ tprs_combo[i] - tprs_combo[0]
                                for i in range(1, len(tprs_combo)) ])
          """
          # Slap (0,0) and (1,1) onto mean_fprs, mean_tprs to complete the
          # ROC curve. Also put corresponding 0 stdevs in std_fprs, std_tprs.
          mean_fprs.extend([0,1])
          mean_tprs.extend([0,1])
          std_fprs.extend([0,0])
          std_tprs.extend([0,0])

          point_sizes = [ s * 8 for s in np.array(range(0, len(mean_fprs))) + 0.1 ]
          point_sizes.extend([0.1,0.1])

          # Sort from left to right.
          mean_fprs_ltor_enum = sorted(enumerate(mean_fprs),
                                       key = lambda x:x[1])
          mean_fprs_ltor = [ mean_fprs[i] 
                             for (i,v) in mean_fprs_ltor_enum ]
          mean_tprs_ltor = [ mean_tprs[i]
                             for (i,v) in mean_fprs_ltor_enum ]
          std_fprs_ltor = [ std_fprs[i] 
                            for (i,v) in mean_fprs_ltor_enum ]
          std_tprs_ltor = [ std_tprs[i]
                            for (i,v) in mean_fprs_ltor_enum ]

          point_sizes_ltor = [ point_sizes[i]
                               for (i,v) in mean_fprs_ltor_enum ]

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
                # Don't take the first and last if we've put a dummy 0 and 1 at
                # each end of the means lists.
                plt.subplot(121)
                plt.errorbar(mean_fprs_ltor, mean_tprs_ltor,
                             xerr = std_fprs_ltor, yerr = std_tprs_ltor,
                             color = 'k', linestyle = 'None',
                             ms = 1, marker = 'o',
                             elinewidth = 0.5)
                plt.hold(True)
                plt.scatter(mean_fprs_ltor, mean_tprs_ltor,
                            s = point_sizes_ltor, c = 'k')
                plt.title(const_attr_str + '\n' + var_attr + '=' + \
                            str(var_attr_values),
                          fontsize = 11)
                plt.xlim([-0.1,1.1])
                plt.ylim([-0.1,1.1])

              # +-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
              # | PLOT LINES  
              # +-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
              plot_curves = False
              if plot_curves:
                num_unique = len(set([ (mean_fprs_ltor[i], mean_tprs_ltor[i]) 
                                       for i in range(len(mean_fprs_ltor)) ]))
                if num_unique > 2:
                  # Plot bezier curves.
                  plot_poly = True
                  if plot_poly:
                    verts = [ (mean_fprs_ltor[i], mean_tprs_ltor[i])
                              for i in range(len(mean_fprs_ltor)) ]
                    connection = Path.LINETO
                    codes = [ connection ] * (len(verts) - 1)
                    verts.insert(0, (1,1))
                    verts.insert(1, (1,0))
                    codes.insert(0, Path.MOVETO)
                    codes.insert(1, Path.LINETO)
                    codes.insert(2, Path.LINETO)
                    path = Path(verts, codes)

                    plt.subplot(1,2,2)
                    ax = plt.gca()
                    patch = patches.PathPatch(path, facecolor='k', lw=5, alpha = 1)
                    # Manually clear axes. This isn't a plotting command, so hold
                    # = False has no effect.
                    if not plt.ishold():
                      plt.cla()
                    ax.add_patch(patch)

                  plot_lines = False
                  if plot_lines:
                    # Plot lines.
                    plt.plot(mean_fprs_ltor, mean_tprs_ltor, color = 'k',
                             linewidth = 1)

                  plt.xlim([-0.1,1.1])
                  plt.ylim([-0.1,1.1])
                  plt.hold(True)
                  #raw_input()
              
          # Reset variables for next ROC curve.
          mean_fprs = []
          mean_tprs = []
          std_fprs = []
          std_tprs = []
          all_fprs = []
          all_tprs = []
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
          mean_fprs.append(mfprs)
          mean_tprs.append(mtprs)
          std_fprs.append(sfprs)
          std_tprs.append(stprs)

          # Record full list of fprs and tprs
          all_fprs.append(fprs)
          all_tprs.append(tprs)

      var_attr_count += 1

    # Plot deltas in fpr and tpr as 2d histogram.
    if plot:
      plot_delta_dist = True
      if plot_delta_dist and delta_fprs and delta_tprs:
        print delta_fprs, '\n'
        print delta_tprs
        print var_attr
        plt.figure()
        heatmap, xedges, yedges = np.histogram2d(delta_fprs, delta_tprs, bins=80)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        plt.subplot(121)
        n, bins, hpatches = plt.hist(delta_fprs, bins = 35, normed = True,
                                    histtype = 'stepfilled', color = 'k')
        plt.setp(hpatches, 'facecolor', 'k')
        plt.title('delta fpr')

        plt.subplot(122)
        n, bins, hpatches = plt.hist(delta_tprs, bins = 35, normed = True,
                                    histtype = 'stepfilled', color = 'k')
        plt.setp(hpatches, 'facecolor', 'k')
        plt.title('delta tpr')

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
  
