import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pprint
import string
import matplotlib.patches as patches
import rumor

from math import exp
from matplotlib import rc
from matplotlib.path import Path
from operator import attrgetter
from params import *

rc('text', usetex = False)
np.seterr(all = 'raise')
pp = pprint.PrettyPrinter(indent = 2)

offset_filter_hrs = 3

fpr_partition_threshold = 0.25
tpr_partition_threshold = 0.75

attr_math_name = {'gamma': '$\gamma$',
                  'threshold': r'$\theta$',
                  'cmpr_window': '$N_{obs}$',
                  'detection_window_hrs': '$h$',
                  'req_consec_detections': '$D_{req}$',
                  'w_smooth': '$N_{smooth}$'}

def fix_detection_times(paramsets, statsets):
  for si in xrange(len(statsets)):
    cmpr_window = paramsets[si].cmpr_window
    offset = cmpr_window * 100000

    # Offset due to comparison window
    for stats in statsets[si]:
      if not stats:
        continue
      earlies = stats['earlies']
      lates = stats['lates']
      earlies_offset = [ early - offset for early in earlies ]
      lates_offset = [ late + offset for late in lates ]
      correct_earlies = [ early for early in earlies_offset
                          if early > 0 ]
      correct_lates = lates_offset
      correct_lates.extend([ -early for early in earlies_offset 
                             if early <= 0 ])
      stats['earlies'] = correct_earlies
      stats['lates'] = correct_lates
      """
      print 'old earlies', sorted(earlies)
      print 'new earlies', sorted(correct_earlies)
      raw_input()
      print 'old lates', sorted(lates)
      print 'new lates', sorted(correct_lates)
      raw_input()
      """
  """
  # Offset due to other onsets
  offsets = rumor.parsing.parse_onset_offsets('data/offsets.tsv')
  for si in xrange(len(statsets)):
    for stats in statsets[si]:
      if not stats:
        continue
      earlies = stats['earlies']
      lates = stats['lates']

      offset = np.mean([ v for v in offsets.values() if v <= offset_filter_hrs ]) * 3600000
      print offset / 3600000.0
      print np.array(sorted(earlies)) / 3600000.0
      print np.array(sorted(lates)) / 3600000.0
      offset_earlies = np.array(earlies) - offset
      offset_lates = np.array(lates) + offset
      true_earlies = []
      true_lates = []
      for e in offset_earlies:
        if e < 0:
          true_lates.append(-e)
        else:
          true_earlies.append(e)
      for e in offset_lates:
        if e < 0:
          true_earlies.append(-e)
        else:
          true_lates.append(e)
          
      true_earlies = [ t for t in true_earlies if t < offset_filter_hrs ] 
      stats['earlies'] = true_earlies
      stats['lates'] = true_lates
      """

def point_to_category(fpr, tpr, fpr_partition_threshold = 0.5,
                      tpr_partition_threshold = 0.5):
  if fpr > fpr_partition_threshold and tpr > tpr_partition_threshold:
    return 'top'
  elif fpr <= fpr_partition_threshold and tpr <= tpr_partition_threshold:
    return 'bottom'
  else:
    return 'center'

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

  fix_detection_times(paramsets, statsets)

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
    fig = plt.figure(figsize = (10,4.5))
    plt.show()

  if save_fig:
    if not os.path.exists('fig'):
      os.mkdir('fig')  

  # TODO: Get this from params somehow... though the relevant subset still has
  # to be specified.
  all_attrs = ['gamma', 'cmpr_window', 'threshold', 'w_smooth',
               'detection_window_hrs', 'req_consec_detections']
  const_attrs_allowed_values = { 'gamma': [0.1, 1, 10],
                                 'cmpr_window': [10, 80, 115, 150],
                                 #'cmpr_window': [80],
                                 #'threshold': [1],
                                 'threshold': [0.65, 1, 3],
                                 'w_smooth': [10, 80, 115, 150],
                                 'detection_window_hrs': [3, 5, 7, 9],
                                 'req_consec_detections': [1, 3, 5] }

  # Assuming parameter combinations are sorted, go through them sequentially and
  # note the index of the parameter that changes. When that index changes, we
  # make a new plot that show the variation of the variable at the new index.
  for var_attr in all_attrs:
    if save_fig:
      if not os.path.exists(os.path.join('fig', var_attr)):
        os.mkdir(os.path.join('fig', var_attr))
    if plot:
      print 'Press any key'
      raw_input()
      plt.figure(figsize = (10,4.5))
      
      # plt.suptitle('Varying ' + attr_math_name[var_attr], size = 15)
      plt.subplot(121)
      plt.hold(False)
      plt.subplot(122)
      plt.hold(False)
    if pnt:
      print 'Varying', var_attr

    delta_fprs = []
    delta_tprs = []

    # Mapping from paramset to percentage of tpr and fpr deltas above 0.
    delta_rank = {}
    # Mapping from paramset to a measure of how far up or down the ROC curve we are.
    updown_rank = {}
    # Partition deltas by where the ROC curve starts on the FPR/TPR plane.
    deltas_partitioned = {}
    deltas_partitioned['top'] = {}
    deltas_partitioned['center'] = {}
    deltas_partitioned['bottom'] = {}
    # Partition fprs and tprs by where they are in the FPR/TPR plane
    rates_partitioned = {}
    rates_partitioned['top'] = {}
    rates_partitioned['center'] = {}
    rates_partitioned['bottom'] = {}
    # Record the earliness relative to true onset
    earliness = {}
    earliness['top'] = {}
    earliness['top']['fprs'] = []
    earliness['top']['tprs'] = []
    earliness['top']['earlies'] = []
    earliness['top']['lates'] = []
    earliness['top']['params'] = []
    earliness['center'] = {}
    earliness['center']['fprs'] = []
    earliness['center']['tprs'] = []
    earliness['center']['earlies'] = []
    earliness['center']['lates'] = []
    earliness['center']['params'] = []
    earliness['bottom'] = {}
    earliness['bottom']['fprs'] = []
    earliness['bottom']['tprs'] = []
    earliness['bottom']['earlies'] = []
    earliness['bottom']['lates'] = []
    earliness['bottom']['params'] = []

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
    all_earlies = []
    all_lates = []
    
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
          #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
          # NEW ROC CURVE! PLOT RELEVANT THINGS FOR OLD ROC CURVE AND RESET
          # VARIABLE IN PREPARATION FOR NEW ROC CURVE.
          #=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

          # Record values for plotting 2d histograms of ROC curve deltas for
          # var_attr.
          """
          delta_fprs.extend([ (mean_fprs[i] - mean_fprs[i-1]) / \
                                (var_attr_values[i] - var_attr_values[i-1])
                              for i in range(1, len(mean_fprs)) ])
          delta_tprs.extend([ (mean_tprs[i] - mean_tprs[i-1]) / \
                                (var_attr_values[i] - var_attr_values[i-1])
                              for i in range(1, len(mean_tprs)) ])
          """
          # Alternate method for computing delta_fprs and delta_tprs. Compute
          # across all combinations of ROC for different trials.

          # zip on so much data is expensive!
          all_tprs_prod = list(itertools.product(*all_tprs))
          all_fprs_prod = list(itertools.product(*all_fprs))
          # Separate storage of deltas for prev params only (corresponding to
          # the ROC curve we just finished looking at, and are about to
          # analyze..
          delta_fprs_prev_params = []
          delta_tprs_prev_params = []

          for combo_i in xrange(len(all_fprs_prod)):
            fprs_combo = all_fprs_prod[combo_i]
            tprs_combo = all_tprs_prod[combo_i]

            # Don't include deltas that are "stuck" in the 0,0 or 1,1 corner.
            """
            fprs_combo_not_stuck = [ fprs_combo[j]
                                     for j in range(len(fprs_combo))
                                     if fprs_combo[j] != 0 and \
                                       fprs_combo[j] != 1 and \
                                       tprs_combo[j] != 0 and \
                                       tprs_combo[j] != 1 ]
            tprs_combo_not_stuck = [ tprs_combo[j]
                                     for j in range(len(tprs_combo))
                                     if fprs_combo[j] != 0 and \
                                       fprs_combo[j] != 1 and \
                                       tprs_combo[j] != 0 and \
                                       tprs_combo[j] != 1 ]
            """
            fprs_combo_not_stuck = [ fprs_combo[j]
                                     for j in range(1, len(fprs_combo))
                                     if fprs_combo[j] != fprs_combo[j-1] and \
                                       tprs_combo[j] != tprs_combo[j-1] ]
            if fprs_combo:
              fprs_combo_not_stuck.insert(0, fprs_combo[0])
            tprs_combo_not_stuck = [ tprs_combo[j]
                                     for j in range(1, len(tprs_combo))
                                     if fprs_combo[j] != fprs_combo[j-1] and \
                                       tprs_combo[j] != tprs_combo[j-1] ]
            if tprs_combo:
              tprs_combo_not_stuck.insert(0, tprs_combo[0])

            new_delta_fprs = [ (fprs_combo[i] - fprs_combo[i-1]) / \
                                 (var_attr_values[i] - var_attr_values[i-1])
                               for i in range(1, len(fprs_combo_not_stuck)) ]
            new_delta_tprs = [ (tprs_combo[i] - tprs_combo[i-1]) / \
                                 (var_attr_values[i] - var_attr_values[i-1])
                               for i in range(1, len(tprs_combo_not_stuck)) ]
            delta_fprs.extend(new_delta_fprs)
            delta_tprs.extend(new_delta_tprs)
            delta_fprs_prev_params.extend(new_delta_fprs)
            delta_tprs_prev_params.extend(new_delta_tprs)
                
            # Categorize ROC curves by whether they start in the top center or
            # bottom and record the corersponding delta fprs and tprs.
            category = None
            if fprs_combo and tprs_combo:
              category = point_to_category(fprs_combo[0], tprs_combo[0],
                fpr_partition_threshold = fpr_partition_threshold,
                tpr_partition_threshold = tpr_partition_threshold)
              deltas_partitioned[category][prev_params] = \
                  (new_delta_fprs, new_delta_tprs)

            for pidx in xrange(len(fprs_combo)):
              category = point_to_category(fprs_combo[pidx],
                tprs_combo[pidx], fpr_partition_threshold = fpr_partition_threshold,
                tpr_partition_threshold = tpr_partition_threshold)
              # This should really be done at the individual point level, not at
              # the ROC level...
              actual_params_dict = prev_params._asdict()
              actual_params_dict[var_attr] = var_attr_values[pidx]
              actual_params = Params(**actual_params_dict)
              rates_partitioned[category][actual_params] = \
                  (fprs_combo[pidx], tprs_combo[pidx])

          # Record a measure of change in tpr and fpr for the purpose of
          # ranking which parameters cause the most positive and negative
          # deltas.
          """
          f_num_geq_zero = len([ ndfpr for ndfpr in delta_fprs_prev_params
                                 if ndfpr >= 0 ])
          t_num_geq_zero = len([ ndtpr for ndtpr in delta_tprs_prev_params
                                 if ndtpr >= 0 ])
          """
          if len(delta_fprs_prev_params) > 0 and \
                len(delta_tprs_prev_params) > 0:
            """
            f_frac_geq_zero = \
              float(f_num_geq_zero) / len(delta_fprs_prev_params)
            t_frac_geq_zero = \
              float(t_num_geq_zero) / len(delta_tprs_prev_params)
            """
            # Store copies of all_fprs and all_tprs as well so we can plot ROC
            # curves at the end in order of rank.
            delta_rank[prev_params] = (np.mean(delta_fprs_prev_params),
                                       np.mean(delta_tprs_prev_params),
                                       all_fprs[:], all_tprs[:],
                                       delta_fprs_prev_params[:],
                                       delta_tprs_prev_params[:])
            
          # Partition ROC points into top, bottom, center. For each one, record
          # the FPR,TPR coordinates and the earlies and lates.
          if all_fprs and all_tprs and all_earlies and all_lates:
            for point_trials_idx in xrange(len(all_fprs)):
              fprs_point_trials = all_fprs[point_trials_idx]
              tprs_point_trials = all_tprs[point_trials_idx]
              earlies_point_trials = all_earlies[point_trials_idx]
              lates_point_trials = all_lates[point_trials_idx]
              for trial_idx in xrange(len(fprs_point_trials)):
                fpr_trial = fprs_point_trials[trial_idx]
                tpr_trial = tprs_point_trials[trial_idx]
                earlies_trial = earlies_point_trials[trial_idx]
                lates_trial = lates_point_trials[trial_idx]

                category = point_to_category(fpr_trial, tpr_trial,
                  fpr_partition_threshold = fpr_partition_threshold,
                  tpr_partition_threshold = tpr_partition_threshold)

                earliness[category]['fprs'].append(fpr_trial)
                earliness[category]['tprs'].append(tpr_trial)
                earliness[category]['earlies'].append(earlies_trial)
                earliness[category]['lates'].append(lates_trial)
                earliness[category]['params'].append(prev_params)

          # Record measures of 'position' for this ROC curve to use for ranking
          # ROC curves by how far 'up' or 'down' they are. %TODO: awkward
          # phrasing.
          # Store copies of all_fprs and all_tprs as well so we can plot ROC
          # curves at the end in order of rank.
          if all_fprs and all_tprs:
            max_mean_fpr = max([ np.mean(all_fprs_i)
                                 for all_fprs_i in all_fprs ])
            max_mean_tpr = max([ np.mean(all_tprs_i)
                                 for all_tprs_i in all_tprs ])
            min_mean_fpr = min([ np.mean(all_fprs_i)
                                 for all_fprs_i in all_fprs ])
            min_mean_tpr = min([ np.mean(all_tprs_i)
                                 for all_tprs_i in all_tprs ])
            updown_rank[prev_params] = (min_mean_fpr, min_mean_tpr,
                                        max_mean_fpr, max_mean_tpr,
                                        all_fprs[:], all_tprs[:])

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
              if len(mean_fprs) > 3:
                # There were Points other than the manually added (0,0) and (1,1).
                plt.savefig(os.path.join('fig', var_attr,
                                         const_attr_str) + '.png')
            else:
              # +-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
              # | PLOT SCATTER  
              # +-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
              plot_scatter = False
              if plot_scatter and len(mean_fprs) > 3:
                # Don't take the first and last if we've put a dummy 0 and 1 at
                # each end of the means lists.
                plt.subplot(121)
                """
                plt.errorbar(mean_fprs_ltor, mean_tprs_ltor,
                             xerr = std_fprs_ltor, yerr = std_tprs_ltor,
                             color = 'k', linestyle = '-', linewidth = 0.5,
                             marker = 'o', elinewidth = 0.25)
                """
                plt.plot(mean_fprs_ltor[1:-1], mean_tprs_ltor[1:-1], color = 'k', lw = 0.5)
                plt.hold(True)
                # For increasing point sizes.
                # Turn on scatter for markers at points
                """
                plt.scatter(mean_fprs_ltor, mean_tprs_ltor,
                            s = point_sizes_ltor, c = 'b')
                """
                """
                plt.scatter(mean_fprs_ltor, mean_tprs_ltor,
                            s = 15, c = 'k')
                """
                plt.title('All ROC Curves')
                plt.xlabel('$FPR$')
                plt.ylabel('$TPR$')
                plt.grid(True)
                """
                plt.title(const_attr_str + '\n' + var_attr + '=' + \
                            str(var_attr_values),
                          fontsize = 11)
                """
                plt.xlim([-0.1,1.1])
                plt.ylim([-0.1,1.1])
                # Enable for sequential viewing
                #plt.hold(False)
                #raw_input()
                #plt.draw()

              # +-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
              # | PLOT LINES  
              # +-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
              plot_curves = False
              if plot_curves:
                num_unique = len(set([ (mean_fprs_ltor[i], mean_tprs_ltor[i]) 
                                       for i in range(len(mean_fprs_ltor)) ]))
                if num_unique > 3:
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
                  plt.title('ROC Curve Envelope', size = 16)
                  plt.xlabel('$FPR$', size = 16)
                  plt.ylabel('$TPR$', size = 16)
                  plt.grid(True)
                  # Enable for sequential viewing
                  #raw_input()
                  #plt.draw()
            plt.gcf().subplots_adjust(top = 0.85)

          # Reset variables for next ROC curve.
          mean_fprs = []
          mean_tprs = []
          std_fprs = []
          std_tprs = []
          all_fprs = []
          all_tprs = []
          all_earlies = []
          all_lates = []
          var_attr_count = 0
          var_attr_values = []
    
      var_attr_values.append(curr_params._asdict()[var_attr])

      # Save the relevant stats about the current point on the ROC curve.
      if plot:
        fprs = [ stats['fpr']
                 for stats in statsets_sorted[psi]
                 if stats ]
        tprs = [ stats['tpr']
                 for stats in statsets_sorted[psi]
                 if stats ]
        
        earlies_point_trials = []
        lates_point_trials = []
        for stats in statsets_sorted[psi]:
          if not stats:
            continue
          earlies_trial = stats['earlies']
          lates_trial = stats['lates']
          earlies_point_trials.append(earlies_trial)
          lates_point_trials.append(lates_trial)

        if pnt:
          print fprs, tprs
        if fprs and tprs:
          mfprs = np.mean(fprs)
          mtprs = np.mean(tprs)
          sfprs = np.std(fprs)
          stprs = np.std(tprs)

          # Record these so we plot them before moving on to the next ROC curve.
          curr_params_dict = curr_params._asdict()
          if all([ curr_params_dict[const_attr]
                   in const_attrs_allowed_values[const_attr]
                   for const_attr in const_attrs ]):
            mean_fprs.append(mfprs)
            mean_tprs.append(mtprs)
            std_fprs.append(sfprs)
            std_tprs.append(stprs)

            # Record full list of fprs and tprs
            all_fprs.append(fprs)
            all_tprs.append(tprs)

            # Record earlies and lates
            all_earlies.append(earlies_point_trials)
            all_lates.append(lates_point_trials)

      var_attr_count += 1

    print var_attr  

    plot_twitter_killer = True
    if plot_twitter_killer:
      nbins = 4
      ecatd = earliness['center']
      ecat_for_params = []
      lcat_for_params = []
      for i in xrange(len(ecatd['fprs'])):
        ecat = [ -e / 3600000.0 for e in ecatd['earlies'][i] ]
        lcat = [ l / 3600000.0 for l in ecatd['lates'][i] ]
        params = ecatd['params'][i]

        """
        param_str = string.join(
          [ '%s$=%s$' % (attr_math_name[attr], str(params._asdict()[attr])) 
            for attr in all_attrs ],
          ', ')
        """

        param_str = string.join(
          [ '%s$=%s$' % (attr_math_name[attr], str(params._asdict()[attr])) 
            for attr in all_attrs
            if attr not in ['req_consec_detections', 'threshold'] ],
          ', ')

        fpr = ecatd['fprs'][i]
        tpr = ecatd['tprs'][i]
        print i, ':', param_str, len(ecat), len(lcat), len(ecat) / float(len(lcat) + len(ecat)), fpr, tpr

        #cond = len(ecat) > 1.45 * len(lcat) and fpr < 0.10 and tpr > 0.92 and \
        #    abs(np.mean(ecat)) > abs(np.mean(lcat))

        cond = fpr < 0.05 and tpr > 0.94

        param_avg = None
        param_count = 0
        
        if cond:
          if param_avg is None:
            param_avg = params
          else:
            param_avg += params
          param_count += 1

          plt.figure(figsize = (9, 2.75))
          n, bins, hpatches = plt.hist(ecat, bins = nbins,
                                       histtype = 'stepfilled', color = 'k',
                                       align = 'mid', label = 'early')

          print bins
          plt.setp(hpatches, 'facecolor', 'w')
          nmax = max(n)
          plt.hold(True)
          n, bins, hpatches = plt.hist(lcat, bins = nbins,
                                       histtype = 'stepfilled', color = 'k',
                                       align = 'mid', label = 'late')
          print bins
          plt.setp(hpatches, 'facecolor', 'k')
          nmax = max(max(n), nmax)
          plt.ylim([0, 1.2 * nmax])

          xmax = max(lcat)
          plate = len(lcat) / float(len(lcat) + len(ecat))
          """
          plt.text(0.4 * xmax, nmax, '$P(early) = %.2f$' % (1 - plate),
                    size = 10)
          plt.text(0.4 * xmax, 0.8 * nmax, 
                    '$P(late) = %.2f$' % plate, size = 10)
          plt.text(0.4 * xmax, 0.6 * nmax,
                    r'$\langle early \rangle = %.2f \; hrs.$' % (-np.mean(ecat)),
                    size = 10)
          plt.text(0.4 * xmax, 0.4 * nmax,
                    r'$\langle late \rangle = %.2f \; hrs.$' % (np.mean(lcat)),
                    size = 10)
          """

          plt.title('$FPR=%.2f$, $TPR=%.2f$, $P(early)=%.2f$, $\langle early \\rangle=%.2f hrs$\n%s' % 
                    (ecatd['fprs'][i], ecatd['tprs'][i], 1 - plate,
                     -np.mean(ecat), param_str), size = 16)

          plt.xlabel('hours late', size = 16)
          plt.ylabel('count', size = 16)
          plt.legend(loc = 1)
          plt.gcf().subplots_adjust(left = 0.07, bottom = 0.20, right = 0.95,
                                    top = 0.80)
          #raw_input()              

    plot_earliness = False
    if plot_earliness:
      #plt.figure(figsize = (7,14))
      #plt.suptitle('Early detection vs. position on ROC curve', size = 15)
      """
      plt.subplot(211)
      for (ci, category) in enumerate(earliness):
        ecatd = earliness[category]
        plt.scatter(ecatd['fprs'], ecatd['tprs'], s = 5, c = (0.75,0.75,0.75),
                    edgecolors = 'none')
        plt.hold(True)
      """
      cat_means = {}
      for (ci, category) in enumerate(earliness):
        ecatd = earliness[category]
        mfprs = np.mean(ecatd['fprs'])
        sfprs = np.std(ecatd['fprs'])
        mtprs = np.mean(ecatd['tprs'])
        stprs = np.std(ecatd['tprs'])
        cat_means[category] = (mfprs, mtprs)
        """
        plt.errorbar(mfprs, mtprs, xerr = sfprs, yerr = stprs,
                     linestyle = 'None', color = 'k', linewidth = 1.5)
        plt.hold(True)
        plt.scatter(mfprs, mtprs, s = 100, c = 'k')
        plt.text(mfprs + 0.05, mtprs - 0.05, category, size = 15)
        plt.axvline(0, linestyle = ':', color = 'k', linewidth = 1)
        plt.axvline(1, linestyle = ':', color = 'k', linewidth = 1)
        plt.axhline(0, linestyle = ':', color = 'k', linewidth = 1)
        plt.axhline(1, linestyle = ':', color = 'k', linewidth = 1)
        plt.axvline(fpr_partition_threshold, linestyle = ':', color = 'k')
        plt.axhline(tpr_partition_threshold, linestyle = ':', color = 'k')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        
        plt.xlabel('$FPR$')
        plt.ylabel('$TPR$')
        """
      nbins = 10
      
      for (ci, category) in enumerate(earliness):
        ecatd = earliness[category]
        ecat = []
        lcat = []
        for e in ecatd['earlies']:
          ecat.extend([ -ei / 3600000.0 for ei in e ])
        for l in ecatd['lates']:
          lcat.extend([ li / 3600000.0 for li in l])

        #plt.subplot(6, 1, ci + 4)
        plt.subplot(1, 3, ci + 1)
        n, bins, hpatches = plt.hist(ecat, bins = nbins,
                                     histtype = 'stepfilled', color = 'k',
                                     align = 'mid', label = 'early')
        plt.setp(hpatches, 'facecolor', 'w')
        nmax = max(n)
        plt.hold(True)
        n, bins, hpatches = plt.hist(lcat, bins = nbins,
                                     histtype = 'stepfilled', color = 'k',
                                     align = 'mid', label = 'late')
        plt.setp(hpatches, 'facecolor', 'k')
        nmax = max(max(n), nmax)
        plt.ylim([0, 1.2 * nmax])

        plate = len(lcat) / float(len(lcat) + len(ecat))
        """
        plt.text(-9.75, nmax, '$P(early) = %.2f$' % (1 - plate),
                  size = 10)
        plt.text(-9.75, 0.8 * nmax, 
                  '$P(late) = %.2f$' % plate, size = 10)
        plt.text(-9.75, 0.6 * nmax,
                  r'$\langle early \rangle = %.2f \; hrs.$' % (-np.mean(ecat)),
                  size = 10)
        plt.text(-9.75, 0.4 * nmax,
                  r'$\langle late \rangle = %.2f \; hrs.$' % (np.mean(lcat)),
                  size = 10)
        """
        plt.title(r'$FPR=%.2f$, $TPR=%.2f$, $P(early)=%.2f$, $\langle early \rangle=%.2f hrs$' % 
                  (cat_means[category][0], cat_means[category][1], 1 - plate,
                  -np.mean(ecat)), size = 16)
        if ci == 2:
          plt.xlabel('hours late', size = 16)
        plt.ylabel('count', size = 16)
        
        if ci == 0:
          plt.legend(loc = 1)

      plt.gcf().subplots_adjust(top = 0.96, bottom = 0.05, hspace = 0.30)
      #plt.savefig('fig/final/early_vs_roc.eps')
      break

    plot_rates_partitioned = False
    if plot_rates_partitioned:
      for category in rates_partitioned:
        print category
        cat_params = rates_partitioned[category].keys()
        attr_to_values = {}
        for attr in all_attrs:
          attr_to_values[attr] = []
          for params in cat_params:
            attr_to_values[attr].append(params._asdict()[attr])

        cat_mean_attr_values = dict([ [attr, np.mean(attr_to_values[attr])]
                                      for attr in attr_to_values
                                      if attr_to_values[attr] ])
        pp.pprint(cat_mean_attr_values)
      break

    # Partition delta ranked results into top, bottom, center.
    plot_deltas_partitioned = False
    if plot_deltas_partitioned:
      plt.figure(figsize = (6,9))
      plt.suptitle(var_attr)
      for (ci, category) in enumerate(deltas_partitioned):
        print category
        deltas_cat = deltas_partitioned[category]
        delta_fprs_cat = []
        for params in deltas_cat:
          delta_fprs_cat.extend(deltas_cat[params][0]) 
        delta_tprs_cat = []
        for params in deltas_cat:
          delta_tprs_cat.extend(deltas_cat[params][1]) 

        nbins = 30
        if delta_fprs_cat:
          delta_fprs_cat = np.array(delta_fprs_cat)
          print 'mean delta fprs = %.4f' % np.mean(delta_fprs_cat)

          """
          pos_ind = np.where(delta_fprs_cat > 0)[0]
          neg_ind = np.where(delta_fprs_cat < 0)[0]

          print 'overall:\t%.4f +/- %.4f' % (np.mean(delta_fprs_cat),
                                             np.std(delta_fprs_cat))
          if len(pos_ind) > 0:
            print 'positive change:\t%.4f +/- %.4f\t(%.2f%%)' % (np.mean(delta_fprs_cat[pos_ind]),
              np.std(delta_fprs_cat[pos_ind]),
              100 * float(len(delta_fprs_cat[pos_ind])) / len(delta_fprs_cat))
          if len(neg_ind) > 0:
            print 'negative change:\t%.4f +/- %.4f\t(%.2f%%)' % (np.mean(delta_fprs_cat[neg_ind]),
              np.std(delta_fprs_cat[neg_ind]),
              100 * float(len(delta_fprs_cat[neg_ind])) / len(delta_fprs_cat))
          print 'no change:\t(%.2f%%)' % \
              (100 * float(len(delta_fprs_cat) - len(pos_ind) - len(neg_ind)) / len(delta_fprs_cat))
          """

          """
          plt.subplot(3, 2, 2 * ci + 1)
          plt.hist(delta_fprs_cat, bins = nbins)
          plt.title(category + '$\Delta_p^{FPR}$')
          """
        else:
          print 'None'

        if delta_tprs_cat:
          delta_tprs_cat = np.array(delta_tprs_cat)
          print 'mean delta tprs = %.4f' % np.mean(delta_tprs_cat)

          """
          pos_ind = np.where(delta_tprs_cat > 0)[0]
          neg_ind = np.where(delta_tprs_cat < 0)[0]

          print 'overall:\t%.4f +/- %.4f' % (np.mean(delta_tprs_cat),
                                             np.std(delta_tprs_cat))
          
          if len(pos_ind) > 0:
            print 'positive change:\t%.4f +/- %.4f\t(%.2f%%)' % (np.mean(delta_tprs_cat[pos_ind]),
              np.std(delta_tprs_cat[pos_ind]),
              100 * float(len(delta_tprs_cat[pos_ind])) / len(delta_tprs_cat))
          if len(neg_ind) > 0:
            print 'negative change:\t%.4f +/- %.4f\t(%.2f%%)' % (np.mean(delta_tprs_cat[neg_ind]),
              np.std(delta_tprs_cat[neg_ind]),
              100 * float(len(delta_tprs_cat[neg_ind])) / len(delta_tprs_cat))
          print 'no change:\t(%.2f%%)' % \
              (100 * float(len(delta_tprs_cat) - len(pos_ind) - len(neg_ind)) / len(delta_tprs_cat))
          """

          """
          plt.subplot(3, 2, 2 * ci + 2)
          plt.hist(delta_tprs_cat, bins = nbins)
          plt.title(category + '$\Delta_p^{TPR}$')
          """
        else:
          print 'None'

      plt.gcf().subplots_adjust(bottom = 0.08, hspace = 0.40)

    # Plot a heatmap of positions in plane for each var_attr
    plot_roc_plane_scatter = False
    if plot_roc_plane_scatter:
      expname = 'threshold_3'
      if not os.path.exists(os.path.join('fig/final/position', expname)):
        os.mkdir(os.path.join('fig/final/position', expname))
      if not os.path.exists(os.path.join('fig/final/position', expname, var_attr)):
        os.mkdir(os.path.join('fig/final/position', expname, var_attr))
      for (cai, const_attr) in enumerate(const_attrs):
        const_attr_values = const_attrs_allowed_values[const_attr]
        plt.figure(figsize = (13, 3))
        for (cavi, const_attr_value) in enumerate(const_attr_values):
          print const_attr, '=', const_attr_value
          udr_fprs = []
          udr_tprs = []
          fprs_for_const_attr_value = []
          fprs_for_const_attr_value = []
          entries_for_const_attr_value = \
              [ updown_rank[udr] for udr in updown_rank 
                if udr._asdict()[const_attr] == const_attr_value ]
          if not entries_for_const_attr_value:
            # It is possible that for the given const_attr_value, none of deltas
            # made the criterion for inclusion. For example, they might have
            # been skipped because they corresponded to points on the ROC curve
            # "stuck" near (0,0) or (1,1), for which deltas are meaningless.
            print 'No matches for ', const_attr, '=', const_attr_value
            continue
          for udr in entries_for_const_attr_value:
            udr_fprs.extend(np.array(udr[4]).flatten())
            udr_tprs.extend(np.array(udr[5]).flatten())
          """
          heatmap, xedges, yedges = np.histogram2d(udr_tprs,
            udr_fprs, bins=40, range = [[0,1], [0,1]])
          heatmap = np.log(heatmap + 0.1)
          extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
          plt.imshow(np.flipud(heatmap), extent = extent)
          """
          plt.subplot(1, 4, cavi + 1)
          plt.scatter(udr_fprs, udr_tprs)
          plt.hold(True)
          mean_udr_fprs = np.mean(udr_fprs)
          mean_udr_tprs = np.mean(udr_tprs)
          print '%.2f,%.2f' % (mean_udr_fprs, mean_udr_tprs)
          plt.axvline(mean_udr_fprs, lw = 1, ls = '--', color = 'k')
          plt.axhline(mean_udr_tprs, lw = 1, ls = '--', color = 'k')
          plt.text(0.4, 0.15, '$FPR_{mean} = %.2f$' % mean_udr_fprs)
          plt.text(0.4, 0.05, '$TPR_{mean} = %.2f$' % mean_udr_tprs)
          plt.xlim([-.1,1.1])
          plt.ylim([-.1,1.1])
          if cavi == 0:
            plt.ylabel('$TPR$')
          plt.xlabel('$FPR$')
          plt.title(const_attr + '=' + str(const_attr_value))
          plt.grid(True)
          plt.hold(False)
          plt.draw()
        plt.gcf().subplots_adjust(left = 0.08, bottom = 0.15, right = 0.94,
                                  wspace = 0.30, hspace = 0.20)
        plt.savefig(os.path.join('fig/final/position/', expname,
                                 var_attr, const_attr + '.eps'))
        #raw_input()

    # For current var_attr, compute rank of all other parameters by how far "up"
    # or "down" the ROC curve lies in the continuum of FPR/TPR tradeoffs.
    updown_rank = [ [udr, updown_rank[udr]] for udr in updown_rank ]
    updown_rank_f_min = sorted(updown_rank, key = lambda x: x[1][0], reverse = True)
    updown_rank_t_min = sorted(updown_rank, key = lambda x: x[1][1], reverse = True)
    updown_rank_f_max = sorted(updown_rank, key = lambda x: x[1][2], reverse = True)
    updown_rank_t_max = sorted(updown_rank, key = lambda x: x[1][3], reverse = True)
    
    plot_roc_updown_ranked = False
    if plot_roc_updown_ranked:
      for updown_rank_sorted in [ updown_rank_f_min, updown_rank_f_max ]:
        plt.figure(figsize = (10,10))
        print '\n\n'
        for udri, udr in enumerate(updown_rank_sorted):
          if udri >= 25:
            break

          udr_params = udr[0]
          udr_f_min_score = udr[1][0]
          udr_t_min_score = udr[1][1]
          udr_f_max_score = udr[1][2]
          udr_t_max_score = udr[1][3]
          print 'upr_f_min_score', udr_f_min_score
          print 'upr_t_min_score', udr_t_min_score
          print 'upr_f_max_score', udr_f_max_score
          print 'upr_t_max_score', udr_t_max_score
          udr_all_fprs = udr[1][4]
          udr_all_tprs = udr[1][5]
          udr_mean_fprs = [ np.mean(daf) for daf in udr_all_fprs ]
          udr_mean_tprs = [ np.mean(dat) for dat in udr_all_tprs ]
          udr_std_fprs = [ np.std(daf) for daf in udr_all_fprs ]
          udr_std_tprs = [ np.std(dat) for dat in udr_all_tprs ]
          
          # Plot thumbnails.
          plt.subplot(5, 5, udri + 1)
          plt.errorbar(udr_mean_fprs, udr_mean_tprs, udr_std_fprs, udr_std_tprs,
                       linestyle = 'None', color = 'b')
          plt.hold(True)
          plt.scatter(udr_mean_fprs, udr_mean_tprs,
                      s = [ 2 + 15 * i for i in range(len(udr_mean_tprs)) ],
                      c = 'b')
          # __str__() doesn't work for some reason... TODO
          plt.title(str([udr_params._asdict()[k]
                         for k in const_attrs]), fontsize = 11)
          plt.setp(plt.gca(), xticklabels=[])
          plt.setp(plt.gca(), yticklabels=[])
          print udr_params.__str__
          plt.xlim([-0.1, 1.1])
          plt.ylim([-0.1, 1.1])
          plt.grid(True)
          plt.hold(False)
          plt.draw()

        raw_input()

    # For current var_attr, compute rank of all other parameters by how much
    # positive and negative delta fprs and delta tprs they cause.
    delta_rank = [ [drk, delta_rank[drk]] for drk in delta_rank ]
    delta_rank_f = sorted(delta_rank, key = lambda x: x[1][0],
                          reverse = True)
    delta_rank_t = sorted(delta_rank, key = lambda x: x[1][1],
                          reverse = True)
    # print '\n\ndelta_rank_f'
    # pp.pprint(delta_rank_f)
    # print '\n\ndelta_rank_t'
    # pp.pprint(delta_rank_t)

    plot_roc_delta_ranked = False
    if plot_roc_delta_ranked:
      for delta_rank_sorted in [ delta_rank_f ]:
        print '\n\n'
        # plt.figure(figsize = (10,10))
        plt.figure(figsize = (10,5))
        plt.suptitle('Top $\Delta_p^{FPR}$ for ' + var_attr + \
                       '\nconstant params = ' + str(const_attrs))
        for dri, dr in enumerate(delta_rank_sorted):
          if dri >= 8:
            break
          dr_params = dr[0]
          dr_fscore = dr[1][0]
          dr_tscore = dr[1][1]
          print 'delta fpr score', dr_fscore, 'delta tpr score', dr_tscore
          dr_all_fprs = dr[1][2]
          dr_all_tprs = dr[1][3]
          dr_mean_fprs = [ np.mean(daf) for daf in dr_all_fprs ]
          dr_mean_tprs = [ np.mean(dat) for dat in dr_all_tprs ]
          dr_std_fprs = [ np.std(daf) for daf in dr_all_fprs ]
          dr_std_tprs = [ np.std(dat) for dat in dr_all_tprs ]

          plt.subplot(2, 4, dri + 1)
          #plt.subplot(4, 4, dri + 1)
          plt.errorbar(dr_mean_fprs, dr_mean_tprs, xerr=dr_std_fprs, yerr=dr_std_tprs,
                       linestyle = 'None', color = 'b')
          plt.hold(True)
          plt.scatter(dr_mean_fprs, dr_mean_tprs,
                      s = [ 2 + 15 * i for i in range(len(dr_mean_tprs)) ],
                      c = 'b')
          plt.title(str([dr_params._asdict()[k]
                         for k in const_attrs]), fontsize = 11)
          # __str__() doesn't work for some reason... TODO
          print dr_params.__str__
          plt.xlim([-0.1, 1.1])
          plt.ylim([-0.1, 1.1])
          plt.xlabel('$FPR$')
          if dri == 0:
            plt.ylabel('$TPR$')
          plt.grid(True)
          plt.draw()
          #plt.setp(plt.gca(), xticklabels=[])
          #plt.setp(plt.gca(), yticklabels=[])
          plt.hold(False)
        plt.gcf().subplots_adjust(top = 0.80, wspace = 0.45, hspace = 0.45)
        plt.savefig('fig/final/top_fpr/' + var_attr + '.eps')
        plt.draw()

    plot_secondary_effect = False
    if plot_secondary_effect:
      for const_attr in const_attrs:
        const_attr_values_f = [ dr[0]._asdict()[const_attr]
                              for dr in delta_rank_f ]
        rank_scores_f = [dr[1][0] for dr in delta_rank_f]
        const_attr_values_t = [ dr[0]._asdict()[const_attr]
                                for dr in delta_rank_t ]
        rank_scores_t = [dr[1][1] for dr in delta_rank_t]

        plt.subplot(121)
        plt.scatter(const_attr_values_f, rank_scores_f)
        plt.hold(True)
        plt.axhline(0, linestyle = '--', color = 'k')
        plt.title('fpr ' + const_attr)
        plt.hold(False)

        plt.subplot(122)
        plt.scatter(const_attr_values_t, rank_scores_t)
        plt.hold(True)
        plt.axhline(0, linestyle = '--', color = 'k')
        plt.title('tpr ' + const_attr)
        plt.hold(False)
        raw_input()

    plot_secondary_effect_hist = False
    if plot_secondary_effect_hist:
      for (cai, const_attr) in enumerate(const_attrs):
        plt.figure(figsize = (14, 3))
        const_attr_values = const_attrs_allowed_values[const_attr]
        # Store all heatmaps and extents, get a common extent, and plot all
        # heatmaps together at the end.
        dr_all_delta_fprs = []
        dr_all_delta_tprs = []
        ranges = []
        for (cavi, const_attr_value) in enumerate(const_attr_values):
          print const_attr, '=', const_attr_value
          dr_delta_fprs = []
          dr_delta_tprs = []
          entries_for_const_attr_value = \
              [ drf[1] for drf in delta_rank_f 
                if drf[0]._asdict()[const_attr] == const_attr_value ]
          if not entries_for_const_attr_value:
            # It is possible that for the given const_attr_value, none of deltas
            # made the criterion for inclusion. For example, they might have
            # been skipped because they corresponded to points on the ROC curve
            # "stuck" near (0,0) or (1,1), for which deltas are meaningless.
            continue
          for dr in entries_for_const_attr_value:
            dr_delta_fprs.extend(dr[4])
            dr_delta_tprs.extend(dr[5])

          #plt.subplot(223)
          dr_all_delta_fprs.append(dr_delta_fprs)
          dr_all_delta_tprs.append(dr_delta_tprs)
          ranges.append([[min(dr_delta_fprs), max(dr_delta_fprs)],
                         [min(dr_delta_tprs), max(dr_delta_tprs)]])

          """
          plt.subplot(334)
          n, bins, hpatches = plt.hist(dr_delta_tprs, bins = 30, normed = False,
                                       histtype = 'stepfilled', color = 'k',
                                       align = 'mid', orientation = 'horizontal')
          plt.setp(hpatches, 'facecolor', 'm')
          plt.axhline(0, linestyle = '--', color = 'k')
          plt.title('$\Delta_p^{TPR}$')
          plt.setp(plt.gca(), xticklabels=[])
          plt.setp(plt.gca(), yticklabels=[])

          plt.subplot(221)
          n, bins, hpatches = plt.hist(dr_delta_fprs, bins = 30, normed = False,
                                       histtype = 'stepfilled', color = 'k',
                                       align = 'mid')
          plt.setp(hpatches, 'facecolor', 'm')
          plt.axvline(0, linestyle = '--', color = 'k')
          plt.title('$\Delta_p^{FPR}$')
          plt.setp(plt.gca(), xticklabels=[])
          plt.setp(plt.gca(), yticklabels=[])
          """
        """
        miny = min([ extent[0] for extent in extents ])
        maxy = max([ extent[1] for extent in extents ])
        minx = min([ extent[2] for extent in extents ])
        maxx = max([ extent[3] for extent in extents ])
        
        extent = [miny, maxy, minx, maxx]
        """
        xmin = min([ rangei[0][0] for rangei in ranges ])
        xmax = max([ rangei[0][1] for rangei in ranges ])
        ymin = min([ rangei[1][0] for rangei in ranges ])
        ymax = max([ rangei[1][1] for rangei in ranges ])
        common_range = [[ymin, ymax], [xmin, xmax]]
        for ri in xrange(len(ranges)):
          dr_delta_fprs = dr_all_delta_fprs[ri]
          dr_delta_tprs = dr_all_delta_tprs[ri]
          plt.subplot(1, 4, ri + 1)
          heatmap, xedges, yedges = np.histogram2d(dr_delta_tprs,
            dr_delta_fprs, bins=25, range = common_range)
          extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
          plt.contour(heatmap, 20, extent = extent)
          plt.hold(True)
          plt.axvline(0, ls = '--', lw = 1.5, color = 'k')
          plt.axhline(0, ls = '--', lw = 1.5, color = 'k')
          plt.axvline(np.mean(dr_delta_fprs), ls = ':',
                      color = 'k')
          plt.axhline(np.mean(dr_delta_tprs), ls = ':',
                      color = 'k')
          plt.title(const_attr + '=' + str(const_attr_values[ri]))
          if ri == 0:
            plt.ylabel('$\Delta_p^{TPR}$', fontsize = 15)
          plt.xlabel('$\Delta_p^{FPR}$', fontsize = 15)
        plt.gcf().subplots_adjust(bottom = 0.20, wspace = 0.35)
        plt.draw()
        #plt.savefig('fig/final/delta_hist_sec/' + var_attr + '/' + \
        #              const_attr + '.eps')
        raw_input()

    # Plot deltas in fpr and tpr as 2d histogram.
    plot_delta_dist = False
    if plot_delta_dist and delta_fprs and delta_tprs:
      plt.figure(figsize = (6,6))
      plt.suptitle(var_attr)

      plt.subplot(223)
      heatmap, xedges, yedges = np.histogram2d(delta_tprs, delta_fprs, bins=30)
      extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
      plt.contour(heatmap, 35, extent=extent)
      plt.hold(True)
      plt.axvline(0, linestyle = '--', color = 'k')
      plt.axhline(0, linestyle = '--', color = 'k')

      plt.subplot(224)
      n, bins, hpatches = plt.hist(delta_tprs, bins = 30, normed = False,
                                   histtype = 'stepfilled', color = 'k',
                                   align = 'mid', orientation = 'horizontal')
      plt.setp(hpatches, 'facecolor', 'm')
      plt.axhline(0, linestyle = '--', color = 'k')
      plt.title('$\Delta_p^{TPR}$')
      #plt.title('\Delta_p^{TPR}')

      plt.subplot(221)
      n, bins, hpatches = plt.hist(delta_fprs, bins = 30, normed = False,
                                   histtype = 'stepfilled', color = 'k',
                                   align = 'mid')
      plt.setp(hpatches, 'facecolor', 'm')
      plt.axvline(0, linestyle = '--', color = 'k')
      plt.title('$\Delta_p^{FPR}$')
      #plt.title('\Delta_p^{FPR}')

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
  
