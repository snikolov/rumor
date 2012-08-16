import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pprint
import string
import matplotlib.patches as patches

from math import exp
from matplotlib import rc
from matplotlib.path import Path
from operator import attrgetter
from params import *

rc('text', usetex = False)
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
          # Separate storage of deltas for current params only.
          delta_fprs_curr_params = []
          delta_tprs_curr_params = []

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
            tprs_combo_not_stuck = [ tprs_combo[j]
                                     for j in range(1, len(tprs_combo))
                                     if fprs_combo[j] != fprs_combo[j-1] and \
                                       tprs_combo[j] != tprs_combo[j-1] ]

            tprs_combo = tprs_combo_not_stuck
            fprs_combo = fprs_combo_not_stuck
            new_delta_fprs = [ (fprs_combo[i] - fprs_combo[i-1]) / \
                                 (var_attr_values[i] - var_attr_values[i-1])
                               for i in range(1, len(fprs_combo)) ]
            new_delta_tprs = [ (tprs_combo[i] - tprs_combo[i-1]) / \
                                 (var_attr_values[i] - var_attr_values[i-1])
                               for i in range(1, len(tprs_combo)) ]
            delta_fprs.extend(new_delta_fprs)
            delta_tprs.extend(new_delta_tprs)
            delta_fprs_curr_params.extend(new_delta_fprs)
            delta_tprs_curr_params.extend(new_delta_tprs)

          # Record the fraction of deltas greater than zero for the purpose of
          # ranking which parameters cause the most positive and negative
          # deltas.
          """
          f_num_geq_zero = len([ ndfpr for ndfpr in delta_fprs_curr_params
                                 if ndfpr >= 0 ])
          t_num_geq_zero = len([ ndtpr for ndtpr in delta_tprs_curr_params
                                 if ndtpr >= 0 ])
          """
          if len(delta_fprs_curr_params) > 0 and \
                len(delta_tprs_curr_params) > 0:
            """
            f_frac_geq_zero = \
              float(f_num_geq_zero) / len(delta_fprs_curr_params)
            t_frac_geq_zero = \
              float(t_num_geq_zero) / len(delta_tprs_curr_params)
            """
            # Store copies of all_fprs and all_tprs as well so we can plot ROC
            # curves at the end in order of rank.
            
            delta_rank[curr_params] = (np.mean(delta_fprs_curr_params),
                                       np.mean(delta_tprs_curr_params),
                                       all_fprs[:], all_tprs[:],
                                       delta_fprs_curr_params[:],
                                       delta_tprs_curr_params[:])
            
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
            updown_rank[curr_params] = (min_mean_fpr, min_mean_tpr,
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
                # For increasing point sizes.
                """
                plt.scatter(mean_fprs_ltor, mean_tprs_ltor,
                            s = point_sizes_ltor, c = 'b')
                """
                plt.scatter(mean_fprs_ltor, mean_tprs_ltor,
                            s = 15, c = 'k')
                plt.suptitle('Varying ' + var_attr)
                plt.title('Scatterplot')
                plt.xlabel('FPR')
                plt.ylabel('TPR')
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
                  plt.title('Best-case envelope')
                  plt.xlabel('FPR')
                  plt.ylabel('TPR')
                  plt.grid(True)
                  # Enable for sequential viewing
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

      # Save the relevant stats about the current point on the ROC curve.
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

      var_attr_count += 1

    print var_attr  
    # Plot a heatmap of positions in plane for each var_attr
    plot_roc_plane_scatter = False
    if plot_roc_plane_scatter:
      plt.figure(figsize = (2.5,2.5))
      for const_attr in const_attrs:
        const_attr_values = const_attrs_allowed_values[const_attr]
        for const_attr_value in const_attr_values:
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
          heatmap, xedges, yedges = np.histogram2d(udr_fprs,
                                                   udr_tprs, bins=30)
          extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
          plt.imshow(np.flipud(heatmap), extent = extent)
          """
          plt.scatter(udr_fprs, udr_tprs)
          plt.hold(True)
          plt.axvline(np.mean(udr_fprs), lw = 1, ls = '--')
          plt.axhline(np.mean(udr_tprs), lw = 1, ls = '--')
          plt.xlim([-.1,1.1])
          plt.ylim([-.1,1.1])
          plt.grid(True)
          plt.hold(False)
          raw_input()

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
                          reverse = False)
    # print '\n\ndelta_rank_f'
    # pp.pprint(delta_rank_f)
    # print '\n\ndelta_rank_t'
    # pp.pprint(delta_rank_t)

    plot_roc_delta_ranked = True
    if plot_roc_delta_ranked:
      for delta_rank_sorted in [ delta_rank_t ]:
        print '\n\n'
        plt.figure(figsize = (10,5))
        plt.suptitle('Bottom $\Delta_p^{TPR}$ for ' + var_attr + \
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
          plt.errorbar(dr_mean_fprs, dr_mean_tprs, dr_std_fprs, dr_std_tprs,
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
          plt.ylabel('$TPR$')
          plt.grid(True)
          plt.draw()
          #plt.setp(plt.gca(), xticklabels=[])
          #plt.setp(plt.gca(), yticklabels=[])
          plt.hold(False)
        plt.gcf().subplots_adjust(top = 0.80, wspace = 0.45, hspace = 0.45)
        plt.savefig('fig/final/bottom_tpr/' + var_attr + '.eps')
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
          #plt.savefig('fig/final/delta_hist_sec/' + var_attr + '_' + \
          #              const_attr + '_' + const_attr_value + '.eps')
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
          plt.draw()
        plt.gcf().subplots_adjust(bottom = 0.20, wspace = 0.35)
        plt.savefig('fig/final/delta_hist_sec/' + var_attr + '/' + \
                      const_attr + '.eps')
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
  
