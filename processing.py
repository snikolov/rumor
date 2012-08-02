import colorsys
import copy
import datetime
import numpy as np
import matplotlib.pyplot as plt
import sys
import util

from gexf import Gexf
from math import log, exp
from subprocess import call
from timeseries import *

np.seterr(all = 'raise')

#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
def compute_rumor_tree_edges(statuses, edges, window):
  rumor_edges=[]
  parent={}
  for edge in edges:
    u = edge[0]
    v = edge[1]
    status_v = statuses[v]
    status_u = statuses[u]
    if status_v == None or status_u == None:
      continue
    if len(status_v) < 4 or len(status_u) < 4:
      continue
    if status_v[3] == '' or status_u[3] == '':
      continue
    # Compare timestamps
    try:
      t_v = util.datetime_to_epoch_seconds(status_v[3])
      t_u = util.datetime_to_epoch_seconds(status_u[3])
    except ValueError:
      print "Can't convert one or both of these to a timestamp:\n", \
          status_v[3], '\n', status_u[3]
    t_diff = t_u - t_v
    if t_diff <= window and t_diff > 0:
      if u not in parent:
        parent[u] = (v, t_v, t_u)
      else:
        parent_u = parent[u]
        # Replace parent if there is a more recent parent
        if t_v > parent_u[1]:
          parent[u] = (v, t_v, t_u)
    elif -t_diff <= window and t_diff < 0:
      if v not in parent:
        parent[v] = (u, t_u, t_v)
      else:
        parent_v = parent[v]
        # Replace parent if there is a more recent parent
        if t_u > parent_v[1]:
          parent[v] = (u, t_u, t_v)

  rumor_edges = [ (parent[a][0],a,parent[a][2]) for a in parent ]
  for r in rumor_edges:
    print r
  rumor_edges.sort(util.timestamped_edge_comparator)
  return rumor_edges

#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
# An edge (v,u) is a rumor edge iff (u,v) is in edges (i.e. u follows
# v) and if t_u - t_v <= window
def compute_rumor_edges(statuses, edges, window):
  rumor_edges = []
  for edge in edges:
    u = edge[0]
    v = edge[1]
    status_v = statuses[v]
    status_u = statuses[u]
    if status_v == None or status_u == None:
      continue
    if len(status_v) < 4 or len(status_u) < 4:
      continue
    if status_v[3] == '' or status_u[3] == '':
      continue
    # Compare timestamps
    try:
      t_v = util.datetime_to_epoch_seconds(status_v[3])
      t_u = util.datetime_to_epoch_seconds(status_u[3])
    except ValueError:
      print "Can't convert one or both of these to a timestamp:\n", \
        status_v[3], '\n', status_u[3]
    t_diff = t_u - t_v
    if t_diff <= window and t_diff > 0:
      rumor_edges.append((v, u, t_u))
    elif -t_diff <= window and t_diff < 0:
      rumor_edges.append((u, v, t_v))

  rumor_edges.sort(util.timestamped_edge_comparator,'descend')
  return rumor_edges

#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
# Take statuses and edges sorted by timestamp and simulate the rumor
# forward in time.
def simulate(rumor, step_mode = 'time', step = 10, limit = 2400):
  rumor_edges = rumor['edges']
  rumor_statuses = rumor['statuses']
  trend_onset = rumor['trend_onset']

  # Figure
  plt.figure()

  # Time series
  max_sizes = []
  total_sizes = []
  component_nums = []
  entropies = []
  max_component_ratios = []
  timestamps = []

  min_time = min([ edge[2] for edge in rumor_edges ])
  if step_mode == 'time':
    next_time = min_time
  max_pos = limit

  print 'time\t\teid\t\tpos\t\t|C_max|\t\tN(C)\t\ttime-trend_onset'

  components = {}
  node_to_component_id = {}
  adj={}

  # Set to keep track of statuses that gain many inbound edges at the same
  # time. This happens when a user follows lots of people that have mentioned
  # the topic, then tweets about the topic gets all of those followees as
  # parents, causing a sharp spike in the component growth

  # spikeset = set()

  for eid, edge in enumerate(rumor_edges):
    # print edge
    # print components
    # print node_to_component_id

    # Update adjacency list
    if edge[0] in adj:
      adj[edge[0]].append(edge[1])
    else:
      adj[edge[0]]=[edge[1]]
    
    # Update components
    if edge[0] not in node_to_component_id and edge[1] not in \
        node_to_component_id:
      # Create new component with id edge[0] (i.e. first node belonging to that
      #  component)
      component_id = edge[0]
      # print 'Creating new component ', component_id, ' from ', edge[0], ' and
      # ', edge[1]
      members = set([edge[0], edge[1]])
      components[edge[0]] = members
      node_to_component_id[edge[0]] = component_id
      node_to_component_id[edge[1]] = component_id
    elif edge[0] not in node_to_component_id:
      c1 = node_to_component_id[edge[1]]
      # print 'Adding ', edge[0], ' to ', c1, ': ', components[c1]
      # raw_input('')
      components[c1].add(edge[0])
      node_to_component_id[edge[0]] = c1
    elif edge[1] not in node_to_component_id:
      c0 = node_to_component_id[edge[0]]
      # print 'Adding ', edge[1], ' to ', c0, ': ', components[c0]
      # raw_input('')
      components[c0].add(edge[1])
      node_to_component_id[edge[1]] = c0
    else:
      c0 = node_to_component_id[edge[0]]
      c1 = node_to_component_id[edge[1]]
      if c0 != c1:
        # Merge components.
        members = components[c1]
        # print 'Merging\n', c0, ': ', components[c0], '\ninto\n', c1, ': ',
        # components[c1], '\n' raw_input('')
        for member in components[c0]:
          members.add(member)
          node_to_component_id[member] = c1
        components.pop(c0)
    
    """
    # Pause when you have some number of repeat statuses in a row (meaning that
    # lots of edges that terminate in that status suddenly got created)
    repeat_num = 2
    status_id = rumor_statuses[rumor_edges[eid][1]][0]
    if eid > repeat_num and \ 
        last_k_statuses_equal(status_id, rumor_statuses,rumor_edges, eid, repeat_num) and \
        status_id not in spikeset:
      print (rumor_statuses[rumor_edges[eid][0]], \ 
        rumor_statuses[rumor_edges[eid][1]])
      spikeset.add(status_id)
      raw_input()
    """

    if step_mode == 'index':
      pos = eid
    elif step_mode == 'time':
      pos = edge[2] - min_time
        
    if pos > limit:
      break

    if step_mode == 'index' and eid % step:
      continue
    if step_mode == 'time':
      if edge[2] < next_time:
        continue
      else:
        next_time = edge[2] + step

    component_sizes = []
    # raw_input('======================================================'
    for cid, members in components.items():
      component_sizes.append(len(members))
      # print 'component ', cid, ' size: ', len(members)  
      # raw_input('-------------------')

    time_after_onset = None
    if trend_onset is not None:
      time_after_onset = edge[2] - trend_onset

    print edge[2] - min_time, '\t\t', eid, '\t\t', pos, '/', limit, '\t\t', max(component_sizes), '\t\t', len(components), '\t\t', time_after_onset
    # Print largest adjacency list sizes.
    neighbor_counts=[ len(adj[k]) for k in adj ]
    sorted_idx=range(len(neighbor_counts))
    sorted_idx.sort(lambda x, y: neighbor_counts[y] - neighbor_counts[x])
    for itop in xrange(10):
      if itop>=len(sorted_idx):
        break
      print adj.keys()[sorted_idx[itop]], ':', neighbor_counts[sorted_idx[itop]]
    raw_input()

    # Desc sort of component sizes
    component_sizes.sort()
    component_sizes.reverse()

    # Append to timeseries
    max_sizes.append(max(component_sizes))
    total_sizes.append(sum(component_sizes))
    component_nums.append(len(component_sizes))
    entropies.append(util.entropy(component_sizes))
    if trend_onset is None:
      trend_onset = 0
    timestamps.append((edge[2] - trend_onset) / (60 * 60))
    max_component_ratios.append(float(max(component_sizes))/sum(component_sizes))
    shifted_ind = np.linspace(1, 1 + len(component_sizes), len(component_sizes))

    if eid > 0:
      color = util.step_to_color(pos, max_pos)
      plt.subplot(331)
      plt.loglog(shifted_ind, component_sizes, color = color, hold = 'on')
      plt.title('Loglog desc component sizes')

      plt.subplot(332)
      plt.semilogy(timestamps[-1], max_sizes[-1], 'ro', color = color,
                   hold = 'on')
      plt.title('Max component size')
      plt.xlabel('time (hours)')

      plt.subplot(333)
      plt.semilogy(timestamps[-1], total_sizes[-1], 'ro', color = color,
                   hold = 'on')
      plt.title('Total network size')
      plt.xlabel('time (hours)')

      plt.subplot(334)
      plt.plot(timestamps[-1], entropies[-1], 'go', color = color, hold = 'on')
      plt.title('Entropy of desc component sizes')
      plt.xlabel('time (hours)')

      plt.subplot(335)
      plt.semilogy(timestamps[-1], component_nums[-1], 'ko', color = color,
                   hold = 'on')
      plt.title('Number of components')
      plt.xlabel('time (hours)')

      plt.subplot(336)
      plt.loglog(shifted_ind, np.cumsum(component_sizes), color = color,
                 hold = 'on')
      plt.title('Cum. sum. of desc component sizes')

      plt.subplot(337)
      plt.plot(timestamps[-1], max_component_ratios[-1], 'ko', color = color,
               hold = 'on')
      plt.title('Max comp size / Total network Size')
      plt.xlabel('time (hours)')

    # plt.hist(component_sizes, np.linspace(0.5, 15.5, 15))
    # plt.plot(np.cumsum(np.histogram(component_sizes, bins = np.linspace(0.5,
    # 15.5, 15))[0]), hold = 'on')
    if not eid % 15*step:
      pass#plt.pause(0.001)
  plt.show()
  return components

#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
def last_k_statuses_equal(equals_val, rumor_statuses, rumor_edges,
                          curr_idx, k):
  for i in xrange(k):
    if rumor_statuses[rumor_edges[curr_idx-i][1]][0] is not equals_val:
      return False
  return True

#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
# DETECTION
#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
def ts_eval_for_param(ts_info_pos, ts_info_neg, threshold, trend_times = None):
  ts_norm_func = ts_mean_median_norm_func(1, 0)
  detection_results = ts_detect(ts_info_pos = ts_info_pos,
                                ts_info_neg = ts_info_neg,
                                threshold = threshold,
                                test_frac = 0.25,
                                ts_norm_func = ts_norm_func)
  detections = detection_results['detection']
  lates = []
  earlies = []
  tp = 0
  fn = 0
  for topic in detections['pos']:
    if len(detections['pos'][topic]['times']) > 0:
      tp += 1
      detection_time = min(detections['pos'][topic]['times'])
      onset_time = ts_info_pos[topic]['trend_start']
      if detection_time > onset_time:
        lates.append(detection_time - onset_time)
      else:
        earlies.append(onset_time - detection_time)
    else:
      fn += 1
  fp = 0
  tn = 0
  for topic in detections['neg']:
    if len(detections['neg'][topic]['times']) > 0:
      fp += 1
    else:
      tn += 1

  print 'total pos = ', len(detections['pos'])
  print 'total neg = ', len(detections['neg'])
  print 'total = ', len(detections['neg']) + len(detections['pos'])
  print 'tp = ', tp 
  print 'fn = ', fn 
  print 'fp = ', fp 
  print 'tn = ', tn
  print 'fpr = ', (float(fp) / (fp + tn))
  print 'tpr = ', (float(tp) / (fn + tp))
  avg_early = None
  std_early = None
  avg_late = None
  std_late = None
  if len(earlies) > 0:
    avg_early = np.mean(earlies) / (3600 * 1000)
    std_early = np.std(earlies) / (3600 * 1000)
  if len(lates) > 0:
    avg_late = np.mean(lates) / (3600 * 1000)
    std_late = np.std(lates) / (3600 * 1000)
  print 'avg. early = ', avg_early, 'hrs'
  print 'stdev. early = ', std_early, 'hrs'
  print 'avg. late = ', avg_late, 'hrs'
  print 'stdev. late = ', std_late, 'hrs'
  print 'earlies\n', earlies, 'hrs'
  print 'lates\n', lates, 'hrs'
  
  viz = True
  if viz:
    viz_detection(detection_results = detection_results,
                  trend_times = trend_times)
  
#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
def viz_detection(ts_info_pos = None, ts_info_neg = None, trend_times = None,
                  detection_results = None):
  # Get raw and normalized rates
  # Compare trend times, detection times, rates, normalized rates, scores
  
  if detection_results is None:
    ts_norm_func = ts_mean_median_norm_func(1, 0)
    detection_results = ts_detect(ts_info_pos, ts_info_neg, threshold = 5,
                                  test_frac = 0.25, ts_norm_func = ts_norm_func)

  detections = detection_results['detection']
  scores = detection_results['scores']

  tests = { 'pos': 'b', 'neg': 'r' }
  for type in tests:
    # Plot detection times vs actual trend times.
    for topic in detections[type]:
      topic_detection_scores = detections[type][topic]['scores']
      topic_detection_times = detections[type][topic]['times']
      if len(topic_detection_times) > 0:
        markerline, stemlines, baseline = \
            plt.stem(np.array(topic_detection_times),
                     1.2 * np.ones((len(topic_detection_times), 1)),
                     hold = 'on')
        plt.setp(markerline, 'markerfacecolor', tests[type])
        plt.setp(markerline, 'markeredgecolor', tests[type])
        plt.setp(stemlines, 'color', tests[type])

      plt.plot(scores[type][topic].times, scores[type][topic].values,
               hold = 'on', color = 'g')
      plt.title(topic)

      if trend_times is not None:
        if type is 'pos':
          topic_trending_times = trend_times[topic]
          markerline, stemlines, baseline = \
              plt.stem(np.array(topic_trending_times),
                       np.ones((len(topic_trending_times), 1)),
                       hold = 'on')
          plt.setp(markerline, 'markerfacecolor', 'k')
          plt.setp(markerline, 'markeredgecolor', 'k')
          plt.setp(stemlines, 'color', 'k')

      plt.show()

#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
def ts_balance_data(ts_info_pos, ts_info_neg):
  topics_pos = ts_info_pos.keys()
  topics_neg = ts_info_neg.keys()

  # Balance the data
  if len(ts_info_pos) > len(ts_info_neg):
    more_pos = True
    r = (len(ts_info_pos) - len(ts_info_neg)) / float(len(ts_info_pos))
    for topic in topics_pos:
      if np.random.rand() < r:
        ts_info_pos.pop(topic)
  else:
    more_pos = False
    r = (len(ts_info_neg) - len(ts_info_pos)) / float(len(ts_info_neg))
    topics = ts_info_neg.keys()
    for topic in topics_neg:
      if np.random.rand() < r:
        ts_info_neg.pop(topic)

  return ts_info_pos, ts_info_neg

#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
def ts_normalize(ts_info, ts_norm_func):
  # TODO: This became really slow. Profile!!
  # Normalize all timeseries
  method = 'offline'
  for (i, topic) in enumerate(ts_info):
    print topic, ' ', (i + 1), '/', len(ts_info)
    ts = ts_info[topic]['ts']
    if method is 'online':
      norm_values = np.zeros((len(ts.values), 1))
      norm_values = \
          [ (ts.values[i] + 0.01) / (ts_norm_func(ts.values[0:i + 1]) + 0.01)
            for i in range(len(ts.values)) ]
    else:
      ts_norm = ts_norm_func(ts.values)
      norm_values = \
          [ (v + 0.01) / (ts_norm + 0.01) for v in ts.values ]
    ts_info[topic]['ts'] = Timeseries(ts.times, norm_values)
  return ts_info

#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
def ts_mean_median_norm_func(mean_weight, median_weight):
  func = lambda x: median_weight * np.median(x) + mean_weight * np.mean(x)
  return func

#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
# Create timeseries bundles.
def ts_bundle(ts_info, detection_window_time):
  bundle = {}
  for topic in ts_info:
    ts = ts_info[topic]['ts']

    if ts_info[topic]['trend_start'] is None or \
          ts_info[topic]['trend_end'] is None:
      start = ts.tmin + \
          np.random.rand() * (ts.tmax - ts.tmin - detection_window_time)
      end = start + detection_window_time
    else:
      start = ts_info[topic]['trend_start'] - detection_window_time
      end =  ts_info[topic]['trend_start']

    tsw = ts.ts_in_window(start,end)
    # Add 1 as a fudge factor, since we're taking log. TODO
    bundle[topic] = Timeseries(tsw.times, np.cumsum(tsw.values) + 0.01)
  return bundle

#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
def ts_split_training_test(ts_info, test_frac):
  # Split into training and test
  ts_info_train = ts_info
  topics = ts_info.keys()
  ts_info_test = {}
  for topic in topics:
    if np.random.rand() < test_frac:
      ts_info_test[topic] = ts_info_train.pop(topic)
  return ts_info_train, ts_info_test

#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
def ts_detect(ts_info_pos, ts_info_neg, threshold = 1, test_frac = 0.05,
              ts_norm_func = None):
  np.random.seed(31953)

  print 'Creating deep copies...'
  ts_info_pos = copy.deepcopy(ts_info_pos)
  ts_info_neg = copy.deepcopy(ts_info_neg)

  # Sample data
  print 'Sampling data...'
  topics_pos = ts_info_pos.keys()
  topics_neg = ts_info_neg.keys()
  p_sample = 0.05
  for t in topics_pos:
    if np.random.rand() < p_sample:
      ts_info_pos.pop(t)
  for t in topics_neg:
    if np.random.rand() < p_sample:
      ts_info_neg.pop(t)

  # TODO: different norm_funcs?

  print 'Balancing data...'
  ts_info_pos, ts_info_neg = ts_balance_data(ts_info_pos, ts_info_neg)
  
  ## Normalize whole timeseries a priori.
  # ts_info_pos = ts_normalize(ts_info_pos)
  # ts_info_neg = ts_normalize(ts_info_neg)
  
  print 'Splitting into training and test...'
  ts_info_pos_train, ts_info_pos_test = ts_split_training_test(ts_info_pos,
                                                               test_frac)
  ts_info_neg_train, ts_info_neg_test = ts_split_training_test(ts_info_neg,
                                                               test_frac)
  detection_interval_time = 5 * 60 * 1000
  detection_window_time = 12 * detection_interval_time

  ts_norm_func = ts_mean_median_norm_func(0.5, 0.5)
  # Normalize only training timeseries a priori (TODO: do it online)
  print 'Normalizing...'
  ts_info_pos_train = ts_normalize(ts_info_pos_train, ts_norm_func)
  ts_info_neg_train = ts_normalize(ts_info_neg_train, ts_norm_func)
  ts_norm_func = ts_mean_median_norm_func(0.5, 0.5)

  print 'Creating bundles...'
  bundle_pos = ts_bundle(ts_info_pos_train, detection_window_time)
  bundle_neg = ts_bundle(ts_info_neg_train, detection_window_time)

  results = {}
  results['scores'] = {}
  results['detection'] = {}
  detection = {}
  scores = {}

  # Test
  tests = {'pos' : {'ts_info' : ts_info_pos_test, 'color' : 'b'},
           'neg' : {'ts_info' : ts_info_neg_test, 'color' : 'r'}}

  stop_when_detected = False
  ignore_detection_far_from_onset = False
  ignore_detection_window = 6 * 1000 * 3600
  plot_hist = True
  plot_scores = False
  if plot_hist or plot_scores:
    plt.close('all')
    plt.ion()
    plt.figure()
    plt.hold(False)
    plt.show()
  # Number of contiguous samples to use to compare two volume trajectories.
  cmpr_window = 1
  for type in tests:
    detection[type] = {}
    scores[type] = {}
    for ti, topic in enumerate(tests[type]['ts_info']):
      print 'Topic: ', topic, '\t', ti + 1, '/', len(tests[type]['ts_info'])
      indices_tested = set()
      ts_test = tests[type]['ts_info'][topic]['ts']
      # Store scores at the end of each window
      topic_scores = []
      topic_score_times = []
      # Detection variables
      detected = False
      detection[type][topic] = {}
      detection[type][topic]['scores'] = []
      detection[type][topic]['times'] = []
      scores[type][topic] = []

      t_window_starts = np.arange(ts_test.tmin,
        ts_test.tmax - detection_window_time - detection_interval_time,
        detection_interval_time)
      for t_window_start in t_window_starts:
        if detected and stop_when_detected:
          break
        i_window_start = ts_test.time_to_index(t_window_start)
        # print 'Start index: ', i_window_start
        dt_detects = np.arange(detection_interval_time,
                               detection_window_time,
                               detection_interval_time)
        for dt_detect in dt_detects:
          if detected and stop_when_detected:
            break
          di_detect = ts_test.dtime_to_dindex(dt_detect)
          i_detect = i_window_start + di_detect
          if i_detect in indices_tested:
            continue
          indices_tested.add(i_detect)

          # print 'Offset: ', di_detect, '\tAbsolute: ', \
          #   (i_window_start + di_detect)

          # Compute score and do detection
          score_end_of_window_only = True
          test_rate = ts_test.values[0:i_window_start + di_detect]
          # TODO: decaying weights for online background model.
          test_rate_norm = ts_norm_func(test_rate)
          test_rate_in_window = np.array(
            ts_test.values[i_window_start:i_window_start + di_detect])
          # TODO: abstract out the 0.01 trick in a separate normalization
          # method.
          test_trajectory = np.cumsum(
            (test_rate_in_window + 0.01) / (test_rate_norm + 0.01))
          
          test_val = test_trajectory[-1]
          if dt_detect == max(dt_detects) or not score_end_of_window_only:
            score = detection_func(bundle_pos, bundle_neg, test_trajectory,
                                   len(test_trajectory) - 1, cmpr_window)
            topic_scores.append(score)
            topic_score_times.append(dt_detect + t_window_start)
            if score > threshold:
              detection_time = t_window_start + dt_detect
              onset_time = tests[type]['ts_info'][topic]['trend_start']
              record_detection = True
              if onset_time is not None and \
                  ignore_detection_far_from_onset and \
                  abs(detection_time - onset_time) > ignore_detection_window:
                record_detection = False
              if record_detection:
                detection[type][topic]['times'].append(detection_time)
                detection[type][topic]['scores'].append(score)
                detected = True
              sys.stdout.write('.')
            else:
              sys.stdout.write(' ')

          # Plots
          if plot_scores:
            if dt_detect == max(dt_detects) and \
                  t_window_start == max(t_window_starts):
              plt.plot(topic_score_times, topic_scores,
                       color = tests[type]['color'])
              plt.title(topic)
              plt.show()

          if plot_hist:
            if dt_detect == max(dt_detects):
              # Plot histogram of positive and negative values at
              # i_window_start + di_detect and vertical line for test value
              values_pos = [bundle_pos[t].values[di_detect] for t in bundle_pos]
              values_neg = [bundle_neg[t].values[di_detect] for t in bundle_neg]

              n, bins, patches = plt.hist([log(v) for v in values_pos],
                                          bins = 25,
                                          color = (0,0,1),
                                          normed = True)
              plt.hold(True)
              plt.setp(patches, 'facecolor', (0,0,1), 'alpha', 0.25)

              n, bins, patches = plt.hist([log(v) for v in values_neg],
                                          bins = 25,
                                          color = (1,0,0),
                                          normed = True)

              plt.setp(patches, 'facecolor', (1,0,0), 'alpha', 0.25)
              #print 'Test value: ', log(test_val)
              """
              print 'Score: ', detection_func(bundle_pos, bundle_neg,
                                              test_trajectory,
                                              len(test_trajectory) - 1,
                                              cmpr_window)
              """
              plt.axvline(log(test_val), hold = 'on',
                          color = tests[type]['color'], linewidth = 2)
              plt.title(topic)
              plt.hold(False)
              plt.draw()
      scores[type][topic] = Timeseries(topic_score_times, topic_scores)
  results['detection'] = detection
  results['scores'] = scores
  return results

#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
def detection_func(bundle_pos, bundle_neg, trajectory_test, idx, cmpr_window):
  gamma = 1

  if cmpr_window > 1:
    dists_pos = []
    dists_neg = []

    # Make the min index 1, since we are taking log of values of a cumsum and the
  # value at 0 will be 0.
    imin = max(idx - cmpr_window + 1, 1)
    imax = idx

    if imin == imax:
      return 0

    trajectory_test_cmpr = trajectory_test[imin:imax]

    # Convex distance function. SLOW
    # dist = lambda x, y: abs(log(x) - log(y))

    bundle_pos_cmpr = [ bundle_pos[topic].values[imin:imax] 
                        for topic in bundle_pos ]

    bundle_neg_cmpr = [ bundle_neg[topic].values[imin:imax] 
                        for topic in bundle_neg ]

    dists_pos = [
      np.mean(
        [
          abs(log(trajectory_test_cmpr[i]) - log(trajectory_pos_cmpr[i]))
          for i in range(len(trajectory_pos_cmpr))
        ]
      )
      for trajectory_pos_cmpr in bundle_pos_cmpr
    ]

    dists_neg = [
      np.mean(
        [
          abs(log(trajectory_test_cmpr[i]) - log(trajectory_neg_cmpr[i]))
          for i in range(len(trajectory_neg_cmpr))
        ]
      )
      for trajectory_neg_cmpr in bundle_neg_cmpr
    ]

    prob_pos = np.mean([exp(-gamma * d) for d in dists_pos])
    prob_neg = np.mean([exp(-gamma * d) for d in dists_neg])
  else:
    prob_pos = np.mean( [ exp(-gamma * abs(log(trajectory_test[idx]) - \
                          log(bundle_pos[t].values[idx])))
                          for t in bundle_pos ] )
    prob_neg = np.mean( [ exp(-gamma * abs(log(trajectory_test[idx]) - \
                          log(bundle_neg[t].values[idx])))
                          for t in bundle_neg ] )

  return prob_pos / prob_neg

#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
def viz_timeseries(ts_infos):
  """
  for (i, ts_info) in enumerate(ts_infos):
    ts_infos[i] = copy.deepcopy(ts_info)
    topics = ts_info.keys()
    for t in topics:
      if np.random.rand() < 0.9:
        ts_infos[i].pop(t)
  """

  colors = [(0,0,1), (1,0,0)]
  detection_window_time = 0.75 * 3600 * 1000
  ts_norm_func = ts_mean_median_norm_func(0, 1)
  bundles = {}
  for (i, ts_info) in enumerate(ts_infos):
    # Normalize.
    ts_info = ts_normalize(ts_info, ts_norm_func)
    # Create bundles.
    bundle = ts_bundle(ts_info, detection_window_time)
    bundles[i] = bundle
    # Plot.
    color = colors[i]
    for t in bundle:
      plt.semilogy(np.array(bundle[t].times) - bundle[t].tmin,
                   bundle[t].values, hold = 'on', linewidth = 1,
                   color = color)
  plt.show()

  plot_hist = True
  if plot_hist:
    for time in np.linspace(0, detection_window_time - 1, 20):
      for i in bundles:
        hist = []
        for topic in bundles[i]:
          ts = bundles[i][topic]
          idx = ts.dtime_to_dindex(time)
          if ts.values[idx] == 0:
            print 'zero encountered', topic
          else:
            hist.append(log(ts.values[idx]))
        
        n, bins, patches = plt.hist(hist, bins = 25, color = colors[i],
                                    normed = True, hold = 'on')
        plt.setp(patches, 'facecolor', colors[i], 'alpha', 0.25)
      
      plt.title(str((detection_window_time - time) / (60 * 1000)) + \
                    ' minutes before onset')
      plt.show()

def build_gexf(edges, out_name, p_sample = 1):
  gexf = Gexf("snikolov", out_name)
  graph = gexf.addGraph('directed', 'dynamic', out_name)
  end = str(max([edge[2] for edge in edges]))
  for (src, dst, time) in edges:
    if np.random.rand() > p_sample:
      continue
    # Assumes time is in epoch seconds
    #d = datetime.datetime.fromtimestamp(int(time))    
    #date = d.isoformat()
    start = str(time)
    if not graph.nodeExists(src):
      graph.addNode(id = src, label = '', start = start, end = end)
    if not graph.nodeExists(dst):
      graph.addNode(id = dst, label = '', start = start, end = end)
    graph.addEdge(id = str(src) + ',' + str(dst), source = src,
                  target = dst, start = start, end = end)
  out = open('data/graphs/' + out_name + '.gexf', 'w')
  gexf.write(out)
  out.close()
