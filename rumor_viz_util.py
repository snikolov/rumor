import os
import random
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import math
import colorsys

# What to add to convert from local time GMT time.
TIME_OFFSET_SECONDS = -5 * 60 * 60
WINDOW = 1 * 60 * 60

def setup_rumor(statuses_path, edges_path, p_sample=0.01):
  # TODO: Look for canonically-named pickles. Load them if they exist
  # Parse edges and statuses.
  edges = parse_edges_sampled(edges_path,p_sample)    
  print 'Done.', (len(edges))
  edge_nodes = [ edge[0] for edge in edges ]
  edge_nodes.extend([ edge[1] for edge in edges ])
  edge_nodes = set(edge_nodes)
  statuses = parse_statuses_edge_sampled(statuses_path, edge_nodes)
  # Compute rumor edges and sort them by timestamp.
  rumor_edges = compute_rumor_edges(statuses, edges)
  rumor_edges.sort(timestamped_edge_comparator)
  return { 'statuses':statuses, 'edges': rumor_edges }

def parse_edges_sampled(path, p):
  files = os.listdir(path)
  edges = []
  append = edges.append
  for fi, file in enumerate(files):
    if not fi % 25:
      print os.path.join(path,file), len(edges)
    f = open(os.path.join(path,file), 'r')
    line = f.readline()
    while line:
      if random.random() < p:
        edge = line_to_fields(line)
        append((edge[0], edge[1]))
      line = f.readline()
    f.close()
  return edges

def line_to_fields(line):
  clean_fields = []
  fields = re.split('\t', line)
  for field in fields:
    clean_fields.append(re.sub('\n', '', field))
  return clean_fields

def parse_edges_node_sampled(path, sample_nodes):
  files = os.listdir(path)
  edges = []
  append = edges.append
  for fi, file in enumerate(files):
    if not fi % 25:
      print os.path.join(path,file)
    f = open(os.path.join(path,file), 'r')
    line = f.readline()
    while line:
      edge = line_to_fields(line)
      if (edge[0] in sample_nodes and edge[1] in sample_nodes):
        append((edge[0], edge[1]))
      line = f.readline()
    f.close()
  return edges

def parse_edges(path):
  files = os.listdir(path)
  edges = []
  for fi, file in enumerate(files):
    if not fi % 25:
      print os.path.join(path,file)
    f = open(os.path.join(path,file), 'r')
    line = f.readline()
    while line:
      edge = line_to_fields(line)
      append((edge[0], edge[1]))
      line = f.readline()
    f.close()
  return edges
    
def parse_statuses_sampled(path, p):
  files = os.listdir(path)
  statuses = {}
  count_bad_lines = 0
  for fi, file in enumerate(files):
    if not fi % 25:
      print os.path.join(path, file)
    f = open(os.path.join(path, file), 'r')
    line = f.readline()
    while line:
      if random.random() < p:
        status_fields = line_to_fields(line)
        if len(status_fields) >= 2 and status_fields[1] != '':
          # Note: if more than one status per user, we just get the 
          # last one in the list.
          statuses[status_fields[1]] = tuple(status_fields)
        else:
          count_bad_lines += 1
      line = f.readline()
  return statuses

def parse_statuses_edge_sampled(path, sample_edge_nodes):
  files = os.listdir(path)
  statuses = {}
  for fi, file in enumerate(files):
    if not fi % 25:
      print os.path.join(path, file)
    f = open(os.path.join(path, file), 'r')
    line = f.readline()
    while line:
      status_fields = line_to_fields(line)
      if len(status_fields) >= 2 and status_fields[1] != '':
        if status_fields[1] in sample_edge_nodes:
          # Note: if more than one status per user, we just get the 
          # last one in the list.
          statuses[status_fields[1]] = tuple(status_fields)
      line = f.readline()
  return statuses

# An edge (v,u) is a rumor edge iff (u,v) is in edges (i.e. u follows v)
# and if t_u - t_v <= WINDOW 
def compute_rumor_edges(statuses, edges):
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
      t_v = datetime_to_epoch_seconds(status_v[3])
      t_u = datetime_to_epoch_seconds(status_u[3])
    except ValueError:
      print "Can't convert one or both of these to a timestamp:\n", status_v[3], '\n', status_u[3]
    if t_u - t_v <= WINDOW:
      rumor_edges.append((v, u, t_u))
  return rumor_edges

# Take statuses and edges sorted by timestamp and simulate the rumor
# forward in time.
def simulate(statuses, rumor_edges, step_mode = 'index', step = 1000, limit = 10000):
  components = {}

  # Figure
  plt.figure()

  # Time series
  node_to_component_id = {}
  max_sizes = []
  total_sizes = []
  component_nums = []
  entropies = []
  max_component_ratios = []
  timestamps = []

  if step_mode == 'time':
    min_time = min([ edge[2] for edge in rumor_edges ])
    next_time = min_time
  max_pos = limit

  for eid, edge in enumerate(rumor_edges):
    # print edge
    # print components
    # print node_to_component_id
    
    if edge[0] not in node_to_component_id and edge[1] not in node_to_component_id:
      # Create new component with id edge[0] 
      #  (i.e. first node belonging to that component)
      component_id = edge[0]
      # print 'Creating new component ', component_id, ' from ', edge[0], ' and ', edge[1]
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
        # print 'Merging\n', c0, ': ', components[c0], '\ninto\n', c1, ': ', components[c1], '\n'
        # raw_input('')
        for member in components[c0]:
          members.add(member)
          node_to_component_id[member] = c1
        components.pop(c0)
    
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
        next_time += step

    component_sizes = []
    # raw_input('======================================================'
    for cid, members in components.items():
      component_sizes.append(len(members))
      # print 'component ', cid, ' size: ', len(members)  
      # raw_input('-------------------')
    print eid, '\t', max(component_sizes), '\t', len(components)

    # Desc sort of component sizes
    component_sizes.sort()
    component_sizes.reverse()

    # Append to timeseries
    max_sizes.append(max(component_sizes))
    total_sizes.append(sum(component_sizes))
    component_nums.append(len(component_sizes))
    entropies.append(entropy(component_sizes))
    timestamps.append(edge[2])
    max_component_ratios.append(float(max(component_sizes))/sum(component_sizes))
    shifted_ind = np.linspace(1, 1 + len(component_sizes), len(component_sizes))

    if eid > 0:
      color = step_to_color(pos, max_pos)
      plt.subplot(331)
      plt.loglog(shifted_ind, component_sizes, color = color, hold = 'on')
      plt.title('Loglog desc component sizes')

      plt.subplot(332)
      plt.semilogy(timestamps[-1], max_sizes[-1], 'ro', color = color, hold = 'on')
      plt.title('Max component size')

      plt.subplot(333)
      plt.semilogy(timestamps[-1], total_sizes[-1], 'ro', color = color, hold = 'on')
      plt.title('Total network size')

      plt.subplot(334)
      plt.plot(timestamps[-1], entropies[-1], 'go', color = color, hold = 'on')
      plt.title('Entropy of desc component sizes')

      plt.subplot(335)
      plt.semilogy(timestamps[-1], component_nums[-1], 'ko', color = color, hold = 'on')
      plt.title('Number of components')

      plt.subplot(336)
      plt.loglog(shifted_ind, np.cumsum(component_sizes), color = color, hold = 'on')
      plt.title('Cum. sum. of desc component sizes')

      plt.subplot(337)
      plt.plot(timestamps[-1], max_component_ratios[-1], 'ko', color = color, hold = 'on')
      plt.title('Max comp size / Total network Size')

    # plt.hist(component_sizes, np.linspace(0.5, 15.5, 15))
    # plt.plot(np.cumsum(np.histogram(component_sizes, bins = np.linspace(0.5, 15.5, 15))[0]), hold = 'on')
    plt.pause(0.001)
  plt.show()
  return components

def step_to_color(pos, max_pos):
  p = float(pos) / max_pos
  return colorsys.hsv_to_rgb(0.75*p, .75, 1)

def datetime_to_epoch_seconds(t):
  return time.mktime(time.strptime(t, '%Y-%m-%d %H:%M:%S')) + TIME_OFFSET_SECONDS

def timestamped_edge_comparator(e1, e2):
  # Timestamps are always a whole number of seconds.
  return int(e1[2]) - int(e2[2])

def entropy(x):
  sx = sum(x)
  p = [ float(xi) / sx for xi in x ]
  total = 0
  for pi in p:
    if pi > 0:
      total += pi * math.log(pi)
  return -total
