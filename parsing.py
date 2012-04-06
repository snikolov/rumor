import os
import random
import re
import time
import util

def parse_trend_onset(path):
  try:
    f = open(path, 'r')
    line = f.readline()
    return int(line)
  except IOError:
    return None

# Parses edges from the file at path.
def parse_edges_sampled(path, p):
  files = os.listdir(path)
  edges = []
  append = edges.append
  
  # Determine the earliest time
  for fi, file in enumerate(files):
    if not re.match('part-.-[0-9]{5}',file):
      print 'Filename', file, 'not valid data'
      continue
    if os.path.isdir(file):
      print 'File', file, 'is directory'
      continue
    f = open(os.path.join(path,file), 'r')
    line = f.readline()
    while line:
      if random.random() < p:
        if not len(edges) % 25000:
          print os.path.join(path,file), len(edges)
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
    f = open(os.path.join(path,file), 'r')
    line = f.readline()
    while line:
      if not len(edges) % 25000:
        print os.path.join(path,file), len(edges)
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
    f = open(os.path.join(path,file), 'r')
    line = f.readline()
    while line:
      if not len(edges) % 25000:
        print os.path.join(path,file), len(edges)
      edge = line_to_fields(line)
      append((edge[0], edge[1]))
      line = f.readline()
    f.close()
  return edges
    
def parse_statuses_edges_in_window(statuses_dir, edges_dir, p, tmin, tmax):
  # Parse statuses within time window [tmin,tmax]. Note that these are relative
  # to the minimum time across all statuses. Statuses are assumed sorted in
  # ascending order.
  statuses_files = os.listdir(statuses_dir)
  statuses = {}
  for fi, file in enumerate(statuses_files):
    f = open(os.path.join(statuses_dir,file), 'r')
    line = f.readline()
    min_time = None
    while line:
      # Hacky way to quickly detect if line starts with a valid id
      status_fields = line_to_fields(line)
      if not re.match('[0-9]{18}',status_fields[0]):
        print status_fields[0], '======================'
        print 'Skipping', line
        raw_input()
        line = f.readline()
        continue
      try: 
        timestamp = util.datetime_to_epoch_seconds(status_fields[3])
      except ValueError:
        print "Can't convert this to a timestamp:", status_fields[3]
        print "Whole line:", line
        line = f.readline()
        continue
      if min_time is None:
        min_time = timestamp
      rel_time = timestamp - min_time
      if rel_time < tmin:
        print 'time too low'
        line = f.readline()
        continue
      if rel_time > tmax:
        print 'time too high'
        break
      if not len(statuses) % 10:
        print os.path.join(statuses_dir,file), len(statuses), (rel_time / 3600)
      statuses[status_fields[1]] = tuple(status_fields)
      line = f.readline()
    f.close()

  # Based on the parsed statuses, sample a fraction p of the edges emanating
  # from those statuses.
  edges = []
  edges_files = os.listdir(edges_dir)
  for fi, file in enumerate(edges_files):
    f = open(os.path.join(edges_dir,file))
    line = f.readline()
    while line:
      if random.random() > p:
        line = f.readline()
        continue
      edge_fields = line_to_fields(line)
      if edge_fields[0] in statuses and edge_fields[1] in statuses:
        edges.append((edge_fields[0], edge_fields[1]))
      if not len(edges) % 250:
        pass #print os.path.join(edges_dir,file), len(edges)
      line = f.readline()
    f.close()
  return (statuses, edges)

def parse_statuses_sampled(path, p):
  files = os.listdir(path)
  statuses = {}
  count_bad_lines = 0
  for fi, file in enumerate(files):
    f = open(os.path.join(path, file), 'r')
    line = f.readline()
    while line:
      if random.random() < p:
        status_fields = line_to_fields(line)
        if len(status_fields) >= 2 and status_fields[1] != '':
          if not len(statuses) % 25000:
            print os.path.join(path, file), len(statuses)
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
    f = open(os.path.join(path, file), 'r')
    line = f.readline()
    while line:
      status_fields = line_to_fields(line)
      if len(status_fields) >= 2 and status_fields[1] != '':
        if status_fields[1] in sample_edge_nodes:
          if not len(statuses) % 25000:
            print os.path.join(path, file), len(statuses)
          # Note: if more than one status per user, we just get the 
          # last one in the list.
          statuses[status_fields[1]] = tuple(status_fields)
      line = f.readline()
  return statuses
