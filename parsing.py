import random
import re
import os
import time
import util

from gexf import Gexf
from subprocess import call
from timeseries import *

#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
def parse_trend_onset(path):
  try:
    f = open(path, 'r')
    line = f.readline()
    return int(line)
  except IOError:
    return None

#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
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

#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
def line_to_fields(line):
  clean_fields = []
  fields = re.split('\t', line)
  for field in fields:
    clean_fields.append(re.sub('\n', '', field))
  return clean_fields

#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
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

#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
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

#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~    
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

#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
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

#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
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

#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
def parse_timeseries(path):
  files = os.listdir(path)
  topic_info = {}
  for file in files:
    if not re.match('part-.-[0-9]{5}',file):
      print 'Filename', file, 'not valid data. Skipping...'
      continue
    if os.path.isdir(file):
      print 'File', file, 'is directory. Skipping...'
      continue
    f = open(os.path.join(path,file), 'r')
    line = f.readline()
    while line:
      fields = line_to_fields(line)
      if len(fields) != 6:
        # print 'Bad line', line, '. Skipping...'
        line = f.readline()
        continue
      if any([field is '' for field in fields]):
        # print 'Bad line', line, '. Skipping...'
        line = f.readline()
        continue
      topic = fields[0]
      time = float(fields[1])
      trend_start = float(fields[2])
      trend_end = float(fields[3])
      ts_value = float(fields[4])
      if topic not in topic_info:
        topic_info[topic] = {}
        topic_info[topic]['ts_dict'] = {}
      if int(time) in topic_info[topic]['ts_dict']:
        topic_info[topic]['ts_dict'][int(time)] += ts_value
      else:
        topic_info[topic]['ts_dict'][int(time)] = ts_value
      topic_info[topic]['trend_start'] = trend_start
      topic_info[topic]['trend_end'] = trend_end
      line = f.readline()

  all_times = sorted(list(set([ t for l in [ times.keys() for times in [ topic_info[topic]['ts_dict'] for topic in topic_info] ] for t in l])))

  # Hardcode timestep for now
  tstep = 300000
  
  for ti, topic in enumerate(topic_info):
    topic_info[topic]['ts'] = Timeseries(ts_dict = topic_info[topic]['ts_dict'], 
                                         tmin = min(all_times),
                                         tmax = max(all_times) + tstep,
                                         tstep = tstep)    
  return topic_info

#=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
# Builds gexf graph from timestamped edges
def build_gexf(path, out_name, p_sample = 1):
  graphs = {}
  gexfs = {}
  call(['mkdir', 'data/graphs/' + out_name])
  files = os.listdir(path)
  for (fi, file) in enumerate(files):
    print file
    if not re.match('part-.-[0-9]{5}',file):
      print 'Filename', file, 'not valid data. Skipping...'
      continue
    if os.path.isdir(file):
      print 'File', file, 'is directory. Skipping...'
      continue
    f = open(os.path.join(path,file), 'r')
    line = f.readline()
    while line:
      if np.random.rand() > p_sample:
        line = f.readline()
        continue
      if np.random.rand() < 0.0001:
        print sorted([ (len(graphs[t]._nodes), len(graphs[t]._edges), t) for t in graphs ])
      fields = line_to_fields(line)
      if len(fields) != 12:
        # print 'Bad line', line, '. Skipping...'
        line = f.readline()
        continue
      dst_status_id = fields[0]
      src_status_id = fields[4]
      time = fields[5]
      topic = fields[8]

      # Clean non ASCII chars from topic name, since XML complains.
      topic = ''.join(c for c in topic if ord(c) < 128)

      if topic not in gexfs:
        gexfs[topic] = Gexf("Stanislav Nikolov", topic)
        graphs[topic] = gexfs[topic].addGraph("directed", "dynamic", topic)
      if not graphs[topic].nodeExists(src_status_id):
        graphs[topic].addNode(id = str(src_status_id), label = "",
                              start = time, end = "Infinity")
      if not graphs[topic].nodeExists(dst_status_id):
        graphs[topic].addNode(id = str(dst_status_id), label = "",
                              start = time, end = "Infinity")
      graphs[topic].addEdge(id = "(" + str(src_status_id) + "," + str(dst_status_id) + ")",
                            source = src_status_id,
                            target = dst_status_id,
                            start = time,
                            end = "Infinity")
      line = f.readline()

    for topic in gexfs.keys():
      clean_topic = ''.join([ c for c in topic if (c != '#' and c != ' ') ])
      out = open("data/graphs/" + out_name + "/" + clean_topic + ".gexf", "w")
      gexfs[topic].write(out)
      out.close()
