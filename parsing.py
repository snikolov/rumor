import os
import random
import re
import time

def parse_trend_onset(path):
  try:
    f = open(path, 'r')
    line = f.readline()
    return int(line)
  except IOError:
    return None

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
