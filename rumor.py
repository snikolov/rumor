import parsing
import processing
import os

from constants import *

def setup_rumor(path, p_sample=0.01, window=WINDOW):
  statuses_path = os.path.join(path,'statuses')
  edges_path = os.path.join(path, 'edges')
  trend_onset_path = os.path.join(path, 'trend_onset.txt')
  # TODO: Look for canonically-named pickles. Load them if they exist
  # Parse edges and statuses.
  edges = parsing.parse_edges_sampled(edges_path,p_sample)    
  print 'Parsed', (len(edges)), 'edges'
  edge_nodes = [ edge[0] for edge in edges ]
  edge_nodes.extend([ edge[1] for edge in edges ])
  edge_nodes = set(edge_nodes)
  statuses = parsing.parse_statuses_edge_sampled(statuses_path, edge_nodes)
  print 'Parsed', len(statuses), 'statuses'
  # Compute rumor edges and sort them by timestamp.
  rumor_edges = processing.compute_rumor_edges(statuses, edges, window)
  trend_onset = parsing.parse_trend_onset(trend_onset_path)
  return { 'statuses':statuses, 'edges':rumor_edges, 'trend_onset':trend_onset }
