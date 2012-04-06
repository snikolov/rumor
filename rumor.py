import parsing
import processing
import os

from constants import *

"""
%load_ext autoreload
%autoreload 2
"""

def setup_rumor(path, p_sample=1, rumor_window=WINDOW):
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
  
  """
  (statuses, edges) = parsing.parse_statuses_edges_in_window(statuses_path,
    edges_path, p_sample, tmin, tmax)
  """
  # Compute rumor edges and sort them by timestamp.
  rumor_edges = processing.compute_rumor_edges(statuses, edges, rumor_window)
  trend_onset = parsing.parse_trend_onset(trend_onset_path)
  return { 'statuses':statuses, 'edges':rumor_edges, 'trend_onset':trend_onset }
