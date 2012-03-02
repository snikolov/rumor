import colorsys
import math
import time

from constants import *

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
