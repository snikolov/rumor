#!/usr/bin/python

import colorsys
import numpy
import matplotlib.pyplot as plt
import os
import re
import time as clock

def step_to_color(pos, max_pos):
  p = float(pos) / max_pos
  return colorsys.hsv_to_rgb(0.75*p, .75, 1)

path = '../data/tech_rates'
files = os.listdir(path)
for file in files:
  if not re.match('part-.-[0-9]{5}',file):
    print 'Filename', file, 'not valid data. Skipping...'
    continue
  if os.path.isdir(file):
    print 'File', file, 'is directory. Skipping...'
    continue
  f = open(os.path.join(path,file), 'r')
  line = f.readline()
  topic_rates_map = {}
  while line:
    fields = re.split('\t',line)
    if len(fields) != 6:
      # print 'Bad line', line, '. Skipping...'
      line = f.readline()
      continue
    if any([field is '' for field in fields]):
      # print 'Bad line', line, '. Skipping...'
      line = f.readline()
      continue
    topic = fields[0]
    topic_time = float(fields[1])
    trending_start = float(fields[2])
    trending_end = float(fields[3])
    topic_count = int(fields[4])
    if topic not in topic_rates_map:
      topic_rates_map[topic] = {}
      topic_rates_map[topic]['times'] = {}
    topic_rates_map[topic]['times'][topic_time] = topic_count
    topic_rates_map[topic]['start'] = trending_start
    topic_rates_map[topic]['end'] = trending_end
    line = f.readline()

topic_times = sorted(list(set([ t for l in [ times.keys() for times in [ topic_rates_map[topic]['times'] for topic in topic_rates_map] ] for t in l])))
min_time = topic_times[0]
max_time = topic_times[-1]
bucket_size = float('inf')
for tidx in xrange(len(topic_times)-1):
  diff = topic_times[tidx + 1] - topic_times[tidx]
  if diff < bucket_size:
    bucket_size = diff
print min_time,max_time,bucket_size

time = numpy.linspace(start=min_time,stop=max_time,num=(max_time-min_time)//bucket_size+1)
print time
time_to_idx = dict([(time[i],i) for i in range(len(time))])
topic_rates = {}
plt.figure()

topic_rates_in_detection_window = {}

from_topic = 0
to_topic = 50

# Build the timeseries and extract a slice preceding trend onset.
for ti,topic in enumerate(topic_rates_map):
  trend_start_idx = 0
  trend_end_idx = len(time) - 1
  trend_start = topic_rates_map[topic]['start']
  trend_end = topic_rates_map[topic]['end']
  for t in sorted(topic_rates_map[topic]['times'].keys()):
    if t > trend_start and trend_start_idx == 0:
      trend_start_idx = time_to_idx[t]
    if t > trend_end and trend_end_idx == len(time) - 1:
      trend_end_idx = time_to_idx[t]
    if topic not in topic_rates:
      topic_rates[topic] = numpy.zeros(len(time))
    topic_rates[topic][time_to_idx[t]] = topic_rates_map[topic]['times'][t]

  detect_start_idx = int(max(0,trend_start_idx - 12 * 3600000 / bucket_size))

  if detect_start_idx == trend_start_idx:
    continue

  topic_rates_in_detection_window[topic] = topic_rates[topic][detect_start_idx:trend_start_idx]
  
  if ti < from_topic or ti > to_topic:
    continue

  plt.plot(time[0:trend_start_idx], topic_rates[topic][0:trend_start_idx], 'b')

  plt.plot(time[trend_start_idx:trend_end_idx],
           topic_rates[topic][trend_start_idx:trend_end_idx], 'r')

  plt.plot(time[trend_end_idx::], topic_rates[topic][trend_end_idx::], 'b')

  plt.axvline(x=topic_rates_map[topic]['start'], linestyle='--', color='k')

  plt.axvline(x=topic_rates_map[topic]['end'], linestyle='--', color='k')
  plt.xlabel('time')
  plt.title(topic)
  plt.show()

"""
topics_sorted = topic_rates_in_detection_window.keys()
topics_sorted.sort(lambda x,y: int(sum(topic_rates_in_detection_window[x]) - sum(topic_rates_in_detection_window[y])))
topics_sorted.reverse()

topic_volumes = [sum(topic_rates_in_detection_window[x]) for x in topics_sorted]
max_topic_vol = max(topic_volumes)
min_topic_vol = min(topic_volumes)

for ti, topic in enumerate(topics_sorted):
  if ti < from_topic or ti > to_topic:
    continue
  detect_window_cumul = numpy.cumsum(topic_rates_in_detection_window[topic])
  if max(detect_window_cumul) < 100:
    continue  
  plt.plot(detect_window_cumul,
           hold = 'on',
           linewidth = 2)
           # color = step_to_color(sum(topic_rates_in_detection_window[topic]) - min_topic_vol, max_topic_vol - min_topic_vol)
  plt.draw()

plt.legend(topics_sorted[from_topic:to_topic + 1], loc=2)
plt.xlabel('time')
plt.show()
"""
