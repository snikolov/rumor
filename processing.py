import numpy as np
import matplotlib.pyplot as plt
import util

# An edge (v,u) is a rumor edge iff (u,v) is in edges (i.e. u follows v)
# and if t_u - t_v <= window
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
      print "Can't convert one or both of these to a timestamp:\n", status_v[3], '\n', status_u[3]
    t_diff = t_u - t_v
    if t_diff <= window and t_diff > 0:
      rumor_edges.append((v, u, t_u))
    elif -t_diff <= window and t_diff < 0:
      rumor_edges.append((u, v, t_v))

  rumor_edges.sort(util.timestamped_edge_comparator)
  return rumor_edges

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

  spikeset = set()

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
    
    # Pause when you have some number of repeat statuses in a row 
    # (meaning that lots of edges that terminate in that status suddenly got created)
    repeat_num = 50
    status_id = rumor_statuses[rumor_edges[eid][1]][0]
    if eid > repeat_num and last_k_statuses_equal(status_id, rumor_statuses, rumor_edges, eid, repeat_num) and status_id not in spikeset:
      print (rumor_statuses[rumor_edges[eid][0]], rumor_statuses[rumor_edges[eid][1]])
      spikeset.add(status_id)
      raw_input()

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

    time_after_onset = None
    if trend_onset is not None:
      time_after_onset = edge[2] - trend_onset

    print edge[2] - min_time, '\t\t', eid, '\t\t', pos, '/', limit, '\t\t', max(component_sizes), '\t\t', len(components), '\t\t', time_after_onset
    
    # Desc sort of component sizes
    component_sizes.sort()
    component_sizes.reverse()

    # Append to timeseries
    max_sizes.append(max(component_sizes))
    total_sizes.append(sum(component_sizes))
    component_nums.append(len(component_sizes))
    entropies.append(util.entropy(component_sizes))
    timestamps.append((edge[2] - min_time) / (60 * 60))
    max_component_ratios.append(float(max(component_sizes))/sum(component_sizes))
    shifted_ind = np.linspace(1, 1 + len(component_sizes), len(component_sizes))

    if eid > 0:
      color = util.step_to_color(pos, max_pos)
      plt.subplot(331)
      plt.loglog(shifted_ind, component_sizes, color = color, hold = 'on')
      plt.title('Loglog desc component sizes')

      plt.subplot(332)
      plt.semilogy(timestamps[-1], max_sizes[-1], 'ro', color = color, hold = 'on')
      plt.title('Max component size')
      plt.xlabel('time (hours)')

      plt.subplot(333)
      plt.semilogy(timestamps[-1], total_sizes[-1], 'ro', color = color, hold = 'on')
      plt.title('Total network size')
      plt.xlabel('time (hours)')

      plt.subplot(334)
      plt.plot(timestamps[-1], entropies[-1], 'go', color = color, hold = 'on')
      plt.title('Entropy of desc component sizes')
      plt.xlabel('time (hours)')

      plt.subplot(335)
      plt.semilogy(timestamps[-1], component_nums[-1], 'ko', color = color, hold = 'on')
      plt.title('Number of components')
      plt.xlabel('time (hours)')

      plt.subplot(336)
      plt.loglog(shifted_ind, np.cumsum(component_sizes), color = color, hold = 'on')
      plt.title('Cum. sum. of desc component sizes')

      plt.subplot(337)
      plt.plot(timestamps[-1], max_component_ratios[-1], 'ko', color = color, hold = 'on')
      plt.title('Max comp size / Total network Size')
      plt.xlabel('time (hours)')

    # plt.hist(component_sizes, np.linspace(0.5, 15.5, 15))
    # plt.plot(np.cumsum(np.histogram(component_sizes, bins = np.linspace(0.5, 15.5, 15))[0]), hold = 'on')
    if not eid % 15*step:
      pass#plt.pause(0.001)
  plt.show()
  return components

def last_k_statuses_equal(equals_val, rumor_statuses, rumor_edges, curr_idx, k):
  for i in xrange(k):
    if rumor_statuses[rumor_edges[curr_idx-i][1]][0] is not equals_val:
      return False
  return True
