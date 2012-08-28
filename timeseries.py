import numpy as np

class Timeseries:

  def __init__(self, times = None, values = None, ts_dict = None, tmin = None,
               tmax = None, tstep = None):
    if times is None or values is None:
      self.tmin = tmin
      self.tmax = tmax
      self.tstep = tstep
      self.times = [ tmin + i * tstep for i in range((tmax - tmin) // tstep) ] 
      self.values = [0] * len(self.times)
      for t in ts_dict:
        self.values[self.time_to_index(t)] = ts_dict[t]
    else:
      self.times = times
      self.values = values
      if len(times) > 1:
        self.tstep = times[1] - times[0]
      else:
        self.tstep = None
      self.tmin = times[0]
      self.tmax = times[-1]

  def time_to_index(self, t):
    return int((t - self.tmin) // self.tstep)

  def dtime_to_dindex(self, dtime):
    return self.time_to_index(dtime + self.tmin)

  # Return a new timeseries in the window defined by [start,end], or as much of
  # it as possible. If there is no data available, the result will be padded
  # with zeros.
  def ts_in_window(self, start, end):
    num_bins = int((end - start) // self.tstep)
    istart = 0
    iend = len(self.times)

    istart = max(min(self.time_to_index(start), len(self.times)), 0)
    iend = max(min(self.time_to_index(end), len(self.times)), 0)

    start = int(self.tstep * (start // self.tstep))
    end = int(self.tstep * (end // self.tstep))
    times = [0] * num_bins
    values = [0] * num_bins
    times = [ start + i * self.tstep for i in range(num_bins) ]

    if istart == 0:
      values[num_bins - iend + istart : num_bins] = self.values[istart : iend]
    elif iend == len(self.times):
      values[0 : iend - istart] = self.values[istart : iend]
    else:
      values[0 : num_bins] = self.values[istart : iend]
    return Timeseries(times = times, values = values)  

  def ddt(self):
    values = np.array(self.values)
    ddt_values = (values[1:-1] - values[0:-2]) # / self.tstep
    return Timeseries(self.times[:-2], list(ddt_values))

  def pow(self, p): 
    return Timeseries(self.times, np.array(self.values) ** p)

  def abs(self): 
    return Timeseries(self.times, np.abs(np.array(self.values)))

