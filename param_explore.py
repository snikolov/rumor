import cloud
import itertools
import numpy as np
import rumor

from collections import namedtuple
from cStringIO import StringIO

Params = namedtuple('Params',
                    ['pos_path',
                     'neg_path',
                     'threshold',
                     'test_frac',
                     'cmpr_window',
                     'cmpr_step',
                     'w_smooth',
                     'gamma',
                     'p_sample',
                     'detection_step',
                     'min_dist_step',
                     'detection_window_hrs',
                     'req_consec_detections'])

# TODO: use Params namedtuple rather than full argument list
def detect_trial(pos_path, neg_path, threshold, test_frac, cmpr_window, cmpr_step,
                  w_smooth, gamma, p_sample, detection_step, min_dist_step,
                  detection_window_hrs, req_consec_detections):
  ts_pos = rumor.parsing.parse_timeseries_from_file(cloud.files.getf(pos_path), {})
  ts_neg = rumor.parsing.parse_timeseries_from_file(cloud.files.getf(neg_path), {})
  rumor.parsing.insert_timeseries_objects(ts_pos)
  rumor.parsing.insert_timeseries_objects(ts_neg)

  tstep = ts_pos[ts_pos.keys()[0]]['ts'].tstep
  # It doesn't make sense for the comparison window to be as big or bigger
  # than the detection window.
  if cmpr_window >= detection_window_hrs * 3600 * 1000 / float(tstep):
    return None
  return rumor.processing.ts_shift_detect(ts_pos, ts_neg, threshold,
                                          test_frac, cmpr_window,
                                          cmpr_step, w_smooth, gamma,
                                          p_sample, detection_step,
                                          min_dist_step, detection_window_hrs,
                                          req_consec_detections)
  
def detect_trials(pos_path, neg_path, threshold, test_frac, cmpr_window, cmpr_step,
           w_smooth, gamma, p_sample, detection_step, min_dist_step,
           detection_window_hrs, req_consec_detections):
  
  trials = 2
  pos_path_ = [pos_path] * trials
  neg_path_ = [neg_path] * trials
  threshold_ = [threshold] * trials
  test_frac_ = [test_frac] * trials
  cmpr_window_ = [cmpr_window] * trials
  cmpr_step_ = [cmpr_step] * trials
  w_smooth_ = [w_smooth] * trials
  gamma_ = [gamma] * trials
  p_sample_ = [p_sample] * trials
  detection_step_ = [detection_step] * trials
  min_dist_step_ = [min_dist_step] * trials
  detection_window_hrs_ = [detection_window_hrs] * trials
  req_consec_detections_ = [req_consec_detections] * trials
  
  jids = cloud.map(detect_trial,
                   pos_path_,
                   neg_path_,
                   threshold_,
                   test_frac_,
                   cmpr_window_,
                   cmpr_step_,
                   w_smooth_,
                   gamma_,
                   p_sample_,
                   detection_step_,
                   min_dist_step_,
                   detection_window_hrs_,
                   req_consec_detections_,
                   _type = 'f2')

  params = Params(pos_path, neg_path, threshold, test_frac, cmpr_window,
                  cmpr_step, w_smooth, gamma, p_sample, detection_step,
                  min_dist_step, detection_window_hrs, req_consec_detections)  

  return params, jids

def write_results(paramsets, results):
  num_paramsets = len(paramsets)
  num_results = len(results)
  if num_results % num_paramsets:
    print 'Something\'s wrong here: %d resultsets from %d paramsets.' % \
      (num_results, num_paramsets) 
    return
  num_trials = num_results / num_paramsets
  results_iter = iter(results)

  for i in xrange(len(paramsets)):
    paramset = paramsets[i]
    print paramset
    resultset = []
    for j in xrange(num_trials):
      resultset.append(results_iter.next())
    
    fprs = [ result['fpr']
             for result in resultset
             if result['fpr']]
    tprs = [ result['tpr']
             for result in resultset
             if result['tpr']]
    mean_earlies = [ np.mean(result['earlies'])
                     for result in resultset
                     if len(result['earlies']) > 0 ]
    std_earlies = [ np.std(result['earlies'])
                    for result in resultset
                    if len(result['earlies']) > 0 ]
    mean_lates = [ np.mean(result['lates'])
                   for result in resultset
                   if len(result['lates']) > 0 ]
    std_lates = [ np.std(result['lates'])
                  for result in resultset
                  if len(result['lates']) > 0 ]
    print 'mean fpr: ', np.mean(fprs)
    print 'std fpr: ', np.std(fprs)
    print 'mean tpr: ', np.mean(tprs)
    print 'std tpr: ', np.std(tprs)
    print 'mean_earlies: ', [ v / float(3600 * 1000) for v in mean_earlies ]
    print 'std_earlies: ', [ v / float(3600 * 1000) for v in std_earlies ]
    print 'mean_lates: ', [ v / float(3600 * 1000) for v in mean_lates ]
    print 'std_lates: ', [ v / float(3600 * 1000) for v in std_lates ]

# Launch.
pos_path = ['statuses_news_rates_2m.tsv']
neg_path = ['statuses_nonviral_rates_2m.tsv']
threshold = [1,2] #[1,3]
test_frac = [0.5]
cmpr_window = [80] #[10, 80, 150]
cmpr_step = [None]
w_smooth = [80] #[10, 80, 150]
gamma = [1] #[0.1, 1, 10]
p_sample = [0.5]
detection_step = [None]
min_dist_step = [None]
detection_window_hrs = [5] #[3, 5, 7]
req_consec_detections = [1] #[1,3]

param_product = itertools.product(pos_path,
                                  neg_path,
                                  threshold,
                                  test_frac,
                                  cmpr_window,
                                  cmpr_step,
                                  w_smooth,
                                  gamma,
                                  p_sample,
                                  detection_step,
                                  min_dist_step,
                                  detection_window_hrs,
                                  req_consec_detections)

jids = cloud.map(detect_trials,
                 *zip(*param_product),
                 _type = 'f2')

params_sub_jids = cloud.result(jids)
params = [ elt[0] for elt in params_sub_jids ]
sub_jids = [ elt[1] for elt in params_sub_jids ]
print (params, cloud.result(sub_jids))
