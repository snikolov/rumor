from collections import namedtuple

"""
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
"""

# TODO: This __str__ overriding doesn't seem to work properly...
params_list = ['pos_path',
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
               'req_consec_detections']

class Params(namedtuple("Params", params_list)):
  __slots__ = ()
  @property
  def __str__(self):
    return 'threshold=' + str(self.threshold) + ',' + \
      'cmpr_window=' + str(self.cmpr_window) + ',' + \
      'w_smooth=' + str(self.w_smooth) + ',' + \
      'gamma=' + str(self.gamma) + ',' + \
      'detection_window_hrs=' + str(self.detection_window_hrs) + ',' + \
      'req_consec_detections=' + str(self.req_consec_detections)
