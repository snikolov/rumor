from collections import namedtuple

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
