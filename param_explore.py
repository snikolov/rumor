import rumor
import cloud

def detect(pos_path, neg_path, **kwargs):
  pos = rumor.parsing.parse_timeseries_from_file(cloud.files.getf(pos_path), {})
  neg = rumor.parsing.parse_timeseries_from_file(cloud.files.getf(neg_path), {})
  rumor.parsing.insert_timeseries_objects(pos)
  rumor.parsing.insert_timeseries_objects(neg)

  jid = cloud.call(rumor.processing.ts_shift_detect, pos, neg, **kwargs)  
  return cloud.result(jid)

jid = cloud.call(detect, 'statuses_news_rates_2m', 'statuses_nonviral_rates_2m',
                 cmpr_window = 100,
                 w_smooth = 80,
                 detection_step = 15,
                 min_dist_step = 10,
                 detection_window_hrs = 5,
                 cmpr_step = 10,
                 gamma = 1,
                 p_sample = 0.15,
                 test_frac = 0.1,
                 _type = 'm1')

print cloud.result(jid)
