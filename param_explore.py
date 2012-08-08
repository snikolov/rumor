import rumor
import cloud

def detect(pos_path, neg_path, **kwargs):
  pos = rumor.parsing.parse_timeseries_from_file(cloud.files.getf(pos_path), {})
  neg = rumor.parsing.parse_timeseries_from_file(cloud.files.getf(neg_path), {})
  rumor.parsing.insert_timeseries_objects(pos)
  rumor.parsing.insert_timeseries_objects(neg)
  return rumor.processing.ts_shift_detect(pos, neg, **kwargs)  

jid = cloud.call(detect,
                 'statuses_news_rates_2m.tsv',
                 'statuses_nonviral_rates_2m.tsv',
                 cmpr_window = 50,
                 w_smooth = 80,
                 detection_step = 1,
                 min_dist_step = 1,
                 detection_window_hrs = 2,
                 cmpr_step = 1,
                 gamma = 1,
                 p_sample = 1,
                 test_frac = 0.75,
                 _type = 'f2')

print cloud.result(jid)
