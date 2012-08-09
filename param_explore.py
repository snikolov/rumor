import rumor
import cloud

def detect(pos_path, neg_path, **kwargs):
  pos = rumor.parsing.parse_timeseries_from_file(cloud.files.getf(pos_path), {})
  neg = rumor.parsing.parse_timeseries_from_file(cloud.files.getf(neg_path), {})
  rumor.parsing.insert_timeseries_objects(pos)
  rumor.parsing.insert_timeseries_objects(neg)
  
  return rumor.processing.ts_shift_detect(pos, neg, **kwargs)  

jids = []
for i in xrange(5):
  jid = cloud.call(detect,
                   'statuses_news_rates_2m.tsv',
                   'statuses_nonviral_rates_2m.tsv',
                   cmpr_window = 70,
                   w_smooth = 100,
                   detection_window_hrs = 3,
                   gamma = 1,
                   p_sample = 0.25,
                   test_frac = 0.5,
                   _type = 'f2')
  jids.append(jid)

for jid in jids:
  print cloud.result(jid)
