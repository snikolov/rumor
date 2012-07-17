#!/usr/bin/python

import rumor

ts_topic = rumor.parsing.parse_timeseries('data/topic_rates')
ts_nonviral = rumor.parsing.parse_timeseries('data/nonviral_bottom_rates')

rumor.processing.detect(ts_topic, ts_nonviral)
