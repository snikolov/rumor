#!/usr/bin/python

import rumor
import sys

ts_viral = rumor.parsing.parse_timeseries('data/' + sys.argv[1])
ts_nonviral = rumor.parsing.parse_timeseries('data/' + sys.argv[2])

rumor.processing.detect(ts_viral, ts_nonviral)
