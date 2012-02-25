import matplotlib

from rumor_viz_util import *

PATH_EDGES_POS = 'data/steve_jobs/pos/edges/'
PATH_STATUSES_POS = 'data/steve_jobs/pos/statuses/'
PATH_EDGES_NEG = 'data/steve_jobs/neg/edges/'
PATH_STATUSES_NEG = 'data/steve_jobs/neg/statuses/'

simulate(statuses, rumor_edges)

"""
import xmlrpclib

# Create an object to represent our server.                                                                                                                  
server_url = 'http://127.0.0.1:20738/RPC2'
server = xmlrpclib.Server(server_url)

server.ubigraph.clear()

# Create a graph                                                                                                                                             
for i in range(0,10):
    server.ubigraph.new_vertex_w_id(i)

# Make some edges                                                                                                                                            
for i in range(0,10):
    server.ubigraph.new_edge(i, (i+1)%10)
"""
