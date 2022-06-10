import cudf
import cugraph

import sys

import numpy as np
import pickle

from tqdm import tqdm
import os
import argparse

try:
    with open('../pretrains/graph_test.pkl', 'rb') as handle:
        saved_dict = pickle.load(handle)
except:
    import pickle5 as pickle
    with open('../pretrains/graph_test.pkl', 'rb') as handle:
        saved_dict = pickle.load(handle)

edges = saved_dict['edges']
xyz = saved_dict['xyz']
rgb = saved_dict['rgb']
pivots_inds = saved_dict['pivots_inds']
pivots = saved_dict['pivots']

n_pivots = pivots.shape[0]
gdf = cudf.DataFrame()
# print(edges.shape)
gdf['src'] = edges[:, 0].astype(int)
gdf['dst'] = edges[:, 1].astype(int)
gdf["data"] = edges[:, 2]

# print(np.max(edges[:, 0]), np.max(edges[:, 1]))
num_vertice = len(np.unique(edges[:, 0:2]))
# st = time.time()
G = cugraph.Graph(symmetrized=True)

# print('edges', edges.shape, edges[:10, :])
G.from_cudf_edgelist(gdf, source='src', destination='dst', edge_attr='data')


# print(edges.shape)

# geo_distance = np.zeros((128, num_vertice-128)) + 10000
geo_dict = {}
for i, ind in enumerate(pivots_inds):
    # print(ind)
    df = cugraph.sssp(G, i)

    # print('num_vertices', num_vertices)
    df = cugraph.filter_unreachable(df)
    # print(df)
    # quit()
    dis = np.array(df['distance'].to_array())
    vertex = np.array(df['vertex'].to_array())
    # num_vertices = len(vertex)
    # 
    mask = (vertex >= n_pivots)
    vertex = vertex[mask] - n_pivots
    dis = dis[mask]

    # geo_dict = 
    geo_dict[i] = {
        'vertex': vertex,
        'dis': dis,
    }

save_scene_path = os.path.join('../pretrains/path_test.pkl')
with open(save_scene_path, 'wb') as handle:
    pickle.dump(geo_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)