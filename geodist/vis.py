# import pickle
import numpy as np
import mayavi.mlab as mlab
import pickle5 as pickle


color_r = np.linspace(0, 1, 5)
color_g = np.linspace(0, 1, 5)
color_b = np.linspace(0, 1, 5)

color_rgb = np.meshgrid(color_r, color_g, color_b)
# print(color_rgb)
color_rgb = np.stack(color_rgb, axis=-1).reshape(-1, 3)
# print(color_rgb.shape)
COLOR_DETECTRON2 = color_rgb
# scene_name = 'scene0700_01'
idx = range(0, 64, 1)

# for s in geo_dict.keys():

with open('../pretrains/graph_test.pkl', 'rb') as handle:
    saved_dict = pickle.load(handle)

with open('../pretrains/path_test.pkl', 'rb') as handle:
    geo_dist_pkl = pickle.load(handle)

# 
# scene = scene_info[scene_name]




# print('geo_distance', geo_distance.shape)

edges = saved_dict['edges']
xyz = saved_dict['xyz']
rgb = saved_dict['rgb']
pivots_inds = saved_dict['pivots_inds']
pivots = saved_dict['pivots']
# print(query.shape)

min_z = np.min(xyz[:, 2])
idx = [i for i in idx if xyz[pivots_inds[i],2] > min_z + 0.2]

geo_dist = []
max_len = xyz.shape[0]
for i, k in enumerate(geo_dist_pkl.keys()):
    geo = np.ones((max_len)) * 100
    geo[geo_dist_pkl[k]['vertex'].astype(np.long)] = geo_dist_pkl[k]['dis']
    geo_dist.append(geo)
geo_distance = np.stack(geo_dist, axis=0) # 128, N


# print(geo_dist_pkl[idx]['dis'])
sing_geo = geo_distance[idx] # N
pivot = pivots[idx]

# sort_inds = np.argsort(sing_geo)
# top_sort = sort_inds[0:5000]
# bot_sort = sort_inds[5000:]

# sing_eu = np.square(locs - query).sum(1)
# sort_inds = np.argsort(sing_eu)
# top_sort = sort_inds[0:5000]
# bot_sort = sort_inds[5000:]


# locs_filter = locs[top_sort]
mask = (sing_geo < 1.5) # P x N

mask = (np.sum(mask, 0) >= 1) # N

points_near = xyz[mask] # N


points_far = xyz[~mask] 


points_pivot_inds = np.ones((max_len)) * -1
points_pivot_dis = np.ones((max_len)) * 1000
for i in idx:
    single_geo = geo_distance[i] # N
    mask_valid = (single_geo < points_pivot_dis) & (single_geo < 1)
    # print(mask_valid.shape)
    points_pivot_dis[mask_valid] = single_geo[mask_valid]
    points_pivot_inds[mask_valid] = i


fig = mlab.figure(figure=None, bgcolor=(1.0, 1.0, 1.0), size=((800, 800)))

mlab.points3d(pivot[:, 0],pivot[:, 1],pivot[:, 2], color=(1,0,0), mode='sphere', scale_factor=0.05, figure=fig)


for i in idx:
    mask_points = (points_pivot_inds == i)
    points_neighbor = xyz[mask_points]
    color_p = tuple(COLOR_DETECTRON2[i%COLOR_DETECTRON2.shape[0]])
    mlab.points3d(points_neighbor[:,0],points_neighbor[:,1],points_neighbor[:,2], color=color_p, mode='sphere', scale_factor=0.02, figure=fig)
# locs_filter_dist = np.square(locs - query).sum(1)
# locs_filter_dist_mask = (locs_filter_dist < 2.5)
# locs_filter = locs[locs_filter_dist_mask, :]

# locs_nofilter = locs[~locs_filter_dist_mask, :]

# mlab.points3d(locs_filter[:,0],locs_filter[:,1],locs_filter[:,2], color=(0,1,0), mode='sphere', scale_factor=0.02, figure=fig)
# mlab.points3d(locs_nofilter[:,0],locs_nofilter[:,1],locs_nofilter[:,2], color=(0.3,0.3,0.3), mode='sphere', scale_factor=0.02, figure=fig)


# mlab.points3d(points_near[:,0],points_near[:,1],points_near[:,2], color=(0,1,0), mode='sphere', scale_factor=0.02, figure=fig)


# # locs_out_dist = np.square(locs_out - query).sum(1)
# # locs_out_dist_mask = (locs_out_dist < 4)
# # locs_out = locs_out[locs_out_dist_mask, :]

mlab.points3d(points_far[:,0],points_far[:,1],points_far[:,2], color=(171/255.0, 198/255.0, 230/255.0), mode='sphere', scale_factor=0.02, figure=fig)

mlab.show()
