import pickle
import numpy as np

import sys
# sys.path.append( './' )
import faiss                     # make faiss available
import faiss.contrib.torch_utils
from tqdm import tqdm
import argparse
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import glob
# np.random.seed(1234)
import time

def knn():
    res = faiss.StandardGpuResources()  # use a single GPU
    # index_flat = faiss.IndexFlatL2(3)  # build a flat (CPU) index
    # geo_knn = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(3))
    geo_knn = faiss.IndexFlatL2(3)


    input_file = '/home/ubuntu/fewshot3d_ws/SoftGroup/dataset/scannetv2/train/scene0000_00_inst_nostuff.pth'
    # for input_file in tqdm(input_files):
    xyz, rgb, semantic, _ = torch.load(input_file)
    scene_name = os.path.basename(input_file)[:12]
    xyz = torch.from_numpy(xyz)

    points_dict = torch.load('/home/ubuntu/fewshot3d_ws/SoftGroup/dataset/scannetv2/points/points20')

    pivots_num = 20
    # pivots_inds = np.random.choice(xyz.shape[0], pivots_num, False)
    # pivots_inds = torch.from_numpy(pivots_inds).cuda()
    

    # print(np.unique(semantic))
    # print(xyz.shape[0], np.count_nonzero(semantic==0), np.count_nonzero(semantic==1))
    fg_inds = np.nonzero(semantic > 1)[0]
    fg_inds = torch.from_numpy(fg_inds)
    # # print(fg_inds.shape)
    fg_num = fg_inds.shape[0]
    # fg_num = xyz.shape[0]

    xyz = xyz[fg_inds, :]
    rgb = rgb[fg_inds, :]
    semantic = semantic[fg_inds]

    rgb = torch.from_numpy(rgb)
    # print(rgb)
    pivots_inds = torch.Tensor(points_dict[scene_name]).long()

    new_pivot_inds = []
    for ind in pivots_inds:
        if ind in fg_inds:
            new_pivot_inds.append(torch.nonzero(fg_inds==ind)[0][0])
    pivots_inds = torch.stack(new_pivot_inds,-1).squeeze(-1)

    # print(pivots_inds)
    pivots = xyz[pivots_inds]
    pivots_labels = torch.from_numpy(semantic[pivots_inds])

    pivots_dists = torch.zeros((pivots_num, fg_num))
    pivots_visited = torch.zeros((pivots_num, fg_num), dtype=torch.bool)
    # pivots_labels = torch.ones((pivots_num, fg_num)) * -1
    

    geo_knn.add(xyz)


    D_geo, I_geo = geo_knn.search(xyz, 24)
    D_geo = D_geo[:, 1:] # n_points ,7
    I_geo = I_geo[:, 1:]

    D_geo = torch.sqrt(D_geo)

    # print(I_geo)
    # I_geo = torch.rand((fg_num, 7)).cuda()
    # points_dim0 = torch.arange(fg_num).repeat_interleave(7).cuda()
    # # points_dim0 = points_dim0. # n_points * 7
    # I_geo = I_geo.reshape(-1)

    # indices = torch.stack([points_dim0, I_geo], dim=0)
    # values = torch.ones((indices.shape[1])).cuda()
    # for i in range(fg_num):
    #     temp = torch
    #     indices.append()
    # points_selfmat = torch.sparse_coo_tensor(indices, values, (fg_num, fg_num))

    # points_pivotmat = torch.zeros((fg_num, pivots_num)).to_sparse().cuda()

    total_step = 1

    t_start = time.time()
    for p in range(pivots_inds.shape[0]):
        indices = I_geo[pivots_inds[p], :]

        pivot_rgb = rgb[[pivots_inds[p]]]
        for i in range(total_step):
            # print(i)
            # print(indices)

            check_visited = pivots_visited[p, indices]
            # print(indices, check_visited, check_visited.shape)
            # quit()
            # print(indices.shape, check_visited.shape)
            new_indices1 = indices[(check_visited==0).type(torch.bool)]

            # new_indices = indices[check_visited.type(torch.bool)]

            # print(indices[[True,True,True,False,False,False,True,True,True]])
            # print(indices, check_visited)
            # print(torch.masked_select(indices, check_visited))
            # new_indices = indices[[True,True,True,False,False,False,True,True,True]]
            # new_indices = indices[[0,2]].clone()
            # print(new_indices, new_indices.shape, check_visited)

            # print(new_indices)

            # print(indices)
            unique_indices = torch.unique(new_indices1)

            
            # p0rint('unique_indices', unique_indices.shape)
            pivots_dists[p, unique_indices] = total_step - i
            pivots_visited[p, unique_indices] = 1
            # pivots_dists.scatter_(dim=1, index=unique_indices, value=1)

            new_indices = I_geo[unique_indices,:].reshape(-1)

            # print('prev', new_indices.shape)
            new_indices_dis = D_geo[unique_indices,:].reshape(-1)

            prev_rgb = torch.repeat_interleave(rgb[unique_indices], 23, dim=0) # prev, 3
            post_rgb = rgb[new_indices] # post, 3
            # new_indices_rgb = rgb[new_indices, :]
            color_diff = torch.sqrt(torch.sum((post_rgb - prev_rgb)**2, axis=-1)) # N
            # print(torch.mean(color_diff))
            valid_new_indices = (new_indices_dis < 0.04) & (color_diff < 0.1)
            # valid_new_indices = ((new_indices_dis <= 0.04) & (color_diff <= 0.4)) | ((color_diff <= 0.2) & (new_indices_dis <= 0.08))
            new_indices = new_indices[valid_new_indices]


            # print('post', new_indices.shape)

            indices = torch.cat([new_indices, unique_indices])

            # if i == 127:
            #     print(indices.shape, unique_indices.shape)
        
        # points_pivotmat = torch.sparse.mm(points_selfmat, points_pivotmat)
    
    t_end = time.time()

    print('time', t_end - t_start)
    print('pivots_dists', pivots_dists.shape)
    max_dist, max_pivot_inds = torch.max(pivots_dists, dim=0)

    print(max_pivot_inds.shape, pivots_labels.shape)
    # max_pivot_inds[max_dist==0] = -1

    group_labels = pivots_labels[max_pivot_inds]
    group_labels[max_dist==0] = -100

    # print(group_labels.shape, semantic_scores_b_.shape)
    uni_label, count = torch.unique(max_pivot_inds, return_counts=True)
    print(uni_label, count)

    save_dict = {
        # 'rgb': rgb.cpu().numpy()
        'xyz': xyz.cpu().numpy(),
        'pivots': pivots.cpu().numpy(),
        'max_pivot_inds': max_pivot_inds.cpu().numpy(),
        'group_labels': group_labels.cpu().numpy(),
    }


    with open('test_knn.pkl', 'wb') as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # print(torch.unique(max_pivot_inds))
    # for p in range(pivots_num):
    #     mask = (max_dist > 0) & (max_pivot_inds == p)

    # points_selfmat.scatter_(dim=0, index=I_geo, src=1)

    # while True:
    #     for p in range(pivots_num):
    #         pivots_dists[p, I_geo[pivots[p]]] += D_geo[pivots[p]]

knn()