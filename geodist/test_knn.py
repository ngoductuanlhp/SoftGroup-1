import pickle
import numpy as np

import sys
sys.path.append( './' )
import faiss                     # make faiss available
import faiss.contrib.torch_utils
from tqdm import tqdm
import argparse
import os
import torch
import glob
np.random.seed(1234)


def knn():
    res = faiss.StandardGpuResources()  # use a single GPU
    # index_flat = faiss.IndexFlatL2(3)  # build a flat (CPU) index
    geo_knn = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(3))


    input_file = '/home/tuan/workspace/SoftGroup/dataset/scannetv2/val/scene0046_01_inst_nostuff.pth'
    # for input_file in tqdm(input_files):
    xyz, rgb, semantic, _ = torch.load(input_file)
    scene_name = os.path.basename(input_file)[:12]

    # print(np.unique(semantic))
    # print(xyz.shape[0], np.count_nonzero(semantic==0), np.count_nonzero(semantic==1))
    fg_inds = np.nonzero(semantic > 1)[0]
    # print(fg_inds.shape)
    fg_num = fg_inds.shape[0]


    xyz = xyz[fg_inds, :]
    rgb = rgb[fg_inds, :]

    xyz = torch.from_numpy(xyz).cuda()

    pivots_num = 20
    pivots_inds = np.random.choice(xyz.shape[0], pivots_num, False)
    pivots_inds = torch.from_numpy(pivots_inds).cuda()
    pivots = xyz[pivots_inds]

    pivots_dists = torch.ones((pivots_num, fg_num)).cuda() * 1000
    pivots_labels = torch.ones((pivots_num, fg_num)).cuda() * -1

    geo_knn.add(xyz)


    D_geo, I_geo = geo_knn.search(xyz, 8)
    D_geo = D_geo[:, 1:] # n_points ,7
    I_geo = I_geo[:, 1:]

    points_dim0 = torch.arange(fg_num).repeat_interleave(7).cuda()
    # points_dim0 = points_dim0. # n_points * 7
    I_geo = I_geo.reshape(-1)

    indices = torch.stack([points_dim0, I_geo], dim=0)
    values = torch.ones((indices.shape[1])).cuda()
    # for i in range(fg_num):
    #     temp = torch
    #     indices.append()
    points_selfmat = torch.sparse_coo_tensor(indices, values, (fg_num, fg_num))

    points_pivotmat = torch.zeros((fg_num, pivots_num)).to_sparse().cuda()

    ans = torch.sparse.mm(points_selfmat, points_pivotmat)
    # points_selfmat.scatter_(dim=0, index=I_geo, src=1)

    # while True:
    #     for p in range(pivots_num):
    #         pivots_dists[p, I_geo[pivots[p]]] += D_geo[pivots[p]]

knn()