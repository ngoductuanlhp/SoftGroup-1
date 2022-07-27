import os, sys, glob, math, numpy as np
import torch

from typing import Dict, List, Sequence, Tuple, Union
import open3d as o3d
import torch_scatter

sys.path.append('../')

import time

import segmentator


def get_superpoint(mesh_file):
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
    faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
    superpoint = segmentator.segment_mesh(vertices, faces)
    # self.superpoints[scene] = superpoint
    print(mesh_file, superpoint.shape)
    return superpoint



# files = glob.glob('/home/ubuntu/3dis_ws/SoftGroup/dataset/scannetv2/scans/*_vh_clean_2.ply')

# for file in files:
#     spp = get_superpoint(file)

#     scene_name = file.split('/')[-1][:12]

#     torch.save(spp, os.path.join('/home/ubuntu/3dis_ws/SoftGroup/dataset/scannetv2/superpoints', scene_name+'.pth'))
#     # np.save(os.path.join('/home/ubuntu/3dis_ws/SoftGroup/dataset/scannetv2/superpoints', scene_name+'.npy'), spp)

spp = torch.load('/home/ubuntu/3dis_ws/SoftGroup/dataset/scannetv2/superpoints/scene0000_00.pth')
# print(torch.unique(spp, return_counts=True))
n_points = spp.shape[0]

ssp_unique, ssp_ids, spp_len = torch.unique(spp, return_inverse=True, return_counts=True)

n_ssp = ssp_unique.shape[0]

inst_mask = (torch.rand(n_points) > 0.5).int()

sum_spp_inst = torch_scatter.scatter(inst_mask, ssp_ids) # n_ssp

ssp_mask = (2*sum_spp_inst >= spp_len)

refine_mask = ssp_mask[ssp_ids]
print(ssp_mask.shape, refine_mask.shape)
print(torch.count_nonzero(inst_mask), torch.count_nonzero(refine_mask))
# print(n_points, n_ssp)
# print(ssp_unique)