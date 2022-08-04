import os, sys, glob, math, numpy as np
import torch

from typing import Dict, List, Sequence, Tuple, Union
import open3d as o3d
import torch_scatter

sys.path.append('../')

import time

import segmentator
import shutil

def get_superpoint(mesh_file):
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
    faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
    superpoint = segmentator.segment_mesh(vertices, faces)
    # self.superpoints[scene] = superpoint
    print(mesh_file, superpoint.shape)
    return superpoint


    # NOTE copy scans_test
    # test_scenes = glob.glob('/vinai-public-dataset/ScanNetv2/scans_test/scene*')

    # for scene in test_scenes:
    #     scene_name = scene.split('/')[-1]
    #     # ply_files = glob.glob(scene + '/*_vh_clean_2.ply')
    #     ply_file = scene + '/' + scene_name + '_vh_clean_2.ply'

    #     assert os.path.exists(ply_file)
    #     print(ply_file)
    #     dst_file = os.path.join('/home/ubuntu/DATASET/scannetv2/scans_test', scene_name + '_vh_clean_2.ply')
    #     shutil.copy(ply_file, dst_file)

files = glob.glob('/home/ubuntu/3dis_ws/SoftGroup/dataset/scannetv2/scans_test/*_vh_clean_2.ply')

for file in files:
    spp = get_superpoint(file)

    scene_name = file.split('/')[-1][:12]

    save_path = os.path.join('/home/ubuntu/3dis_ws/SoftGroup/dataset/scannetv2/superpoints', scene_name+'.pth')
    assert not os.path.exists(save_path)
    torch.save(spp, save_path)
    # np.save(os.path.join('/home/ubuntu/3dis_ws/SoftGroup/dataset/scannetv2/superpoints', scene_name+'.npy'), spp)

# spp = torch.load('/home/ubuntu/3dis_ws/SoftGroup/dataset/scannetv2/superpoints/scene0000_00.pth')
# # print(torch.unique(spp, return_counts=True))
# n_points = spp.shape[0]

# ssp_unique, ssp_ids, spp_len = torch.unique(spp, return_inverse=True, return_counts=True)

# n_ssp = ssp_unique.shape[0]

# inst_mask = (torch.rand(n_points) > 0.5).int()

# sum_spp_inst = torch_scatter.scatter(inst_mask, ssp_ids) # n_ssp

# ssp_mask = (2*sum_spp_inst >= spp_len)

# refine_mask = ssp_mask[ssp_ids]
# print(ssp_mask.shape, refine_mask.shape)
# print(torch.count_nonzero(inst_mask), torch.count_nonzero(refine_mask))
# # print(n_points, n_ssp)
# # print(ssp_unique)