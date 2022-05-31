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



def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """

    N, C = xyz.shape
    # print("DEBUG", N)
    centroids = np.zeros(npoint, dtype=np.int32)
    distance = np.ones(N) * 1e10
    farthest = np.random.choice(N, 1, False).astype(np.int32)[0]
    # print(farthest, xyz.shape, xyz[farthest, :].view(1,3))
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :].reshape(1, 3)
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return centroids

def main():
    res = faiss.StandardGpuResources()  # use a single GPU
    # index_flat = faiss.IndexFlatL2(3)  # build a flat (CPU) index
    geo_knn = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(3))


    radius = 0.02
    k = 16

    train_folder = '/home/ubuntu/fewshot3d_ws/SoftGroup/dataset/scannetv2/train'
    save_folder = '/home/ubuntu/fewshot3d_ws/SoftGroup/dataset/scannetv2/graph'
    os.makedirs(save_folder, exist_ok=True)

    input_files = glob.glob(train_folder + '/scene*')
    # input_file = '/home/tuan/workspace/SoftGroup/dataset/scannetv2/val/scene0046_01_inst_nostuff.pth'
    for input_file in tqdm(input_files):
        xyz, rgb, _, _ = torch.load(input_file)
        scene_name = os.path.basename(input_file)[:12]

        edges = []
        n_points = xyz.shape[0]
        n_pivots = 256

        pivots_inds = farthest_point_sample(xyz, n_pivots)
        pivots = xyz[pivots_inds]
        
        geo_knn.add(xyz) 
        D_geo, I_geo = geo_knn.search(pivots, 2)  # actual search

        # print(D_geo[:, 0])
        D_geo = D_geo[:, 1:]
        I_geo = I_geo[:, 1:]
        D_geo = np.sqrt(D_geo)
        D_geo[D_geo >= radius] = 100

        pivots_edges = np.concatenate([np.arange(n_pivots)[:, None], I_geo + n_pivots, D_geo], axis=-1)
        # pivots_edges = []
        # for i, p in enumerate(pivots_inds):
        #     pivots_edges.append(np.array([p, I_geo[i, 0], D_geo[i, 0]]))

        #     for j in range(1,k-1):
        #         if D_geo[i,j] < radius:
        #             pivots_edges.append(np.array([p, I_geo[i, j], D_geo[i, j]]))




        # we want to see 3 nearest neighbors
        # geo_knn.add(xyz) 
        D_geo, I_geo = geo_knn.search(xyz, k)  # actual search
        D_geo = D_geo[:, 1:]
        I_geo = I_geo[:, 1:]
        D_geo = np.sqrt(D_geo)
        # D[D >= radius] = 100

        # we want to see 3 nearest neighbors
        # color_knn.add(rgb)
        # D_rgb, I_rgb = color_knn.search(rgb, 4)  # actual search
        # D_rgb = D_rgb[:, 1:]
        # I_rgb = I_rgb[:, 1:]
        # D_rgb = np.sqrt(D_rgb)
        # D[D >= radius] = 100

        for src_idx in range(n_points):
            # if src_idx in pivots_inds:
            #     continue

            radius_mask = (D_geo[src_idx] <= radius)

            src_rgb = rgb[[src_idx]] # 3
            neighbors_rgb = rgb[I_geo[src_idx]] # N, 3
            color_diff = np.sqrt(np.sum(np.abs(src_rgb - neighbors_rgb)**2, axis=-1)) # N
            # print(np.mean(color_diff))
            color_mask = (color_diff <= 0.05)

            radius_mask = ((D_geo[src_idx] <= 0.02) & (color_diff <= 0.2)) | ((color_diff <= 0.1) & (D_geo[src_idx] <= 0.04))

            # if np.count_nonzero(D_geo[src_idx]==-1) or np.count_nonzero(I_geo[src_idx]==-1):
            #     print('bug')
            count_valid = np.count_nonzero(radius_mask)
            if count_valid == 0:
                temp = np.array([[src_idx+n_pivots, I_geo[src_idx,0]+n_pivots, 100]])
            else:
                temp = np.zeros((count_valid,3))
                temp[:, 0] = src_idx + n_pivots
                temp[:, 1] = I_geo[src_idx][radius_mask] + n_pivots
                temp[:, 2] = D_geo[src_idx][radius_mask]

            edges.append(temp)


        edges = np.concatenate(edges, axis=0)
        edges = np.concatenate([edges, pivots_edges], axis=0)
        
        saved_dict = {
            'edges': edges,
            'xyz': xyz,
            'rgb': rgb,
            'pivots_inds': pivots_inds,
            'pivots': pivots
        }
        saved_path = os.path.join(save_folder, scene_name + '.pkl')
        with open(saved_path, 'wb') as handle:
            pickle.dump(saved_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        geo_knn.reset()

        # print('Graph save to', saved_path)

if __name__ == '__main__':
    main()