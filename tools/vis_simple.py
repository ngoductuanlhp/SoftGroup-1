import argparse
import os
from operator import itemgetter

import numpy as np
import torch
import math

import open3d as o3d


def main():
    input_file  = "/Users/tuan/Workspace/DATASETS/stpls3d/val/10_points_GTv3_080_inst_nostuff.pth"
    points, colors, label, inst_label = torch.load(input_file)

    print(points.shape)
    points = points[:,:3]

    colors = (colors + 1) * 127.5
    colors = colors / 255.0

    m = np.eye(3)
    theta = 0.35 * math.pi
    m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0],
                        [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(colors)

    

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    
    vis.add_geometry(pc)
    vis.get_render_option().point_size = 6
    vis.run()
    vis.destroy_window()

main()