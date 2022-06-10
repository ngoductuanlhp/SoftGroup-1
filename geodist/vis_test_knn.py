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
SEMANTIC_NAMES = np.array([
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
    'picture', 'counter', 'desk', 'curtain', 'refridgerator', 'shower curtain', 'toilet', 'sink',
    'bathtub', 'otherfurniture'
])
CLASS_COLOR = {
    'unannotated': [0, 0, 0],
    'floor': [143, 223, 142],
    'wall': [171, 198, 230],
    'cabinet': [0, 120, 177],
    'bed': [255, 188, 126],
    'chair': [189, 189, 57],
    'sofa': [144, 86, 76],
    'table': [255, 152, 153],
    'door': [222, 40, 47],
    'window': [197, 176, 212],
    'bookshelf': [150, 103, 185],
    'picture': [200, 156, 149],
    'counter': [0, 190, 206],
    'desk': [252, 183, 210],
    'curtain': [219, 219, 146],
    'refridgerator': [255, 127, 43],
    'bathtub': [234, 119, 192],
    'shower curtain': [150, 218, 228],
    'toilet': [0, 160, 55],
    'sink': [110, 128, 143],
    'otherfurniture': [80, 83, 160]
}

with open('/home/tuan/workspace/SoftGroup/pretrains/test_knn_pred.pkl', 'rb') as handle:
    save_dict = pickle.load(handle)

fig = mlab.figure(figure=None, bgcolor=(1.0, 1.0, 1.0), size=((800, 800)))


pivot = save_dict['pivots']
xyz = save_dict['xyz']
max_pivot_inds = save_dict['max_pivot_inds']
group_labels = save_dict['group_labels']


mlab.points3d(pivot[:, 0],pivot[:, 1],pivot[:, 2], color=(1,0,0), mode='sphere', scale_factor=0.1, figure=fig)

# unique_inds = np.unique(max_pivot_inds)

# for ind in unique_inds:
#     if ind >= 0:
#         xyz_local = xyz[max_pivot_inds==ind]


#         color_local = tuple(COLOR_DETECTRON2[ind*10])
#         mlab.points3d(xyz_local[:, 0],xyz_local[:, 1],xyz_local[:, 2], color=color_local, mode='sphere', scale_factor=0.05, figure=fig)

unique_labels = np.unique(group_labels)
for ind in unique_labels:
    if ind >= 0:
        xyz_local = xyz[group_labels==ind]


        # color_local = tuple(COLOR_DETECTRON2[ind*10])?
        print(ind)
        color_local = tuple(np.array(CLASS_COLOR[SEMANTIC_NAMES[int(ind)]])/255.0) 
        mlab.points3d(xyz_local[:, 0],xyz_local[:, 1],xyz_local[:, 2], color=color_local, mode='sphere', scale_factor=0.05, figure=fig)

xyz_far = xyz[group_labels==-100]

color_local = tuple(COLOR_DETECTRON2[1])
mlab.points3d(xyz_far[:, 0],xyz_far[:, 1],xyz_far[:, 2], color=color_local, mode='sphere', scale_factor=0.05, figure=fig)
mlab.show()
