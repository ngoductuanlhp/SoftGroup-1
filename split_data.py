import os
import numpy as np
import shutil
# split_files = open('/home/ubuntu/Workspace/tuannd42-dev/3dis_ws/SoftGroup/dataset/scannetv2/scannetv2_train.txt', 'r')
# split_names = split_files.read().splitlines()

# total_samples = len(split_names)
# sub_samples = total_samples // 10

# rands = np.random.choice(total_samples, size=sub_samples, replace=False)

# val_split = sorted([split_names[i] for i in rands])
# # big_split = new_split

# # val_split =[n for n in split_names if 'Scene20' in n]

# with open('/home/ubuntu/Workspace/tuannd42-dev/3dis_ws/SoftGroup/dataset/scannetv2/scannetv2_trainsmall.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % name for name in val_split)

# train_split = ([s for s in split_names if s not in val_split])
# # big_split = new_split

# with open('/home/ubuntu/Workspace/tuannd42-dev/3dod_ws/MonoDETR/splits/vkitti_2/train.txt', 'w') as filehandle:
#     filehandle.writelines("%s\n" % name for name in train_split)

# print(f'train: {len(train_split)}')
# print(f'val: {len(val_split)}')

split_files = open('dataset/scannetv2/scannetv2_trainsmall.txt', 'r')
split_names = split_files.read().splitlines()

files = sorted([f'dataset/scannetv2/train/{s}_inst_nostuff.pth' for s in split_names])

dst_files = sorted([f'dataset/scannetv2/trainsmall/{s}_inst_nostuff.pth' for s in split_names])

for i in range(len(dst_files)):
    shutil.copy(files[i], dst_files[i])