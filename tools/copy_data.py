import shutil
import os
import copy
import glob

src_dir = '/vinai/tuannd42/DATASET/scannetv2/scans'
dst_dir = '/home/ubuntu/DATASET/scannetv2/scans'

for file in glob.glob("/vinai/tuannd42/DATASET/scannetv2/scans/*_vh_clean_2.ply"):
    file_name = file.split('/')[-1]
    shutil.copy(file, os.path.join(dst_dir, file_name))

    print(file_name)