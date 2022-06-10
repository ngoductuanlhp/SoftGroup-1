import math
import os.path as osp
from glob import glob

import numpy as np
import scipy.interpolate
import scipy.ndimage
import torch
from torch.utils.data import Dataset

from ..ops import voxelization_idx


class CustomDataset(Dataset):

    CLASSES = None

    def __init__(self,
                 data_root,
                 prefix,
                 suffix,
                 voxel_cfg=None,
                 training=True,
                 repeat=1,
                 logger=None,
                 limit_anno=False,
                 limit_type=None,
                 extend_by_euclidean=False,
                 consist=True):
        self.data_root = data_root
        self.prefix = prefix
        self.suffix = suffix
        self.voxel_cfg = voxel_cfg
        self.training = training
        self.repeat = repeat
        self.logger = logger
        self.mode = 'train' if training else 'test'
        self.filenames = self.get_filenames()

        self.limit_anno = limit_anno
        self.limit_type = limit_type
        self.extend_by_euclidean = extend_by_euclidean
        self.consist = consist
        if self.limit_anno:
            self.limit_points_dict = torch.load(osp.join(self.data_root, 'points', self.limit_type))

        self.logger.info(f'Load {self.mode} dataset: {len(self.filenames)} scans')

    def get_filenames(self):
        filenames = glob(osp.join(self.data_root, self.prefix, '*' + self.suffix))
        assert len(filenames) > 0, 'Empty dataset.'
        filenames = sorted(filenames * self.repeat)
        return filenames

    def load(self, filename):
        return torch.load(filename)

    def __len__(self):
        return len(self.filenames)

    def elastic(self, x, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        bb = np.abs(x).max(0).astype(np.int32) // gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [
            scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0)
            for n in noise
        ]

        def g(x_):
            return np.hstack([i(x_)[:, None] for i in interp])

        return x + g(x) * mag

    def getInstanceInfo(self, xyz, instance_label, semantic_label):
        pt_mean = np.ones((xyz.shape[0], 3), dtype=np.float32) * -100.0
        instance_pointnum = []
        instance_cls = []
        instance_num = int(instance_label.max()) + 1
        for i_ in range(instance_num):
            inst_idx_i = np.where(instance_label == i_)
            xyz_i = xyz[inst_idx_i]
            pt_mean[inst_idx_i] = xyz_i.mean(0)
            instance_pointnum.append(inst_idx_i[0].size)
            cls_idx = inst_idx_i[0][0]
            instance_cls.append(semantic_label[cls_idx])
        pt_offset_label = pt_mean - xyz
        return instance_num, instance_pointnum, instance_cls, pt_offset_label

    def dataAugment(self, xyz, jitter=False, flip=False, rot=False, prob=0.9):
        m = np.eye(3)
        if jitter and np.random.rand() < prob:
            m += np.random.randn(3, 3) * 0.1
        if flip and np.random.rand() < prob:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1
        if rot and np.random.rand() < prob:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0],
                              [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
        else:
            # Empirically, slightly rotate the scene can match the results from checkpoint
            theta = 0.35 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0],
                              [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])

        return np.matmul(xyz, m)

    def crop(self, xyz, step=32):
        xyz_offset = xyz.copy()
        valid_idxs = xyz_offset.min(1) >= 0
        assert valid_idxs.sum() == xyz.shape[0]
        spatial_shape = np.array([self.voxel_cfg.spatial_shape[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while (valid_idxs.sum() > self.voxel_cfg.max_npoint):
            step_temp = step
            if valid_idxs.sum() > 1e6:
                step_temp = step * 2
            offset = np.clip(spatial_shape - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < spatial_shape).sum(1) == 3)
            spatial_shape[:2] -= step_temp
        return xyz_offset, valid_idxs

    def getCroppedInstLabel(self, instance_label, valid_idxs):
        instance_label = instance_label[valid_idxs]
        j = 0
        while (j < instance_label.max()):
            if (len(np.where(instance_label == j)[0]) == 0):
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label

    def transform_train(self, xyz, rgb, semantic_label, aug_prob=0.9, valid_idxs=None):
        xyz_middle = self.dataAugment(xyz, True, True, True, aug_prob)
        xyz = xyz_middle * self.voxel_cfg.scale
        if np.random.rand() < aug_prob:
            xyz = self.elastic(xyz, 6 * self.voxel_cfg.scale // 50, 40 * self.voxel_cfg.scale / 50)
            xyz = self.elastic(xyz, 20 * self.voxel_cfg.scale // 50,
                               160 * self.voxel_cfg.scale / 50)
        xyz_middle = xyz / self.voxel_cfg.scale
        xyz = xyz - xyz.min(0)

        if valid_idxs is None:
            max_tries = 5
            while (max_tries > 0):
                xyz_offset, valid_idxs = self.crop(xyz)
                if valid_idxs.sum() >= self.voxel_cfg.min_npoint:
                    xyz = xyz_offset
                    break
                max_tries -= 1
            if valid_idxs.sum() < self.voxel_cfg.min_npoint:
                return None

        xyz = xyz[valid_idxs]
        xyz_middle = xyz_middle[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]

        return xyz, xyz_middle, rgb, semantic_label, valid_idxs

    def transform_test(self, xyz, rgb, semantic_label):
        xyz_middle = self.dataAugment(xyz, False, False, False)
        xyz = xyz_middle * self.voxel_cfg.scale
        xyz -= xyz.min(0)
        valid_idxs = np.ones(xyz.shape[0], dtype=bool)

        return xyz, xyz_middle, rgb, semantic_label

    def __getitem__(self, index):
        filename = self.filenames[index]
        scan_id = osp.basename(filename).replace(self.suffix, '')
        data = self.load(filename)

        xyz, rgb, semantic_label, instance_label = data

        if self.limit_anno and self.training:
            points_idx = self.limit_points_dict[scan_id]

            # NOTE extend labels
            if self.extend_by_euclidean:
                # print('buggggggggg')
                extend_idx = []
                pivots_xyz = xyz[points_idx, :]

                distance = np.sqrt(np.sum((pivots_xyz[:, None, :] - xyz[None, :, :])**2, axis=-1)) # N_pivot, N

                for p_i in range(pivots_xyz.shape[0]):
                    distance_to_pivot = distance[p_i, :] # N

                    inds = np.nonzero(distance_to_pivot < 0.2)[0]

                    pivot_label = semantic_label[points_idx[p_i]]
                    semantic_label[inds] = pivot_label

                    # print(inds.shape)
                    extend_idx.append(inds)

                extend_idx = np.concatenate(extend_idx, axis=-1)
                # points_idx.extend(extend_idx)
                points_idx = np.concatenate([points_idx, extend_idx])
                # print(points_idx, extend_idx)
                points_idx = np.unique(points_idx)

            mask_invalid = np.ones_like(semantic_label, dtype=bool)
            mask_invalid[points_idx] = False
            semantic_label[mask_invalid] = -100
            instance_label[mask_invalid] = -100


        if self.training:
            # NOTE first sample
            data1 = self.transform_train(xyz, rgb, semantic_label, aug_prob=1.0, valid_idxs=None)
            if data1 is None:
                return None
            xyz1, xyz_middle1, rgb1, semantic_label1, valid_idxs = data1
            # info = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32), semantic_label)
            # inst_num, inst_pointnum, inst_cls, pt_offset_label = info
            coord1 = torch.from_numpy(xyz1).long()
            coord_float1 = torch.from_numpy(xyz_middle1)
            feat1 = torch.from_numpy(rgb1).float()
            if self.training:
                feat1 += torch.randn(3) * 0.1
            semantic_label1 = torch.from_numpy(semantic_label1)

            # NOTE second sample
            data2 = self.transform_train(xyz, rgb, semantic_label, aug_prob=1.0, valid_idxs=valid_idxs)
            xyz2, xyz_middle2, rgb2, semantic_label2, valid_idxs = data2
            # info = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32), semantic_label)
            # inst_num, inst_pointnum, inst_cls, pt_offset_label = info
            coord2 = torch.from_numpy(xyz2).long()
            coord_float2 = torch.from_numpy(xyz_middle2)
            feat2 = torch.from_numpy(rgb2).float()
            if self.training:
                feat2 += torch.randn(3) * 0.1
            semantic_label2 = torch.from_numpy(semantic_label2)


            # NOTE stack samples
            coord = torch.stack([coord1, coord2], dim=0)
            coord_float = torch.stack([coord_float1, coord_float2], dim=0)
            feat = torch.stack([feat1, feat2], dim=0)
            semantic_label = torch.stack([semantic_label1, semantic_label2], dim=0)

            return (scan_id, coord, coord_float, feat, semantic_label)

        data = self.transform_test(xyz, rgb, semantic_label)
        if data is None:
            return None
        xyz, xyz_middle, rgb, semantic_label = data
        # info = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32), semantic_label)
        # inst_num, inst_pointnum, inst_cls, pt_offset_label = info
        coord = torch.from_numpy(xyz).long()
        coord_float = torch.from_numpy(xyz_middle)
        feat = torch.from_numpy(rgb).float()
        semantic_label = torch.from_numpy(semantic_label)

        return (scan_id, coord, coord_float, feat, semantic_label)

    def collate_fn_train(self, batch):
        scan_ids = []
        coords = []
        coords_float = []
        feats = []
        semantic_labels = []

        batch_offsets = [0]

        batch_id = 0

        for i in range(2):
            for data in batch:
                if data is None:
                    continue
                (scan_id, coord, coord_float, feat, semantic_label) = data # (2, N, c)

                
                scan_ids.append(scan_id)

                coords.append(torch.cat([coord.new_full((coord[i].size(0), 1), batch_id), coord[i]], 1))
                coords_float.append(coord_float[i])
                feats.append(feat[i])
                semantic_labels.append(semantic_label[i])
                batch_id += 1
                batch_offsets.append(batch_offsets[-1] + semantic_label[i].shape[0])

            if not self.consist:
                break

                    # coords.append(torch.cat([coord.new_full((coord.size(0), coord.size(1), 1), batch_id), coord], 2))
                    # coords_float.append(coord_float)
                    # feats.append(feat)
                    # semantic_labels.append(semantic_label)
                    # batch_id += 1
                    # batch_offsets.append(torch.tensor([batch_offsets[-1][0] + semantic_label[0].shape[0], batch_offsets[-1][1] + semantic_label[1].shape[0]]))
        assert batch_id > 0 and batch_id % 2 == 0, 'empty batch'
        if batch_id < len(batch):
            self.logger.info(f'batch is truncated from size {len(batch)} to {batch_id}')

            # # merge all the scenes in the batch
            # coords = torch.cat(coords, 1)  # long (2, N, 1 + 3), the batch item idx is put in coords[:, 0]
            # batch_idxs = coords[..., 0].int() # (2, N)
            # coords_float = torch.cat(coords_float, 1).to(torch.float32)  # float (N, 3)
            # feats = torch.cat(feats, 1)  # float (N, C)
            # semantic_labels = torch.cat(semantic_labels, 1).long()  # long (N)
            # batch_offsets = torch.stack(batch_offsets, dim=-1) # B+1, 2


            # spatial_shape1 = np.clip(
            #     coords[0].max(0)[0][1:].numpy() + 1, self.voxel_cfg.spatial_shape[0], None)
            # spatial_shape2 = np.clip(
            #     coords[1].max(0)[0][1:].numpy() + 1, self.voxel_cfg.spatial_shape[0], None)

            # voxel_coords1, v2p_map1, p2v_map1 = voxelization_idx(coords[0], batch_id)
            # voxel_coords2, v2p_map2, p2v_map2 = voxelization_idx(coords[1], batch_id)
            # voxel_coords = [voxel_coords1, voxel_coords2]
            # p2v_map = [p2v_map1, p2v_map2]
            # v2p_map = [v2p_map1, v2p_map2]
            # spatial_shape = [spatial_shape1, spatial_shape2]

        # merge all the scenes in the batch
        coords = torch.cat(coords, 0)  # long (N, 1 + 3), the batch item idx is put in coords[:, 0]
        batch_idxs = coords[:, 0].int()
        coords_float = torch.cat(coords_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0)  # float (N, C)
        semantic_labels = torch.cat(semantic_labels, 0).long()  # long (N)

        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)# B+1s

        spatial_shape = np.clip(
            coords.max(0)[0][1:].numpy() + 1, self.voxel_cfg.spatial_shape[0], None)
        voxel_coords, v2p_map, p2v_map = voxelization_idx(coords, batch_id)

        return {
            'scan_ids': scan_ids,
            'coords': coords,
            'batch_idxs': batch_idxs,
            'voxel_coords': voxel_coords,
            'p2v_map': p2v_map,
            'v2p_map': v2p_map,
            'coords_float': coords_float,
            'feats': feats,
            'semantic_labels': semantic_labels,
            'spatial_shape': spatial_shape,
            'batch_size': batch_id,
            'batch_offsets': batch_offsets,
        }

    def collate_fn_test(self, batch):
        scan_ids = []
        coords = []
        coords_float = []
        feats = []
        semantic_labels = []
        batch_offsets = [0]

        batch_id = 0
        for data in batch:
            if data is None:
                continue
            (scan_id, coord, coord_float, feat, semantic_label) = data
            scan_ids.append(scan_id)
            coords.append(torch.cat([coord.new_full((coord.size(0), 1), batch_id), coord], 1))
            coords_float.append(coord_float)
            feats.append(feat)
            semantic_labels.append(semantic_label)
            batch_id += 1
            batch_offsets.append(batch_offsets[-1] + semantic_label.shape[0])
        assert batch_id > 0, 'empty batch'
        if batch_id < len(batch):
            self.logger.info(f'batch is truncated from size {len(batch)} to {batch_id}')

        # merge all the scenes in the batch
        coords = torch.cat(coords, 0)  # long (N, 1 + 3), the batch item idx is put in coords[:, 0]
        batch_idxs = coords[:, 0].int()
        coords_float = torch.cat(coords_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0)  # float (N, C)
        semantic_labels = torch.cat(semantic_labels, 0).long()  # long (N)

        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int) # B+1s

        spatial_shape = np.clip(
            coords.max(0)[0][1:].numpy() + 1, self.voxel_cfg.spatial_shape[0], None)
        voxel_coords, v2p_map, p2v_map = voxelization_idx(coords, batch_id)
        return {
            'scan_ids': scan_ids,
            'coords': coords,
            'batch_idxs': batch_idxs,
            'voxel_coords': voxel_coords,
            'p2v_map': p2v_map,
            'v2p_map': v2p_map,
            'coords_float': coords_float,
            'feats': feats,
            'semantic_labels': semantic_labels,
            'spatial_shape': spatial_shape,
            'batch_size': batch_id,
            'batch_offsets': batch_offsets,
        }
