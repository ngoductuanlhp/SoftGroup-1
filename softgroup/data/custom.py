import numpy as np
import scipy.interpolate
import scipy.ndimage
import torch
from torch.utils.data import Dataset

import math
import os.path as osp
from glob import glob
from ..ops import voxelization_idx


class CustomDataset(Dataset):

    CLASSES = None

    def __init__(self, data_root, prefix, suffix, voxel_cfg=None, training=True, repeat=1, logger=None, lite=False):
        self.data_root = data_root
        self.prefix = prefix
        self.suffix = suffix
        self.voxel_cfg = voxel_cfg
        self.training = training
        self.repeat = repeat
        self.logger = logger
        self.mode = "train" if training else "test"
        self.lite = lite
        self.filenames = self.get_filenames()

        if self.lite:
            self.logger.info("Only load lite dataset (/10)")
            self.filenames = self.filenames[::10]
        self.logger.info(f"Load {self.mode} dataset: {len(self.filenames)} scans")

        # self.filenames = self.filenames[:50]

    def get_filenames(self):
        filenames = glob(osp.join(self.data_root, self.prefix, "*" + self.suffix))
        assert len(filenames) > 0, "Empty dataset."
        filenames = sorted(filenames * self.repeat)
        return filenames

    def load(self, filename):
        return torch.load(filename)

    def __len__(self):
        return len(self.filenames)

    def elastic(self, x, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype("float32") / 3
        blur1 = np.ones((1, 3, 1)).astype("float32") / 3
        blur2 = np.ones((1, 1, 3)).astype("float32") / 3

        bb = np.abs(x).max(0).astype(np.int32) // gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype("float32") for _ in range(3)]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode="constant", cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode="constant", cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode="constant", cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode="constant", cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode="constant", cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode="constant", cval=0) for n in noise]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]

        def g(x_):
            return np.hstack([i(x_)[:, None] for i in interp])

        return x + g(x) * mag

    def getInstanceInfo(self, xyz, instance_label, semantic_label):
        pt_mean = np.ones((xyz.shape[0], 3), dtype=np.float32) * -100.0
        instance_pointnum = []
        instance_cls = []
        instance_box = []
        instance_num = int(instance_label.max()) + 1

        pt_offset_vertices_label = np.ones((xyz.shape[0], 3 * 2), dtype=np.float32) * -100.0

        # pt_offset_vertices_label = np.ones((xyz.shape[0], 3*8), dtype=np.float32) * -100.0

        for i_ in range(instance_num):
            inst_idx_i = np.where(instance_label == i_)
            xyz_i = xyz[inst_idx_i]

            centroid = xyz_i.mean(0)
            pt_mean[inst_idx_i] = centroid

            min_xyz_i = xyz_i.min(0)
            max_xyz_i = xyz_i.max(0)

            pt_offset_vertices_label[inst_idx_i, 0:3] = min_xyz_i - xyz_i
            pt_offset_vertices_label[inst_idx_i, 3:6] = max_xyz_i - xyz_i

            # min_x_i = xyz_i[np.argmin(xyz_i[:,0])]
            # min_y_i = xyz_i[np.argmin(xyz_i[:,1])]
            # min_z_i = xyz_i[np.argmin(xyz_i[:,2])]
            # max_x_i = xyz_i[np.argmax(xyz_i[:,0])]
            # max_y_i = xyz_i[np.argmax(xyz_i[:,1])]
            # max_z_i = xyz_i[np.argmax(xyz_i[:,2])]

            # pt_offset_vertices_label[inst_idx_i, 6:9] = min_x_i - xyz_i
            # pt_offset_vertices_label[inst_idx_i, 9:12] = min_y_i - xyz_i
            # pt_offset_vertices_label[inst_idx_i, 12:15] = min_z_i - xyz_i

            # pt_offset_vertices_label[inst_idx_i, 15:18] = max_x_i - xyz_i
            # pt_offset_vertices_label[inst_idx_i, 18:21] = max_y_i - xyz_i
            # pt_offset_vertices_label[inst_idx_i, 21:24] = max_z_i - xyz_i

            instance_box.append(np.concatenate([min_xyz_i, max_xyz_i], axis=0))

            instance_pointnum.append(inst_idx_i[0].size)
            cls_idx = inst_idx_i[0][0]
            instance_cls.append(semantic_label[cls_idx])

        pt_offset_label = pt_mean - xyz

        instance_box = np.stack(instance_box, axis=0)  # N, 6
        return instance_num, instance_pointnum, instance_cls, instance_box, pt_offset_label, pt_offset_vertices_label

    def dataAugment(self, xyz, jitter=False, flip=False, rot=False, prob=1.0):
        m = np.eye(3)
        if jitter and np.random.rand() < prob:
            m += np.random.randn(3, 3) * 0.1
        if flip and np.random.rand() < prob:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1
        if rot and np.random.rand() < prob:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(
                m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]]
            )
        else:
            # Empirically, slightly rotate the scene can match the results from checkpoint
            theta = 0.35 * math.pi
            m = np.matmul(
                m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]]
            )

        return np.matmul(xyz, m)

    def crop(self, xyz, step=32):
        xyz_offset = xyz.copy()
        valid_idxs = xyz_offset.min(1) >= 0
        assert valid_idxs.sum() == xyz.shape[0]
        spatial_shape = np.array([self.voxel_cfg.spatial_shape[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while valid_idxs.sum() > self.voxel_cfg.max_npoint:
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
        while j < instance_label.max():
            if len(np.where(instance_label == j)[0]) == 0:
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label

    def transform_train(self, xyz, rgb, semantic_label, instance_label, spp, aug_prob=1.0):
        xyz_middle = self.dataAugment(xyz, True, True, True, aug_prob)
        xyz = xyz_middle * self.voxel_cfg.scale
        if np.random.rand() < aug_prob:
            xyz = self.elastic(xyz, 6 * self.voxel_cfg.scale // 50, 40 * self.voxel_cfg.scale / 50)
            xyz = self.elastic(xyz, 20 * self.voxel_cfg.scale // 50, 160 * self.voxel_cfg.scale / 50)
        # xyz_middle = xyz / self.voxel_cfg.scale

        xyz = xyz - xyz.min(0)
        max_tries = 5
        while max_tries > 0:
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
        instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)

        spp = spp[valid_idxs]
        return xyz, xyz_middle, rgb, semantic_label, instance_label, spp

    def transform_test(self, xyz, rgb, semantic_label, instance_label, spp):
        xyz_middle = self.dataAugment(xyz, False, False, False)
        xyz = xyz_middle * self.voxel_cfg.scale
        xyz -= xyz.min(0)
        valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)
        return xyz, xyz_middle, rgb, semantic_label, instance_label, spp

    def __getitem__(self, index):
        filename = self.filenames[index]
        scan_id = osp.basename(filename).replace(self.suffix, "")
        xyz, rgb, semantic_label, instance_label = self.load(filename)

        spp_filename = osp.join("dataset/scannetv2/superpoints", scan_id + ".pth")
        spp = self.load(spp_filename)

        data = (
            self.transform_train(xyz, rgb, semantic_label, instance_label, spp)
            if self.training
            else self.transform_test(xyz, rgb, semantic_label, instance_label, spp)
        )
        if data is None:
            return None
        xyz, xyz_middle, rgb, semantic_label, instance_label, spp = data
        info = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32), semantic_label)
        inst_num, inst_pointnum, inst_cls, inst_box, pt_offset_label, pt_offset_vertices_label = info
        coord = torch.from_numpy(xyz).long()
        coord_float = torch.from_numpy(xyz_middle)
        feat = torch.from_numpy(rgb).float()
        if self.training:
            feat += torch.randn(3) * 0.1
        semantic_label = torch.from_numpy(semantic_label)
        instance_label = torch.from_numpy(instance_label)
        pt_offset_label = torch.from_numpy(pt_offset_label)
        pt_offset_vertices_label = torch.from_numpy(pt_offset_vertices_label)

        # spp = torch.from_numpy(spp)

        inst_box = torch.from_numpy(inst_box)

        # NOTE debug

        # inds = torch.nonzero(instance_label != -100).view(-1)
        # sem_inds = semantic_label[inds]
        # assert torch.all(torch.tensor(inst_cls) >= 0)
        # assert torch.all(sem_inds >= 2)

        return (
            scan_id,
            coord,
            coord_float,
            feat,
            semantic_label,
            instance_label,
            spp,
            inst_num,
            inst_pointnum,
            inst_cls,
            inst_box,
            pt_offset_label,
            pt_offset_vertices_label,
        )

    def collate_fn(self, batch):
        scan_ids = []
        coords = []
        coords_float = []
        feats = []
        semantic_labels = []
        instance_labels = []

        spps = []

        instance_pointnum = []  # (total_nInst), int
        instance_cls = []  # (total_nInst), long
        instance_box = []
        instance_batch_offsets = [0]

        pt_offset_labels = []
        pt_offset_vertices_labels = []

        pc_mins = []
        pc_maxs = []

        total_inst_num = 0
        batch_id = 0
        for data in batch:
            if data is None:
                continue
            (
                scan_id,
                coord,
                coord_float,
                feat,
                semantic_label,
                instance_label,
                spp,
                inst_num,
                inst_pointnum,
                inst_cls,
                inst_box,
                pt_offset_label,
                pt_offset_vertices_label,
            ) = data
            instance_label[np.where(instance_label != -100)] += total_inst_num
            total_inst_num += inst_num
            scan_ids.append(scan_id)
            coords.append(torch.cat([coord.new_full((coord.size(0), 1), batch_id), coord], 1))
            coords_float.append(coord_float)
            feats.append(feat)
            semantic_labels.append(semantic_label)
            instance_labels.append(instance_label)
            spps.append(spp)

            instance_pointnum.extend(inst_pointnum)
            instance_cls.extend(inst_cls)
            instance_box.append(inst_box)
            instance_batch_offsets.append(total_inst_num)
            pt_offset_labels.append(pt_offset_label)
            pt_offset_vertices_labels.append(pt_offset_vertices_label)

            pc_mins.append(torch.min(coord_float, dim=0)[0])
            pc_maxs.append(torch.max(coord_float, dim=0)[0])

            batch_id += 1
        assert batch_id > 0, "empty batch"
        if batch_id < len(batch):
            self.logger.info(f"batch is truncated from size {len(batch)} to {batch_id}")

        # merge all the scenes in the batch
        coords = torch.cat(coords, 0)  # long (N, 1 + 3), the batch item idx is put in coords[:, 0]
        batch_idxs = coords[:, 0].int()
        coords_float = torch.cat(coords_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0)  # float (N, C)
        semantic_labels = torch.cat(semantic_labels, 0).long()  # long (N)
        instance_labels = torch.cat(instance_labels, 0).long()  # long (N)
        spps = torch.cat(spps, 0).long()

        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int (total_nInst)
        instance_cls = torch.tensor(instance_cls, dtype=torch.long)  # long (total_nInst)
        instance_box = torch.cat(instance_box, dim=0).float()  # (total_nInst, 6)
        instance_batch_offsets = torch.tensor(instance_batch_offsets, dtype=torch.long)
        pt_offset_labels = torch.cat(pt_offset_labels).float()
        pt_offset_vertices_labels = torch.cat(pt_offset_vertices_labels).float()

        del pt_offset_vertices_labels, pt_offset_labels
        pt_offset_labels = None
        pt_offset_vertices_labels = None

        pc_mins = torch.stack(pc_mins)
        pc_maxs = torch.stack(pc_maxs)  # batch, 3

        pc_dims = torch.stack([pc_mins, pc_maxs], dim=0)  # 2, batch, 3

        # spatial_shape = np.array([768, 768, 400])
        spatial_shape = np.clip(coords.max(0)[0][1:].numpy() + 1, self.voxel_cfg.spatial_shape[0], None)
        voxel_coords, v2p_map, p2v_map = voxelization_idx(coords, batch_id)
        return {
            "scan_ids": scan_ids,
            "coords": coords,
            "batch_idxs": batch_idxs,
            "voxel_coords": voxel_coords,
            "p2v_map": p2v_map,
            "v2p_map": v2p_map,
            "coords": coords,
            "coords_float": coords_float,
            "feats": feats,
            "semantic_labels": semantic_labels,
            "instance_labels": instance_labels,
            "spps": spps,
            "instance_pointnum": instance_pointnum,
            "instance_cls": instance_cls,
            "instance_box": instance_box,
            "instance_batch_offsets": instance_batch_offsets,
            "pt_offset_labels": pt_offset_labels,
            "pt_offset_vertices_labels": pt_offset_vertices_labels,
            "spatial_shape": spatial_shape,
            "batch_size": batch_id,
            "pc_dims": pc_dims,
        }
