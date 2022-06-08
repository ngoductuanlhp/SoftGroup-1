import functools
from logging import logProcesses

import spconv.pytorch as spconv
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import faiss                     # make faiss available
import faiss.contrib.torch_utils

from ..ops import (ballquery_batch_p, bfs_cluster, get_mask_iou_on_cluster, get_mask_iou_on_pred,
                   get_mask_label, global_avg_pool, sec_max, sec_min, voxelization,
                   voxelization_idx)
from ..util import cuda_cast, force_fp32, rle_encode
from .blocks import MLP, ResidualBlock, UBlock
import pickle

class SoftGroup(nn.Module):

    def __init__(self,
                 channels=32,
                 num_blocks=7,
                 semantic_only=False,
                 semantic_classes=20,
                 instance_classes=18,
                 sem2ins_classes=[],
                 ignore_label=-100,
                 grouping_cfg=None,
                 instance_voxel_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 fixed_modules=[]):
        super().__init__()
        self.channels = channels
        self.num_blocks = num_blocks
        self.semantic_only = semantic_only
        self.semantic_classes = semantic_classes
        self.instance_classes = instance_classes
        self.sem2ins_classes = sem2ins_classes
        self.ignore_label = ignore_label
        self.grouping_cfg = grouping_cfg
        self.instance_voxel_cfg = instance_voxel_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fixed_modules = fixed_modules

        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        # backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                6, channels, kernel_size=3, padding=1, bias=False, indice_key='subm1'))
        block_channels = [channels * (i + 1) for i in range(num_blocks)]
        self.unet = UBlock(block_channels, norm_fn, 2, block, indice_key_id=1)
        self.output_layer = spconv.SparseSequential(norm_fn(channels), nn.ReLU())

        # point-wise prediction
        self.semantic_linear = MLP(channels, semantic_classes, norm_fn=norm_fn, num_layers=2)

        self.init_weights()

        for mod in fixed_modules:
            mod = getattr(self, mod)
            for param in mod.parameters():
                param.requires_grad = False

        
        # print('rank', torch.distributed.get_rank())

    def init_knn(self):
        
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = rank

        # self.knn_res = faiss.StandardGpuResources()
        # self.geo_knn = faiss.index_cpu_to_gpu(self.knn_res, 0, faiss.IndexFlatL2(3))
        self.geo_knn = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), 3, cfg)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MLP):
                m.init_weights()
        # if not self.semantic_only:
        #     for m in [self.cls_linear, self.iou_score_linear]:
        #         nn.init.normal_(m.weight, 0, 0.01)
        #         nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super().train(mode)
        for mod in self.fixed_modules:
            mod = getattr(self, mod)
            for m in mod.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def forward(self, batch, return_loss=False, scale_factor=1):
        if return_loss:
            return self.forward_train(**batch, scale_factor=scale_factor)
        else:
            return self.forward_test(**batch)

    @cuda_cast
    def forward_train(self, batch_idxs, batch_offsets, voxel_coords, p2v_map, v2p_map, coords_float, feats,
                      semantic_labels, spatial_shape, batch_size, scale_factor=1, **kwargs):

        scale_consistent = (1-scale_factor)
        scale_group = scale_factor
        losses = {}
            # if feats.dim == 2:
            #     feats = torch.cat((feats, coords_float), 1)
            #     voxel_feats = voxelization(feats, p2v_map)
            #     input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
            #     semantic_scores = self.forward_backbone(input, v2p_map)

            #     # point wise losses
            #     point_wise_loss = self.point_wise_loss(semantic_scores, semantic_labels)
            #     losses.update(point_wise_loss)
            # else:
            #     # breakpoint()
            #     feats1 = torch.cat((feats[0], coords_float[0]), 1)
            #     voxel_feats1 = voxelization(feats1, p2v_map[0].cuda())
            #     input1 = spconv.SparseConvTensor(voxel_feats1, voxel_coords[0].int().cuda(), spatial_shape[0], batch_size)
            #     semantic_scores1 = self.forward_backbone(input1, v2p_map[0].cuda())

            #     feats2 = torch.cat((feats[1], coords_float[1]), 1)
            #     voxel_feats2 = voxelization(feats2, p2v_map[1].cuda())
            #     input2 = spconv.SparseConvTensor(voxel_feats2, voxel_coords[1].int().cuda(), spatial_shape[1], batch_size)
            #     semantic_scores2 = self.forward_backbone(input2, v2p_map[1].cuda())

            #     # point wise losses
            #     losses['semantic_loss1'] = self.point_wise_loss(semantic_scores1, semantic_labels[0]) * 0.5
            #     # losses['semantic_loss2'] = self.point_wise_loss(semantic_scores2, semantic_labels[1]) * 0.5
            #     # losses.update(point_wise_loss)

            #     consis_loss_dict = self.consistent_loss(semantic_scores1, semantic_scores2)
            #     losses.update(consis_loss_dict)

            #     losses['group_loss'] = self.grouping_loss(coords_float[0], feats[0], semantic_scores1, semantic_labels[0], batch_offsets[0]) * 0.5
                # losses['group_loss2'] = self.grouping_loss(coords_float[1], feats[1], semantic_scores2, semantic_labels[1], batch_offsets[1]) * 0.5
        
        middle_batch = batch_size // 2

        start = 0
        middle = batch_offsets[middle_batch]
        end = batch_offsets[-1]

        rgb_feats = feats.clone().detach()
        feats = torch.cat((feats, coords_float), 1)
        voxel_feats = voxelization(feats, p2v_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        semantic_scores = self.forward_backbone(input, v2p_map)

        # breakpoint()
        # point wise losses
        # losses['semantic_loss'] = self.point_wise_loss(semantic_scores, semantic_labels)
        # losses.update(point_wise_loss)
        # print('batch_size', batch_size)
        # losses['group_loss'] = self.grouping_loss(coords_float, semantic_scores, semantic_labels, batch_offsets)
        
        

        assert (middle - start) == (end - middle) and end == semantic_scores.shape[0]

        semantic_scores1 = semantic_scores[start:middle, :]
        semantic_scores2 = semantic_scores[middle:end, :]

        semantic_labels1 = semantic_labels[start:middle]
        semantic_labels2 = semantic_labels[middle:end]

        semantic_scores_detach = semantic_scores1.clone().detach()
        with torch.no_grad():
            semantic_labels1_group = self.simple_grouping(coords_float[start:middle,:], semantic_scores_detach, semantic_labels1, rgb_feats[start:middle,:], batch_offsets[:middle_batch])

        losses['semantic_loss'] = self.point_wise_loss(semantic_scores1, semantic_labels1_group) * scale_group

            
            # # point wise losses
            # # losses['semantic_loss1'] = self.point_wise_loss(semantic_scores1, semantic_labels1) * 0.5
            # # losses['semantic_loss2'] = self.point_wise_loss(semantic_scores2, semantic_labels[1]) * 0.5
            # # losses.update(point_wise_loss)

        consis_loss_dict = self.consistent_loss(semantic_scores1, semantic_scores2, scale=scale_consistent)
        losses.update(consis_loss_dict)

            # losses['group_loss'] = self.grouping_loss(coords_float[start:middle,:], semantic_scores1, semantic_labels1, batch_offsets[:middle_batch])
            # # losses['group_loss2'] = self.grouping_loss(coords_float[1], feats[1], semantic_scores2, semantic_labels[1], batch_offsets[1]) * 0.5

            
                

        return self.parse_losses(losses)

    def simple_grouping(self, coords_float, semantic_scores, semantic_labels, rgb_feats, batch_offsets):

        semantic_labels1 = semantic_labels.clone()
        # print('prev', torch.nonzero(semantic_labels1!= -100).view(-1))
        # print('prev label', semantic_labels1[torch.nonzero(semantic_labels!= -100).view(-1)])
        for idx in range(batch_offsets.shape[0]-1):

            coords_float_b = coords_float[batch_offsets[idx]:batch_offsets[idx+1], :]
            # feats_b = feats[batch_offsets[idx]:batch_offsets[idx+1], :]
            # semantic_scores_b = semantic_scores[batch_offsets[idx]:batch_offsets[idx+1], :]
            semantic_labels_b = semantic_labels[batch_offsets[idx]:batch_offsets[idx+1]]
            rgb_feats_b = rgb_feats[batch_offsets[idx]:batch_offsets[idx+1], :]

            # semantic_preds = semantic_scores_b.max(1)[1].detach()
            # fg_conditions = (semantic_preds > 1) | (semantic_labels_b != self.ignore_label)

            # fg_inds = torch.nonzero(fg_conditions).view(-1)

            # fg_num = fg_inds.shape[0]
            # if fg_num == 0:
            #     continue

            # coords_float_b_ = coords_float_b[fg_inds, :]
            # semantic_labels_b_ = semantic_labels_b[fg_inds]
            # # semantic_scores_b_ = semantic_scores_b[fg_inds, :]
            # rgb_feats_b_ = rgb_feats_b[fg_inds, :]
            
            fg_num = coords_float_b.shape[0]
            pivots_inds = torch.nonzero(semantic_labels_b != self.ignore_label).view(-1)
            pivots_num = pivots_inds.shape[0]
            if pivots_num == 0:
                continue
            # print(pivots_inds.shape, pivots_inds)
            pivots_labels = semantic_labels_b[pivots_inds]
            pivots = coords_float_b[pivots_inds, :]

            self.geo_knn.add(coords_float_b)
            # D_geo, I_geo = self.geo_knn.search(pivots, 16)
            # D_geo = torch.sqrt(D_geo)

            pivots_dists = torch.zeros((pivots_num, fg_num)).to(coords_float_b.device)
            pivots_visited = torch.zeros((pivots_num, fg_num), dtype=torch.bool).to(coords_float_b.device)

                # pivots_dists = torch.ones((fg_num), dtype=torch.float32).to(coords_float_b.device) * 100
                # pivots_assigners = torch.zeros((fg_num), dtype=torch.long).to(coords_float_b.device)

            max_step = 4
            for p in range(pivots_num):
                D_geo, I_geo = self.geo_knn.search(pivots[[p]], 16)
                D_geo = torch.sqrt(D_geo)
            
                pivots_dists[p, pivots_inds[p]] = 10000
                indices = I_geo[0, :]
                for i in range(max_step):
                    check_visited = pivots_visited[p, indices]
                    new_indices1 = indices[(check_visited==0).type(torch.bool)]
                    unique_indices = torch.unique(new_indices1)
                    # unique_indices = torch.unique(indices)
                    pivots_dists[p, unique_indices] = max_step - i
                    pivots_visited[p, unique_indices] = 1
                    # print(I_geo[unique_indices,:].shape)
                    # indices = torch.cat([I_geo[unique_indices,:].reshape(-1), unique_indices])

                    D_geo, I_geo = self.geo_knn.search(coords_float_b[unique_indices], 16)
                    D_geo = torch.sqrt(D_geo)

                    new_indices = I_geo.reshape(-1)

                    # print('prev', new_indices.shape)
                    new_indices_dis = D_geo.reshape(-1)


                    prev_rgb = torch.repeat_interleave(rgb_feats_b[unique_indices], 16, dim=0) # prev, 3
                    post_rgb = rgb_feats_b[new_indices] # post, 3
                    # # new_indices_rgb = rgb[new_indices, :]
                    color_diff = torch.sqrt(torch.sum((post_rgb - prev_rgb)**2, axis=-1)) # N
                    # # print(torch.mean(color_diff))
                    valid_new_indices = (new_indices_dis < 0.06) & (color_diff < 0.1)
                    # valid_new_indices = (new_indices_dis < 0.1)
                    new_indices = new_indices[valid_new_indices]

                    indices = torch.cat([new_indices, unique_indices])

            max_dist, max_pivot_inds = torch.max(pivots_dists, dim=0)
            # max_pivot_inds[max_dist==0] = -1

            pseudo_labels = pivots_labels[max_pivot_inds]
            pseudo_labels[max_dist==0] = self.ignore_label
                # for p in range(pivots_num):
                #     neighbors = I_geo[p].type(torch.long)
                #     neighbors_dist = D_geo[p].type(torch.float32)

                #     prev_dist = pivots_dists[neighbors]
                #     valid = (neighbors_dist < 0.1) & (neighbors_dist < prev_dist)
                    
                #     neighbors = neighbors[valid]
                #     neighbors_dist = neighbors_dist[valid]

                #     # print()
                #     # print(valid, neighbors_dist)
                #     pivots_dists[neighbors] = neighbors_dist
                #     pivots_assigners[neighbors] = p

                # pseudo_labels = pivots_labels[pivots_assigners]
                # pseudo_labels[pivots_dists==100] = self.ignore_label

            # print(pseudo_labels.shape, torch.count_nonzero(pseudo_labels!=-100), torch.count_nonzero(pivots_dists==100))

            # temp = torch.ones((coords_float_b.shape[0]), dtype=torch.long).to(coords_float_b.device) * self.ignore_label
            # temp[fg_inds] = pseudo_labels

            semantic_labels1[batch_offsets[idx]:batch_offsets[idx+1]] = pseudo_labels
            self.geo_knn.reset()

        return semantic_labels1

    def grouping_loss(self, coords_float, semantic_scores, semantic_labels, rgb_feats, batch_offsets):
        losses = {}
        loss_groups = torch.tensor([0.0], dtype=torch.float, requires_grad=True).to(coords_float.device)
        # print(batch_offsets)
        
        semantic_labels1 = semantic_labels.clone()
        # print('prev', torch.nonzero(semantic_labels1!= -100).view(-1))
        # print('prev label', semantic_labels1[torch.nonzero(semantic_labels!= -100).view(-1)])
        for idx in range(batch_offsets.shape[0]-1):

            coords_float_b = coords_float[batch_offsets[idx]:batch_offsets[idx+1], :]
            # feats_b = feats[batch_offsets[idx]:batch_offsets[idx+1], :]
            semantic_scores_b = semantic_scores[batch_offsets[idx]:batch_offsets[idx+1], :]
            semantic_labels_b = semantic_labels[batch_offsets[idx]:batch_offsets[idx+1]]
            rgb_feats_b = rgb_feats[batch_offsets[idx]:batch_offsets[idx+1], :]
            # breakpoint()
            semantic_preds = semantic_scores_b.max(1)[1].detach()

            semantic_softmax = F.softmax(semantic_scores_b, dim=-1).detach()
            fg_conditions = (semantic_preds > 1) | (semantic_labels_b != self.ignore_label)

            # fg_conditions = (semantic_preds > 1)


            fg_inds = torch.nonzero(fg_conditions).view(-1)
            # print(fg_inds.shape)

            # breakpoint()
            # print(fg_inds.shape, fg_inds)
            fg_num = fg_inds.shape[0]
            if fg_num == 0:
                continue

            coords_float_b_ = coords_float_b[fg_inds, :]
            semantic_labels_b_ = semantic_labels_b[fg_inds]
            semantic_scores_b_ = semantic_scores_b[fg_inds, :]
            rgb_feats_b_ = rgb_feats_b[fg_inds, :]
            
            # print('debug fb', torch.count_nonzero(semantic_labels_b > 1), torch.count_nonzero(semantic_labels_b_ > 1))
            # print('origin label', semantic_labels_b[semantic_labels_b!= -100])
            with torch.no_grad():

                pivots_inds = torch.nonzero(semantic_labels_b_ != self.ignore_label).view(-1)

                pivots_num = pivots_inds.shape[0]
                if pivots_num == 0:
                    continue
                # print(pivots_inds.shape, pivots_inds)
                pivots_labels = semantic_labels_b_[pivots_inds]
                pivots = coords_float_b_[pivots_inds, :]

                
                # breakpoint()
                self.geo_knn.add(coords_float_b_)

                pivots_dists = torch.zeros((pivots_num, fg_num)).to(coords_float_b_.device)
                pivots_visited = torch.zeros((pivots_num, fg_num), dtype=torch.bool).to(coords_float_b_.device)

                D_geo, I_geo = self.geo_knn.search(coords_float_b_, 12)
                D_geo = D_geo[:, 1:] # n_points ,7
                I_geo = I_geo[:, 1:]

                D_geo = torch.sqrt(D_geo)


                # for p in range(pivots_num):
                #     pivots_dists[p, I_geo[pivots_inds[p]]] = 100
                #     pivots_dists[p, pivots_inds[p]] = 1000
                max_step = 4
                for p in range(pivots_num):
                    pivots_dists[p, pivots_inds[p]] = 10000
                    indices = I_geo[pivots_inds[p], :]
                    for i in range(max_step):
                        check_visited = pivots_visited[p, indices]
                        new_indices1 = indices[(check_visited==0).type(torch.bool)]
                        unique_indices = torch.unique(new_indices1)
                        # unique_indices = torch.unique(indices)
                        pivots_dists[p, unique_indices] = max_step - i
                        pivots_visited[p, unique_indices] = 1
                        # print(I_geo[unique_indices,:].shape)
                        # indices = torch.cat([I_geo[unique_indices,:].reshape(-1), unique_indices])

                        new_indices = I_geo[unique_indices,:].reshape(-1)

                        # print('prev', new_indices.shape)
                        new_indices_dis = D_geo[unique_indices,:].reshape(-1)


                        prev_rgb = torch.repeat_interleave(rgb_feats_b_[unique_indices], 11, dim=0) # prev, 3
                        post_rgb = rgb_feats_b_[new_indices] # post, 3
                        # new_indices_rgb = rgb[new_indices, :]
                        color_diff = torch.sqrt(torch.sum((post_rgb - prev_rgb)**2, axis=-1)) # N
                        # print(torch.mean(color_diff))
                        # valid_new_indices = (new_indices_dis < 0.04) & (color_diff < 0.1)
                        valid_new_indices = (new_indices_dis < 0.04)



                        new_indices = new_indices[valid_new_indices]


                #         # print('post', new_indices.shape)

                        indices = torch.cat([new_indices, unique_indices])
                self.geo_knn.reset()
        

                max_dist, max_pivot_inds = torch.max(pivots_dists, dim=0)
                # max_pivot_inds[max_dist==0] = -1

                group_labels = pivots_labels[max_pivot_inds]
                group_labels[max_dist==0] = self.ignore_label
                group_labels = group_labels.detach()
                # print(group_labels.shape)

                temp = torch.ones((coords_float_b)).to(coords_float_b.device) * self.ignore_label
                temp[fg_inds] = group_labels
                semantic_labels[batch_offsets[idx]:batch_offsets[idx+1]] = temp

                # save_dict = {
                #     # 'rgb': rgb.cpu().numpy()
                #     'xyz': coords_float_b_.cpu().numpy(),
                #     'pivots': pivots.cpu().numpy(),
                #     'max_pivot_inds': max_pivot_inds.cpu().numpy(),
                #     'group_labels': group_labels.cpu().numpy(),
                # }

                # print(group_labels.shape, semantic_scores_b_.shape)
                # uni_label, count = torch.unique(group_labels, return_counts=True)
                # print(uni_label, count)

                # with open('geodist/test_knn_pred.pkl', 'wb') as handle:
                #     pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                # quit()

        # print('post', torch.nonzero(semantic_labels!= -100).view(-1))
        # print('post label', semantic_labels[torch.nonzero(semantic_labels1!= -100).view(-1)])

        

        return semantic_labels
            

            # for u in uni_label:
            #     print(torch.count_nonzero(grou))
            # if torch.sum(group_labels != self.ignore_label) > 0:
            #     loss_groups += F.cross_entropy(
            #         semantic_scores_b_, group_labels, ignore_index=self.ignore_label)

        # loss_groups = loss_groups / (batch_offsets.shape[0] - 1)
        # return loss_groups

    def point_wise_loss(self, semantic_scores, semantic_labels):
        losses = {}
        if torch.sum(semantic_labels != self.ignore_label) == 0:
            semantic_loss = torch.Tensor(0, requires_grad=True)
        else:
            semantic_loss = F.cross_entropy(
                semantic_scores, semantic_labels, ignore_index=self.ignore_label)
        return semantic_loss
        # losses['semantic_loss'] = semantic_loss

        # # pos_inds = instance_labels != self.ignore_label
        # # if pos_inds.sum() == 0:
        # #     offset_loss = 0 * pt_offsets.sum()
        # # else:
        # #     offset_loss = F.l1_loss(
        # #         pt_offsets[pos_inds], pt_offset_labels[pos_inds], reduction='sum') / pos_inds.sum()
        # # losses['offset_loss'] = offset_loss
        # return losses


    def consistent_loss(self, semantic_scores1, semantic_scores2, thresh_consistent=0.9, scale=1):
        # semantic_scores1: N x classes
        losses = {}

        pseudo_sem_scores2, pseudo_sem_labels2 = semantic_scores2.max(1)
        pseudo_sem_scores2 = pseudo_sem_scores2.detach()
        pseudo_sem_labels2 = pseudo_sem_labels2.detach()
        consistent_mask2 = (pseudo_sem_scores2 >= thresh_consistent)
        consis_loss1 = F.cross_entropy(
                semantic_scores1[consistent_mask2], pseudo_sem_labels2[consistent_mask2], ignore_index=self.ignore_label)

        # pseudo_sem_scores1, pseudo_sem_labels1 = semantic_scores1.max(1)
        # pseudo_sem_scores1 = pseudo_sem_scores1.detach()
        # pseudo_sem_labels1 = pseudo_sem_labels1.detach()
        # consistent_mask1 = (pseudo_sem_scores1 >= thresh_consistent)
        # consis_loss2 = F.cross_entropy(
        #         semantic_scores2[consistent_mask1], pseudo_sem_labels1[consistent_mask1], ignore_index=self.ignore_label)
        
        losses['consis_loss1'] = consis_loss1 * scale
        # losses['consis_loss2'] = consis_loss2 * 0.5

        return losses


    def parse_losses(self, losses):
        loss = sum(v for v in losses.values())
        losses['loss'] = loss
        for loss_name, loss_value in losses.items():
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            losses[loss_name] = loss_value.item()
        return loss, losses

    @cuda_cast
    def forward_test(self, batch_idxs, voxel_coords, p2v_map, v2p_map, coords_float, feats,
                     semantic_labels, spatial_shape, batch_size,
                     scan_ids, **kwargs):
        feats = torch.cat((feats, coords_float), 1)
        voxel_feats = voxelization(feats, p2v_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        semantic_scores = self.forward_backbone(
            input, v2p_map, x4_split=self.test_cfg.x4_split)
        if self.test_cfg.x4_split:
            coords_float = self.merge_4_parts(coords_float)
            semantic_labels = self.merge_4_parts(semantic_labels)
            # instance_labels = self.merge_4_parts(instance_labels)
            # pt_offset_labels = self.merge_4_parts(pt_offset_labels)
        semantic_preds = semantic_scores.max(1)[1]
        ret = dict(
            scan_id=scan_ids[0],
            coords_float=coords_float.cpu().numpy(),
            semantic_preds=semantic_preds.cpu().numpy(),
            semantic_labels=semantic_labels.cpu().numpy(),
            # offset_preds=pt_offsets.cpu().numpy(),
            # offset_labels=pt_offset_labels.cpu().numpy(),
            # instance_labels=instance_labels.cpu().numpy()
            )

        return ret

    def forward_backbone(self, input, input_map, x4_split=False):
        if x4_split:
            output_feats = self.forward_4_parts(input, input_map)
            output_feats = self.merge_4_parts(output_feats)
        else:
            output = self.input_conv(input)
            output = self.unet(output)
            output = self.output_layer(output)
            output_feats = output.features[input_map.long()]

        semantic_scores = self.semantic_linear(output_feats)
        # pt_offsets = self.offset_linear(output_feats)
        return semantic_scores

    def forward_4_parts(self, x, input_map):
        """Helper function for s3dis: devide and forward 4 parts of a scene."""
        outs = []
        for i in range(4):
            inds = x.indices[:, 0] == i
            feats = x.features[inds]
            coords = x.indices[inds]
            coords[:, 0] = 0
            x_new = spconv.SparseConvTensor(
                indices=coords, features=feats, spatial_shape=x.spatial_shape, batch_size=1)
            out = self.input_conv(x_new)
            out = self.unet(out)
            out = self.output_layer(out)
            outs.append(out.features)
        outs = torch.cat(outs, dim=0)
        return outs[input_map.long()]

    def merge_4_parts(self, x):
        """Helper function for s3dis: take output of 4 parts and merge them."""
        inds = torch.arange(x.size(0), device=x.device)
        p1 = inds[::4]
        p2 = inds[1::4]
        p3 = inds[2::4]
        p4 = inds[3::4]
        ps = [p1, p2, p3, p4]
        x_split = torch.split(x, [p.size(0) for p in ps])
        x_new = torch.zeros_like(x)
        for i, p in enumerate(ps):
            x_new[p] = x_split[i]
        return x_new


    @force_fp32(apply_to='feats')
    def clusters_voxelization(self,
                              clusters_idx,
                              clusters_offset,
                              feats,
                              coords,
                              scale,
                              spatial_shape,
                              rand_quantize=False):
        batch_idx = clusters_idx[:, 0].cuda().long()
        c_idxs = clusters_idx[:, 1].cuda()
        feats = feats[c_idxs.long()]
        coords = coords[c_idxs.long()]

        coords_min = sec_min(coords, clusters_offset.cuda())
        coords_max = sec_max(coords, clusters_offset.cuda())

        # 0.01 to ensure voxel_coords < spatial_shape
        clusters_scale = 1 / ((coords_max - coords_min) / spatial_shape).max(1)[0] - 0.01
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        coords_min = coords_min * clusters_scale[:, None]
        coords_max = coords_max * clusters_scale[:, None]
        clusters_scale = clusters_scale[batch_idx]
        coords = coords * clusters_scale[:, None]

        if rand_quantize:
            # after this, coords.long() will have some randomness
            range = coords_max - coords_min
            coords_min -= torch.clamp(spatial_shape - range - 0.001, min=0) * torch.rand(3).cuda()
            coords_min -= torch.clamp(spatial_shape - range + 0.001, max=0) * torch.rand(3).cuda()
        coords_min = coords_min[batch_idx]
        coords -= coords_min
        assert coords.shape.numel() == ((coords >= 0) * (coords < spatial_shape)).sum()
        coords = coords.long()
        coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), coords.cpu()], 1)

        out_coords, inp_map, out_map = voxelization_idx(coords, int(clusters_idx[-1, 0]) + 1)
        out_feats = voxelization(feats, out_map.cuda())
        spatial_shape = [spatial_shape] * 3
        voxelization_feats = spconv.SparseConvTensor(out_feats,
                                                     out_coords.int().cuda(), spatial_shape,
                                                     int(clusters_idx[-1, 0]) + 1)
        return voxelization_feats, inp_map

    def get_batch_offsets(self, batch_idxs, bs):
        batch_offsets = torch.zeros(bs + 1).int().cuda()
        for i in range(bs):
            batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
        assert batch_offsets[-1] == batch_idxs.shape[0]
        return batch_offsets

    @force_fp32(apply_to=('x'))
    def global_pool(self, x, expand=False):
        indices = x.indices[:, 0]
        batch_counts = torch.bincount(indices)
        batch_offset = torch.cumsum(batch_counts, dim=0)
        pad = batch_offset.new_full((1, ), 0)
        batch_offset = torch.cat([pad, batch_offset]).int()
        x_pool = global_avg_pool(x.features, batch_offset)
        if not expand:
            return x_pool

        x_pool_expand = x_pool[indices.long()]
        x.features = torch.cat((x.features, x_pool_expand), dim=1)
        return x
