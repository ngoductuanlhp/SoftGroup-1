import functools

import spconv.pytorch as spconv
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..ops import (ballquery_batch_p, ballquery_batch_p_boxiou, bfs_cluster, get_mask_iou_on_cluster, get_mask_iou_on_pred,
                   get_mask_label, global_avg_pool, sec_max, sec_min, sec_mean, voxelization,
                   voxelization_idx)
from ..util import cuda_cast, force_fp32, rle_encode
from .blocks import MLP, ResidualBlock, UBlock, conv_with_kaiming_uniform, PositionalEmbedding
from .model_utils import iou_aabb, compute_dice_loss, sigmoid_focal_loss, non_maximum_cluster, non_maximum_cluster2
from .model_utils import dice_coefficient, compute_dice_loss, sigmoid_focal_loss, get_bounding_vertices
from .geodesic_utils import cal_geodesic_single, cal_geodesic_vectorize

import numpy as np

import faiss                     # make faiss available
import faiss.contrib.torch_utils

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
                 fixed_modules=[],
                 embedding_coord=False):
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
        self.embedding_coord = embedding_coord

        self.init_knn()
        

        if self.embedding_coord:
            n_freqs = 4
            self.pos_embed = PositionalEmbedding(3, n_freqs)
            in_channels = self.pos_embed.out_channels + 3
        else:
            in_channels = 6

        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        # backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels, channels, kernel_size=3, padding=1, bias=False, indice_key='subm1'))
        block_channels = [channels * (i + 1) for i in range(num_blocks)]
        self.unet = UBlock(block_channels, norm_fn, 2, block, indice_key_id=1)
        self.output_layer = spconv.SparseSequential(norm_fn(channels), nn.ReLU())

        # point-wise prediction
        self.semantic_linear = MLP(channels, semantic_classes, norm_fn=norm_fn, num_layers=2)
        self.offset_linear = MLP(channels, 3, norm_fn=norm_fn, num_layers=2)

        # BBox
        self.offset_vertices_linear = MLP(channels, 3*2, norm_fn=norm_fn, num_layers=2)
        # self.offset_vertices_linear = MLP(channels, 3*8, norm_fn=norm_fn, num_layers=2)
        self.box_conf_linear = MLP(channels, 1, norm_fn=norm_fn, num_layers=2)

        # topdown refinement path
        if not semantic_only:
            self.tiny_unet = UBlock([channels, 2 * channels], norm_fn, 2, block, indice_key_id=11)
            self.tiny_unet_outputlayer = spconv.SparseSequential(norm_fn(channels), nn.ReLU())
            self.cls_linear = nn.Linear(channels, instance_classes + 1)
            self.iou_score_linear = nn.Linear(channels, 1)

            ### convolution before the condinst take place (convolution num before the generated parameters take place)
            conv_block = conv_with_kaiming_uniform("BN", activation=True)

                # if self.embedding_relative_coord:
                #     n_freqs = 4
                #     self.pos_embed = PositionalEmbedding(3, n_freqs)
                #     in_channels = channels + self.pos_embed.out_channels
                # else:
                #     in_channels = (channels + 3)

            # in_channels = channels + 3*9
            in_channels = channels + 3 + 1

            self.weight_nums = [in_channels * (channels), (channels) * 1]
            self.bias_nums = [(channels), 1]
            self.num_gen_params = sum(self.weight_nums) + sum(self.bias_nums)
            self.controller = nn.Sequential(*[conv_block(channels, channels), conv_block(channels, channels), nn.Conv1d(channels, self.num_gen_params, kernel_size=1)])
            self.mask_tower = nn.Sequential(*[conv_block(channels, channels), conv_block(channels, channels), conv_block(channels, channels), nn.Conv1d(channels, channels, 1)])

        self.init_weights()

        for mod in fixed_modules:
            mod = getattr(self, mod)
            for param in mod.parameters():
                param.requires_grad = False

        if 'input_conv' in self.fixed_modules and 'unet' in self.fixed_modules:
            self.freeze_backbone = True
        else:
            self.freeze_backbone = False
        # self.freeze_backbone = False



    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MLP):
                m.init_weights()
        if not self.semantic_only:
            for m in [self.cls_linear, self.iou_score_linear]:
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def init_knn(self):
        faiss_cfg = faiss.GpuIndexFlatConfig()
        faiss_cfg.useFloat16 = True
        faiss_cfg.device = 0

        self.geo_knn = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), 3, faiss_cfg)

    def train(self, mode=True):
        super().train(mode)
        for mod in self.fixed_modules:
            mod = getattr(self, mod)
            for m in mod.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def forward(self, batch, return_loss=False):
        if return_loss:
            return self.forward_train(**batch)
        else:
            return self.forward_test(**batch)

    @cuda_cast
    def forward_train(self, batch_idxs, voxel_coords, p2v_map, v2p_map, coords_float, feats,
                      semantic_labels, instance_labels, instance_pointnum, instance_cls,
                      pt_offset_labels, pt_offset_vertices_labels, spatial_shape, batch_size, **kwargs):
        losses = {}

        if self.embedding_coord:
            feats = torch.cat((feats, self.pos_embed(coords_float)), 1)
            # print('feats', feats.shape)
        else:
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = voxelization(feats, p2v_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        semantic_scores, pt_offsets, pt_offsets_vertices, box_conf, output_feats = self.forward_backbone(input, v2p_map)

        # point wise losses

        if not self.freeze_backbone:
            point_wise_loss = self.point_wise_loss(semantic_scores, pt_offsets, pt_offsets_vertices, box_conf, semantic_labels,
                                                instance_labels, pt_offset_labels, pt_offset_vertices_labels, coords_float)
            losses.update(point_wise_loss)

        # instance losses
        if not self.semantic_only:
            proposals_idx, proposals_offset, proposals_box, proposals_conf, proposals_pivots, proposals_cls, proposals_batch_idxs, per_cls_object_idxs = \
                                            self.forward_grouping(semantic_scores, pt_offsets, pt_offsets_vertices, box_conf,
                                                                    batch_idxs, coords_float,
                                                                    self.grouping_cfg)
            if proposals_offset.shape[0] > self.train_cfg.max_proposal_num:
                proposals_offset = proposals_offset[:self.train_cfg.max_proposal_num + 1]
                proposals_idx = proposals_idx[:proposals_offset[-1]]
                proposals_pivots = proposals_pivots[:self.train_cfg.max_proposal_num]
                assert proposals_idx.shape[0] == proposals_offset[-1]

            inst_feats, inst_map, inst_coords_mean, coords_bounding = self.clusters_voxelization(
                proposals_idx.cpu(),
                proposals_offset.cpu(),
                output_feats,
                coords_float,
                pt_offsets,
                rand_quantize=True,
                **self.instance_voxel_cfg)


            instance_batch_idxs, cls_scores, iou_scores, mask_logits_dict, inst_idxs_dict = self.forward_instance(
                inst_feats, inst_map, inst_coords_mean, coords_bounding, coords_float, pt_offsets, output_feats, per_cls_object_idxs, batch_idxs, proposals_batch_idxs)

            # print('mask_logits_dict', mask_logits_dict)

            instance_loss = self.instance_loss(cls_scores, mask_logits_dict, inst_idxs_dict, per_cls_object_idxs, iou_scores, proposals_idx,
                                               proposals_offset, proposals_batch_idxs, semantic_labels, instance_labels, instance_pointnum,
                                               instance_cls, instance_batch_idxs, batch_idxs)
            losses.update(instance_loss)
        return self.parse_losses(losses)

    def point_wise_loss(self, semantic_scores, pt_offsets, pt_offsets_vertices, box_conf, semantic_labels, instance_labels,
                        pt_offset_labels, pt_offset_vertices_labels, coords_float):
        losses = {}
        semantic_loss = F.cross_entropy(
            semantic_scores, semantic_labels, ignore_index=self.ignore_label)
        losses['semantic_loss'] = semantic_loss

        pos_inds = instance_labels != self.ignore_label
        total_pos_inds = pos_inds.sum()
        if total_pos_inds == 0:
            offset_loss = 0 * pt_offsets.sum()
            offset_vertices_loss = 0 * pt_offsets_vertices.sum()
            point_iou_loss = 0 * box_conf.sum()
        else:
            offset_loss = F.l1_loss(
                pt_offsets[pos_inds], pt_offset_labels[pos_inds], reduction='sum') / total_pos_inds

            # print('pt_offsets_vertices', pt_offsets_vertices.shape)
            offset_vertices_loss = F.l1_loss(
                pt_offsets_vertices[pos_inds], pt_offset_vertices_labels[pos_inds], reduction='sum') / total_pos_inds

            iou_gt = iou_aabb(pt_offsets_vertices[pos_inds], pt_offset_vertices_labels[pos_inds], coords_float[pos_inds])

            # breakpoint()
            point_iou_loss = F.mse_loss(box_conf[pos_inds], iou_gt, reduction='none')
            point_iou_loss = point_iou_loss.sum() / total_pos_inds

            # breakpoint()

        losses['point_iou_loss'] = point_iou_loss

        losses['offset_loss'] = offset_loss
        losses['offset_vertices_loss'] = offset_vertices_loss
        return losses

    @force_fp32(apply_to=('cls_scores', 'mask_logits_dict', 'iou_scores'))
    def instance_loss(self, cls_scores, mask_logits_dict, inst_idxs_dict, per_cls_object_idxs, iou_scores, proposals_idx, proposals_offset, proposals_batch_idxs,
                      semantic_labels, instance_labels, instance_pointnum, instance_cls, instance_batch_idxs, batch_idxs):
        losses = {}
        proposals_idx = proposals_idx[:, 1].cuda().contiguous()
        proposals_offset = proposals_offset.cuda().contiguous()

        # cal iou of clustered instance
        ious_on_cluster = get_mask_iou_on_cluster(proposals_idx, proposals_offset, instance_labels,
                                                  instance_pointnum) # N_inst_pred x N_inst_gt


        # filter out background instances
        # print('instance_cls', torch.unique(instance_cls))
        # fg_inds = (instance_cls != self.ignore_label)
        fg_inds = torch.nonzero(instance_cls != self.ignore_label).view(-1)

        # print('instance_cls', instance_cls)
        fg_instance_cls = instance_cls[fg_inds]
        fg_ious_on_cluster = ious_on_cluster[:, fg_inds] # N_inst_pred x N_fg_inst_gt

        # overlap > thr on fg instances are positive samples
        max_iou, gt_inds = fg_ious_on_cluster.max(1) # N_inst_pred
        # pos_inds = max_iou >= self.train_cfg.pos_iou_thr
        pos_inds = torch.nonzero(max_iou >= self.train_cfg.pos_iou_thr).view(-1) # N_pos_inst_pred: inds of positive pred
        pos_gt_inds = gt_inds[pos_inds] # # N_pos_inst_pred: corresponding fg_inst_gt index

        # compute cls loss. follow detection convention: 0 -> K - 1 are fg, K is bg
        labels = fg_instance_cls.new_full((fg_ious_on_cluster.size(0), ), self.instance_classes) # N_pred, 1
        labels[pos_inds] = fg_instance_cls[pos_gt_inds]

        # print('labels', torch.unique(labels))
        cls_loss = F.cross_entropy(cls_scores, labels)

        # breakpoint()
        losses['cls_loss'] = cls_loss
        # return cls_loss
            # # compute mask loss
            # mask_cls_label = labels[instance_batch_idxs.long()] # Num_points: label for each point 
            # slice_inds = torch.arange(
            #     0, mask_cls_label.size(0), dtype=torch.long, device=mask_cls_label.device)

            # # mask_scores: num_points, N_classes
            # mask_scores_sigmoid_slice = mask_scores.sigmoid()[slice_inds, mask_cls_label]
            # mask_label = get_mask_label(proposals_idx, proposals_offset, instance_labels, instance_cls,
            #                             instance_pointnum, ious_on_cluster, self.train_cfg.pos_iou_thr)
            # mask_label_weight = (mask_label != -1).float()
            # mask_label[mask_label == -1.] = 0.5  # any value is ok
            # mask_loss = F.binary_cross_entropy(
            #     mask_scores_sigmoid_slice, mask_label, weight=mask_label_weight, reduction='sum')
            # mask_loss /= (mask_label_weight.sum() + 1)
            # losses['mask_loss'] = mask_loss

        dice_loss = torch.tensor([0.0], dtype=torch.float, requires_grad=True).cuda()
        valid_num_dice_loss = 0

        mask_loss_total = torch.tensor([0.0], dtype=torch.float, requires_grad=True).cuda()
        mask_weight_total = 0
        # max_ious_on_cluster, inds_max_ious_on_cluster = ious_on_cluster.max(1)


        # print('debug', torch.unique(instance_labels), torch.unique(inds_max_ious_on_cluster))
        gt_ious = torch.zeros_like(iou_scores)
        gt_ious_mask = torch.zeros_like(iou_scores)

        # print('pos_inds', pos_inds)

        for class_id in range(self.semantic_classes):
            if class_id in self.grouping_cfg.ignore_classes:
                continue
            if class_id not in mask_logits_dict:
                continue

            mask_logits = mask_logits_dict[class_id] # N_inst, N_points_of_classX
            inst_idxs = inst_idxs_dict[class_id]
            object_idxs = per_cls_object_idxs[class_id]

            # print('inst_idxs', inst_idxs)
            if mask_logits is None:
                continue

            batch_idxs_ = batch_idxs[object_idxs]

            inst_num = mask_logits.shape[0]
            mask_labels = torch.zeros_like(mask_logits)
            weights = torch.zeros_like(mask_logits)

            proposals_batch_idxs_ = proposals_batch_idxs[inst_idxs]

            gt_ious_ = torch.zeros((inst_idxs.shape[0])).cuda()
            gt_ious_mask_ = torch.zeros((inst_idxs.shape[0])).cuda()

            # breakpoint()
            
            for n in range(inst_num):
                if inst_idxs[n] not in pos_inds:
                    continue

                inst_batch_idx = proposals_batch_idxs_[n]
                
                weights[n] = (batch_idxs_ == inst_batch_idx).int()

                corresponding_inst_label = fg_inds[gt_inds[inst_idxs[n]]] # get inst label of this inst_pred
                mask_labels[n] = (instance_labels[object_idxs] == corresponding_inst_label)

                valid_id = torch.nonzero(weights[n] > 0).view(-1)
                
                if valid_id.shape[0] > 0:
                    valid_mask_logits = mask_logits[n, valid_id].view(-1)
                    valid_mask_labels =  mask_labels[n, valid_id].view(-1)
                    # dice_loss += dice_coefficient(valid_mask_logits, valid_mask_labels).mean()
                    dice_loss += compute_dice_loss(valid_mask_logits, valid_mask_labels)
                    valid_num_dice_loss += 1

                    intersection = ((valid_mask_logits >= 0.5) * valid_mask_labels).sum()
                    union = (valid_mask_logits >= 0.5).sum() + valid_mask_labels.sum() - intersection
                    gt_iou = intersection / (union + 1e-4)
                    # gt_ious.append(gt_iou)

                    gt_ious_[n] = gt_iou
                gt_ious_mask_[n] = 1

            # print(class_id, mask_labels.sum())
            # mask_loss = F.binary_cross_entropy(
            #     mask_logits.sigmoid().type(torch.float32), mask_labels, weight=weights, reduction='sum')

            mask_loss = sigmoid_focal_loss(mask_logits, mask_labels, weights)

            # print(class_id, weights.sum(), mask_loss)
            if weights.sum() > 0:
                mask_loss_total += mask_loss
                mask_weight_total += weights.sum()

            gt_ious[inst_idxs] = gt_ious_
            gt_ious_mask[inst_idxs] = gt_ious_mask_

        # print('mask_loss_total', mask_loss_total, mask_weight_total+1)
        losses['mask_loss'] = mask_loss_total / (mask_weight_total + 1)
        losses['dice_loss'] = dice_loss / (valid_num_dice_loss + 1)
        # quit()
            # # compute iou score loss
            # ious = get_mask_iou_on_pred(proposals_idx, proposals_offset, instance_labels,
            #                             instance_pointnum, mask_scores_sigmoid_slice.detach())
            # fg_ious = ious[:, fg_inds]
            # gt_ious, _ = fg_ious.max(1)
            # slice_inds = torch.arange(0, labels.size(0), dtype=torch.long, device=labels.device)
            # iou_score_weight = (labels < self.instance_classes).float()
            # iou_score_slice = iou_scores[slice_inds, labels]

        # slice_inds = torch.arange(0, iou_scores.size(0), dtype=torch.long, device=iou_scores.device)
        # iou_score_slice = iou_scores[slice_inds, labels]
        # iou_score_loss = F.mse_loss(iou_score_slice, gt_ious, reduction='none')

        # breakpoint()

        iou_score_loss = F.mse_loss(iou_scores, gt_ious, reduction='none')
        iou_score_loss = (iou_score_loss * gt_ious_mask).sum() / (gt_ious_mask.sum() + 1e-4)
        losses['iou_score_loss'] = iou_score_loss
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
                      semantic_labels, instance_labels, instance_pointnum, instance_cls,
                      pt_offset_labels, pt_offset_vertices_labels, spatial_shape, batch_size, scan_ids, **kwargs):

        if self.embedding_coord:
            feats = torch.cat((feats, self.pos_embed(coords_float)), 1)
        else:
            feats = torch.cat((feats, coords_float), 1)
        # feats = torch.cat((feats, coords_float), 1)
        voxel_feats = voxelization(feats, p2v_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        semantic_scores, pt_offsets, pt_offsets_vertices, box_conf, output_feats = self.forward_backbone(
            input, v2p_map, x4_split=self.test_cfg.x4_split)
        if self.test_cfg.x4_split:
            coords_float = self.merge_4_parts(coords_float)
            semantic_labels = self.merge_4_parts(semantic_labels)
            instance_labels = self.merge_4_parts(instance_labels)
            pt_offset_labels = self.merge_4_parts(pt_offset_labels)
        semantic_preds = semantic_scores.max(1)[1]
        
        ret = dict(
            scan_id=scan_ids[0],
            coords_float=coords_float.cpu().numpy(),
            semantic_preds=semantic_preds.cpu().numpy(),
            semantic_labels=semantic_labels.cpu().numpy(),
            offset_preds=pt_offsets.cpu().numpy(),
            offset_vertices_preds=pt_offsets_vertices.cpu().numpy(),
            offset_labels=pt_offset_labels.cpu().numpy(),
            instance_labels=instance_labels.cpu().numpy())

        save_dict = dict(
            coords_float=coords_float.cpu().numpy(),
            semantic_scores=semantic_scores.cpu().numpy(),
            semantic_labels=semantic_labels.cpu().numpy(),
            instance_labels=instance_labels.cpu().numpy(),
            box_conf=box_conf.cpu().numpy(),
        )

        with open(f'/home/ubuntu/fewshot3d_ws/SoftGroup/results/bbox_context/sem_info/info_{scan_ids[0]}.pkl', 'wb') as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if not self.semantic_only:
            proposals_idx, proposals_offset, proposals_box, proposals_conf, proposals_pivots, proposals_cls, proposals_batch_idxs, per_cls_object_idxs = \
                                            self.forward_grouping(semantic_scores, pt_offsets, pt_offsets_vertices, box_conf,
                                                                    batch_idxs, coords_float,
                                                                    self.grouping_cfg)

            inst_feats, inst_map, inst_coords_mean, coords_bounding = self.clusters_voxelization(
                proposals_idx.cpu(),
                proposals_offset.cpu(),
                output_feats,
                coords_float,
                pt_offsets,
                rand_quantize=True,
                **self.instance_voxel_cfg)


            instance_batch_idxs, cls_scores, iou_scores, mask_logits_dict, inst_idxs_dict = self.forward_instance(
                inst_feats, inst_map, inst_coords_mean, coords_bounding, coords_float, pt_offsets, output_feats, per_cls_object_idxs, batch_idxs, proposals_batch_idxs)


            pred_instances = self.get_instances(scan_ids[0], proposals_idx, semantic_scores,
                                                cls_scores, iou_scores, mask_logits_dict, inst_idxs_dict, per_cls_object_idxs)
            gt_instances = self.get_gt_instances(semantic_labels, instance_labels)
            ret.update(dict(pred_instances=pred_instances, gt_instances=gt_instances))
            # ret.update(dict(gt_instances=gt_instances))

            # debug_accu = self.debug_accu_classification(cls_scores, mask_scores, iou_scores, proposals_idx,
            #                                    proposals_offset, instance_labels, instance_pointnum,
            #                                    instance_cls)
            # ret.update(dict(debug_accu=debug_accu.cpu().numpy()))
        return ret

    def forward_backbone(self, input, input_map, x4_split=False):

        context = torch.no_grad if self.freeze_backbone else torch.enable_grad
        with context():
            if x4_split:
                output_feats = self.forward_4_parts(input, input_map)
                output_feats = self.merge_4_parts(output_feats)
            else:
                output = self.input_conv(input)
                output = self.unet(output)
                output = self.output_layer(output)
                output_feats = output.features[input_map.long()]

            semantic_scores = self.semantic_linear(output_feats)
            pt_offsets = self.offset_linear(output_feats)
            pt_offsets_vertices = self.offset_vertices_linear(output_feats)
            box_conf = self.box_conf_linear(output_feats).squeeze(-1)
            return semantic_scores, pt_offsets, pt_offsets_vertices, box_conf, output_feats

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

    @torch.no_grad()
    @force_fp32(apply_to=('semantic_scores, pt_offsets'))
    def forward_grouping(self,
                         semantic_scores,
                         pt_offsets,
                         pt_offsets_vertices, box_conf,
                         batch_idxs,
                         coords_float,
                         grouping_cfg=None):


        proposals_idx_list = []
        proposals_offset_list = []
        proposals_box_list = []
        proposals_cls_list = []
        proposals_conf_list = []
        proposal_pivots_list = []
        proposals_batch_idxs_list = []

        batch_size = batch_idxs.max() + 1
        semantic_scores = semantic_scores.softmax(dim=-1)

        # box_iou_thresh = self.grouping_cfg.box_iou_thresh
        radius = self.grouping_cfg.radius
        mean_active = self.grouping_cfg.mean_active
        npoint_thr = self.grouping_cfg.npoint_thr
        class_numpoint_mean = torch.tensor(
            self.grouping_cfg.class_numpoint_mean, dtype=torch.float32)

        per_cls_object_idxs = {}
        for class_id in range(self.semantic_classes):
            if class_id in self.grouping_cfg.ignore_classes:
                continue
            scores = semantic_scores[:, class_id].contiguous()

            # object_idxs = (scores > self.grouping_cfg.score_thr).nonzero().view(-1)
            object_idxs = torch.nonzero(scores > self.grouping_cfg.score_thr).view(-1)
            per_cls_object_idxs[class_id] = object_idxs

            if object_idxs.size(0) < self.test_cfg.min_npoint:
                continue
            batch_idxs_ = batch_idxs[object_idxs]
            batch_offsets_ = self.get_batch_offsets(batch_idxs_, batch_size)
            coords_ = coords_float[object_idxs]
            pt_offsets_ = pt_offsets[object_idxs]
            box_conf_ = box_conf[object_idxs]
            pt_offsets_vertices_ = pt_offsets_vertices[object_idxs]

            # NOTE NMC
            box_conf_ = box_conf_ * scores[object_idxs]
            nmc_outputs = non_maximum_cluster(box_conf_, coords_, pt_offsets_, pt_offsets_vertices_, batch_offsets_, mean_active=self.grouping_cfg.mean_active_nmc, iou_thresh=self.grouping_cfg.iou_thresh)

            if nmc_outputs is not None:
                proposals_idx, proposals_offset, proposals_box, proposals_conf, proposal_pivots = nmc_outputs

                # breakpoint()

            # if len(proposals_idx) > 0:
                proposal_pivots  = object_idxs[proposal_pivots.long()]
                proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()
                proposals_batch_idxs = batch_idxs[proposals_idx[proposals_offset[:-1].long(), 1].long()]

                if len(proposals_offset_list) > 0:
                    proposals_idx[:, 0] += sum([x.size(0) for x in proposals_offset_list]) - 1
                    proposals_offset += proposals_offset_list[-1][-1]
                    proposals_offset = proposals_offset[1:]

                proposals_idx_list.append(proposals_idx)
                proposals_offset_list.append(proposals_offset)
                proposals_box_list.append(proposals_box)
                proposals_conf_list.append(proposals_conf)
                proposal_pivots_list.append(proposal_pivots)
                proposals_batch_idxs_list.append(proposals_batch_idxs)

                proposals_cls_list.append(torch.ones((proposals_box.shape[0]), dtype=torch.int, device=proposals_box.device) * (class_id - 2 + 1))

        proposals_idx = torch.cat(proposals_idx_list, dim=0)
        proposals_offset = torch.cat(proposals_offset_list)
        proposals_box_list = torch.cat(proposals_box_list, dim=0) # nTotal, 6
        proposals_conf_list = torch.cat(proposals_conf_list, dim=0) # nTotal
        proposal_pivots_list = torch.cat(proposal_pivots_list, dim=0)
        proposals_cls_list = torch.cat(proposals_cls_list, dim=0) # nTotal
        proposals_batch_idxs = torch.cat(proposals_batch_idxs_list, dim=0)

        return proposals_idx, proposals_offset, proposals_box_list, proposals_conf_list, proposal_pivots_list, proposals_cls_list, proposals_batch_idxs, per_cls_object_idxs

    def parse_dynamic_params(self, params, out_channels):
        assert params.dim()==2
        assert len(self.weight_nums) == len(self.bias_nums)
        assert params.size(1) == sum(self.weight_nums) + sum(self.bias_nums)

        num_instances = params.size(0)
        num_layers = len(self.weight_nums)
        params_splits = list(torch.split_with_sizes(params, self.weight_nums+self.bias_nums, dim=1))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_instances*out_channels, -1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_instances*out_channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_instances, -1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_instances)

        return weight_splits, bias_splits

    def dyco_head(self, mask_features, weights, biases, inst_coords_mean_, coords_bounding_, coords_, relative_geo_dist, use_coords=True):
        num_insts = inst_coords_mean_.shape[0]
        assert mask_features.dim() == 3
        n_layers = len(weights)
        c = mask_features.size(1)
        n_mask = mask_features.size(0)
        x = mask_features.permute(2,1,0).repeat(num_insts, 1, 1) ### num_inst * c * N_mask


        relative_coords = inst_coords_mean_[:, None, :] - coords_[None, :, :]### N_inst * N_mask * 3
        relative_coords = relative_coords.permute(0,2,1) ### num_inst * 3 * n_mask

        relative_geo_dist = relative_geo_dist[:, None,:]

        # print('relative_geo_dist', relative_geo_dist.shape, x.shape, relative_coords.shape)
        if use_coords:
            x = torch.cat([relative_coords, relative_geo_dist, x], dim=1) ### num_inst * (3+c) * N_mask

        x = x.reshape(1, -1, n_mask) ### 1 * (num_inst*c') * Nmask
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv1d(x, w, bias=b, stride=1, padding=0, groups=num_insts)
            if i < n_layers - 1:
                x = F.relu(x)
        x = x.squeeze(0)
        return x

    def forward_instance(self, inst_feats, inst_map, inst_coords_mean, coords_bounding, coords_float, pt_offsets, output_feats, per_cls_object_idxs, batch_idxs, proposals_batch_idxs):

        label_shift = (self.semantic_classes - self.instance_classes)
        feats = self.tiny_unet(inst_feats)
        feats = self.tiny_unet_outputlayer(feats)

        # predict mask scores
        # mask_scores = self.mask_linear(feats.features)
        # mask_scores = mask_scores[inst_map.long()]
        instance_batch_idxs = feats.indices[:, 0][inst_map.long()]

        # predict instance cls and iou scores
        feats = self.global_pool(feats)

        cls_scores = self.cls_linear(feats)
        iou_scores = self.iou_score_linear(feats).squeeze(-1)

        controller = self.controller(feats.unsqueeze(-1)).squeeze(-1) # N, num_params

        cls_scores_softmax = cls_scores.softmax(dim=-1)
        inst_cls_pred = torch.max(cls_scores, dim=1)[1] # 0 -> 17

        mask_feats = self.mask_tower(output_feats.unsqueeze(-1).permute(2,1,0)).permute(2,1,0)

        batch_size = batch_idxs.max() + 1
        # final_proposals_idx_list = []
        # final_proposals_offset_list = []
        mask_logits_dict = {}
        inst_idxs_dict = {}
        for class_id in range(self.semantic_classes):
            if class_id in self.grouping_cfg.ignore_classes:
                continue

            object_idxs = per_cls_object_idxs[class_id]

            coords_ = coords_float[object_idxs, :]

            mask_feats_ = mask_feats[object_idxs, :]

            pt_offsets_ = pt_offsets[object_idxs, :]

            scores = cls_scores_softmax[:, class_id-label_shift].contiguous()
            inst_idxs = torch.nonzero(scores > self.grouping_cfg.score_thr).view(-1)
            # inst_idxs = torch.nonzero(inst_cls_pred == (class_id - label_shift)).view(-1)

            # inst_idxs = torch.arange(scores.shape[0], device=scores.device)

            if inst_idxs.shape[0] == 0 or object_idxs.shape[0] == 0:
                continue


            controller_ = controller[inst_idxs, :]
            weight_params_, bias_params_ = self.parse_dynamic_params(controller_, self.channels)

            inst_coords_mean_ = inst_coords_mean[inst_idxs]
            coords_bounding_ = coords_bounding[inst_idxs]

            batch_idxs_ = batch_idxs[object_idxs]
            proposals_batch_idxs_ = proposals_batch_idxs[inst_idxs]

            inst_coords_mean_catbatch = torch.cat([inst_coords_mean_, proposals_batch_idxs_[:, None]*1000.0], dim=-1)
            coords_catbatch = torch.cat([coords_, batch_idxs_[:, None]*1000.0], dim=-1)
            distances = torch.sum((inst_coords_mean_catbatch[:, None, :] - coords_catbatch[None, :, :])**2, dim=-1)
            pivot_inds = torch.argmin(distances, dim=-1)

            batch_offsets_ = self.get_batch_offsets(batch_idxs_, batch_size)
            # proposal_batch_offsets_ = self.get_batch_offsets(proposals_batch_idxs_, batch_size)


            # pivot_inds = proposals_pivots[inst_idxs]
            # pivot_coords_ = coords_float[pivot_inds]

            # breakpoint()
            # NOTE process geodist
            relative_geo_dist = cal_geodesic_vectorize(self.geo_knn, pivot_inds, coords_, batch_offsets_, proposals_batch_idxs_, batch_size,
                                                    max_step=128 if self.training else 256,
                                                    neighbor=64,
                                                    radius=0.05) # n_inst, n_points

            mask_logits = self.dyco_head(mask_feats_, weight_params_, bias_params_, inst_coords_mean_, coords_bounding_, coords_, relative_geo_dist) # N_inst_of_classX, N_points_of_classX
            mask_logits_dict[class_id] = mask_logits
            inst_idxs_dict[class_id] = inst_idxs
        # print('inst_idxs_dict', inst_idxs_dict)
        return instance_batch_idxs, cls_scores, iou_scores, mask_logits_dict, inst_idxs_dict

    @force_fp32(apply_to=('semantic_scores', 'cls_scores', 'iou_scores', 'mask_scores'))
    def get_instances(self, scan_id, proposals_idx, semantic_scores, cls_scores, iou_scores,
                      mask_logits_dict, inst_idxs_dict, per_cls_object_idxs):
        num_instances = cls_scores.size(0)
        num_points = semantic_scores.size(0)
        cls_scores = cls_scores.softmax(1)
        semantic_pred = semantic_scores.max(1)[1]
        cls_pred_list, score_pred_list, mask_pred_list = [], [], []

        # for i in range(self.instance_classes):
        #     if i in self.sem2ins_classes:
        #         cls_pred = cls_scores.new_tensor([i + 1], dtype=torch.long)
        #         score_pred = cls_scores.new_tensor([1.], dtype=torch.float32)
        #         mask_pred = (semantic_pred == i)[None, :].int()
        #     else:
        #         cls_pred = cls_scores.new_full((num_instances, ), i + 1, dtype=torch.long)
        #         cur_cls_scores = cls_scores[:, i]
        #         cur_iou_scores = iou_scores
                
        #         cur_mask_scores = mask_scores[:, i]

        #         score_pred = cur_cls_scores * cur_iou_scores.clamp(0, 1)
        #         mask_pred = torch.zeros((num_instances, num_points), dtype=torch.int, device='cuda')
        #         mask_inds = cur_mask_scores > self.test_cfg.mask_score_thr
        #         cur_proposals_idx = proposals_idx[mask_inds].long()
        #         mask_pred[cur_proposals_idx[:, 0], cur_proposals_idx[:, 1]] = 1

        #         # filter low score instance
        #         inds = cur_cls_scores > self.test_cfg.cls_score_thr
        #         cls_pred = cls_pred[inds]
        #         score_pred = score_pred[inds]
        #         mask_pred = mask_pred[inds]

        #         # filter too small instances
        #         npoint = mask_pred.sum(1)
        #         inds = npoint >= self.test_cfg.min_npoint
        #         cls_pred = cls_pred[inds]
        #         score_pred = score_pred[inds]
        #         mask_pred = mask_pred[inds]
        #     cls_pred_list.append(cls_pred)
        #     score_pred_list.append(score_pred)
        #     mask_pred_list.append(mask_pred)

        label_shift = self.semantic_classes - self.instance_classes
        for class_id in range(self.semantic_classes): # 0->19
            if class_id in self.grouping_cfg.ignore_classes:
                continue
            if class_id not in mask_logits_dict:
                continue
            
            

            object_idxs = per_cls_object_idxs[class_id] # fg points
            inst_idxs = inst_idxs_dict[class_id] # fg inst
            cur_mask_scores = mask_logits_dict[class_id]

            cur_cls_scores = cls_scores[inst_idxs, class_id-label_shift]

            # cur_cls_scores, cur_cls_inds = torch.max(cls_scores[inst_idxs], dim=1)
            # assert torch.all(cur_cls_inds == class_id-label_shift)

            cur_iou_scores = iou_scores[inst_idxs]
            

            num_instances = inst_idxs.shape[0]

            cls_pred = cur_cls_scores.new_full((num_instances, ), class_id-label_shift+1, dtype=torch.long)

            score_pred = cur_cls_scores * cur_iou_scores.clamp(0, 1)
            # score_pred = cur_cls_scores
            # score_pred = cur_cls_scores.sigmoid() * cur_iou_scores.sigmoid()

            mask_pred = torch.zeros((num_instances, num_points), dtype=torch.int, device='cuda')
            mask_inds = (cur_mask_scores > self.test_cfg.mask_score_thr).type(torch.int)
            mask_pred[:, object_idxs] = mask_inds

            # filter low score instance
            # print('cur_cls_scores', cur_cls_scores.shape, inds.shape)
            inds = cur_cls_scores > self.test_cfg.cls_score_thr
            cls_pred = cls_pred[inds]
            score_pred = score_pred[inds]
            mask_pred = mask_pred[inds]

            # filter too small instances
            npoint = mask_pred.sum(1)
            inds = npoint >= self.test_cfg.min_npoint
            cls_pred = cls_pred[inds]
            score_pred = score_pred[inds]
            mask_pred = mask_pred[inds]

            cls_pred_list.append(cls_pred)
            score_pred_list.append(score_pred)
            mask_pred_list.append(mask_pred)

        cls_pred = torch.cat(cls_pred_list).cpu().numpy()
        score_pred = torch.cat(score_pred_list).cpu().numpy()
        mask_pred = torch.cat(mask_pred_list).cpu().numpy()

        instances = []
        for i in range(cls_pred.shape[0]):
            pred = {}
            pred['scan_id'] = scan_id
            pred['label_id'] = cls_pred[i]
            pred['conf'] = score_pred[i]
            # rle encode mask to save memory
            pred['pred_mask'] = rle_encode(mask_pred[i])
            instances.append(pred)
        return instances

    def get_gt_instances(self, semantic_labels, instance_labels):
        """Get gt instances for evaluation."""
        # convert to evaluation format 0: ignore, 1->N: valid
        label_shift = self.semantic_classes - self.instance_classes
        semantic_labels = semantic_labels - label_shift + 1
        semantic_labels[semantic_labels < 0] = 0
        instance_labels += 1
        ignore_inds = instance_labels < 0
        # scannet encoding rule
        gt_ins = semantic_labels * 1000 + instance_labels
        gt_ins[ignore_inds] = 0
        gt_ins = gt_ins.cpu().numpy()
        return gt_ins

    @force_fp32(apply_to='feats')
    def clusters_voxelization(self,
                              clusters_idx,
                              clusters_offset,
                              feats,
                              coords,
                              pt_offsets,
                              scale,
                              spatial_shape,
                              rand_quantize=False):
        batch_idx = clusters_idx[:, 0].cuda().long()
        c_idxs = clusters_idx[:, 1].cuda()
        feats = feats[c_idxs.long()]
        coords = coords[c_idxs.long()]

        clusters_offset_cuda = clusters_offset.cuda()
        coords_mean = sec_mean(coords + pt_offsets[c_idxs.long()], clusters_offset_cuda) # N, 3

        # breakpoint()

        coords_min = sec_min(coords, clusters_offset_cuda) # N, 3
        coords_max = sec_max(coords, clusters_offset_cuda) # N, 3

        coords_bounding = get_bounding_vertices(coords_min, coords_max, coords_mean)



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
        return voxelization_feats, inp_map, coords_mean, coords_bounding

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

    @force_fp32(apply_to=('cls_scores', 'mask_scores', 'iou_scores'))
    def debug_accu_classification(self, cls_scores, mask_scores, iou_scores, proposals_idx, proposals_offset,
                      instance_labels, instance_pointnum, instance_cls):
        with torch.no_grad():
            proposals_idx = proposals_idx[:, 1].cuda()
            proposals_offset = proposals_offset.cuda()

            # cal iou of clustered instance
            ious_on_cluster = get_mask_iou_on_cluster(proposals_idx, proposals_offset, instance_labels,
                                                    instance_pointnum)

            # filter out background instances
            fg_inds = (instance_cls != self.ignore_label)
            fg_instance_cls = instance_cls[fg_inds]
            fg_ious_on_cluster = ious_on_cluster[:, fg_inds]

            # overlap > thr on fg instances are positive samples
            max_iou, gt_inds = fg_ious_on_cluster.max(1)
            pos_inds = max_iou >= self.train_cfg.pos_iou_thr
            pos_gt_inds = gt_inds[pos_inds]

            # compute cls loss. follow detection convention: 0 -> K - 1 are fg, K is bg
            labels = fg_instance_cls.new_full((fg_ious_on_cluster.size(0), ), self.instance_classes)
            labels[pos_inds] = fg_instance_cls[pos_gt_inds]

            cls_scores_conf, cls_scores_pred = torch.max(cls_scores, -1)

            # inds = cur_cls_scores > self.test_cfg.cls_score_thr
            cls_scores_pred[cls_scores_conf <= self.test_cfg.cls_score_thr] = self.instance_classes
            # iou_scores_pred = torch.index_select(iou_scores, 1, cls_scores_pred)
            # score_pred = cls_scores_conf * iou_scores_pred.clamp(0, 1)
            # cls_loss = F.cross_entropy(cls_scores, labels)
            accuracy = ((cls_scores_pred == labels).sum() / cls_scores_pred.shape[0])

            

            return accuracy
