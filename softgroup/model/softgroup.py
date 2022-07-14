import functools

import spconv.pytorch as spconv
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..ops import (ballquery_batch_p, ballquery_batch_p_boxiou, bfs_cluster, get_mask_iou_on_cluster, get_mask_iou_on_pred,
                   get_mask_label, global_avg_pool, sec_max, sec_min, voxelization,
                   voxelization_idx)
from ..util import cuda_cast, force_fp32, rle_encode
from .blocks import MLP, ResidualBlock, UBlock, PositionalEmbedding
from .model_utils import iou_aabb, compute_dice_loss, sigmoid_focal_loss, non_maximum_cluster, non_maximum_cluster2
import numpy as np



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
        # self.offset_vertices_linear = MLP(channels, 3*2, norm_fn=norm_fn, num_layers=2)
        self.offset_vertices_linear = MLP(channels, 3*8, norm_fn=norm_fn, num_layers=2)
        self.box_conf_linear = MLP(channels, 1, norm_fn=norm_fn, num_layers=2)

        # topdown refinement path
        if not semantic_only:
            self.tiny_unet = UBlock([channels, 2 * channels], norm_fn, 2, block, indice_key_id=11)
            self.tiny_unet_outputlayer = spconv.SparseSequential(norm_fn(channels), nn.ReLU())
            self.cls_linear = nn.Linear(channels, instance_classes + 1)
            self.mask_linear = MLP(channels, instance_classes + 1, norm_fn=None, num_layers=2)
            self.iou_score_linear = nn.Linear(channels, instance_classes + 1)

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
            proposals_idx, proposals_offset = self.forward_grouping(semantic_scores, pt_offsets, pt_offsets_vertices, box_conf,
                                                                    batch_idxs, coords_float,
                                                                    self.grouping_cfg)
            if proposals_offset.shape[0] > self.train_cfg.max_proposal_num:
                proposals_offset = proposals_offset[:self.train_cfg.max_proposal_num + 1]
                proposals_idx = proposals_idx[:proposals_offset[-1]]
                assert proposals_idx.shape[0] == proposals_offset[-1]
            inst_feats, inst_map = self.clusters_voxelization(
                proposals_idx.cpu(),
                proposals_offset.cpu(),
                output_feats,
                coords_float,
                rand_quantize=True,
                **self.instance_voxel_cfg)
            instance_batch_idxs, cls_scores, iou_scores, mask_scores = self.forward_instance(
                inst_feats, inst_map)
            instance_loss = self.instance_loss(cls_scores, mask_scores, iou_scores, proposals_idx,
                                               proposals_offset, instance_labels, instance_pointnum,
                                               instance_cls, instance_batch_idxs)
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

    @force_fp32(apply_to=('cls_scores', 'mask_scores', 'iou_scores'))
    def instance_loss(self, cls_scores, mask_scores, iou_scores, proposals_idx, proposals_offset,
                      instance_labels, instance_pointnum, instance_cls, instance_batch_idxs):
        losses = {}
        proposals_idx = proposals_idx[:, 1].cuda()
        proposals_offset = proposals_offset.cuda()

        # cal iou of clustered instance
        ious_on_cluster = get_mask_iou_on_cluster(proposals_idx, proposals_offset.int(), instance_labels,
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
        cls_loss = F.cross_entropy(cls_scores, labels)
        losses['cls_loss'] = cls_loss

        # compute mask loss
        mask_cls_label = labels[instance_batch_idxs.long()]
        slice_inds = torch.arange(
            0, mask_cls_label.size(0), dtype=torch.long, device=mask_cls_label.device)
        mask_scores_sigmoid_slice = mask_scores.sigmoid()[slice_inds, mask_cls_label]
        mask_label = get_mask_label(proposals_idx, proposals_offset, instance_labels, instance_cls,
                                    instance_pointnum, ious_on_cluster, self.train_cfg.pos_iou_thr)
        mask_label_weight = (mask_label != -1).float()
        mask_label[mask_label == -1.] = 0.5  # any value is ok
        
        if self.train_cfg.focal_loss:
            mask_loss = sigmoid_focal_loss(mask_scores_sigmoid_slice, mask_label, mask_label_weight)
        else:
            mask_loss = F.binary_cross_entropy(
                mask_scores_sigmoid_slice, mask_label, weight=mask_label_weight, reduction='sum')

        mask_loss /= (mask_label_weight.sum() + 1)
        losses['mask_loss'] = mask_loss

        # compute iou score loss
        ious = get_mask_iou_on_pred(proposals_idx, proposals_offset, instance_labels,
                                    instance_pointnum, mask_scores_sigmoid_slice.detach())
        fg_ious = ious[:, fg_inds]
        gt_ious, _ = fg_ious.max(1)
        slice_inds = torch.arange(0, labels.size(0), dtype=torch.long, device=labels.device)
        iou_score_weight = (labels < self.instance_classes).float()
        iou_score_slice = iou_scores[slice_inds, labels]
        iou_score_loss = F.mse_loss(iou_score_slice, gt_ious, reduction='none')
        iou_score_loss = (iou_score_loss * iou_score_weight).sum() / (iou_score_weight.sum() + 1)
        losses['iou_score_loss'] = iou_score_loss

        if self.train_cfg.dice_loss:
            dice_loss = compute_dice_loss(mask_scores_sigmoid_slice[mask_label_weight.bool()], mask_label[mask_label_weight.bool()])
            losses['dice_loss'] = dice_loss

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

    # @cuda_cast
    # def forward_test(self, batch_idxs, voxel_coords, p2v_map, v2p_map, coords_float, feats,
    #                  semantic_labels, instance_labels, pt_offset_labels, spatial_shape, batch_size,
    #                  scan_ids, **kwargs):
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
        if not self.semantic_only:
            proposals_idx, proposals_offset, proposals_box, proposals_conf, proposals_cls = self.forward_grouping(semantic_scores, pt_offsets, pt_offsets_vertices, box_conf,
                                                                    batch_idxs, coords_float,
                                                                    self.grouping_cfg)
            # proposals_idx, proposals_offset, proposals_box, proposals_conf, proposals_cls = self.forward_grouping_origin(semantic_scores, pt_offsets, pt_offsets_vertices, box_conf,
            #                                                         batch_idxs, coords_float,
            #                                                         self.grouping_cfg)

            # FIXME DEBUG
            # nmc_clusters = torch.LongTensor(feats.shape[0], 1).fill_(-100)
            # for idx in range(proposals_offset.shape[0]-1):
            #     nmc_clusters[proposals_idx[proposals_offset[idx]:proposals_offset[idx+1],1].long()] = idx
            # ret['nmc_clusters'] = nmc_clusters.cpu().numpy()

            # proposals_box = proposals_box.cpu().numpy()
            # proposals_conf = proposals_conf.cpu().numpy()
            # proposals_cls = proposals_cls.cpu().numpy()

            # nmc_instances = []
            # for i in range(proposals_cls.shape[0]):
            #     pred = {}
            #     pred['scan_id'] = scan_ids[0]
            #     pred['label_id'] = proposals_cls[i]
            #     pred['conf'] = proposals_conf[i]
            #     # rle encode mask to save memory
            #     mask_pred = torch.zeros((feats.shape[0]), dtype=torch.int).to(feats.device)
            #     mask_pred[proposals_idx[proposals_offset[i]:proposals_offset[i+1],1].long()] = 1

            #     pred['pred_mask'] = rle_encode(mask_pred.cpu().numpy())
            #     pred['box'] = proposals_box[i]
            #     nmc_instances.append(pred)
            # ret.update(dict(nmc_instances=nmc_instances))


            inst_feats, inst_map = self.clusters_voxelization(proposals_idx.cpu(), proposals_offset.cpu(),
                                                                output_feats, coords_float,
                                                                **self.instance_voxel_cfg)
            _, cls_scores, iou_scores, mask_scores = self.forward_instance(inst_feats, inst_map)
            pred_instances = self.get_instances(scan_ids[0], proposals_idx, semantic_scores,
                                                cls_scores, iou_scores, mask_scores, coords_float)
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
        batch_size = batch_idxs.max() + 1
        semantic_scores = semantic_scores.softmax(dim=-1)

        # box_iou_thresh = self.grouping_cfg.box_iou_thresh
        radius = self.grouping_cfg.radius
        mean_active = self.grouping_cfg.mean_active
        npoint_thr = self.grouping_cfg.npoint_thr
        class_numpoint_mean = torch.tensor(
            self.grouping_cfg.class_numpoint_mean, dtype=torch.float32)


        for class_id in range(self.semantic_classes):
            if class_id in self.grouping_cfg.ignore_classes:
                continue
            scores = semantic_scores[:, class_id].contiguous()

            # object_idxs = (scores > self.grouping_cfg.score_thr).nonzero().view(-1)
            object_idxs = torch.nonzero(scores > self.grouping_cfg.score_thr).view(-1)
            if object_idxs.size(0) < self.test_cfg.min_npoint:
                continue
            batch_idxs_ = batch_idxs[object_idxs]
            batch_offsets_ = self.get_batch_offsets(batch_idxs_, batch_size)
            coords_ = coords_float[object_idxs]
            pt_offsets_ = pt_offsets[object_idxs]
            box_conf_ = box_conf[object_idxs]
            pt_offsets_vertices_ = pt_offsets_vertices[object_idxs]

                # NOTE BALL QUERY
                # idx, start_len = ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_, batch_offsets_,
                #                                    radius, mean_active)
                # coords_box_ = coords_.repeat(1, 2) + pt_offsets_vertices_

                # coords_box_ = torch.cat([coords_ + pt_offsets_vertices_[:, :3], coords_ + pt_offsets_vertices_[:, 3:]], -1).contiguous()
                # coords_min_ = (coords_ + pt_offsets_vertices_[:, :3]).contiguous().cuda()
                # coords_max_ = (coords_ + pt_offsets_vertices_[:, 3:]).contiguous().cuda()
                # # breakpoint()
                # idx, start_len = ballquery_batch_p_boxiou(coords_min_, coords_max_, batch_idxs_, batch_offsets_,
                #                                 box_iou_thresh, mean_active)
                # proposals_idx, proposals_offset = bfs_cluster(class_numpoint_mean, idx.cpu(),
                #                                             start_len.cpu(), npoint_thr, class_id)

            # NOTE NMC
            box_conf_ = box_conf_ * scores[object_idxs]
            proposals_idx, proposals_offset, proposals_box, proposals_conf = non_maximum_cluster(box_conf_, coords_, pt_offsets_, pt_offsets_vertices_, batch_offsets_, mean_active=self.grouping_cfg.mean_active_nmc, iou_thresh=self.grouping_cfg.iou_thresh)

            # TODO measure mAP, mAR

            if len(proposals_idx) > 0:
                proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()
                if len(proposals_offset_list) > 0:
                    proposals_idx[:, 0] += sum([x.size(0) for x in proposals_offset_list]) - 1
                    proposals_offset += proposals_offset_list[-1][-1]
                    proposals_offset = proposals_offset[1:]

                proposals_idx_list.append(proposals_idx)
                proposals_offset_list.append(proposals_offset)
                proposals_box_list.append(proposals_box)
                proposals_conf_list.append(proposals_conf)
                proposals_cls_list.append(torch.ones((proposals_box.shape[0]), dtype=torch.int, device=proposals_box.device) * (class_id - 2 + 1))

            # score_pred, class_pred = torch.max(semantic_scores, 1)
            # object_idxs = torch.nonzero(class_pred > 1).view(-1)
            # batch_idxs_ = batch_idxs[object_idxs]
            # batch_offsets_ = self.get_batch_offsets(batch_idxs_, batch_size)
            # coords_ = coords_float[object_idxs]
            # pt_offsets_ = pt_offsets[object_idxs]
            # box_conf_ = box_conf[object_idxs]
            # pt_offsets_vertices_ = pt_offsets_vertices[object_idxs]

            # # idx, start_len = ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_, batch_offsets_,
            # #                                     radius, mean_active)

            # coords_min_ = (coords_ + pt_offsets_vertices_[:, :3]).contiguous().cuda()
            # coords_max_ = (coords_ + pt_offsets_vertices_[:, 3:]).contiguous().cuda()
            # # breakpoint()
            # idx, start_len = ballquery_batch_p_boxiou(coords_min_, coords_max_, batch_idxs_, batch_offsets_,
            #                                 box_iou_thresh, mean_active)

            # class_id = 2
            # proposals_idx, proposals_offset = bfs_cluster(class_numpoint_mean, idx.cpu(),
            #                                             start_len.cpu(), npoint_thr, class_id)

            # # box_con_ = score_pred[object_idxs]
            # # proposals_idx, proposals_offset = non_maximum_cluster(box_conf_, coords_, pt_offsets_vertices_, batch_idxs_, batch_offsets_)
            # if len(proposals_idx) > 0:
            #     proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()
            #     if len(proposals_offset_list) > 0:
            #         proposals_idx[:, 0] += sum([x.size(0) for x in proposals_offset_list]) - 1
            #         proposals_offset += proposals_offset_list[-1][-1]
            #         proposals_offset = proposals_offset[1:]

            #     proposals_idx_list.append(proposals_idx)
            #     proposals_offset_list.append(proposals_offset)
        
        proposals_idx = torch.cat(proposals_idx_list, dim=0)
        proposals_offset = torch.cat(proposals_offset_list)
        proposals_box_list = torch.cat(proposals_box_list, dim=0) # nTotal, 6
        proposals_conf_list = torch.cat(proposals_conf_list, dim=0) # nTotal
        proposals_cls_list = torch.cat(proposals_cls_list, dim=0) # nTotal

        # breakpoint()
        # print('num nmc clusters', proposals_offset.shape)
        return proposals_idx, proposals_offset, proposals_box_list, proposals_conf_list, proposals_cls_list

    @torch.no_grad()
    @force_fp32(apply_to=('semantic_scores, pt_offsets'))
    def forward_grouping_origin(self,
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
        batch_size = batch_idxs.max() + 1
        semantic_scores = semantic_scores.softmax(dim=-1)

        # box_iou_thresh = self.grouping_cfg.box_iou_thresh
        radius = self.grouping_cfg.radius
        mean_active = self.grouping_cfg.mean_active
        npoint_thr = self.grouping_cfg.npoint_thr
        class_numpoint_mean = torch.tensor(
            self.grouping_cfg.class_numpoint_mean, dtype=torch.float32)


        for class_id in range(self.semantic_classes):
            if class_id in self.grouping_cfg.ignore_classes:
                continue
            scores = semantic_scores[:, class_id].contiguous()

            # object_idxs = (scores > self.grouping_cfg.score_thr).nonzero().view(-1)
            object_idxs = torch.nonzero(scores > self.grouping_cfg.score_thr).view(-1)
            if object_idxs.size(0) < self.test_cfg.min_npoint:
                continue
            batch_idxs_ = batch_idxs[object_idxs]
            batch_offsets_ = self.get_batch_offsets(batch_idxs_, batch_size)
            coords_ = coords_float[object_idxs]
            pt_offsets_ = pt_offsets[object_idxs]
            box_conf_ = box_conf[object_idxs]
            pt_offsets_vertices_ = pt_offsets_vertices[object_idxs]

                # NOTE BALL QUERY
                # idx, start_len = ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_, batch_offsets_,
                #                                    radius, mean_active)
                # coords_box_ = coords_.repeat(1, 2) + pt_offsets_vertices_

                # coords_box_ = torch.cat([coords_ + pt_offsets_vertices_[:, :3], coords_ + pt_offsets_vertices_[:, 3:]], -1).contiguous()
                # coords_min_ = (coords_ + pt_offsets_vertices_[:, :3]).contiguous().cuda()
                # coords_max_ = (coords_ + pt_offsets_vertices_[:, 3:]).contiguous().cuda()
                # # breakpoint()
                # idx, start_len = ballquery_batch_p_boxiou(coords_min_, coords_max_, batch_idxs_, batch_offsets_,
                #                                 box_iou_thresh, mean_active)
                # proposals_idx, proposals_offset = bfs_cluster(class_numpoint_mean, idx.cpu(),
                #                                             start_len.cpu(), npoint_thr, class_id)

            # NOTE NMC
            box_conf_ = box_conf_ * scores[object_idxs]
            proposals_idx, proposals_offset, _, _ = non_maximum_cluster(box_conf_, coords_, pt_offsets_, pt_offsets_vertices_, batch_offsets_, mean_active=self.grouping_cfg.mean_active_nmc, iou_thresh=self.grouping_cfg.iou_thresh)

            # TODO measure mAP, mAR

            if len(proposals_idx) > 0:
                proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()

                proposals_box = []
                proposals_conf = []
                for idx in range(len(proposals_offset) - 1):
                    inst_indices = proposals_idx[proposals_offset[idx]:proposals_offset[idx+1], 1].long()
                    inst = coords_float[inst_indices]
                    score_inst = torch.mean(semantic_scores[inst_indices, class_id])
                    box_min = torch.min(inst, 0)[0]
                    box_max = torch.max(inst, 0)[0]
                    box = torch.cat([box_min, box_max])
                    proposals_box.append(box)
                    proposals_conf.append(score_inst)
                proposals_box = torch.stack(proposals_box)
                proposals_conf = torch.tensor(proposals_conf)


                if len(proposals_offset_list) > 0:
                    proposals_idx[:, 0] += sum([x.size(0) for x in proposals_offset_list]) - 1
                    proposals_offset += proposals_offset_list[-1][-1]
                    proposals_offset = proposals_offset[1:]

                proposals_idx_list.append(proposals_idx)
                proposals_offset_list.append(proposals_offset)

                
                proposals_box_list.append(proposals_box)
                proposals_conf_list.append(proposals_conf)
                proposals_cls_list.append(torch.ones((proposals_box.shape[0]), dtype=torch.int, device=proposals_box.device) * (class_id - 2 + 1))

        
        proposals_idx = torch.cat(proposals_idx_list, dim=0)
        proposals_offset = torch.cat(proposals_offset_list)
        proposals_box_list = torch.cat(proposals_box_list, dim=0) # nTotal, 6
        proposals_conf_list = torch.cat(proposals_conf_list, dim=0) # nTotal
        proposals_cls_list = torch.cat(proposals_cls_list, dim=0) # nTotal

        # breakpoint()
        # print('num nmc clusters', proposals_offset.shape)
        return proposals_idx, proposals_offset, proposals_box_list, proposals_conf_list, proposals_cls_list

    def forward_instance(self, inst_feats, inst_map):
        feats = self.tiny_unet(inst_feats)
        feats = self.tiny_unet_outputlayer(feats)

        # predict mask scores
        mask_scores = self.mask_linear(feats.features)
        mask_scores = mask_scores[inst_map.long()]
        instance_batch_idxs = feats.indices[:, 0][inst_map.long()]

        # predict instance cls and iou scores
        feats = self.global_pool(feats)
        cls_scores = self.cls_linear(feats)
        iou_scores = self.iou_score_linear(feats)

        return instance_batch_idxs, cls_scores, iou_scores, mask_scores

    @force_fp32(apply_to=('semantic_scores', 'cls_scores', 'iou_scores', 'mask_scores'))
    def get_instances(self, scan_id, proposals_idx, semantic_scores, cls_scores, iou_scores,
                      mask_scores, coords_float):
        num_instances = cls_scores.size(0)
        num_points = semantic_scores.size(0)
        cls_scores = cls_scores.softmax(1)
        semantic_pred = semantic_scores.max(1)[1]
        cls_pred_list, score_pred_list, mask_pred_list = [], [], []

        for i in range(self.instance_classes):
            if i in self.sem2ins_classes:
                cls_pred = cls_scores.new_tensor([i + 1], dtype=torch.long)
                score_pred = cls_scores.new_tensor([1.], dtype=torch.float32)
                mask_pred = (semantic_pred == i)[None, :].int()
            else:
                cls_pred = cls_scores.new_full((num_instances, ), i + 1, dtype=torch.long)
                cur_cls_scores = cls_scores[:, i]
                cur_iou_scores = iou_scores[:, i]
                cur_mask_scores = mask_scores[:, i]
                score_pred = cur_cls_scores * cur_iou_scores.clamp(0, 1)
                mask_pred = torch.zeros((num_instances, num_points), dtype=torch.int, device='cuda')
                mask_inds = cur_mask_scores > self.test_cfg.mask_score_thr
                cur_proposals_idx = proposals_idx[mask_inds].long()
                mask_pred[cur_proposals_idx[:, 0], cur_proposals_idx[:, 1]] = 1

                # filter low score instance
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
