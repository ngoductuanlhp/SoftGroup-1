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
from .blocks import MLP, ResidualBlock, UBlock, PositionalEmbedding, conv_with_kaiming_uniform
from .model_utils import giou_aabb, iou_aabb, compute_dice_loss, sigmoid_focal_loss, non_maximum_cluster, non_maximum_cluster2, non_max_suppression_gpu, non_maximum_queries
# from ..pointnet2 import pointnet2_utils
from softgroup.pointnet2.pointnet2_utils import furthest_point_sample

from softgroup.pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetSAModuleVotesSeparate
from softgroup.pointnet2.pointnet2_utils import furthest_point_sample

from .detr.pos_embedding import PositionEmbeddingCoordsSine
from .detr.transformer_layers import TransformerDecoderLayer, TransformerDecoder
from .detr.helper import GenericMLP

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
                 transformer_cfg=None,
                 grouping_cfg=None,
                 instance_voxel_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 fixed_modules=[],
                 embedding_coord=False,
                 criterion=None):
        super().__init__()
        self.channels = channels
        self.num_blocks = num_blocks
        self.semantic_only = semantic_only
        self.semantic_classes = semantic_classes
        self.instance_classes = instance_classes
        self.sem2ins_classes = sem2ins_classes
        self.ignore_label = ignore_label
        self.transformer_cfg = transformer_cfg
        self.grouping_cfg = grouping_cfg
        self.instance_voxel_cfg = instance_voxel_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fixed_modules = fixed_modules
        self.embedding_coord = embedding_coord

        self.criterion = criterion
        

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
        self.box_conf_linear = MLP(channels, 1, norm_fn=norm_fn, num_layers=2)

        # topdown refinement path
        # if not semantic_only:
        #     self.tiny_unet = UBlock([channels, 2 * channels], norm_fn, 2, block, indice_key_id=11)
        #     self.tiny_unet_outputlayer = spconv.SparseSequential(norm_fn(channels), nn.ReLU())
        #     self.cls_linear = nn.Linear(channels, instance_classes + 1)
        #     self.mask_linear = MLP(channels, instance_classes + 1, norm_fn=None, num_layers=2)
        #     self.iou_score_linear = nn.Linear(channels, instance_classes + 1)

        # NOTE trans feats
        self.trans_channels = 2 * channels
        self.trans_linear = MLP(channels, self.trans_channels, norm_fn=norm_fn, num_layers=3)

        # NOTE dyco
        self.init_dyco()

        # NOTE transformer decoder

        ''' Set aggregate '''
        # set_aggregate_dim_out = 2 * self.channels
        # mlp_dims = [self.channels, 2*self.channels, 2*self.channels, set_aggregate_dim_out]
        # self.set_aggregator = PointnetSAModuleVotesSeparate(
        #     radius=0.2,
        #     nsample=64,
        #     npoint=transformer_cfg.n_context_points,
        #     mlp=mlp_dims,
        #     normalize_xyz=True,
        # )

        ''' Position embedding '''
        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=transformer_cfg.dec_dim, pos_type="fourier", normalize=True
        )

        self.query_projection = GenericMLP(
            input_dim=transformer_cfg.dec_dim,
            hidden_dims=[transformer_cfg.dec_dim],
            output_dim=transformer_cfg.dec_dim,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )

        # self.encoder_to_decoder_projection = GenericMLP(
        #     input_dim=self.trans_channels,
        #     hidden_dims=[self.trans_channels],
        #     output_dim=transformer_cfg.dec_dim,
        #     norm_fn_name="bn1d",
        #     activation="relu",
        #     use_conv=True,
        #     output_use_activation=True,
        #     output_use_norm=True,
        #     output_use_bias=False,
        # )

        self.detr_sem_head = GenericMLP(
            input_dim=transformer_cfg.dec_dim,
            hidden_dims=[transformer_cfg.dec_dim, transformer_cfg.dec_dim],
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_dim=self.instance_classes+1
        )

        self.detr_conf_head = GenericMLP(
            input_dim=transformer_cfg.dec_dim,
            hidden_dims=[transformer_cfg.dec_dim, transformer_cfg.dec_dim],
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_dim=1
        )

        ''' DETR-Decoder '''
        decoder_layer = TransformerDecoderLayer(
            d_model=transformer_cfg.dec_dim,
            nhead=transformer_cfg.dec_nhead,
            dim_feedforward=transformer_cfg.dec_ffn_dim,
            dropout=0.1,
            normalize_before=True,
            use_rel=True,
        )

        self.decoder = TransformerDecoder(
            decoder_layer, num_layers=transformer_cfg.dec_nlayers, return_intermediate=True
        )


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

    def init_dyco(self):
        ################################
        ### for instance embedding
        self.output_dim = self.channels
        # self.output_dim = cfg.dec_dim
        self.mask_conv_num = 3
        conv_block = conv_with_kaiming_uniform("BN", activation=True)
        mask_tower = []
        for i in range(self.mask_conv_num):
            mask_tower.append(conv_block(self.channels, self.channels))
        mask_tower.append(nn.Conv1d(
            self.channels,  self.output_dim, 1
        ))
        self.add_module('mask_tower', nn.Sequential(*mask_tower))

        ### convolution before the condinst take place (convolution num before the generated parameters take place)
        before_embedding_conv_num = 1
        conv_block = conv_with_kaiming_uniform("BN", activation=True)
        before_embedding_tower = []
        for i in range(before_embedding_conv_num-1):
            before_embedding_tower.append(conv_block(64, 64))
        before_embedding_tower.append(conv_block(64, self.output_dim))
        self.add_module("before_embedding_tower", nn.Sequential(*before_embedding_tower))

        ### cond inst generate parameters for
        self.use_coords = True
        self.embedding_conv_num = 2
        weight_nums = []
        bias_nums = []
        for i in range(self.embedding_conv_num):
            if i ==0:
                if self.use_coords:
                    weight_nums.append((self.output_dim+3) * self.output_dim)
                else:
                    weight_nums.append(self.output_dim * self.output_dim)
                bias_nums.append(self.output_dim)
            elif i == self.embedding_conv_num-1:
                weight_nums.append(self.output_dim)
                bias_nums.append(1)
            else:
                weight_nums.append(self.output_dim*self.output_dim)
                bias_nums.append(self.output_dim)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)
        self.controller = nn.Conv1d(self.output_dim, self.num_gen_params, kernel_size=1)
        torch.nn.init.normal_(self.controller.weight, std=0.01)
        torch.nn.init.constant_(self.controller.bias, 0)



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

    def forward(self, batch, return_loss=False):
        if return_loss:
            return self.forward_train(**batch)
        else:
            return self.forward_test(**batch)

    @cuda_cast
    def forward_train(self, batch_idxs, voxel_coords, p2v_map, v2p_map, coords, coords_float, feats,
                      semantic_labels, instance_labels, instance_pointnum, instance_cls, instance_box, instance_batch_offsets,
                      pt_offset_labels, pt_offset_vertices_labels, spatial_shape, batch_size, pc_dims, **kwargs):

        
        feats = torch.cat((feats, coords_float), 1)
        
        voxel_feats = voxelization(feats, p2v_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        semantic_scores, pt_offsets, pt_offsets_vertices, box_conf, output_feats = self.forward_backbone(input, v2p_map)
        # breakpoint()
        batch_inputs = {
            "semantic_labels": semantic_labels,
            "instance_labels": instance_labels,
            "pt_offset_labels": pt_offset_labels,
            "pt_offset_vertices_labels": pt_offset_vertices_labels,
            "coords_float": coords_float,
            'instance_cls': instance_cls,
            'instance_box': instance_box,
            'instance_batch_offsets': instance_batch_offsets,
        }

        model_outputs = {}

        # instance losses
        if not self.semantic_only:
            # n_points = 8192
            # sampling_inds, object_idxs, batch_offsets_ = self.forward_sampling(
            #                         semantic_scores,
            #                         batch_idxs,
            #                         coords_float,
            #                         batch_size,
            #                         n_points=n_points) # batch, n_points

            mask_features  = self.mask_tower(torch.unsqueeze(output_feats, dim=2).permute(2,1,0)).permute(2,1,0)
            trans_features = self.trans_linear(output_feats)

            semantic_scores_max, semantic_scores_pred = torch.max(semantic_scores, dim=1) # N_points

            object_conditions = (semantic_scores_pred >= 2)
            object_idxs = torch.nonzero(object_conditions).view(-1)

            batch_idxs_ = batch_idxs[object_idxs]
            coords_float_ = coords_float[object_idxs]
            # output_feats_ = output_feats[object_idxs]
            # semantic_scores_ = semantic_scores[object_idxs]
            pt_offsets_ = pt_offsets[object_idxs]
            pt_offsets_vertices_ = pt_offsets_vertices[object_idxs]
            mask_features_ = mask_features[object_idxs]
            box_conf_ = box_conf[object_idxs]
            semantic_scores_pred_ = semantic_scores_pred[object_idxs]
            trans_features_ = trans_features[object_idxs]
            batch_offsets_ = self.get_batch_offsets(batch_idxs_, batch_size)

            # NOTE NMC
            box_conf_ = box_conf_ * semantic_scores_max[object_idxs]
            queries_mask, queries_inds, queries_feats, queries_coords = \
                non_maximum_queries(box_conf_, coords_float_, pt_offsets_, pt_offsets_vertices_, semantic_scores_pred_, trans_features_, batch_offsets_)

            # breakpoint()
            context_inds = []
            for b in range(batch_size):
                start, end = batch_offsets_[b], batch_offsets_[b+1]
                coords_float_b = coords_float_[start:end].unsqueeze(0)
                inds_b = furthest_point_sample(coords_float_b, self.transformer_cfg.n_context_points) + start # 1 x n_points 
                context_inds.append(inds_b)
            context_inds = torch.cat(context_inds, dim=0).long() # batch x n_points
            # context_inds = object_idxs[context_inds]

            context_feats = trans_features_[context_inds.flatten()].reshape(batch_size, self.transformer_cfg.n_context_points, -1)
            context_coords = coords_float_[context_inds.flatten()].reshape(batch_size, self.transformer_cfg.n_context_points, -1)

            # contexts = self.forward_aggregator(coords_float_, output_feats_, batch_offsets_, batch_size, pre_enc_inds=None)
            # context_locs, context_feats, pre_enc_inds = contexts

            # breakpoint()
            # NOTE transformer decoder
            dec_outputs = self.forward_decoder(context_coords, context_feats, queries_coords, queries_feats, queries_mask, pc_dims)

            # NOTE subsampling when training

            object_idxs_subsample = []
            for b in range(batch_size):
                start, end = batch_offsets_[b], batch_offsets_[b+1]
                num_points_b = (end - start).cpu()
                new_inds = torch.tensor(np.random.choice(num_points_b, self.transformer_cfg.n_subsample, replace=num_points_b<self.transformer_cfg.n_subsample), dtype=torch.long, device=coords_float.device) + start
                object_idxs_subsample.append(new_inds)
            object_idxs_subsample = torch.cat(object_idxs_subsample) # N_subsample: batch x 20000

            mask_features_subsample = mask_features_[object_idxs_subsample]
            coords_float_subsample = coords_float_[object_idxs_subsample]
            batch_offsets_subsample =  self.get_batch_offsets(batch_idxs_[object_idxs_subsample], batch_size)

            cls_logits_layers, mask_logits_layers = self.forward_head(dec_outputs, mask_features_subsample, coords_float_subsample, queries_coords, batch_offsets_subsample)


            model_outputs.update(dict(
                object_idxs=object_idxs[object_idxs_subsample],
                batch_offsets_=batch_offsets_subsample,
                semantic_scores=semantic_scores,
                pt_offsets=pt_offsets, 
                pt_offsets_vertices=pt_offsets_vertices, 
                box_conf=box_conf,
                cls_logits_layers=cls_logits_layers,
                mask_logits_layers=mask_logits_layers,
                queries_inds=queries_inds,
                queries_mask=queries_mask
            ))


            losses = self.criterion(batch_inputs, model_outputs)


        
        return self.parse_losses(losses)


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
                      pt_offset_labels, pt_offset_vertices_labels, spatial_shape, batch_size, pc_dims, scan_ids, **kwargs):

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
            
            batch_offsets = self.get_batch_offsets(batch_idxs, batch_size)
            mask_features  = self.mask_tower(torch.unsqueeze(output_feats, dim=2).permute(2,1,0)).permute(2,1,0)
            trans_features = self.trans_linear(output_feats)

            semantic_scores_max, semantic_scores_pred = torch.max(semantic_scores, dim=1) # N_points

            object_conditions = (semantic_scores_pred >= 2)
            object_idxs = torch.nonzero(object_conditions).view(-1)

            batch_idxs_ = batch_idxs[object_idxs]
            coords_float_ = coords_float[object_idxs]
            output_feats_ = output_feats[object_idxs]
            semantic_scores_ = semantic_scores[object_idxs]
            pt_offsets_ = pt_offsets[object_idxs]
            pt_offsets_vertices_ = pt_offsets_vertices[object_idxs]
            mask_features_ = mask_features[object_idxs]
            box_conf_ = box_conf[object_idxs]
            semantic_scores_pred_ = semantic_scores_pred[object_idxs]
            trans_features_ = trans_features[object_idxs]
            batch_offsets_ = self.get_batch_offsets(batch_idxs_, batch_size)

            # NOTE NMC
            box_conf_ = box_conf_ * semantic_scores_max[object_idxs]
            queries_mask, queries_inds, queries_feats, queries_coords = \
                non_maximum_queries(box_conf_, coords_float_, pt_offsets_, pt_offsets_vertices_, semantic_scores_pred_, trans_features_, batch_offsets_)

            context_inds = []
            for b in range(batch_size):
                start, end = batch_offsets_[b], batch_offsets_[b+1]
                coords_float_b = coords_float_[start:end].unsqueeze(0)
                inds_b = furthest_point_sample(coords_float_b, self.transformer_cfg.n_context_points) + start # 1 x n_points 
                context_inds.append(inds_b)
            context_inds = torch.cat(context_inds, dim=0).long() # batch x n_points
            # context_inds = object_idxs[context_inds]

            context_feats = trans_features_[context_inds.flatten()].reshape(batch_size, self.transformer_cfg.n_context_points, -1)
            context_coords = coords_float_[context_inds.flatten()].reshape(batch_size, self.transformer_cfg.n_context_points, -1)

            # contexts = self.forward_aggregator(coords_float_, output_feats_, batch_offsets_, batch_size, pre_enc_inds=None)
            # context_locs, context_feats, pre_enc_inds = contexts


            # NOTE transformer decoder
            dec_outputs = self.forward_decoder(context_coords, context_feats, queries_coords, queries_feats, queries_mask, pc_dims)


            cls_logits_layers, mask_logits_layers = self.forward_head(dec_outputs, mask_features_, coords_float_, queries_coords, batch_offsets_)




            pred_instances = self.get_instance(scan_ids[0], mask_logits_layers[-1], cls_logits_layers[-1], object_idxs, batch_offsets, batch_offsets_, semantic_scores_,\
                                                 logit_thresh=0.5, score_thresh=0.5, npoint_thresh=100)

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
            output = self.input_conv(input)

            list_output = self.unet(output)

            # for u in list_output:
            #     f = u.features
            #     print(f.shape)

            output = list_output[-1]

            output = self.output_layer(output)
            output_feats = output.features[input_map.long()]

            semantic_scores = self.semantic_linear(output_feats)
            pt_offsets = self.offset_linear(output_feats)
            pt_offsets_vertices = self.offset_vertices_linear(output_feats)
            box_conf = self.box_conf_linear(output_feats).squeeze(-1)
            return semantic_scores, pt_offsets, pt_offsets_vertices, box_conf, output_feats

    @torch.no_grad()
    def forward_sampling(self,
                         semantic_scores,
                         batch_idxs,
                         coords_float,
                         batch_size,
                         n_points=8192):

        # semantic_scores_sm = F.softmax(semantic_scores, dim=-1)
        semantic_scores_pred = torch.argmax(semantic_scores, dim=1) # N_points
        
        # object_conditions = torch.ones(semantic_scores.shape[0], dtype=torch.bool, device=semantic_scores.device)
        # for class_id in self.grouping_cfg.ignore_classes:
        #     object_conditions = object_conditions & (semantic_scores_pred != class_id)
        object_conditions = (semantic_scores_pred >= 2)
        object_idxs = torch.nonzero(object_conditions).view(-1)


        # if object_idxs.size(0) >= self.test_cfg.min_npoint:
        batch_idxs_ = batch_idxs[object_idxs]
        coords_float_ = coords_float[object_idxs]
        batch_offsets_ = self.get_batch_offsets(batch_idxs_, batch_size)

        inds = []
        for b in range(batch_size):
            start, end = batch_offsets_[b], batch_offsets_[b+1]
            coords_float_b = coords_float_[start:end].unsqueeze(0)
            inds_b = furthest_point_sample(coords_float_b, n_points) + start # 1 x n_points 
            inds.append(inds_b)
        inds = torch.cat(inds, dim=0).long() # batch x n_points

        inds = object_idxs[inds]
        return inds, object_idxs, batch_offsets_


    @force_fp32(apply_to=('locs_float_', 'output_feats_'))
    def forward_aggregator(self, locs_float_, output_feats_, batch_offsets_, batch_size, pre_enc_inds):
        context_locs = []
        grouped_features = []
        grouped_xyz = []
        pre_enc_inds = []

        for b in range(batch_size):
            start = batch_offsets_[b]
            end = batch_offsets_[b+1]
            locs_float_b = locs_float_[start:end, :].unsqueeze(0)
            output_feats_b = output_feats_[start:end, :].unsqueeze(0)
            batch_points = (end - start).item()

            if batch_points == 0:
                return None
            
            context_locs_b, grouped_features_b, grouped_xyz_b, pre_enc_inds_b = self.set_aggregator.group_points(locs_float_b.contiguous(), 
                                                                    output_feats_b.transpose(1,2).contiguous())

            context_locs.append(context_locs_b)
            grouped_features.append(grouped_features_b)
            grouped_xyz.append(grouped_xyz_b)
            pre_enc_inds.append(pre_enc_inds_b)

        context_locs = torch.cat(context_locs)
        grouped_features = torch.cat(grouped_features)
        grouped_xyz = torch.cat(grouped_xyz)
        pre_enc_inds = torch.cat(pre_enc_inds)

        context_feats = self.set_aggregator.mlp(grouped_features, grouped_xyz)
        context_feats = context_feats.transpose(1,2)

        return context_locs, context_feats, pre_enc_inds
        
    def prepare_context_query(self, coords_float, output_feats, sampling_inds, batch_size, n_points=8192):

        n_points_arr = [n_points, n_points//4, n_points//16]
        n_query = 128


        coords_float_context_l3 = torch.stack([coords_float[sampling_inds[b][:n_points]] for b in range(batch_size)], dim=0) # batch, n_points, channel
        output_feats_context_l3 = torch.stack([output_feats[sampling_inds[b][:n_points]] for b in range(batch_size)], dim=0) # batch, n_points, channel

        coords_float_context_l2 = coords_float_context_l3[:,:n_points_arr[1],:]
        output_feats_context_l2 = output_feats_context_l3[:,:n_points_arr[1],:]

        coords_float_context_l1 = coords_float_context_l3[:,:n_points_arr[2],:]
        output_feats_context_l1 = output_feats_context_l3[:,:n_points_arr[2],:]

        coords_float_query = coords_float_context_l3[:, :n_query, :]
        output_feats_query = output_feats_context_l3[:, :n_query, :]

        coords_float_context_arr = [coords_float_context_l1, coords_float_context_l2, coords_float_context_l3]
        output_feats_context_arr = [output_feats_context_l1, output_feats_context_l2, output_feats_context_l3]

        return coords_float_context_arr, output_feats_context_arr, coords_float_query, output_feats_query

    def forward_decoder(self, context_coords, context_feats, queries_coords, queries_feats, queries_mask, pc_dims):
        # batch_size = context_locs.shape[0]
        # queries_feats: batch, npoints, channel
        batch_size, n_queries = queries_coords.shape[:2]

        input_range = [
            pc_dims[1], # max
            pc_dims[0], # min
        ]


        context_embedding_pos = self.pos_embedding(context_coords, input_range=pc_dims)
        # context_feats = self.encoder_to_decoder_projection(
        #     context_feats.permute(0, 2, 1)
        # ) # batch x channel x npoints

        ''' Init dec_inputs by query features '''
        query_embedding_pos = self.pos_embedding(queries_coords, input_range=input_range)
        query_embedding_pos = self.query_projection(query_embedding_pos.float())


        # decoder expects: npoints x batch x channel
        context_embedding_pos   = context_embedding_pos.permute(2, 0, 1)
        query_embedding_pos     = query_embedding_pos.permute(2, 0, 1)
        context_feats           = context_feats.permute(1, 0, 2)
        dec_inputs              = queries_feats.permute(1, 0, 2)

        # Encode relative pos
        relative_coords = torch.abs(queries_coords[:,:,None,:] - context_coords[:,None,:,:])   # b x n_queries x n_contexts x 3
        n_queries, n_contexts = relative_coords.shape[1], relative_coords.shape[2]

        relative_embedding_pos = self.pos_embedding(relative_coords.reshape(batch_size, n_queries*n_contexts, -1), input_range=input_range).reshape(batch_size, -1, n_queries, n_contexts,)
        relative_embedding_pos   = relative_embedding_pos.permute(2,3,0,1)

        # num_layers x n_queries x batch x channel
        dec_outputs = self.decoder(
            tgt=dec_inputs, 
            memory=context_feats, 
            pos=context_embedding_pos, 
            query_pos=query_embedding_pos,
            relative_pos=relative_embedding_pos
        )

        return dec_outputs

    def forward_head(self, dec_outputs, mask_features_, locs_float_, fps_sampling_locs, batch_offsets_):
        num_layers, n_queries, batch, channel,  = dec_outputs.shape
        
        # outputs = []
        cls_logits_layers, mask_logits_layers = [], []
        n_inst_per_layer = batch * n_queries
        for l in range(num_layers):

            dec_output = dec_outputs[l] # n_queries x batch x channel
            # mlp head outputs are (num_layers x batch) x noutput x nqueries, so transpose last two dims
            cls_logits = self.detr_sem_head(dec_output.permute(1,2,0)).transpose(1, 2) # batch x n_queries x n_classes

            param_kernel2 = dec_output.transpose(0,1).flatten(0,1) # (batch * n_queries) * channel
            before_embedding_feature    = self.before_embedding_tower(torch.unsqueeze(param_kernel2, dim=2))
            controllers                  = self.controller(before_embedding_feature).squeeze(dim=2)

            controllers  = controllers.reshape(batch, n_queries, -1)

            mask_logits_list = []
            for b in range(batch):
                start, end = batch_offsets_[b], batch_offsets_[b+1]

                if end - start == 0:
                    mask_logits_list.append(None)
                    continue

                controller      = controllers[b] # n_queries x channel
                weights, biases = self.parse_dynamic_params(controller, self.output_dim)

                mask_feature_b = mask_features_[start:end, :]
                locs_float_b   = locs_float_[start:end, :]
                fps_sampling_locs_b = fps_sampling_locs[b]

                mask_logits         = self.mask_heads_forward(mask_feature_b, weights, biases, n_queries, locs_float_b, 
                                                            fps_sampling_locs_b)
                
                
                mask_logits     = mask_logits.squeeze(dim=0) # (n_queries) x N_mask
                mask_logits_list.append(mask_logits)
                
            # output = {'cls_logits': cls_logits, 'mask_logits': mask_logits_list}
            # outputs.append(output)
            cls_logits_layers.append(cls_logits)
            mask_logits_layers.append(mask_logits_list)
        return cls_logits_layers, mask_logits_layers

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


    def mask_heads_forward(self, mask_features, weights, biases, num_insts, coords_, fps_sampling_coords):
        assert mask_features.dim() == 3
        n_layers = len(weights)
        c = mask_features.size(1)
        n_mask = mask_features.size(0)
        x = mask_features.permute(2,1,0).repeat(num_insts, 1, 1) ### num_inst * c * N_mask

        relative_coords = fps_sampling_coords.reshape(-1, 1, 3) - coords_.reshape(1, -1, 3) ### N_inst * N_mask * 3
        relative_coords = relative_coords.permute(0,2,1)
        x = torch.cat([relative_coords, x], dim=1) ### num_inst * (3+c) * N_mask

        x = x.reshape(1, -1, n_mask) ### 1 * (num_inst*c') * Nmask
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv1d(x, w, bias=b, stride=1, padding=0, groups=num_insts)
            if i < n_layers - 1:
                x = F.relu(x)

        return x

    def get_instance(self, scan_id, mask_logits, cls_logits, object_idxs, batch_offsets, batch_offsets_, semantic_scores_, logit_thresh=0.5, score_thresh=0.5, npoint_thresh=100):

        semantic_scores_ = F.softmax(semantic_scores_,dim=1)
        # cls_logits_pred = cls_logits.max(2)[1] # batch x n_queries x 1 
        
        # NOTE only batch 1 when test
        b = 0

        instances = []
        start   = batch_offsets[b]
        end     = batch_offsets[b+1]
        num_points = end - start

        mask_logit_b = mask_logits[b].sigmoid()
        cls_logits_b = F.softmax(cls_logits[b], dim=-1)
        cls_logits_pred_b = torch.argmax(cls_logits[b], dim=-1)

        n_queries = mask_logit_b.shape[0]

        semantic_scores_b = semantic_scores_[batch_offsets_[b]:batch_offsets_[b+1]]

        cls_preds_cond = (cls_logits_pred_b < self.instance_classes)
        mask_logit_b_bool = (mask_logit_b >= logit_thresh)

        proposals_npoints = torch.sum(mask_logit_b_bool, dim=1)
        npoints_cond = (proposals_npoints >= npoint_thresh)

        mask_logit_scores = torch.sum(mask_logit_b * mask_logit_b_bool.int(), dim=1) / (proposals_npoints + 1e-6)
        mask_logit_scores_cond = (mask_logit_scores >= score_thresh)


        cls_logits_scores = torch.gather(cls_logits_b, 1, cls_logits_pred_b.unsqueeze(-1)).squeeze(-1) 

        sem_scores = torch.sum(semantic_scores_b[None,:,:].expand(n_queries, semantic_scores_b.shape[0], semantic_scores_b.shape[1]) * mask_logit_b_bool.int()[:,:,None], dim=1) / (proposals_npoints[:, None] + 1e-6) # n_pred, n_clas
        sem_scores = torch.gather(sem_scores, 1, cls_logits_pred_b.unsqueeze(-1)).squeeze(-1) 

        scores = mask_logit_scores * torch.pow(cls_logits_scores, 0.5) * sem_scores

        final_cond = cls_preds_cond & npoints_cond & mask_logit_scores_cond

        if torch.count_nonzero(final_cond) == 0:
            return instances

        cls_final = cls_logits_pred_b[final_cond]
        masks_final = mask_logit_b_bool[final_cond]
        scores_final = scores[final_cond]

        num_insts = scores_final.shape[0]
        proposals_pred = torch.zeros((num_insts, num_points), dtype=torch.int, device=mask_logit_b.device)

        inst_inds, point_inds = torch.nonzero(masks_final, as_tuple=True)
        
        point_inds = object_idxs[point_inds]

        proposals_pred[inst_inds, point_inds] = 1

        pick_idxs = non_max_suppression_gpu(proposals_pred, scores_final, threshold=0.2)  # int, (nCluster, N)

        proposals_pred = proposals_pred[pick_idxs].cpu().numpy()
        scores_final = scores_final[pick_idxs].cpu().numpy()
        cls_final = cls_final[pick_idxs].cpu().numpy()

        # proposals_pred = proposals_pred.cpu().numpy()
        # scores_final = scores_final.cpu().numpy()
        # cls_final = cls_final.cpu().numpy()

        
        for i in range(cls_final.shape[0]):
            pred = {}
            pred['scan_id'] = scan_id
            pred['label_id'] = cls_final[i] + 1
            pred['conf'] = scores_final[i]
            # rle encode mask to save memory
            pred['pred_mask'] = rle_encode(proposals_pred[i])
            instances.append(pred)
        return instances

    def get_box(self, scan_id, cls_preds, box_offset_preds, conf_preds, coords_float, sampling_inds):
        # cls_preds N_queries, c
        n_queries = cls_preds.shape[0]
        query_inds = sampling_inds[:n_queries]

        conf_preds = conf_preds.squeeze(-1)

        coords_float_queries = coords_float[query_inds]
        box_preds = coords_float_queries.repeat(1,2) + box_offset_preds # N, 6

        cls_preds_sm = F.softmax(cls_preds, -1)
        cls, cls_scores = torch.max(cls_preds_sm, dim=-1)

        scores = cls_scores * torch.clamp(conf_preds, 0, 1)

        cond = (cls < self.instance_classes) & (scores >= 0.2)

        # breakpoint()
        cls_final = cls[cond].cpu().numpy()
        box_final = box_preds[cond].cpu().numpy()
        score_final = scores[cond].cpu().numpy()

        boxes = []
        for i in range(cls_final.shape[0]):
            pred = {}
            pred['scan_id'] = scan_id
            pred['label_id'] = cls_final[i]+1
            pred['conf'] = score_final[i]
            pred['box'] = box_final[i]
            boxes.append(pred)
        return boxes

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
