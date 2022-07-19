import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .model_utils import giou_aabb, iou_aabb, compute_dice_loss, sigmoid_focal_loss, non_maximum_cluster, non_maximum_cluster2

from .matcher import giou_aabb_matcher

class Criterion(nn.Module):
    def __init__(self,
                matcher,
                semantic_classes=20,
                instance_classes=18,
                ignore_label=-100,
                eos_coef=0.1,
                point_wise_loss=True):
        super(Criterion, self).__init__()

        self.matcher = matcher
        self.point_wise_loss = point_wise_loss

        self.label_shift = semantic_classes - instance_classes
        self.semantic_classes = semantic_classes
        self.instance_classes = instance_classes

        self.eos_coef =eos_coef

        empty_weight = torch.ones(self.instance_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        self.loss_weight = {
            'ce_loss': 1,
            'offset_loss': 1,
            'conf_loss': 1,
            'giou_loss': 1.
        }

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def cal_point_wise_loss(self, semantic_scores, pt_offsets, pt_offsets_vertices, box_conf, semantic_labels, instance_labels,
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
            giou_loss = 0 * box_conf.sum()
        else:
            offset_loss = F.l1_loss(
                pt_offsets[pos_inds], pt_offset_labels[pos_inds], reduction='sum') / total_pos_inds

            # print('pt_offsets_vertices', pt_offsets_vertices.shape)
            offset_vertices_loss = F.l1_loss(
                pt_offsets_vertices[pos_inds], pt_offset_vertices_labels[pos_inds], reduction='sum') / total_pos_inds

            # iou_gt = iou_aabb(pt_offsets_vertices[pos_inds], pt_offset_vertices_labels[pos_inds], coords_float[pos_inds])
            iou_gt, giou = giou_aabb(pt_offsets_vertices[pos_inds], pt_offset_vertices_labels[pos_inds], coords_float[pos_inds])
            iou_gt = iou_gt.detach()
            
            giou_loss = torch.sum(1 - giou) / total_pos_inds
            # breakpoint()
            point_iou_loss = F.mse_loss(box_conf[pos_inds], iou_gt, reduction='none')
            point_iou_loss = point_iou_loss.sum() / total_pos_inds

            # breakpoint()

        losses['point_iou_loss'] = point_iou_loss

        losses['offset_loss'] = offset_loss
        losses['offset_vertices_loss'] = offset_vertices_loss

        losses['giou_loss'] = giou_loss
        return losses

    def loss_labels(self, cls_preds, cls_gt, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        batch_size, n_queries, _ = cls_preds.shape

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(cls_gt, indices)])

        target_classes = torch.full(
            (batch_size, n_queries), self.instance_classes, dtype=torch.int64, device=cls_preds.device
        )

        target_classes[idx] = target_classes_o


        ce_loss = F.cross_entropy(cls_preds.permute(0,2,1), target_classes, self.empty_weight, reduction='sum')
        
        ce_loss = ce_loss / (batch_size * n_queries)
        return dict(ce_loss=ce_loss)


    def loss_boxes(self, box_offset_preds, conf_preds, coords_float_query, box_gt, indices):
        
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        # breakpoint()

        conf_preds_ = conf_preds[src_idx].squeeze(-1)
        box_offset_preds_ = box_offset_preds[src_idx]
        coords_float_query_ = coords_float_query[src_idx]
        box_preds_ = coords_float_query_.repeat(1, 2) + box_offset_preds_

        box_gt_ = torch.cat([t[J] for t, (_, J) in zip(box_gt, indices)])
        # box_gt_ = box_gt[tgt_idx]

        num_inst = box_gt_.shape[0]

        offset_loss = F.l1_loss(box_preds_, box_gt_, reduction='sum') / num_inst

        iou_, giou_ = giou_aabb_matcher(box_preds_, box_gt_)
        iou_ = iou_.detach()

        conf_loss = F.mse_loss(conf_preds_, iou_, reduction='none')
        conf_loss = conf_loss.sum() / num_inst

        giou_loss = torch.sum(1 - giou_) / num_inst

        return dict(
            offset_loss=offset_loss,
            conf_loss=conf_loss,
            giou_loss=giou_loss,
        )


        
        

    def single_layer_loss(self, cls_preds, box_offset_preds, conf_preds, coords_float_query, cls_labels, box_labels, indices):
        loss = torch.tensor(0.0, requires_grad=True).to(cls_preds.device)
        loss_dict = {}
        
        loss_dict.update(self.loss_labels(cls_preds, cls_labels, indices))
        loss_dict.update(self.loss_boxes(box_offset_preds, conf_preds, coords_float_query, box_labels, indices))

        for k, v in loss_dict.items():
            loss += self.loss_weight[k] * v

        return loss

    def forward(self, batch_inputs, model_outputs):
        loss_dict = {}
        # loss = torch.tensor(0.0, requires_grad=True).to(semantic_scores.device)

        # '''semantic loss'''
        semantic_scores = model_outputs['semantic_scores']
        pt_offsets = model_outputs['pt_offsets']
        pt_offsets_vertices = model_outputs['pt_offsets_vertices']
        box_conf = model_outputs['box_conf']

        semantic_labels = batch_inputs['semantic_labels']
        instance_labels = batch_inputs['instance_labels']
        pt_offset_labels = batch_inputs['pt_offset_labels']
        pt_offset_vertices_labels = batch_inputs['pt_offset_vertices_labels']
        coords_float = batch_inputs['coords_float']
        instance_cls = batch_inputs['instance_cls']
        instance_box = batch_inputs['instance_box']
        instance_batch_offsets = batch_inputs['instance_batch_offsets']

        if self.point_wise_loss:

            point_wise_loss = self.cal_point_wise_loss(semantic_scores, pt_offsets, pt_offsets_vertices, box_conf, semantic_labels,
                                                instance_labels, pt_offset_labels, pt_offset_vertices_labels, coords_float)
            
            loss_dict.update(point_wise_loss)
        ''' Main loss '''
        cls_preds = model_outputs["cls_preds"]
        box_offset_preds = model_outputs["box_offset_preds"]
        conf_preds = model_outputs["conf_preds"]
        coords_float_query = model_outputs["coords_float_query"]
        batch_offsets_ = model_outputs["batch_offsets_"]
        object_idxs = model_outputs["object_idxs"]

        semantic_labels_ = semantic_labels[object_idxs]
        instance_labels_ = instance_labels[object_idxs]
        coords_float_ = coords_float[object_idxs]

        n_layers, batch_size = cls_preds.shape[0:2]

        match_indices, cls_labels, box_labels = self.matcher(cls_preds[-1], box_offset_preds[-1], conf_preds[-1], \
                                                            coords_float_query, instance_cls, instance_box, instance_batch_offsets, instance_label_shift=self.label_shift)
        if len(match_indices) < batch_size:
            loss_dict['loss'] = 0.0 * cls_preds.sum()
            return loss_dict
        # NOTE main loss
        main_loss = self.single_layer_loss(cls_preds[-1], box_offset_preds[-1], conf_preds[-1], coords_float_query, cls_labels, box_labels, match_indices)
        loss_dict[f'loss_layer{n_layers-1}'] = main_loss

        ''' Auxilary loss '''
        for l in range(n_layers-1):
            interm_loss = self.single_layer_loss(cls_preds[l], box_offset_preds[l], conf_preds[l], coords_float_query, cls_labels, box_labels, match_indices)
            interm_loss = interm_loss * 0.5

            loss_dict[f'loss_layer{l}'] = interm_loss
            
        return loss_dict
