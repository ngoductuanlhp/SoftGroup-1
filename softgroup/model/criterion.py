import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from .model_utils import giou_aabb, iou_aabb, non_maximum_cluster, non_maximum_cluster2

from .matcher import giou_aabb_matcher

@torch.no_grad()
def get_iou(inputs, targets, thresh=0.5):
    inputs_bool = inputs.detach().sigmoid()
    inputs_bool = ( inputs_bool >= thresh)

    intersection = (inputs_bool * targets).sum(-1)
    union = inputs_bool.sum(-1) + targets.sum(-1) - intersection
    
    iou = intersection / (union + 1e-6)

    return iou

def compute_dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    # inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / (num_boxes + 1e-6)

def compute_sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / (num_boxes + 1e-6)
class Criterion(nn.Module):
    def __init__(self,
                matcher,
                semantic_classes=20,
                instance_classes=18,
                ignore_label=-100,
                eos_coef=0.1,
                point_wise_loss=True,
                total_epoch=40):
        super(Criterion, self).__init__()

        self.matcher = matcher
        self.point_wise_loss = point_wise_loss

        self.ignore_label = ignore_label

        self.label_shift = semantic_classes - instance_classes
        self.semantic_classes = semantic_classes
        self.instance_classes = instance_classes

        self.eos_coef =eos_coef

        self.total_epoch = total_epoch

        empty_weight = torch.ones(self.instance_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        self.loss_weight = {
            'dice_loss': 4,
            'focal_loss': 4,
            'cls_loss': 1,
            'iou_loss': 1,
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


        
        

    def single_layer_loss(self, mask_logits_list, cls_logits, conf_logits, row_indices, cls_labels, inst_labels, batch_size, aux=False, n_main_queries=64):
        # loss = torch.tensor(0.0, requires_grad=True, device=cls_logits.device, dtype=torch.float)
        # loss = 0.0
        loss_dict = {}

        for k in self.loss_weight:
            loss_dict[k] = torch.tensor(0.0, requires_grad=True, device=cls_logits.device, dtype=torch.float)
            # loss_dict[k] = 0
        
        num_gt = 0 
        for b in range(batch_size):
            mask_logit_b = mask_logits_list[b]
            cls_logit_b = cls_logits[b] # n_queries x n_classes
            conf_logits_b = conf_logits[b] # n_queries

            # if aux:
            #     mask_logit_b = mask_logit_b[n_main_queries:]
            #     cls_logit_b = cls_logit_b[n_main_queries:]
            #     conf_logits_b = conf_logits_b[n_main_queries:]
            # else:
            #     mask_logit_b = mask_logit_b[:n_main_queries]
            #     cls_logit_b = cls_logit_b[:n_main_queries]
            #     conf_logits_b = conf_logits_b[:n_main_queries]

            pred_inds, cls_label, inst_label = row_indices[b], cls_labels[b], inst_labels[b]

            n_queries = cls_logit_b.shape[0]

            if mask_logit_b == None:
                continue

            if pred_inds is None:
                continue

            mask_logit_pred = mask_logit_b[pred_inds]
            conf_logits_pred = conf_logits_b[pred_inds]

            num_gt_batch = len(pred_inds)
            num_gt += num_gt_batch
             
            loss_dict['dice_loss'] = loss_dict['dice_loss'] + compute_dice_loss(mask_logit_pred, inst_label, num_gt_batch)

            # breakpoint()
            loss_dict['focal_loss'] = loss_dict['dice_loss'] + compute_sigmoid_focal_loss(mask_logit_pred, inst_label, num_gt_batch)

            gt_iou = get_iou(mask_logit_pred, inst_label)
            
            loss_dict['iou_loss'] = loss_dict['iou_loss'] + F.mse_loss(conf_logits_pred, gt_iou, reduction='sum') / num_gt_batch


            # target_classes = torch.full(
            #     (n_queries), self.instance_classes, dtype=torch.int64, device=cls_logits.device
            # )

            target_classes = torch.ones((n_queries), dtype=torch.int64, device=cls_logits.device) * self.instance_classes

            target_classes[pred_inds] = cls_label

            loss_dict['cls_loss'] = loss_dict['cls_loss'] + F.cross_entropy(
                cls_logit_b,
                target_classes,
                self.empty_weight,
                reduction="mean",
            )

        for k in loss_dict.keys():
            loss_dict[k] = loss_dict[k] / batch_size

        return loss_dict

        # for k, v in self.loss_weight.items():
        #     loss = loss + loss_dict[k] * v

        # # if last_layer:
        # #     print('iou loss', loss_dict['iou_loss'])
        # return loss

    def forward(self, batch_inputs, model_outputs, n_main_queries, epoch=0):

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


        loss_dict = {}
        for k in self.loss_weight:
            loss_dict[k] = torch.tensor(0.0, requires_grad=True, device=semantic_scores.device, dtype=torch.float)
            loss_dict['aux_' + k] = torch.tensor(0.0, requires_grad=True, device=semantic_scores.device, dtype=torch.float)

        if self.point_wise_loss:

            point_wise_loss = self.cal_point_wise_loss(semantic_scores, pt_offsets, pt_offsets_vertices, box_conf, semantic_labels,
                                                instance_labels, pt_offset_labels, pt_offset_vertices_labels, coords_float)
            
            loss_dict.update(point_wise_loss)

        ''' Main loss '''
        mask_logits_layers = model_outputs["mask_logits_layers"]
        cls_logits_layers = model_outputs["cls_logits_layers"]
        conf_logits_layers = model_outputs["conf_logits_layers"]

        batch_offsets_ = model_outputs["batch_offsets_"]
        object_idxs = model_outputs["object_idxs"]

        semantic_labels_ = semantic_labels[object_idxs]
        instance_labels_ = instance_labels[object_idxs]
        # coords_float_ = coords_float[object_idxs]

        n_layers = len(cls_logits_layers) 
        batch_size, n_queries = cls_logits_layers[-1].shape[:2]

        # row_indices, cls_labels, inst_labels = self.matcher(cls_logits_layers[-1], mask_logits_layers[-1], conf_logits_layers[-1],\
        #                                                     instance_cls, semantic_labels_, instance_labels_, batch_offsets_, instance_label_shift=self.label_shift)
        
        gt_dict, aux_gt_dict = self.matcher.forward_dup(cls_logits_layers[-1], mask_logits_layers[-1], conf_logits_layers[-1],\
                                                            instance_cls, semantic_labels_, instance_labels_, batch_offsets_, instance_label_shift=self.label_shift, dup_gt=6, n_main_queries=n_main_queries)

        # NOTE main loss

        row_indices = gt_dict['row_indices']
        inst_labels = gt_dict['inst_labels']
        cls_labels = gt_dict['cls_labels']

        main_loss_dict = self.single_layer_loss(mask_logits_layers[-1], cls_logits_layers[-1],  conf_logits_layers[-1], row_indices, cls_labels, inst_labels, batch_size, n_main_queries=n_main_queries)

        for k, v in self.loss_weight.items():
            loss_dict[k] = loss_dict[k] + main_loss_dict[k] * v

        ''' Auxilary loss '''
        for l in range(n_layers-1):
            interm_loss_dict = self.single_layer_loss(mask_logits_layers[l], cls_logits_layers[l],  conf_logits_layers[l], row_indices, cls_labels, inst_labels, batch_size, n_main_queries=n_main_queries)
            # interm_loss = interm_loss * 0.5
            for k, v in self.loss_weight.items():
                loss_dict[k] = loss_dict[k] + interm_loss_dict[k] * v
            # loss_dict[f'loss_layer{l}'] = interm_loss
        
        # NOTE aux loss

        aux_row_indices = aux_gt_dict['row_indices']
        aux_inst_labels = aux_gt_dict['inst_labels']
        aux_cls_labels = aux_gt_dict['cls_labels']

        aux_main_loss_dict = self.single_layer_loss(mask_logits_layers[-1], cls_logits_layers[-1],  conf_logits_layers[-1], aux_row_indices, aux_cls_labels, aux_inst_labels, batch_size, aux=True, n_main_queries=n_main_queries)

        coef_aux = math.exp((1 - 5 * epoch/self.total_epoch))
        for k, v in self.loss_weight.items():
            loss_dict['aux_' + k] = loss_dict['aux_' + k] + aux_main_loss_dict[k] * v * coef_aux

        ''' Auxilary loss '''
        for l in range(n_layers-1):
            aux_interm_loss_dict = self.single_layer_loss(mask_logits_layers[l], cls_logits_layers[l],  conf_logits_layers[l], aux_row_indices, aux_cls_labels, aux_inst_labels, batch_size, aux=True, n_main_queries=n_main_queries)
            # interm_loss = interm_loss * 0.5
            for k, v in self.loss_weight.items():
                loss_dict['aux_' + k] = loss_dict['aux_' + k] + aux_interm_loss_dict[k] * v * coef_aux

        return loss_dict
