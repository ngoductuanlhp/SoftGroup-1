# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

# from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, generalized_box_cdist
import functools

def compute_dice(inputs, targets):
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
    return loss

def giou_aabb_matcher(box_preds, box_gt):

    coords_min_pred = box_preds[:, 0:3] # n_queries x 3
    coords_max_pred = box_preds[:, 3:6] # n_queries x 3

    coords_min_gt = box_gt[:, 0:3] # n_inst x 3
    coords_max_gt = box_gt[:, 3:6]# n_inst x 3


    upper = torch.min(coords_max_pred, coords_max_gt) # Nx3
    lower = torch.max(coords_min_pred, coords_min_gt) # Nx3

    intersection = torch.prod(torch.clamp((upper - lower), min=0.0), -1) # N

    gt_volumes = torch.prod(torch.clamp((coords_max_gt - coords_min_gt), min=0.0), -1)
    pred_volumes = torch.prod(torch.clamp((coords_max_pred - coords_min_pred), min=0.0), -1)

    union = gt_volumes + pred_volumes - intersection
    iou = intersection / (union + 1e-6)

    upper_bound = torch.max(coords_max_pred, coords_max_gt)
    lower_bound = torch.min(coords_min_pred, coords_min_gt)

    volumes_bound = torch.prod(torch.clamp((upper_bound - lower_bound), min=0.0), -1) # N

    giou = iou - (volumes_bound - union) / (volumes_bound + 1e-6)

    return iou, giou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                ignore_label=-100,
                cost_class=1,
                cost_bbox=1,
                cost_giou=1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()

        self.ignore_label = ignore_label

        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
    
    def get_gt(self, instance_labels_b, instance_cls):
        n_points = instance_labels_b.shape[0]

        unique_inst = torch.unique(instance_labels_b)
        unique_inst = unique_inst[unique_inst != self.ignore_label]
        
        
        cls_label = instance_cls[unique_inst] # n_inst
        cls_label_cond = (cls_label >= 0)
        cls_labels_b = cls_label[cls_label_cond]
        fg_unique_inst = unique_inst[cls_label_cond]

        n_inst_gt = fg_unique_inst.shape[0]

        if n_inst_gt == 0:
            return None

        mask_labels_b = torch.zeros((n_inst_gt, n_points), device=instance_labels_b.device, dtype=torch.float)
        for i in range(n_inst_gt):
            inst_id = fg_unique_inst[i]
            mask_labels_b[i] = (instance_labels_b == inst_id).float()

        return cls_labels_b, mask_labels_b

    def get_match(self, cls_labels_b, mask_labels_b, cls_preds_b, mask_logits_preds_b, conf_preds_b, dup_gt=6, n_main_queries=64):

        n_inst_gt, n_points = mask_labels_b.shape[:2]

        n_queries = cls_preds_b.shape[0]


        # cls_preds_b = cls_preds[b] # n_queries, n_classes
        # mask_logits_preds_b = mask_logits_preds[b]
        # conf_preds_b = conf_preds[b] # n_queries

        dice_cost = compute_dice(mask_logits_preds_b.reshape(-1, 1, n_points).repeat(1, n_inst_gt, 1).flatten(0, 1), 
                                mask_labels_b.reshape(1, -1, n_points).repeat(n_queries, 1, 1).flatten(0, 1))

        dice_cost = dice_cost.reshape(n_queries, n_inst_gt)


        cls_preds_b_sm = torch.nn.functional.softmax(cls_preds_b, dim=-1)

        class_cost = -cls_preds_b_sm[:, cls_labels_b]

        conf_cost = -conf_preds_b[:, None].repeat(1, n_inst_gt)

        final_cost = 1 * class_cost + 5 * dice_cost + 1 * conf_cost
        
        final_cost = final_cost.detach()
        
        main_final_cost = final_cost[:n_main_queries, :] # 96, n_gt
        main_final_cost = main_final_cost.cpu().numpy() # n_queries, n_gt

        row_inds, col_inds = linear_sum_assignment(main_final_cost)



        aux_final_cost = final_cost[n_main_queries:, :].repeat(1, dup_gt).cpu().numpy()

        aux_row_inds, aux_col_inds = linear_sum_assignment(aux_final_cost)

        return row_inds, col_inds, aux_row_inds, aux_col_inds

    @torch.no_grad()
    def forward(self, cls_preds, mask_logits_preds, conf_preds, instance_cls, semantic_labels_, instance_labels_, batch_offsets_, instance_label_shift=2):
        # cls_preds : batch x classes x n_queries
        batch_size, n_queries, _ = cls_preds.shape

        row_indices = []
        inst_labels = []
        cls_labels = []


        for b in range(batch_size):
            start, end = batch_offsets_[b], batch_offsets_[b+1]
            

            cls_preds_b = cls_preds[b] # n_queries, n_classes
            mask_logits_preds_b = mask_logits_preds[b]
            conf_preds_b = conf_preds[b] # n_queries

            instance_labels_b = instance_labels_[start:end]
            # semantic_labels_b = semantic_labels_[start:end]
            # coords_float_b = coords_float_[start:end]

            # n_points = instance_labels_b.shape[0]

            # unique_inst = torch.unique(instance_labels_b)
            # unique_inst = unique_inst[unique_inst != self.ignore_label]
            
            
            # cls_label = instance_cls[unique_inst] # n_inst
            # cls_label_cond = (cls_label >= 0)
            # cls_labels_b = cls_label[cls_label_cond]
            # fg_unique_inst = unique_inst[cls_label_cond]

            # n_inst_gt = fg_unique_inst.shape[0]

            # if n_inst_gt == 0:
            #     row_indices.append(None)
            #     inst_labels.append(None)
            #     cls_labels.append(None)
            #     continue

            # mask_labels_b = torch.zeros((n_inst_gt, n_points), device=cls_preds.device, dtype=torch.float)
            # for i in range(n_inst_gt):
            #     inst_id = fg_unique_inst[i]
            #     mask_labels_b[i] = (instance_labels_b == inst_id).float()

            labels_b = self.get_gt(instance_labels_b, instance_cls)
            if labels_b is None:
                row_indices.append(None)
                inst_labels.append(None)
                cls_labels.append(None)
                continue

            n_inst_gt, n_points = mask_labels_b.shape[:2]

            cls_labels_b, mask_labels_b = labels_b

            dice_cost = compute_dice(mask_logits_preds_b.reshape(-1, 1, n_points).repeat(1, n_inst_gt, 1).flatten(0, 1), 
                                    mask_labels_b.reshape(1, -1, n_points).repeat(n_queries, 1, 1).flatten(0, 1))

            dice_cost = dice_cost.reshape(n_queries, n_inst_gt)


            cls_preds_b_sm = torch.nn.functional.softmax(cls_preds_b, dim=-1)

            class_cost = -cls_preds_b_sm[:, cls_labels_b]

            conf_cost = -conf_preds_b[:, None].repeat(1, n_inst_gt)

            final_cost = 1 * class_cost + 5 * dice_cost + 1 * conf_cost
            
            final_cost = final_cost.detach().cpu().numpy()


            row_inds, col_inds = linear_sum_assignment(final_cost)

            row_indices.append(row_inds)
            inst_labels.append(mask_labels_b[col_inds])
            cls_labels.append(cls_labels_b[col_inds])
            # row_inds, inst_masks[col_inds], sem_labels[col_inds]

        return row_indices, cls_labels, inst_labels


    @torch.no_grad()
    def forward_dup(self, cls_preds, mask_logits_preds, conf_preds, instance_cls, semantic_labels_, instance_labels_, batch_offsets_, instance_label_shift=2, dup_gt=1, n_main_queries=64):
        # cls_preds : batch x classes x n_queries
        batch_size, n_queries, _ = cls_preds.shape

        gt_dict = dict(
            row_indices=[],
            inst_labels=[],
            cls_labels=[]
        )

        aux_gt_dict = dict(
            row_indices=[],
            inst_labels=[],
            cls_labels=[]
        )

        for b in range(batch_size):
            start, end = batch_offsets_[b], batch_offsets_[b+1]

            instance_labels_b = instance_labels_[start:end]

            labels_b = self.get_gt(instance_labels_b, instance_cls)
            if labels_b is None:
                gt_dict['row_indices'].append(None)
                gt_dict['inst_labels'].append(None)
                gt_dict['cls_labels'].append(None)

                aux_gt_dict['row_indices'].append(None)
                aux_gt_dict['inst_labels'].append(None)
                aux_gt_dict['cls_labels'].append(None)
                continue
            
            # NOTE gt
            cls_labels_b, mask_labels_b = labels_b
            row_inds, col_inds, aux_row_inds, aux_col_inds = self.get_match(cls_labels_b, mask_labels_b, cls_preds[b], mask_logits_preds[b], conf_preds[b], dup_gt=dup_gt, n_main_queries=n_main_queries)

            gt_dict['row_indices'].append(row_inds)
            gt_dict['inst_labels'].append(mask_labels_b[col_inds])
            gt_dict['cls_labels'].append(cls_labels_b[col_inds])

            # NOTE aux gt
            aux_cls_labels_b = cls_labels_b.repeat(dup_gt)
            aux_mask_labels_b = mask_labels_b.repeat(dup_gt, 1)

            # aux_row_inds, aux_col_inds = self.get_match(aux_cls_labels_b, aux_mask_labels_b, cls_preds[b], mask_logits_preds[b], conf_preds[b])

            aux_gt_dict['row_indices'].append(aux_row_inds)
            aux_gt_dict['inst_labels'].append(aux_mask_labels_b[aux_col_inds])
            aux_gt_dict['cls_labels'].append(aux_cls_labels_b[aux_col_inds])

        return gt_dict, aux_gt_dict

def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou)