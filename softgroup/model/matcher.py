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


    @torch.no_grad()
    def forward(self, cls_preds, box_offset_preds, conf_preds, coords_float_query, instance_cls, instance_box, instance_batch_offsets, instance_label_shift=2):
        # cls_preds : batch x classes x n_queries
        batch_size, n_queries, _ = cls_preds.shape

        match_indices = []
        cls_labels = []
        box_labels = []
        giou_labels = []
        for b in range(batch_size):
            inst_s, inst_e = instance_batch_offsets[b], instance_batch_offsets[b+1]

            cls_preds_b = cls_preds[b] # n_queries, n_classes
            box_offset_preds_b = box_offset_preds[b]
            coords_float_query_b = coords_float_query[b] # n_queries, 3
            conf_preds_b = conf_preds[b].squeeze(-1)

            instance_cls_b = instance_cls[inst_s:inst_e]
            instance_box_b = instance_box[inst_s:inst_e]

            fg_inst_inds = (instance_cls_b != self.ignore_label)


            n_inst_gt = torch.count_nonzero(fg_inst_inds)

            if n_inst_gt == 0:
                continue

            instance_cls_b = instance_cls_b[fg_inst_inds]
            instance_box_b = instance_box_b[fg_inst_inds]

            cost_class = -cls_preds_b[:, instance_cls_b] # n_queries x n_inst

            cost_conf = -conf_preds_b[:, None].repeat(1, n_inst_gt)


            # breakpoint()
            
            box_preds_b = coords_float_query_b.repeat(1, 2) + box_offset_preds_b

            box_preds_b = box_preds_b[:, None, :].repeat(1, n_inst_gt, 1) # n_queries, n_inst, 6
            instance_box_b = instance_box_b[None, :, :].repeat(n_queries, 1, 1) # n_queries, n_inst, 6


            # breakpoint()

            box_preds_b = box_preds_b.reshape(n_queries*n_inst_gt, -1)
            instance_box_b = instance_box_b.reshape(n_queries*n_inst_gt, -1)
            _, giou_b = giou_aabb_matcher(box_preds_b, instance_box_b)


            # breakpoint()

            giou_b = giou_b.reshape(n_queries, n_inst_gt)

            # cost_offset = torch.cdist(box_preds_b[:, 0:3], box_labels_b[:, 0:3], p=1) + torch.cdist(box_preds_b[:, 3:6], box_labels_b[:, 3:6], p=1)
            # breakpoint()
            cost_offset = torch.norm(box_preds_b - instance_box_b, p=2, dim=-1)
            cost_offset = cost_offset.reshape(n_queries, n_inst_gt)

            # breakpoint()

            cost_box = -giou_b

            # breakpoint()
            final_cost = cost_class * 5 + cost_conf * 2 + cost_box * 2 + cost_offset

            final_cost = final_cost.detach().cpu().numpy()
            
            match_ind = linear_sum_assignment(final_cost)
            match_indices.append(match_ind)
            cls_labels.append(instance_cls_b)
            box_labels.append(instance_box_b)
            giou_labels.append(giou_b)

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in match_indices], cls_labels, box_labels

    
    # @torch.no_grad()
    # def forward(self, cls_preds, box_offset_preds, conf_preds, coords_float_query, instance_labels_, semantic_labels_, coords_float_, batch_offsets_, instance_label_shift=2):
    #     # cls_preds : batch x classes x n_queries
    #     batch_size, n_queries, _ = cls_preds.shape

    #     match_indices = []
    #     cls_labels = []
    #     box_labels = []
    #     giou_labels = []
    #     for b in range(batch_size):
    #         start, end = batch_offsets_[b], batch_offsets_[b+1]

    #         cls_preds_b = cls_preds[b] # n_queries, n_classes
    #         box_offset_preds_b = box_offset_preds[b]
    #         coords_float_query_b = coords_float_query[b] # n_queries, 3
    #         conf_preds_b = conf_preds[b].squeeze(-1)

    #         instance_labels_b = instance_labels_[start:end]
    #         semantic_labels_b = semantic_labels_[start:end]
    #         coords_float_b = coords_float_[start:end]

    #         unique_inst = torch.unique(instance_labels_b)
    #         unique_inst = unique_inst[unique_inst != self.ignore_label]
    #         # n_inst_gt = len(unique_inst)

    #         # breakpoint()

    #         # cls_labels_b = torch.zeros((n_inst_gt), dtype=torch.long, device=cls_preds.device)
    #         # box_labels_b = torch.zeros((n_inst_gt, 6), dtype=torch.float, device=cls_preds.device)
    #         cls_labels_b = []
    #         box_labels_b_min, box_labels_b_max = [], []

    #         # print(b, semantic_labels_b.shape)

    #         for i in range(len(unique_inst)):
    #             inst_idx = unique_inst[i]
    #             inst_points_inds = torch.nonzero(instance_labels_b==inst_idx).view(-1)

    #             # breakpoint()
    #             cls_label = (semantic_labels_b[inst_points_inds[0]] - instance_label_shift).item()

    #             # print(cls_label)
    #             if cls_label >= 0:

    #                 cls_labels_b.append(cls_label)

    #                 coords_min_inst = torch.min(coords_float_b[inst_points_inds], dim=0)[0]
    #                 coords_max_inst = torch.max(coords_float_b[inst_points_inds], dim=0)[0]

    #                 box_labels_b_min.append(coords_min_inst)
    #                 box_labels_b_max.append(coords_max_inst)

    #                 # box_labels_b[i, 0:3] = coords_min_inst
    #                 # box_labels_b[i, 3:6] = coords_max_inst
    #         # print(cls_labels_b)
    #         if len(cls_labels_b) == 0:
    #             continue
            
    #         cls_labels_b = torch.tensor(cls_labels_b, device=cls_preds.device, dtype=torch.long)
    #         box_labels_b_min = torch.stack(box_labels_b_min, dim=0)
    #         box_labels_b_max = torch.stack(box_labels_b_max, dim=0)
    #         box_labels_b = torch.cat([box_labels_b_min, box_labels_b_max], dim=-1)

    #         n_inst_gt = len(cls_labels_b)

    #         cost_class = -cls_preds_b[:, cls_labels_b] # n_queries x n_inst

    #         # breakpoint()

    #         # breakpoint()
    #         cost_conf = -conf_preds_b[:, None].repeat(1, n_inst_gt)


    #         # breakpoint()
            
    #         box_preds_b = coords_float_query_b.repeat(1, 2) + box_offset_preds_b

    #         box_preds_b = box_preds_b[:, None, :].repeat(1, n_inst_gt, 1) # n_queries, n_inst, 6
    #         box_labels_b = box_labels_b[None, :, :].repeat(n_queries, 1, 1) # n_queries, n_inst, 6


    #         # breakpoint()

    #         box_preds_b = box_preds_b.reshape(n_queries*n_inst_gt, -1)
    #         box_labels_b = box_labels_b.reshape(n_queries*n_inst_gt, -1)
    #         _, giou_b = giou_aabb_matcher(box_preds_b, box_labels_b)


    #         # breakpoint()

    #         giou_b = giou_b.reshape(n_queries, n_inst_gt)

    #         # cost_offset = torch.cdist(box_preds_b[:, 0:3], box_labels_b[:, 0:3], p=1) + torch.cdist(box_preds_b[:, 3:6], box_labels_b[:, 3:6], p=1)
    #         # breakpoint()
    #         cost_offset = torch.norm(box_preds_b - box_labels_b, p=2, dim=-1)
    #         cost_offset = cost_offset.reshape(n_queries, n_inst_gt)

    #         # breakpoint()

    #         cost_box = -giou_b

    #         # breakpoint()
    #         final_cost = cost_class * 5 + cost_conf * 2 + cost_box * 2 + cost_offset

    #         final_cost = final_cost.detach().cpu().numpy()
            
    #         match_ind = linear_sum_assignment(final_cost)
    #         match_indices.append(match_ind)
    #         cls_labels.append(cls_labels_b)
    #         box_labels.append(box_labels_b)
    #         giou_labels.append(giou_b)

    #     return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in match_indices], cls_labels, box_labels



    # @torch.no_grad()
    # def forward_seg_single(self, mask_logit, cls_logit, instance_masked, semantic_masked, fewshot=False):
    #     with torch.no_grad():

    #         n_mask = instance_masked.shape[-1]

    #         # if n_mask == 0:
    #         #     return None, None, None

    #         unique_inst = sorted(torch.unique(instance_masked))
    #         unique_inst = [i for i in unique_inst if i != -100]
    #         n_inst_gt = len(unique_inst)
    #         # print('unique_inst', unique_inst)
    #         inst_masks = torch.zeros((n_inst_gt, n_mask)).to(mask_logit.device)
    #         sem_labels = torch.zeros((n_inst_gt)).to(mask_logit.device)
    #         # min_inst_id = min(unique_inst)
    #         count = 0
    #         for idx in unique_inst:
    #             temp = (instance_masked == idx)
    #             inst_masks[count,:] = temp

    #             sem_labels[count] = semantic_masked[torch.nonzero(temp)[0]]
    #             count += 1

    #         # dice_cost = compute_dice(mask_logit.reshape(-1, 1, n_mask).repeat(1, n_inst_gt, 1).flatten(0, 1), 
    #         #                         inst_masks.reshape(1, -1, n_mask).repeat(self.n_queries, 1, 1).flatten(0, 1))

    #         # dice_cost = dice_cost.reshape(self.n_queries, n_inst_gt)

    #         sem_logit = torch.nn.functional.softmax(sem_logit, dim=-1)
    #         class_cost = -torch.gather(cls_logit, 1, sem_labels.unsqueeze(0).expand(self.n_queries, n_inst_gt).long())

    #         final_cost = 1 * class_cost + 1 * dice_cost
                
    #         final_cost = final_cost.detach().cpu().numpy()


    #         row_inds, col_inds = linear_sum_assignment(final_cost)


    #         return row_inds, inst_masks[col_inds], sem_labels[col_inds]

def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou)