import functools

import spconv.pytorch as spconv
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..ops import (ballquery_batch_p, bfs_cluster, get_mask_iou_on_cluster, get_mask_iou_on_pred,
                   get_mask_label, global_avg_pool, sec_max, sec_min, voxelization,
                   voxelization_idx)
from ..util import cuda_cast, force_fp32, rle_encode
from .blocks import MLP, ResidualBlock, UBlock, PositionalEmbedding

def compute_dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    # inputs = inputs.sigmoid()
    # inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum()
    denominator = inputs.sum() + targets.sum()
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

def sigmoid_focal_loss(inputs, targets, weights, alpha: float = 0.25, gamma: float = 2):
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
    prob = inputs
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    
    # return loss.sum()

    loss = (loss * weights).sum()
    return loss

@torch.no_grad()
def iou_aabb(pt_offsets_vertices, pt_offset_vertices_labels, coords):
    coords_min_pred = coords + pt_offsets_vertices[:, 0:3] # N x 3
    coords_max_pred = coords + pt_offsets_vertices[:, 3:6] # N x 3

    coords_min_gt = coords + pt_offset_vertices_labels[:, 0:3] # N x 3
    coords_max_gt = coords + pt_offset_vertices_labels[:, 3:6] # N x 3


    upper = torch.min(coords_max_pred, coords_max_gt) # Nx3
    lower = torch.max(coords_min_pred, coords_min_gt) # Nx3

    intersection = torch.prod(torch.clamp((upper - lower), min=0.0), -1) # N

    gt_volumes = torch.prod(torch.clamp((coords_max_gt - coords_min_gt), min=0.0), -1)
    pred_volumes = torch.prod(torch.clamp((coords_max_pred - coords_min_pred), min=0.0), -1)

    union = gt_volumes + pred_volumes - intersection
    iou = intersection / (union + 1e-6)
    return iou

@torch.no_grad()
def iou_aabb_single(coords_min_pivot, coords_max_pivot, coords_min_neighbors, coords_max_neighbors):
    # coords_min_pred : 1 x 3
    n_neighbors = coords_min_neighbors.shape[0]
    coords_min_pivot = coords_min_pivot.repeat(n_neighbors, 1)
    coords_max_pivot = coords_max_pivot.repeat(n_neighbors, 1)

    coords_min = torch.stack([coords_min_pivot, coords_min_neighbors], -1) # Nx3x2
    coords_max = torch.stack([coords_max_pivot, coords_max_neighbors], -1) # Nx3x2

    coords = torch.cat([coords_min, coords_max], -1) # Nx3x4

    upper = torch.min(coords_max, -1)[0] # Nx3
    lower = torch.max(coords_min, -1)[0] # Nx3

    range = (upper - lower) 
    range[range < 0] = 0

    intersection = torch.prod(range, -1) # N

    union_min = torch.min(coords, -1)[0] # Nx3
    union_max = torch.max(coords, -1)[0] # Nx3
    union = union_max - union_min
    union = torch.prod(union, -1) # N
    union[union<=0] = 1e-6

    iou = intersection / union
    return iou

def cal_iou(volumes, x1, y1, z1, x2, y2, z2, sort_indices, index):
    rem_volumes = torch.index_select(volumes, dim=0, index=sort_indices)

    xx1 = torch.index_select(x1, dim=0, index=sort_indices)
    xx2 = torch.index_select(x2, dim=0, index=sort_indices)
    yy1 = torch.index_select(y1, dim=0, index=sort_indices)
    yy2 = torch.index_select(y2, dim=0, index=sort_indices)
    zz1 = torch.index_select(z1, dim=0, index=sort_indices)
    zz2 = torch.index_select(z2, dim=0, index=sort_indices)

    # centroid_ = torch.index_select(centroid, dim=0, index=sort_indices)
    # pivot = centroid[[index]]


    xx1 = torch.max(xx1, x1[index])
    yy1 = torch.max(yy1, y1[index])
    zz1 = torch.max(zz1, z1[index])
    xx2 = torch.min(xx2, x2[index])
    yy2 = torch.min(yy2, y2[index])
    zz2 = torch.min(zz2, z2[index])


    l = torch.clamp(xx2 - xx1, min=0.0)
    w = torch.clamp(yy2 - yy1, min=0.0)
    h = torch.clamp(zz2 - zz1, min=0.0)

    inter = w*h*l

    union = (rem_volumes - inter) + volumes[index]

    IoU = inter / union

    return IoU

@torch.no_grad()
def non_maximum_cluster(box_conf, coords, pt_offsets, pt_offsets_vertices, batch_offsets, radius=6**2, mean_active=300, iou_thresh=0.3):
    # box_conf: N
    n_points = box_conf.shape[0]
    batch_size = len(batch_offsets) - 1

    coords_centroid = coords + pt_offsets
    coords_min = coords + pt_offsets_vertices[:, :3]
    coords_max = coords + pt_offsets_vertices[:, 3:]

    # proposals_idx = []
    # proposals_offset = [0]
    proposals_conf = []
    proposals_indices = []
    proposal_boxes = []
    proposal_pivots = []

    # cluster_id = 0

    for b in range(batch_size):
        batch_start = batch_offsets[b]
        batch_end = batch_offsets[b + 1]

        centroid = coords_centroid[batch_start:batch_end]

        x1 = coords_min[batch_start:batch_end, 0]
        y1 = coords_min[batch_start:batch_end, 1]
        z1 = coords_min[batch_start:batch_end, 2]
        x2 = coords_max[batch_start:batch_end, 0]
        y2 = coords_max[batch_start:batch_end, 1]
        z2 = coords_max[batch_start:batch_end, 2]

        volumes = (x2 - x1) * (y2 - y1) * (z2 - z1)

        sort_indices = torch.argsort(box_conf[batch_start:batch_end], descending=True)

        # visited = torch.zeros((batch_end - batch_start), dtype=torch.bool, device=box_conf.device)


        while len(sort_indices) > mean_active:
            index = sort_indices[0]
            sort_indices = sort_indices[1:]
                    

            rem_volumes = torch.index_select(volumes, dim=0, index=sort_indices)

            xx1 = torch.index_select(x1, dim=0, index=sort_indices)
            xx2 = torch.index_select(x2, dim=0, index=sort_indices)
            yy1 = torch.index_select(y1, dim=0, index=sort_indices)
            yy2 = torch.index_select(y2, dim=0, index=sort_indices)
            zz1 = torch.index_select(z1, dim=0, index=sort_indices)
            zz2 = torch.index_select(z2, dim=0, index=sort_indices)

            # centroid_ = torch.index_select(centroid, dim=0, index=sort_indices)
            # pivot = centroid[[index]]


            xx1 = torch.max(xx1, x1[index])
            yy1 = torch.max(yy1, y1[index])
            zz1 = torch.max(zz1, z1[index])
            xx2 = torch.min(xx2, x2[index])
            yy2 = torch.min(yy2, y2[index])
            zz2 = torch.min(zz2, z2[index])


            l = torch.clamp(xx2 - xx1, min=0.0)
            w = torch.clamp(yy2 - yy1, min=0.0)
            h = torch.clamp(zz2 - zz1, min=0.0)

            inter = w*h*l

            union = (rem_volumes - inter) + volumes[index]

            IoU = inter / union

            # distances = torch.sum((pivot - centroid_)**2, -1)

            # mask = (IoU >= iou_thresh) | (distances < 0.016)
            mask = (IoU >= iou_thresh)
            neighbor_indices = torch.nonzero(mask).view(-1)
            final_neighbor_indices = sort_indices[neighbor_indices]

            cluster_indices = torch.cat([final_neighbor_indices, index.unsqueeze(0)])
            # cluster_indices_clone = cluster_indices.clone()
            cluster_indices = cluster_indices + batch_start
            len_cluster = len(cluster_indices)

            if len_cluster > mean_active:
                proposal_pivots.append(index+batch_start)
                proposals_conf.append(box_conf[index+batch_start])
                proposals_indices.append(cluster_indices)

                box = torch.tensor([x1[index], y1[index], z1[index], x2[index], y2[index], z2[index]])
                proposal_boxes.append(box)


            sort_indices = sort_indices[~mask]

    if len(proposals_indices) == 0:
        return None

    proposal_pivots = torch.tensor(proposal_pivots, dtype=torch.long, device=coords.device)
    proposals_conf = torch.tensor(proposals_conf, device=coords.device)

    proposals_sort_indices = torch.argsort(proposals_conf, descending=True).long()
    proposal_pivots_final = proposal_pivots[proposals_sort_indices]
    proposals_conf_final = proposals_conf[proposals_sort_indices]

    # breakpoint()
    # proposals_idx = torch.cat(proposals_idx[proposals_sort_indices])
    # proposals_offset = proposals_offset[proposals_sort_indices]
    cluster_id = 0
    proposals_idx_final = []
    proposals_offset_final = [0]
    proposals_box_final = []
    for i in range(len(proposals_sort_indices)):
        proposals_offset_final.append(proposals_offset_final[-1] + len(proposals_indices[proposals_sort_indices[i]]))
        proposals_idx_final.append(torch.stack([torch.ones_like(proposals_indices[proposals_sort_indices[i]]) * cluster_id, proposals_indices[proposals_sort_indices[i]]], -1))
        proposals_box_final.append(proposal_boxes[proposals_sort_indices[i]])
        cluster_id += 1

    proposals_idx_final = torch.cat(proposals_idx_final)
    proposals_offset_final = torch.tensor(proposals_offset_final)
    proposals_box_final = torch.stack(proposals_box_final, 0) # nProposals, 6

    return proposals_idx_final.int(), proposals_offset_final.int(), proposals_box_final, proposals_conf_final, proposal_pivots_final

@torch.no_grad()
def non_maximum_cluster2(box_conf, coords, pt_offsets, pt_offsets_vertices, batch_offsets, radius=6**2, mean_active=300, iou_thresh=0.3):
    # box_conf: N
    n_points = box_conf.shape[0]
    batch_size = len(batch_offsets) - 1

    coords_centroid = coords + pt_offsets
    coords_min = coords + pt_offsets_vertices[:, :3]
    coords_max = coords + pt_offsets_vertices[:, 3:]

    # proposals_idx = []
    # proposals_offset = [0]
    proposals_conf = []
    proposals_indices = []
    proposal_boxes = []

    # cluster_id = 0

    for b in range(batch_size):
        batch_start = batch_offsets[b]
        batch_end = batch_offsets[b + 1]

        xm = coords_centroid[batch_start:batch_end, 0]
        ym = coords_centroid[batch_start:batch_end, 1]
        zm = coords_centroid[batch_start:batch_end, 2]

        x1 = coords_min[batch_start:batch_end, 0]
        y1 = coords_min[batch_start:batch_end, 1]
        z1 = coords_min[batch_start:batch_end, 2]
        x2 = coords_max[batch_start:batch_end, 0]
        y2 = coords_max[batch_start:batch_end, 1]
        z2 = coords_max[batch_start:batch_end, 2]

        volumes1 = (xm - x1) * (ym - y1) * (zm - z1)
        volumes2 = (x2 - xm) * (y2 - ym) * (z2 - zm)
        volumes3 = (x2 - x1) * (y2 - y1) * (z2 - z1)



        sort_indices = torch.argsort(box_conf[batch_start:batch_end], descending=True)

        # visited = torch.zeros((batch_end - batch_start), dtype=torch.bool, device=box_conf.device)


        while len(sort_indices) > mean_active:
            index = sort_indices[0]
            sort_indices = sort_indices[1:]
            # while len(sort_indices) > 0:
            #     index = sort_indices[0]
            #     sort_indices = sort_indices[1:]
            #     if visited[index] == False:
            #         break
            # if len(sort_indices) < 50:
            #     break
                    
            iou1 = cal_iou(volumes1, x1, y1, z1, xm, ym, zm, sort_indices, index)
            iou2 = cal_iou(volumes2, xm, ym, zm, x2, y2, z2, sort_indices, index)
            iou3 = cal_iou(volumes3, x1, y1, z1, x2, y2, z2, sort_indices, index)

            # IoU = (iou1 + iou2) / 2.0
            IoU = torch.maximum(torch.maximum(iou1, iou2), iou3)
            # rem_volumes = torch.index_select(volumes, dim=0, index=sort_indices)

            # xx1 = torch.index_select(x1, dim=0, index=sort_indices)
            # xx2 = torch.index_select(x2, dim=0, index=sort_indices)
            # yy1 = torch.index_select(y1, dim=0, index=sort_indices)
            # yy2 = torch.index_select(y2, dim=0, index=sort_indices)
            # zz1 = torch.index_select(z1, dim=0, index=sort_indices)
            # zz2 = torch.index_select(z2, dim=0, index=sort_indices)

            # # centroid_ = torch.index_select(centroid, dim=0, index=sort_indices)
            # # pivot = centroid[[index]]


            # xx1 = torch.max(xx1, x1[index])
            # yy1 = torch.max(yy1, y1[index])
            # zz1 = torch.max(zz1, z1[index])
            # xx2 = torch.min(xx2, x2[index])
            # yy2 = torch.min(yy2, y2[index])
            # zz2 = torch.min(zz2, z2[index])


            # l = torch.clamp(xx2 - xx1, min=0.0)
            # w = torch.clamp(yy2 - yy1, min=0.0)
            # h = torch.clamp(zz2 - zz1, min=0.0)

            # inter = w*h*l

            # union = (rem_volumes - inter) + volumes[index]

            # IoU = inter / union

            # distances = torch.sum((pivot - centroid_)**2, -1)

            # mask = (IoU >= iou_thresh) | (distances < 0.016)
            mask = (IoU >= iou_thresh)
            # mask = (iou3 >= iou_thresh) | (iou1 >= 0.5) | (iou2 >= 0.5)
            neighbor_indices = torch.nonzero(mask).view(-1)
            final_neighbor_indices = sort_indices[neighbor_indices]

            cluster_indices = torch.cat([final_neighbor_indices, index.unsqueeze(0)])
            # cluster_indices_clone = cluster_indices.clone()
            cluster_indices = cluster_indices + batch_start
            len_cluster = len(cluster_indices)

            if len_cluster > mean_active:

                proposals_conf.append(box_conf[index+batch_start])
                proposals_indices.append(cluster_indices)

                box = torch.tensor([x1[index], y1[index], z1[index], x2[index], y2[index], z2[index]])
                proposal_boxes.append(box)


            sort_indices = sort_indices[~mask]

    if len(proposals_indices) == 0:
        return [], [0], None, None

    proposals_conf = torch.tensor(proposals_conf)
    proposals_sort_indices = torch.argsort(proposals_conf, descending=True).long()
    proposals_conf_final = proposals_conf[proposals_sort_indices]

    # breakpoint()
    # proposals_idx = torch.cat(proposals_idx[proposals_sort_indices])
    # proposals_offset = proposals_offset[proposals_sort_indices]
    cluster_id = 0
    proposals_idx_final = []
    proposals_offset_final = [0]
    proposals_box_final = []
    for i in range(len(proposals_sort_indices)):
        proposals_offset_final.append(proposals_offset_final[-1] + len(proposals_indices[proposals_sort_indices[i]]))
        proposals_idx_final.append(torch.stack([torch.ones_like(proposals_indices[proposals_sort_indices[i]]) * cluster_id, proposals_indices[proposals_sort_indices[i]]], -1))
        proposals_box_final.append(proposal_boxes[proposals_sort_indices[i]])
        cluster_id += 1

    proposals_idx_final = torch.cat(proposals_idx_final)
    proposals_offset_final = torch.tensor(proposals_offset_final)
    proposals_box_final = torch.stack(proposals_box_final, 0) # nProposals, 6

    # breakpoint()
    return proposals_idx_final.cpu().int(), proposals_offset_final.cpu().int(), proposals_box_final, proposals_conf_final



def dice_coefficient(x, target):
    eps = 1e-5
    # n_inst = 1
    # x = x.reshape(n_inst, -1)
    # target = target.reshape(n_inst, -1)
    intersection = (x * target).sum()
    union = (x ** 2.0).sum() + (target ** 2.0).sum() + eps
    loss = 1. - (2 * intersection / union)
    return loss

def compute_dice_loss(inputs, targets):
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
    numerator = 2 * (inputs * targets).sum()
    denominator = inputs.sum() + targets.sum()
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

def sigmoid_focal_loss(inputs, targets, weights, alpha: float = 0.25, gamma: float = 2):
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
    
    # return loss.sum()

    loss = (loss * weights).sum()
    return loss

def get_bounding_vertices(coords_min, coords_max, coords_mean):
    c1 = torch.cat([coords_min[:, [0]], coords_min[:, [1]], coords_min[:, [2]]], 1)
    c2 = torch.cat([coords_min[:, [0]], coords_min[:, [1]], coords_max[:, [2]]], 1)
    c3 = torch.cat([coords_min[:, [0]], coords_max[:, [1]], coords_min[:, [2]]], 1)
    c4 = torch.cat([coords_min[:, [0]], coords_max[:, [1]], coords_max[:, [2]]], 1)
    c5 = torch.cat([coords_max[:, [0]], coords_min[:, [1]], coords_min[:, [2]]], 1)
    c6 = torch.cat([coords_max[:, [0]], coords_min[:, [1]], coords_max[:, [2]]], 1)
    c7 = torch.cat([coords_max[:, [0]], coords_max[:, [1]], coords_min[:, [2]]], 1)
    c8 = torch.cat([coords_max[:, [0]], coords_max[:, [1]], coords_max[:, [2]]], 1)

    c = torch.cat([c1, c2, c3, c4, c5, c6, c7, c8, coords_mean], 1)
    return c






        

