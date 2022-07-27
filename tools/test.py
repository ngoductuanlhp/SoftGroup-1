import argparse
import multiprocessing as mp
import os
import os.path as osp
from functools import partial

import numpy as np
import torch
import yaml
from munch import Munch
from softgroup.data import build_dataloader, build_dataset
from softgroup.evaluation import (ScanNetEval, evaluate_offset_mae, evaluate_semantic_acc,
                                  evaluate_semantic_miou, PointWiseEval)
from softgroup.model import SoftGroup
from softgroup.util import (collect_results_gpu, get_dist_info, get_root_logger, init_dist,
                            is_main_process, load_checkpoint, rle_decode)
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from eval_det import CLASS_LABELS, VALID_CLASS_IDS, eval_sphere

def get_args():
    parser = argparse.ArgumentParser('SoftGroup')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--dist', action='store_true', help='run with distributed parallel')
    parser.add_argument('--out', type=str, help='directory for output results')
    parser.add_argument('--save_lite', action='store_true')
    args = parser.parse_args()
    return args


def save_npy(root, name, scan_ids, arrs):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    paths = [osp.join(root, f'{i}.npy') for i in scan_ids]
    pool = mp.Pool()
    pool.starmap(np.save, zip(paths, arrs))
    pool.close()
    pool.join()


def save_single_instance(root, scan_id, insts):
    f = open(osp.join(root, f'{scan_id}.txt'), 'w')
    os.makedirs(osp.join(root, 'predicted_masks'), exist_ok=True)
    for i, inst in enumerate(insts):
        assert scan_id == inst['scan_id']
        label_id = inst['label_id']
        conf = inst['conf']
        f.write(f'predicted_masks/{scan_id}_{i:03d}.txt {label_id} {conf:.4f}\n')
        mask_path = osp.join(root, 'predicted_masks', f'{scan_id}_{i:03d}.txt')
        mask = rle_decode(inst['pred_mask'])
        np.savetxt(mask_path, mask, fmt='%d')
    f.close()


def save_pred_instances(root, name, scan_ids, pred_insts):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    roots = [root] * len(scan_ids)
    pool = mp.Pool()
    pool.starmap(save_single_instance, zip(roots, scan_ids, pred_insts))
    pool.close()
    pool.join()


def save_gt_instances(root, name, scan_ids, gt_insts):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    paths = [osp.join(root, f'{i}.txt') for i in scan_ids]
    pool = mp.Pool()
    map_func = partial(np.savetxt, fmt='%d')
    pool.starmap(map_func, zip(paths, gt_insts))
    pool.close()
    pool.join()


def main():
    args = get_args()
    cfg_txt = open(args.config, 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    if args.dist:
        init_dist()
    logger = get_root_logger()

    model = SoftGroup(**cfg.model).cuda()
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    logger.info(f'Load state dict from {args.checkpoint}')
    load_checkpoint(args.checkpoint, logger, model)

    dataset = build_dataset(cfg.data.test, logger, lite=args.save_lite)
    dataloader = build_dataloader(dataset, training=False, dist=args.dist, **cfg.dataloader.test)
    results = []
    scan_ids, coords, sem_preds, sem_labels, offset_preds, offset_vertices_preds, offset_labels = [], [], [], [], [], [], []
    nmc_clusters = []
    inst_labels, pred_insts, gt_insts = [], [], []
    nmc_insts = []
    box_preds, box_gt = {}, {}

    _, world_size = get_dist_info()

    # progress_bar = tqdm(total=len(dataloader) * world_size, disable=not is_main_process())
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(dataloader):
            # if args.save_lite and i % 10 != 0:
            #     continue

            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                result = model(batch)
            if isinstance(result, list):
                results.extend(result)
            else:
                results.append(result)

            # if i % 10 == 0:
            logger.info(f'Infer scene {i+1}/{len(dataloader)}')
        #     progress_bar.update(world_size)
        # progress_bar.close()
        results = collect_results_gpu(results, len(dataset))
    if is_main_process():
        point_eval = PointWiseEval()
        for res in results:
            scan_ids.append(res['scan_id'])
            coords.append(res['coords_float'])
            # sem_preds.append(res['semantic_preds'])
            # sem_labels.append(res['semantic_labels'])
            # offset_preds.append(res['offset_preds'])
            # offset_labels.append(res['offset_labels'])
            # inst_labels.append(res['instance_labels'])
            point_eval.update(res['semantic_preds'], res['offset_preds'], res['semantic_labels'], res['offset_labels'], res['instance_labels'])
            if 'debug_accu' in res:
                point_eval.update_debug_acc(res['debug_accu'], res['debug_accu_num_pos'])

            if cfg.save_cfg.offset_vertices:
                offset_vertices_preds.append(res['offset_vertices_preds'])
            if cfg.save_cfg.semantic:
                sem_preds.append(res['semantic_preds'])
            if cfg.save_cfg.offset:
                offset_preds.append(res['offset_preds'])
            if cfg.save_cfg.nmc_clusters:
                nmc_clusters.append(res['nmc_clusters'])

            if not cfg.model.semantic_only:
                pred_insts.append(res['pred_instances'])
                gt_insts.append(res['gt_instances'])

                nmc_insts.append(res['nmc_instances'])

            # if not cfg.model.semantic_only and cfg.model.eval_box:
            #     box_preds[res['scan_id']] = []
            #     for pred in res['pred_instances']:
            #         box_preds[res['scan_id']].append((CLASS_LABELS[int(pred['label_id']-1)], pred['box'], pred['conf']))
                
            #     box_gt[res['scan_id']] = []

            #     instance_num = int(res['instance_labels'].max()) + 1
            #     for i in range(instance_num):
            #         inds = res['instance_labels'] == i
            #         gt_label_loc = np.nonzero(inds)[0][0]
            #         cls_id = int(res['semantic_preds'][gt_label_loc])
            #         if cls_id >= 2:
            #             instance = res['coords_float'][inds]
            #             box_min = instance.min(0)
            #             box_max = instance.max(0)
            #             box = np.concatenate([box_min, box_max])
            #             class_name = CLASS_LABELS[cls_id - 2]
            #             box_gt[res['scan_id']].append((class_name, box))
        
        # NOTE eval final inst mask+box
        if not cfg.model.semantic_only:
            logger.info('Evaluate instance segmentation')
            scannet_eval = ScanNetEval(dataset.CLASSES)
            scannet_eval.evaluate(pred_insts, gt_insts)

            logger.info('Evaluate axis-align box prediction')
            scannet_eval.evaluate_box(pred_insts, gt_insts, coords)

        # # NOTE eval proposal mask_box
        # if not cfg.model.semantic_only:
        #     logger.info('Evaluate instance segmentation nmc')
        #     scannet_eval = ScanNetEval(dataset.CLASSES)
        #     scannet_eval.evaluate(nmc_insts, gt_insts)

        #     logger.info('Evaluate axis-align box prediction nmc')
        #     scannet_eval.evaluate_box(nmc_insts, gt_insts, coords)

        logger.info('Evaluate semantic segmentation and offset MAE')
        ignore_label = cfg.model.ignore_label
        miou, acc, mae = point_eval.get_eval(logger)
        # evaluate_semantic_miou(sem_preds, sem_labels, ignore_label, logger)
        # evaluate_semantic_acc(sem_preds, sem_labels, ignore_label, logger)
        # evaluate_offset_mae(offset_preds, offset_labels, inst_labels, ignore_label, logger)

        # save output
        if not args.out:
            return
        logger.info('Save results')
        # save_npy(args.out, 'coords', scan_ids, coords)
        if cfg.save_cfg.semantic:
            save_npy(args.out, 'semantic_pred', scan_ids, sem_preds)
            # save_npy(args.out, 'semantic_label', scan_ids, sem_labels)
        if cfg.save_cfg.offset:
            save_npy(args.out, 'offset_pred', scan_ids, offset_preds)
            # save_npy(args.out, 'offset_label', scan_ids, offset_labels)
        if cfg.save_cfg.offset_vertices:
            save_npy(args.out, 'offset_vertices_pred', scan_ids, offset_vertices_preds)
        if cfg.save_cfg.instance:
            save_pred_instances(args.out, 'pred_instance', scan_ids, pred_insts)
            # save_gt_instances(args.out, 'gt_instance', scan_ids, gt_insts)
        if cfg.save_cfg.nmc_clusters:
            save_npy(args.out, 'nmc_clusters_ballquery', scan_ids, nmc_clusters)


if __name__ == '__main__':
    main()
