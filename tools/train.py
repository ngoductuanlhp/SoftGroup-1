import numpy as np

import argparse
import datetime
import os
import os.path as osp
import shutil
import time
import gc

np.random.seed(0)

import torch


torch.manual_seed(0)

import yaml
from munch import Munch
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from softgroup.data import build_dataloader, build_dataset
from softgroup.evaluation import (
    PointWiseEval,
    ScanNetEval,
    evaluate_offset_mae,
    evaluate_semantic_acc,
    evaluate_semantic_miou,
)
from softgroup.model import SoftGroup
from softgroup.model.criterion import Criterion
from softgroup.model.matcher import HungarianMatcher
from softgroup.util import (
    AverageMeter,
    SummaryWriter,
    build_optimizer,
    checkpoint_save,
    collect_results_gpu,
    cosine_lr_after_step,
    get_dist_info,
    get_max_memory,
    get_root_logger,
    init_dist,
    is_main_process,
    is_multiple,
    is_power2,
    load_checkpoint,
)


def get_args():
    parser = argparse.ArgumentParser("SoftGroup")
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("--dist", action="store_true", help="run with distributed parallel")
    parser.add_argument("--resume", type=str, help="path to resume from")
    parser.add_argument("--work_dir", type=str, help="working directory")
    parser.add_argument("--skip_validate", action="store_true", help="skip validation")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--exp_name", type=str, default="default")
    args = parser.parse_args()
    return args


def train(epoch, model, optimizer, scaler, train_loader, cfg, logger, writer):
    model.train()
    iter_time = AverageMeter(True)
    data_time = AverageMeter(True)
    meter_dict = {}
    end = time.time()

    if train_loader.sampler is not None and cfg.dist:
        train_loader.sampler.set_epoch(epoch)

    for i, batch in enumerate(train_loader, start=1):
        data_time.update(time.time() - end)
        cosine_lr_after_step(optimizer, cfg.optimizer.lr, epoch - 1, cfg.step_epoch, cfg.epochs)
        with torch.cuda.amp.autocast(enabled=cfg.fp16):
            loss, log_vars = model(batch, return_loss=True, epoch=epoch - 1)

        # meter_dict
        for k, v in log_vars.items():
            if k not in meter_dict.keys():
                meter_dict[k] = AverageMeter()
            meter_dict[k].update(v)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # time and print
        remain_iter = len(train_loader) * (cfg.epochs - epoch + 1) - i
        iter_time.update(time.time() - end)
        end = time.time()
        remain_time = remain_iter * iter_time.avg
        remain_time = str(datetime.timedelta(seconds=int(remain_time)))
        lr = optimizer.param_groups[0]["lr"]

        if is_multiple(i, 10):
            log_str = f"Epoch [{epoch}/{cfg.epochs}][{i}/{len(train_loader)}]  "
            log_str += (
                f"lr: {lr:.2g}, eta: {remain_time}, mem: {get_max_memory()}, "
                f"data_time: {data_time.val:.2f}, iter_time: {iter_time.val:.2f}"
            )
            for k, v in meter_dict.items():
                log_str += f", {k}: {v.val:.4f}"
            logger.info(log_str)
    writer.add_scalar("train/learning_rate", lr, epoch)
    for k, v in meter_dict.items():
        writer.add_scalar(f"train/{k}", v.avg, epoch)
    checkpoint_save(epoch, model, optimizer, cfg.work_dir, cfg.save_freq)


def validate(epoch, model, optimizer, val_loader, cfg, logger, writer):
    logger.info("Validation")
    # results = []
    # all_sem_preds, all_sem_labels, all_offset_preds, all_offset_labels = [], [], [], []
    all_inst_labels, all_pred_insts, all_gt_insts = [], [], []
    all_debug_accu = []
    # coords = []

    # _, world_size = get_dist_info()
    # progress_bar = tqdm(total=len(val_loader) * world_size, disable=not is_main_process())
    val_set = val_loader.dataset

    point_eval = PointWiseEval(num_classes=cfg.model.semantic_classes)
    scannet_eval = ScanNetEval(val_set.CLASSES)

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(val_loader):
            if i % 10 == 0:
                torch.cuda.empty_cache()
                logger.info(f"Infer scene {i+1}/{len(val_set)}")

            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                result = model(batch)

            
            if not isinstance(result, list):
                result = [result]

            for res in result:
                if cfg.model.semantic_only:
                    point_eval.update(res['semantic_preds'], res['offset_preds'], res['semantic_labels'], res['offset_labels'], res['instance_labels'])
                else:
                    all_pred_insts.append(res['pred_instances'])
                    all_gt_insts.append(res['gt_instances'])
            
            # del result
            # gc.collect()
            # torch.cuda.empty_cache()



    global best_metric

    if cfg.model.semantic_only:
        logger.info("Evaluate semantic segmentation and offset MAE")
        miou, acc, mae = point_eval.get_eval(logger)

        writer.add_scalar("val/mIoU", miou, epoch)
        writer.add_scalar("val/Acc", acc, epoch)
        writer.add_scalar("val/Offset MAE", mae, epoch)

        if best_metric < miou:
            best_metric = miou
            checkpoint_save(epoch, model, optimizer, cfg.work_dir, cfg.save_freq, best=True)

    else:
        logger.info("Evaluate instance segmentation")

        # logger.info('Evaluate axis-align box prediction')
        # eval_res = scannet_eval.evaluate_box(all_pred_insts, all_gt_insts, coords)

        eval_res = scannet_eval.evaluate(all_pred_insts, all_gt_insts)
        del all_pred_insts, all_gt_insts

        writer.add_scalar("val/AP", eval_res["all_ap"], epoch)
        writer.add_scalar("val/AP_50", eval_res["all_ap_50%"], epoch)
        writer.add_scalar("val/AP_25", eval_res["all_ap_25%"], epoch)
        logger.info(
            "AP: {:.3f}. AP_50: {:.3f}. AP_25: {:.3f}".format(
                eval_res["all_ap"], eval_res["all_ap_50%"], eval_res["all_ap_25%"]
            )
        )

        # if len(all_debug_accu) > 0:
        #     accu = np.mean(np.array(all_debug_accu))
        #     logger.info('Mean accuracy of classification: {:.3f}'.format(accu))

        if best_metric < eval_res["all_ap"]:
            best_metric = eval_res["all_ap"]
            checkpoint_save(epoch, model, optimizer, cfg.work_dir, cfg.save_freq, best=True)


def main():
    args = get_args()
    cfg_txt = open(args.config, "r").read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))

    if args.dist:
        init_dist()
    cfg.dist = args.dist

    # work_dir & logger
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = osp.join("./work_dirs", osp.splitext(osp.basename(args.config))[0], args.exp_name)

    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)
    logger.info(f"Config:\n{cfg_txt}")
    logger.info(f"Distributed: {args.dist}")
    logger.info(f"Mix precision training: {cfg.fp16}")
    shutil.copy(args.config, osp.join(cfg.work_dir, osp.basename(args.config)))
    writer = SummaryWriter(cfg.work_dir)

    # model
    matcher = HungarianMatcher()
    criterion = Criterion(matcher, point_wise_loss="input_conv" not in cfg.model.fixed_modules, total_epoch=cfg.epochs)
    model = SoftGroup(**cfg.model, criterion=criterion).cuda()

    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    logger.info(f"Total params: {total_params}")
    logger.info(f"Trainable params: {trainable_params}")

    if args.dist:
        model = DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()], find_unused_parameters=(trainable_params < total_params)
        )

    if args.dist:
        model.module.init_knn()
    else:
        model.init_knn()

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)

    # data
    train_set = build_dataset(cfg.data.train, logger)
    val_set = build_dataset(cfg.data.test, logger)

    # val_set2 = build_dataset(cfg.data.test2, logger)

    train_loader = build_dataloader(train_set, training=True, dist=args.dist, **cfg.dataloader.train)

    # NOTE only validate on single GPU
    val_loader = build_dataloader(val_set, training=False, dist=False, **cfg.dataloader.test)
    # val_loader = build_dataloader(val_set, training=False, dist=args.dist, **cfg.dataloader.test)
    # val_loader2 = build_dataloader(val_set2, training=False, dist=args.dist, **cfg.dataloader.test)
    # optim
    optimizer = build_optimizer(model, cfg.optimizer)

    # pretrain, resume
    start_epoch = 1
    if args.resume:
        logger.info(f"Resume from {args.resume}")
        start_epoch = load_checkpoint(args.resume, logger, model, optimizer=optimizer)
    elif cfg.pretrain:
        logger.info(f"Load pretrain from {cfg.pretrain}")
        load_checkpoint(cfg.pretrain, logger, model)

    # train and val
    logger.info("Training")

    global best_metric
    best_metric = 0

    # if is_main_process():
    #     validate(0, model, optimizer, val_loader, cfg, logger, writer)

    for epoch in range(start_epoch, cfg.epochs + 1):
        train(epoch, model, optimizer, scaler, train_loader, cfg, logger, writer)
        if not args.skip_validate and (is_multiple(epoch, cfg.save_freq) or is_power2(epoch)) and is_main_process():
            validate(epoch, model, optimizer, val_loader, cfg, logger, writer)

            # logger.info('\nvalidate on trainsmall set\n')
            # validate(epoch, model, optimizer, val_loader2, cfg, logger, writer)
        writer.flush()


if __name__ == "__main__":
    main()
