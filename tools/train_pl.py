import argparse
import datetime
import os
import os.path as osp
import shutil
import time

import torch
import yaml
from munch import Munch
from softgroup.data import build_dataloader, build_dataset
from softgroup.evaluation import (ScanNetEval, evaluate_offset_mae, evaluate_semantic_acc,
                                  evaluate_semantic_miou, PointWiseEval)
from softgroup.model import SoftGroup
from softgroup.util import (AverageMeter, SummaryWriter, build_optimizer, checkpoint_save,
                            collect_results_gpu, cosine_lr_after_step, get_dist_info,
                            get_max_memory, get_root_logger, init_dist, is_main_process,
                            is_multiple, is_power2, load_checkpoint)

from softgroup.model.model_module import ModelModule
from tqdm import tqdm

import pytorch_lightning as pl
import hydra

# from pytorch_lightning.loggers import TestTubeLogger
# from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer

from pl_utils import CheckpointEveryNSteps

def get_args():
    parser = argparse.ArgumentParser('SoftGroup')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('--dist', action='store_true', help='run with distributed parallel')
    parser.add_argument('--resume', type=str, help='path to resume from')
    parser.add_argument('--work_dir', type=str, help='working directory')
    parser.add_argument('--skip_validate', action='store_true', help='skip validation')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--exp_name", type=str, default='default')
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
            loss, log_vars = model(batch, return_loss=True)

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
        lr = optimizer.param_groups[0]['lr']

        if is_multiple(i, 10):
            log_str = f'Epoch [{epoch}/{cfg.epochs}][{i}/{len(train_loader)}]  '
            log_str += f'lr: {lr:.2g}, eta: {remain_time}, mem: {get_max_memory()}, '\
                f'data_time: {data_time.val:.2f}, iter_time: {iter_time.val:.2f}'
            for k, v in meter_dict.items():
                log_str += f', {k}: {v.val:.4f}'
            logger.info(log_str)
    writer.add_scalar('train/learning_rate', lr, epoch)
    for k, v in meter_dict.items():
        writer.add_scalar(f'train/{k}', v.avg, epoch)
    checkpoint_save(epoch, model, optimizer, cfg.work_dir, cfg.save_freq)


def validate(epoch, model, optimizer, val_loader, cfg, logger, writer):
    logger.info('Validation')
    results = []
    all_sem_preds, all_sem_labels, all_offset_preds, all_offset_labels = [], [], [], []
    all_inst_labels, all_pred_insts, all_gt_insts = [], [], []
    _, world_size = get_dist_info()
    progress_bar = tqdm(total=len(val_loader) * world_size, disable=not is_main_process())
    val_set = val_loader.dataset
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(val_loader):
            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                result = model(batch)
            results.append(result)
            progress_bar.update(world_size)
        progress_bar.close()
        results = collect_results_gpu(results, len(val_set))
    if is_main_process():
        point_eval = PointWiseEval()
        for res in results:
            # all_sem_preds.append(res['semantic_preds'])
            # all_sem_labels.append(res['semantic_labels'])
            # all_offset_preds.append(res['offset_preds'])
            # all_offset_labels.append(res['offset_labels'])
            # all_inst_labels.append(res['instance_labels'])
            point_eval.update(res['semantic_preds'], res['offset_preds'], res['semantic_labels'], res['offset_labels'], res['instance_labels'])
            if not cfg.model.semantic_only:
                all_pred_insts.append(res['pred_instances'])
                all_gt_insts.append(res['gt_instances'])

        if not cfg.model.semantic_only:
            logger.info('Evaluate instance segmentation')
            scannet_eval = ScanNetEval(val_set.CLASSES)
            eval_res = scannet_eval.evaluate(all_pred_insts, all_gt_insts)
            writer.add_scalar('val/AP', eval_res['all_ap'], epoch)
            writer.add_scalar('val/AP_50', eval_res['all_ap_50%'], epoch)
            writer.add_scalar('val/AP_25', eval_res['all_ap_25%'], epoch)
            logger.info('AP: {:.3f}. AP_50: {:.3f}. AP_25: {:.3f}'.format(
                eval_res['all_ap'], eval_res['all_ap_50%'], eval_res['all_ap_25%']))
        logger.info('Evaluate semantic segmentation and offset MAE')

        miou, acc, mae = point_eval.get_eval(logger)
        # miou = evaluate_semantic_miou(all_sem_preds, all_sem_labels, cfg.model.ignore_label, logger)
        # acc = evaluate_semantic_acc(all_sem_preds, all_sem_labels, cfg.model.ignore_label, logger)
        # mae = evaluate_offset_mae(all_offset_preds, all_offset_labels, all_inst_labels,
        #                           cfg.model.ignore_label, logger)
        writer.add_scalar('val/mIoU', miou, epoch)
        writer.add_scalar('val/Acc', acc, epoch)
        writer.add_scalar('val/Offset MAE', mae, epoch)

        global best_metric

        if not cfg.model.semantic_only:
            if best_metric < eval_res['all_ap_50%']:
                best_metric = eval_res['all_ap_50%']
                checkpoint_save(epoch, model, optimizer, cfg.work_dir, cfg.save_freq, best=True)
        else:
            if best_metric < miou:
                best_metric = miou
                checkpoint_save(epoch, model, optimizer, cfg.work_dir, cfg.save_freq, best=True)


def main():
    args = get_args()
    cfg_txt = open(args.config, 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))

    # if args.dist:
    #     init_dist()
    # cfg.dist = args.dist

    # work_dir & logger
    exp_name = osp.splitext(osp.basename(args.config))[0] + '/' +  args.exp_name
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = osp.join('./work_dirs', exp_name)
    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)

    num_gpus = torch.cuda.device_count() if args.dist else 1

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file)
    logger.info(f'Config:\n{cfg_txt}')
    logger.info(f'Distributed: {num_gpus > 1}')
    logger.info(f'Mix precision training: {cfg.fp16}')
    shutil.copy(args.config, osp.join(cfg.work_dir, osp.basename(args.config)))
    # writer = SummaryWriter(cfg.work_dir)


    
    pl_logger = TensorBoardLogger(
        save_dir="work_dirs",
        name=exp_name,
        # debug=False,
        # create_git_tag=False
    )

    # model
    model = SoftGroup(**cfg.model)

    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    logger.info(f'Total params: {total_params}')
    logger.info(f'Trainable params: {trainable_params}')

    if cfg.pretrain:
        logger.info(f'Load pretrain from {cfg.pretrain}')
        load_checkpoint(cfg.pretrain, logger, model)

    # train and val
    logger.info('Training')

    model_module = ModelModule(cfg, model, logger)

    if cfg.model.semantic_only:
        save_metric = 'val/mIoU'
    else:
        save_metric = 'val/AP'
    cp_callback = ModelCheckpoint(dirpath=f'ckpts/{exp_name}',
                                    filename='{epoch:d}',
                                    monitor=save_metric,
                                    mode='max',
                                    save_top_k=1,
                                    save_last=True)
    
    
    # progress_bar = pl.callbacks.progress.TQDMProgressBar(refresh_rate=10)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # trainer = Trainer(max_epochs=cfg.epochs,
    #                 #  checkpoint_callback=[cp_callback],
    #                 callbacks=[cp_callback, CheckpointEveryNSteps(save_epoch=5, dir=f'ckpts/{exp_name}'), progress_bar, lr_monitor],
    #                 # resume_from_checkpoint = args.resume,
    #                 precision=16 if cfg.fp16 else 32,
    #                 accelerator="gpu",
    #                 logger = pl_logger,
    #                 enable_model_summary=False,
    #                 progress_bar_refresh_rate = 10,
    #                 gpus=num_gpus,
    #                 strategy="ddp" if num_gpus > 1 else None,
    #                 num_sanity_val_steps = 1,
    #                 benchmark = True
    #                 )

    trainer = Trainer(max_epochs=cfg.epochs,
                    #  checkpoint_callback=[cp_callback],
                    callbacks=[cp_callback, CheckpointEveryNSteps(save_epoch=5, dir=f'ckpts/{exp_name}'), lr_monitor],
                    # resume_from_checkpoint = hparams.ckpt_path,
                    logger = pl_logger,
                    weights_summary = None,
                    progress_bar_refresh_rate = 10,
                    gpus=num_gpus,
                    distributed_backend = 'ddp' if num_gpus > 1 else None,
                    num_sanity_val_steps = 1,
                    deterministic=True,
                    benchmark = True
                    )

    trainer.fit(model_module)
    
    


if __name__ == '__main__':
    main()
