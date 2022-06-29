import torch
import pytorch_lightning as pl
from softgroup.data import build_dataloader, build_dataset
from torch.utils.data import DataLoader
from softgroup.evaluation import (ScanNetEval, evaluate_offset_mae, evaluate_semantic_acc,
                                  evaluate_semantic_miou, PointWiseEval)
from softgroup.util import (AverageMeter, SummaryWriter, build_optimizer, checkpoint_save,
                            collect_results_gpu, cosine_lr_after_step, get_dist_info,
                            get_max_memory, get_root_logger, init_dist, is_main_process,
                            is_multiple, is_power2, load_checkpoint)

class ModelModule(pl.LightningModule):
    def __init__(self, cfg, model, tlogger):
        super().__init__()

        # self.save_hyperparameters(
        #     cfg,
        #     ignore=['model', 'logger', 'metrics', 'optimizer_args', 'scheduler_args'])

        self.cfg = cfg
        self.model = model
        self.tlogger = tlogger

        self.automatic_optimization = False

        # self.loss_func = loss_func
        # self.metrics = metrics

        # self.optimizer_args = optimizer_args
        # self.scheduler_args = scheduler_args

    # def forward(self, batch):
    #     return self.backbone(batch)

    def prepare_data(self):
        
        self.train_dataset = build_dataset(self.cfg.data.train, self.tlogger)
        self.val_dataset = build_dataset(self.cfg.data.test, self.tlogger)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.dataloader.train.batch_size,
            num_workers=self.cfg.dataloader.train.batch_size,
            collate_fn=self.train_dataset.collate_fn,
            shuffle=True,
            drop_last=True,
            pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=1,
            collate_fn=self.val_dataset.collate_fn,
            shuffle=False,
            drop_last=False,
            pin_memory=True)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        loss, loss_dict = self.model.forward_train(**batch)

        self.manual_backward(loss)
        opt.step()

        self.log('train/loss', loss.detach(), prog_bar = True)
        self.log('train/cls_loss', loss_dict['cls_loss'].detach(), prog_bar = True)
        self.log('train/mask_loss', loss_dict['mask_loss'].detach(), prog_bar = True)
        # return {'loss': loss}

    def training_epoch_end(self, outputs):
        optimizer = self.optimizers().optimizer
        cosine_lr_after_step(optimizer, self.cfg.optimizer.lr, self.current_epoch, self.cfg.step_epoch, self.cfg.epochs)
        return super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        ret = self.model.forward_test(**batch)

        return ret

    # def on_validation_start(self) -> None:
        # self._log_epoch_metrics('train')
        # self._enable_dataloader_shuffle(self.trainer.val_dataloaders)

    def validation_epoch_end(self, outputs):
        point_eval = PointWiseEval()
        all_pred_insts, all_gt_insts = [], []

        for out in outputs:
            point_eval.update(out['semantic_preds'], out['offset_preds'], out['semantic_labels'], out['offset_labels'], out['instance_labels'])
            if not self.cfg.model.semantic_only:
                all_pred_insts.append(out['pred_instances'])
                all_gt_insts.append(out['gt_instances'])

        miou, acc, mae = point_eval.get_eval(self.tlogger)
        self.log('val/mIoU', miou)
        self.log('val/Acc', acc)
        self.log('val/MAE', mae)

        if not self.cfg.model.semantic_only:
            self.tlogger.info('Evaluate instance segmentation')

            try:
                scannet_eval = ScanNetEval(self.val_dataset.CLASSES)
                eval_res = scannet_eval.evaluate(all_pred_insts, all_gt_insts)
                self.log('val/AP', eval_res['all_ap'])
                self.log('val/AP_50', eval_res['all_ap_50%'])
                self.log('val/AP_25', eval_res['all_ap_25%'])
                self.tlogger.info('AP: {:.3f}. AP_50: {:.3f}. AP_25: {:.3f}'.format(
                    eval_res['all_ap'], eval_res['all_ap_50%'], eval_res['all_ap_25%']))
            except:
                self.log('val/AP', -1)
                self.log('val/AP_50', -1)
                self.log('val/AP_25', -1)
                self.tlogger.info("Error in eval instance segmentation")


    def configure_optimizers(self):
        optim_cfg = self.cfg.optimizer
        parameters = [x for x in self.model.parameters() if x.requires_grad]
        # optimizer = torch.optim.AdamW(parameters, **self.optimizer_args)
        optim_type = optim_cfg.pop('type')
        optimizer = getattr(torch.optim, optim_type)

        optimizer = optimizer(parameters, **optim_cfg)
        
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.epochs, eta_min=1e-6, last_epoch=-1, verbose=False)
        # if disable_scheduler or self.scheduler_args is None:
        #     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda lr: 1)
        # else:
        #     scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **self.scheduler_args)

        return [optimizer]
