import os
import tqdm
import wandb

import torch
import torch.nn as nn
import numpy as np
from lib.helpers.save_helper import (get_checkpoint_state, get_full_checkpoint_state,
                                     save_checkpoint, load_checkpoint, load_full_checkpoint)
from lib.losses.loss_function import SoftBoM_Loss,Hierarchical_Task_Learning
from lib.helpers.decode_helper import extract_dets_from_outputs, decode_detections

from tools import eval


class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger,
                 output_path):

        self.cfg_train = cfg['trainer']
        self.cfg_test = cfg['tester']
        self.cfg_data = cfg['dataset']
        self.cfg_enhancements = cfg.get('enhancements', {})
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_name = test_loader.dataset.class_name
        self.eval_cls = cfg['dataset']['eval_cls']
        self.eval_dataset =  cfg['dataset']['eval_dataset'] if 'eval_dataset' in cfg['dataset'].keys() else cfg['dataset']['type']
        self.tester_metrics = 'kitti' if 'tester_metrics' not in self.cfg_test.keys() else self.cfg_test['tester_metrics']

        self.best_e_result = -1
        self.best_e_epoch = -1
        self.best_m_result = -1
        self.best_m_epoch = -1
        self.best_h_result = -1
        self.best_h_epoch = -1
        self.output_path = output_path


        if self.cfg_train.get('resume_model', None):
            assert os.path.exists(self.cfg_train['resume_model'])
            self.epoch = load_checkpoint(self.model, self.optimizer, self.cfg_train['resume_model'], self.logger, map_location=self.device)
            self.lr_scheduler.last_epoch = self.epoch - 1

        self.model = torch.nn.DataParallel(model).to(self.device)

        # A1: Initialize prototype bank if enabled
        self.prototype_bank = None
        pf_cfg = self.cfg_enhancements.get('prototype_filter', {})
        if pf_cfg.get('enabled', False):
            from lib.modules.prototype_filter import PrototypeBank
            self.prototype_bank = PrototypeBank(
                dim=pf_cfg['feature_dim'],
                K=pf_cfg['num_prototypes'],
                m=pf_cfg['members_per_proto'],
                alpha=pf_cfg['ema_alpha'],
                similarity_threshold=pf_cfg['similarity_threshold'],
                depth_reliability_threshold=pf_cfg['depth_reliability_threshold'],
                device=str(self.device)
            )
            self.logger.info(f'[A1] PrototypeBank initialized: {self.prototype_bank}')

    def _make_criterion(self):
        """
        Build SoftBoM_Loss with enhancement configs (C2 + D1).
        The loss class derives prior_weight, affine params, and amb_loss_weight
        internally from the config and model reference.
        """
        base_model = self.model.module if hasattr(self.model, 'module') else self.model
        return SoftBoM_Loss(
            epoch=self.epoch,
            enhancements_cfg=self.cfg_enhancements,
            model_ref=base_model
        )

    def resume_from_checkpoint(self, ckpt_path):
        """Resume full training state from a checkpoint file."""
        self.logger.info("==> Resuming from checkpoint: {}".format(ckpt_path))
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        epoch, best_results, proto_state = load_full_checkpoint(
            checkpoint, self.model, self.optimizer,
            self.lr_scheduler, self.warmup_lr_scheduler,
            logger=self.logger, map_location=self.device
        )
        self.epoch = epoch
        if best_results is not None:
            self.best_e_result = best_results.get('best_e_result', -1)
            self.best_e_epoch = best_results.get('best_e_epoch', -1)
            self.best_m_result = best_results.get('best_m_result', -1)
            self.best_m_epoch = best_results.get('best_m_epoch', -1)
            self.best_h_result = best_results.get('best_h_result', -1)
            self.best_h_epoch = best_results.get('best_h_epoch', -1)
            self.logger.info("  Restored best results: E={:.2f}@{}, M={:.2f}@{}, H={:.2f}@{}".format(
                self.best_e_result, self.best_e_epoch,
                self.best_m_result, self.best_m_epoch,
                self.best_h_result, self.best_h_epoch))
        if proto_state is not None and self.prototype_bank is not None:
            self.prototype_bank.load_state_dict(proto_state)
            self.logger.info("  Restored prototype bank state")
        self.logger.info("==> Resuming training from epoch {}".format(self.epoch + 1))

    def _save_latest_checkpoint(self):
        """Save full training state as latest_checkpoint.pth for auto-resume."""
        os.makedirs(self.output_path + '/checkpoints', exist_ok=True)
        best_results = {
            'best_e_result': self.best_e_result, 'best_e_epoch': self.best_e_epoch,
            'best_m_result': self.best_m_result, 'best_m_epoch': self.best_m_epoch,
            'best_h_result': self.best_h_result, 'best_h_epoch': self.best_h_epoch,
        }
        state = get_full_checkpoint_state(
            model=self.model, optimizer=self.optimizer, epoch=self.epoch,
            lr_scheduler=self.lr_scheduler,
            warmup_lr_scheduler=self.warmup_lr_scheduler,
            best_results=best_results,
            prototype_bank=self.prototype_bank
        )
        ckpt_path = os.path.join(self.output_path, 'checkpoints', 'latest_checkpoint')
        save_checkpoint(state, ckpt_path, self.logger)

    def train(self):
        start_epoch = self.epoch
        ei_loss = self.compute_e0_loss()
        loss_weightor = Hierarchical_Task_Learning(ei_loss)
        for epoch in range(start_epoch, self.cfg_train['max_epoch']):
            # train one epoch
            self.logger.info('------ TRAIN EPOCH %03d ------' %(epoch + 1))
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.logger.info('Learning Rate: %f' % self.warmup_lr_scheduler.get_lr()[0])
            else:
                self.logger.info('Learning Rate: %f' % self.lr_scheduler.get_lr()[0])

            # reset numpy seed.
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            loss_weights = loss_weightor.compute_weight(ei_loss,self.epoch)

            log_str = 'Weights: '
            for key in sorted(loss_weights.keys()):
                log_str += ' %s:%.4f,' %(key[:-4], loss_weights[key])   
            self.logger.info(log_str)

            ei_loss = self.train_one_epoch(loss_weights)
            self.epoch += 1
            
            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()

            if ((self.epoch % self.cfg_train['eval_frequency']) == 0 and \
                self.epoch >= self.cfg_train['eval_start']):
            # if True:
                self.logger.info('------ EVAL EPOCH %03d ------' % (self.epoch))
                os.makedirs(self.output_path+'/checkpoints', exist_ok=True)
                
                Car_res = self.eval_one_epoch(epoch)

                if self.eval_dataset == 'kitti':
                    self.logger.info(str(Car_res))

                    curr_e_result = Car_res['3d@0.70'][0]
                    curr_m_result = Car_res['3d@0.70'][1]
                    curr_h_result = Car_res['3d@0.70'][2]
                    
                    wandb.log({
                        '3D@0.7-easy': curr_e_result,
                        '3D@0.7-moderate': curr_m_result,
                        '3D@0.7-hard':curr_h_result, 
                        'epoch': self.epoch
                    })

                    if curr_e_result > self.best_e_result:
                        self.best_e_result = curr_e_result
                        self.best_e_epoch = self.epoch
                        wandb.log({
                            '[BEST] 3D@0.7-easy': self.best_e_result,
                        })

                    if curr_m_result > self.best_m_result:
                        self.best_m_result = curr_m_result
                        self.best_m_epoch = self.epoch
                        wandb.log({
                            '[BEST] 3D@0.7-moderate': self.best_m_result,
                        })
                        if self.epoch >= self.cfg_train['eval_start']:
                            ckpt_name = os.path.join(self.output_path+'/checkpoints', 'checkpoint_epoch_%d' % self.epoch)
                            save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch), ckpt_name, self.logger)


                    if curr_h_result > self.best_h_result:
                        self.best_h_result = curr_h_result
                        self.best_h_epoch = self.epoch
                        wandb.log({
                            '[BEST] 3D@0.7-hard': self.best_h_result
                        })

                    self.logger.info("Best Easy Result:{}, epoch:{}".format(self.best_e_result, self.best_e_epoch))
                    self.logger.info("Best Moderate Result:{}, epoch:{}".format(self.best_m_result, self.best_m_epoch))
                    self.logger.info("Best Hard Result:{}, epoch:{}".format(self.best_h_result, self.best_h_epoch))
                else:
                    ckpt_name = os.path.join(self.output_path+'/checkpoints', 'checkpoint_epoch_%d' % self.epoch)
                    save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch), ckpt_name, self.logger)

            # Save latest checkpoint every epoch for auto-resume
            self._save_latest_checkpoint()
        return None
    
    def compute_e0_loss(self):
        self.model.train()
        disp_dict = {}
        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=True, desc='pre-training loss stat')
        with torch.no_grad():        
            for batch_idx, (inputs,calibs,coord_ranges, targets, info) in enumerate(self.train_loader):
                if type(inputs) != dict:
                    inputs = inputs.to(self.device)
                else:
                    for key in inputs.keys(): inputs[key] = inputs[key].to(self.device)
                calibs = calibs.to(self.device)
                coord_ranges = coord_ranges.to(self.device)
                for key in targets.keys():
                    targets[key] = targets[key].to(self.device)
    
                # train one batch
                criterion = self._make_criterion()
                outputs = self.model(inputs,coord_ranges,calibs,targets)
                _, loss_terms = criterion(outputs, targets)
                
                trained_batch = batch_idx + 1
                # accumulate statistics
                for key in loss_terms.keys():
                    if key not in disp_dict.keys():
                        disp_dict[key] = 0
                    disp_dict[key] += loss_terms[key]      
                progress_bar.update()
            progress_bar.close()
            for key in disp_dict.keys():
                disp_dict[key] /= trained_batch             
        return disp_dict

    def train_one_epoch(self,loss_weights=None):
        self.model.train()

        disp_dict = {}
        stat_dict = {}
        for batch_idx, (inputs,calibs,coord_ranges, targets, info) in enumerate(self.train_loader):
            if type(inputs) != dict:
                inputs = inputs.to(self.device)
            else:
                for key in inputs.keys(): inputs[key] = inputs[key].to(self.device)
            calibs = calibs.to(self.device)
            coord_ranges = coord_ranges.to(self.device)
            for key in targets.keys(): targets[key] = targets[key].to(self.device)
            # train one batch
            self.optimizer.zero_grad()
            criterion = self._make_criterion()
            outputs = self.model(inputs,coord_ranges,calibs,targets)

            total_loss, loss_terms = criterion(outputs, targets)
            
            if loss_weights is not None:
                total_loss = torch.zeros(1).cuda()
                for key in loss_weights.keys():
                    w = loss_weights[key].detach() if hasattr(loss_weights[key], 'detach') else loss_weights[key]
                    term = loss_terms[key]
                    # Skip NaN loss terms to prevent contamination
                    if isinstance(term, torch.Tensor) and (torch.isnan(term) or torch.isinf(term)):
                        continue
                    # Skip NaN weights
                    if isinstance(w, torch.Tensor) and (torch.isnan(w) or torch.isinf(w)):
                        continue
                    if isinstance(w, float) and (w != w):  # NaN check for float
                        continue
                    total_loss += w * term

            # Guard: skip backward if total_loss is NaN/Inf (prevents model weight corruption)
            if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                self.optimizer.zero_grad()
                continue

            total_loss.backward()
            self.optimizer.step()

            # A1: Update prototype bank with GT RoI features after warmup
            if self.prototype_bank is not None:
                warmup = self.cfg_enhancements['prototype_filter']['warmup_epochs']
                if self.epoch >= warmup and 'roi_features' in outputs:
                    import torch.nn.functional as F_proto
                    roi_feat = outputs['roi_features']  # (N, 64, 7, 7) detached
                    # Global average pool to (N, 64) then L2 normalize
                    pooled = roi_feat.mean(dim=(-2, -1))  # (N, 64)
                    pooled = F_proto.normalize(pooled, dim=-1)
                    self.prototype_bank.update(pooled.to(self.device))

            trained_batch = batch_idx + 1

            # accumulate statistics
            for key in loss_terms.keys():
                if key not in stat_dict.keys():
                    stat_dict[key] = 0

                if isinstance(loss_terms[key], int):
                    stat_dict[key] += (loss_terms[key])
                else:
                    stat_dict[key] += (loss_terms[key]).detach()
            for key in loss_terms.keys():
                if key not in disp_dict.keys():
                    disp_dict[key] = 0
                # disp_dict[key] += loss_terms[key]
                if isinstance(loss_terms[key], int):
                    disp_dict[key] += (loss_terms[key])
                else:
                    disp_dict[key] += (loss_terms[key]).detach()
            # display statistics in terminal
            if trained_batch % self.cfg_train['disp_frequency'] == 0:
                log_str = 'BATCH[%04d/%04d]' % (trained_batch, len(self.train_loader))
                for key in sorted(disp_dict.keys()):
                    disp_dict[key] = disp_dict[key] / self.cfg_train['disp_frequency']
                    log_str += ' %s:%.4f,' %(key, disp_dict[key])
                    wandb.log({
                        key:disp_dict[key]
                    })
                    disp_dict[key] = 0  # reset statistics
                self.logger.info(log_str)
                
        for key in stat_dict.keys():
            stat_dict[key] /= trained_batch
            # Guard: replace NaN/Inf stats with 0 to prevent HTL weightor corruption
            if isinstance(stat_dict[key], torch.Tensor):
                stat_dict[key] = torch.nan_to_num(stat_dict[key], nan=0.0, posinf=0.0, neginf=0.0)
                            
        return stat_dict    

    def eval_one_epoch(self, epoch):
        self.model.eval()

        if self.eval_dataset == "kitti":
            gt_folder = self.cfg_data['root_dir'] + "/training/label_2"
        else:
            raise NotImplementedError

        results = {}
        progress_bar = tqdm.tqdm(total=len(self.test_loader), leave=True, desc='Evaluation Progress')
        with torch.no_grad():
            for batch_idx, (inputs, calibs, coord_ranges, _, info) in enumerate(self.test_loader):
                # load evaluation data and move data to current device.
                if type(inputs) != dict:
                    inputs = inputs.to(self.device)
                else:
                    for key in inputs.keys(): inputs[key] = inputs[key].to(self.device)
                calibs = calibs.to(self.device) 
                coord_ranges = coord_ranges.to(self.device)
    
                outputs = self.model(inputs,coord_ranges,calibs,K=50,mode='val')
                
                dets, amb_scores, dynamic_tau = extract_dets_from_outputs(outputs, K=50)
                dets = dets.detach().cpu().numpy()
                if amb_scores is not None:
                    amb_scores = amb_scores.detach().cpu().numpy()
                if dynamic_tau is not None:
                    dynamic_tau = dynamic_tau.detach().cpu().numpy()
                
                # get corresponding calibs & transform tensor to numpy
                calibs = [self.test_loader.dataset.get_calib(index)  for index in info['img_id']]
                info = {key: val.detach().cpu().numpy() for key, val in info.items()}
                cls_mean_size = self.test_loader.dataset.cls_mean_size
                dets = decode_detections(dets = dets,
                                        info = info,
                                        calibs = calibs,
                                        cls_mean_size=cls_mean_size,
                                        threshold = self.cfg_test['threshold'],
                                        early_exit_cfg=self.cfg_enhancements.get('early_exit'),
                                        amb_scores=amb_scores,
                                        dynamic_tau=dynamic_tau,
                                        prototype_bank=self.prototype_bank,
                                        proto_filter_cfg=self.cfg_enhancements.get('prototype_filter'))
                results.update(dets)
                progress_bar.update()
            progress_bar.close()
        
        
        out_dir = self.output_path + "/epoch_" +str(epoch) +"/data"
        os.makedirs(out_dir, exist_ok=True)

        self.save_results(results, out_dir)

         # Run evaluation
        if self.tester_metrics == 'kitti':
            Car_res = eval.eval_from_scrach(
            gt_folder,
            out_dir,
            self.eval_cls,
            ap_mode=40)
            return Car_res
        else:
            print("Not suppored dataset for evaluation")

    def save_results(self, results, output_dir='./outputs'):
        for img_id in results.keys():
            out_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            f = open(out_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()        
        
      