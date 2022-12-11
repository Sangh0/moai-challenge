import os
import logging
import time
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from util.callback import EarlyStopping, CheckPoint
from util.losses import DiceLoss, OhemCELoss, CombiLoss
from util.scheduler import PolynomialLRDecay, CosineAnnealingWarmUpRestarts
from util.metric import Metrics

"""
TensorBoard Session
"""
from torch.utils.tensorboard import SummaryWriter

class Trainer(object):

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        lr: float,
        end_lr: float,
        epochs: int,
        weight_decay: float,
        dice_loss_weight: float=0.5,
        ohem_loss_weight: float=0.5,
        separate_out: bool=True,
        optimizer: str='adam',
        check_point: bool=True,
        early_stop: bool=False,
        lr_scheduling: bool=True,
        scheduler_type: str='cosine',
        pretrained_weight: str='None',
        train_log_step: int=10,
        valid_log_step: int=3,
        weight_save_folder_dir: str='./weights',
    ):
        self.logger = logging.getLogger('training logs')
        self.logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        self.logger.addHandler(stream_handler)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f'device is {self.device}...')

        self.model = model.to(self.device)
        if pretrained_weight != 'None':
            self.model.load_state_dict(torch.load(pretrained_weight))
        self.epochs = epochs

        self.loss_func = CombiLoss(
            dice_loss_weight=dice_loss_weight,
            ohem_loss_weight=ohem_loss_weight,
            separate_out=separate_out,
        ).to(self.device)
        self.logger.info('loss function ready...')

        self.metric = Metrics(n_classes=num_classes, dim=1)
        self.logger.info('metrics ready...')

        if optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay,
            )
        elif optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay, 
                momentum=0.99,
            )
        elif optimizer == 'radam':
            self.optimizer = optim.RAdam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer == 'nadam':
            self.optimizer = optim.NAdam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f'optimizer {optimizer} does not exists')
        self.logger.info('optimizer ready...')
        
        self.lr_scheduling = lr_scheduling
        if scheduler_type == 'polynomial':
            self.lr_scheduler = PolynomialLRDecay(
                self.optimizer, 
                max_decay_steps=self.epochs, 
                end_learning_rate=end_lr,
            )
        elif scheduler_type == 'cosine':
            self.lr_scheduler = CosineAnnealingWarmUpRestarts(
                self.optimizer,
                T_0=int(epochs*0.01),
                T_mult=2,
                eta_max=0.01,
                T_up=10,
                gamma=0.5,
            )
        else:
            raise ValueError(f'{scheduler_type} does not exists')
        self.logger.info('scheduler ready...')

        os.makedirs(weight_save_folder_dir, exist_ok=True)
        self.check_point = check_point
        self.cp = CheckPoint(verbose=True)
        self.early_stop = early_stop
        self.es = EarlyStopping(patience=20, verbose=True, path=weight_save_folder_dir+'/early_stop.pt')
        self.logger.info('callbacks ready...')

        self.train_log_step = train_log_step
        self.valid_log_step = valid_log_step

        self.writer = SummaryWriter()

        self.weight_save_folder_dir = weight_save_folder_dir

    def fit(self, train_loader, valid_loader):
        self.logger.info('\nStart Training Model...!')
        start_training = time.time()
        for epoch in tqdm(range(self.epochs)):
            init_time = time.time()

            train_total_loss, train_ohem_loss, train_dice_loss, \
                    train_miou, train_pix_acc = self.train_on_batch(
                train_loader, epoch,
            )

            valid_total_loss, valid_ohem_loss, valid_dice_loss, \
                    valid_miou, valid_pix_acc = self.valid_on_batch(
                valid_loader, epoch,
            )

            end_time = time.time()

            self.logger.info(f'\n{"="*40} Epoch {epoch+1}/{self.epochs} {"="*40}'
                             f'\n{" "*10}time: {end_time-init_time:.3f}s'
                             f'  lr = {self.optimizer.param_groups[0]["lr"]}')
            self.logger.info(f'[train] total loss: {train_total_loss:.3f},  ohem loss: {train_ohem_loss:.3f}, dice loss: {train_dice_loss:.3f}, miou: {train_miou:.3f}, pixel acc: {train_pix_acc:.3f}',
                        f'\n[valid] total loss: {valid_total_loss:.3f}, ohem loss: {valid_ohem_loss:.3f}, dice loss: {valid_dice_loss:.3f}, miou: {valid_miou:.3f}, pixel acc: {valid_pix_acc:.3f}')
            
            self.writer.add_scalar('lr', self.optimizer.param_groups[0]["lr"], epoch)
            
            if self.lr_scheduling:
                self.lr_scheduler.step()

            if self.check_point:
                path = self.weight_save_folder_dir+f'/check_point_{epoch+1}.pt'
                self.cp(train_total_loss, self.model, path)

            if self.early_stop:
                self.es(valid_dice_loss, self.model)
                if self.es.early_stop:
                    print('\n##########################\n'
                          '##### Early Stopping #####\n'
                          '##########################')
                    break

        self.writer.close()
        end_training = time.time()
        self.logger.info(f'\nTotal time for training is {end_training-start_training:.2f}s')
        return {
            'model': self.model,
        }


    def valid_on_batch(self, valid_loader, epoch):
        self.model.eval()
        batch_total_loss, batch_dice_loss, batch_ohem_loss, batch_miou, batch_pix_acc = 0, 0, 0, 0, 0
        for batch, (images, labels) in enumerate(valid_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images)

            miou = self.metric.mean_iou(outputs, labels)
            batch_miou += miou.item()
            pix_acc = self.metric.pixel_acc(outputs, labels)
            batch_pix_acc += pix_acc.item()

            ohem_loss, dice_loss = self.loss_func(outputs, labels)
            total_loss = ohem_loss + dice_loss

            batch_ohem_loss += ohem_loss.item()
            batch_dice_loss += dice_loss.item()
            batch_total_loss += total_loss.item()

            if (batch+1) % self.valid_log_step == 0:
                self.logger.info(f'\n{" "*20} Valid Batch {batch+1}/{len(valid_loader)} {" "*20}'
                        f'\nvalid total loss: {total_loss:.3f},  ohem_loss: {ohem_loss:.3f}, dice loss: {dice_loss:.3f}, mean IOU: {miou:.3f}, pixel acc: {pix_acc:.3f}')

            step = len(valid_loader) * epoch + batch
            self.writer.add_scalar('Valid/total loss', total_loss, step)
            self.writer.add_scalar('Valid/ohem loss', ohem_loss.item(), step)
            self.writer.add_scalar('Valid/dice loss', dice_loss.item(), step)
            self.writer.add_scalar('Valid/mean IOU', miou.item(), step)
            self.writer.add_scalar('Valid/pixel accuracy', pix_acc.item(), step)

        return batch_total_loss/(batch+1), batch_ohem_loss/(batch+1), batch_dice_loss/(batch+1), batch_miou/(batch+1), batch_pix_acc/(batch+1)

    def train_on_batch(self, train_loader, epoch):
        self.model.train()
        batch_total_loss,  batch_ohem_loss, batch_dice_loss, batch_miou, batch_pix_acc = 0, 0, 0, 0, 0
        for batch, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            miou = self.metric.mean_iou(outputs, labels)
            batch_miou += miou.item()
            pix_acc = self.metric.pixel_acc(outputs, labels)
            batch_pix_acc += pix_acc.item()
            
            ohem_loss, dice_loss = self.loss_func(outputs, labels)
            total_loss = ohem_loss + dice_loss

            batch_ohem_loss += ohem_loss.item()
            batch_dice_loss += dice_loss.item()
            batch_total_loss += total_loss.item()
            total_loss.backward()
            self.optimizer.step()

            if (batch+1) % self.train_log_step == 0:
                self.logger.info(f'\n{" "*20} Train Batch {batch+1}/{len(train_loader)} {" "*20}'
                        f'\ntrain total loss: {total_loss:.3f}, ohem loss: {ohem_loss:.3f}, dice loss: {dice_loss:.3f}, mean IOU: {miou:.3f}, pixel acc: {pix_acc:.3f}')

            step = len(train_loader) * epoch + batch
            self.writer.add_scalar('Train/total loss', total_loss.item(), step)
            self.writer.add_scalar('Train/ohem loss', ohem_loss.item(), step)
            self.writer.add_scalar('Train/dice loss', dice_loss.item(), step)
            self.writer.add_scalar('Train/mean IOU', miou.item(), step)
            self.writer.add_scalar('Train/pixel accuracy', pix_acc.item(), step)

        return batch_total_loss/(batch+1), batch_ohem_loss/(batch+1), batch_dice_loss/(batch+1), batch_miou/(batch+1), batch_pix_acc/(batch+1)
