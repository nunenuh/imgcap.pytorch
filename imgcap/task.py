import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.metrics import Accuracy


class ImageCaptionTask(pl.LightningModule):
    def __init__(self, model, optimizers, criterion, scheduler=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizers
        self.criterion = criterion
        self.scheduler = scheduler
        self.metric = Accuracy()
        
    def forward(self, imgs, captions):
        outputs = self.model(imgs, captions)
        return outputs
        
    
    def shared_step(self, batch, batch_idx):
        imgs, captions = batch
        outputs = self.model(imgs, captions)
        
        outputs_preprocess = outputs.reshape(-1, outputs.shape[2])
        captions_preprocess = captions.reshape(-1)
        loss = self.criterion(outputs_preprocess, captions_preprocess)
        acc = self.metric(outputs_preprocess.argmax(1), captions_preprocess)
        return loss, acc

    
    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx)
        result = pl.TrainResult(loss)
        result.log_dict({'trn_loss': loss, 'trn_acc':acc})
        
        return result
    
    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({'val_loss': loss, 'val_acc': acc})
        
        return result
    
    def configure_optimizers(self):
        if self.scheduler:
            return [self.optimizer], [self.scheduler]
        return self.optimizer
    
  