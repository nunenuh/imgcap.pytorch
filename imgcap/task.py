import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence


import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.metrics import Accuracy

class ImageCaptionTask(pl.LightningModule):
    def __init__(self, model, optimizer, criterion, vocab_size, scheduler=None, batch_first=True):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.vocab_size = vocab_size
        self.batch_first = batch_first
        self.metric = Accuracy()
        
    def forward(self, imgs, captions):
        outputs = self.model(imgs, captions[:-1])
        return outputs
        
    def shared_step(self, batch, batch_idx):
        imgs, captions, lengths = batch
        packed = pack_padded_sequence(captions, lengths, batch_first=self.batch_first)
        targets, _, _, _ = packed
        
        outputs = self.model(imgs, captions, lengths)
        loss = self.criterion(outputs, targets)
        acc = self.metric(outputs, targets)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx)
        result = pl.TrainResult(loss)
        result.log_dict({'trn_loss': loss, 'trn_acc': acc})
        
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
    
  