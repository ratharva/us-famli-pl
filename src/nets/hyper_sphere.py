import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision 

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import TensorDataset, DataLoader

class LightHouse(pl.LightningModule):

    def __init__(self, args = None):
        super().__init__()
        self.save_hyperparameters()
        
        self.light_house = nn.Parameter(torch.abs(F.normalize(torch.rand(self.hparams.args.n_lights, self.hparams.args.emb_dim))))

        self.loss = nn.MSELoss()


    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(),
                                lr=self.hparams.args.lr,
                                momentum=self.hparams.args.momentum,
                                weight_decay=self.hparams.args.weight_decay)
        return optimizer
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
        #                                                     T_max=self.hparams.args.epochs,
        #                                                     eta_min=self.hparams.args.lr/50)
        # return [optimizer], [lr_scheduler]        


    def compute_loss(self, batch, batch_idx, mode):

        loss_north_c = 0

        for i in range(self.hparams.args.n_lights - 1):

            r = torch.randperm(self.hparams.args.n_lights)
            light_house_r = self.light_house[r]
            
            loss_north_c = loss_north_c + self.loss(self.light_house, light_house_r)

        loss_north_c_mean = torch.mean(loss_north_c)
        loss_north_c_std = torch.std(loss_north_c)
        loss_north_c = torch.mean(loss_north_c)
        
        self.log(mode + '_loss', loss_north_c)
        self.log(mode + '_loss_north_c_mean', loss_north_c_mean)
        self.log(mode + '_loss_north_c_std', loss_north_c_std)

        return 1.0/(loss_north_c + 1e-7)


    def training_step(self, batch, batch_idx):
        self.light_house = nn.Parameter(torch.abs(F.normalize(self.light_house)))
        return self.compute_loss(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.compute_loss(batch, batch_idx, "valid")

    def forward(self):
        return self.n

    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.rand(1000, self.hparams.args.n_lights, self.hparams.args.emb_dim)), batch_size=1)
    def val_dataloader(self):
        return DataLoader(TensorDataset(torch.rand(1, self.hparams.args.n_lights, self.hparams.args.emb_dim)), batch_size=1)
