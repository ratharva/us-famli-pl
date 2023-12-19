
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision 

import pytorch_lightning as pl

import torchmetrics
from nets.classification_old import EfficientNet
from monai.networks.nets import EfficientNetBN

torch.multiprocessing.set_sharing_strategy('file_system')

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1., gamma=2.):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight =  torch.tensor([1, 1, 1, 1, 1]).cuda()

    def forward(self, inputs, targets):
        CE_loss = nn.CrossEntropyLoss()(inputs, targets) #, reduction='none'
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * CE_loss
        return F_loss.mean()


class EfficientNetMonai(pl.LightningModule):
    def __init__(self, args = None, in_channels=1, base_encoder='efficientnet-b0', num_classes=5, lr=1e-4, weight_decay=1e-4, class_weights=None, ckpt_path = "./"):
        super().__init__()
        self.save_hyperparameters()
        self.ckpt_path = ckpt_path

        print("CKPT PATH: ", self.ckpt_path)
        self.encoder = EfficientNetBN(in_channels=in_channels, model_name=base_encoder, num_classes=num_classes, pretrained=False)

        if(class_weights is not None):
            class_weights = torch.tensor(class_weights).to(torch.float32)
        # self.loss = nn.CrossEntropyLoss(weight=class_weights)
        # weights = weights.cuda()
        # device = '
        # myDevices = torch.cuda.set_device(device)
        # print(myDevices)
        self.alpha = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0])
        # self.alpha = self.alpha.cuda()
        self.gamma = 2.0
        # self.loss = FocalLoss(alpha=myAlpha, gamma=2)
        # self.bLoss = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        self.accuracy = torchmetrics.Accuracy(task="multilabel", num_labels=num_classes)
        self.aucRoc = torchmetrics.AUROC(num_labels=num_classes, average='weighted', task="multilabel")
        # self.sigmoid = nn.Sigmoid()

    def focal_loss(self, inputs, targets):
        CE_loss = nn.CrossEntropyLoss()(inputs, targets)  # Calculate the BCE loss
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha.cuda() * (1 - pt)**self.gamma * CE_loss #.cuda()
        return F_loss.mean()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y = y.float()
        y_true = torch.where(y.sum(dim=1, keepdim=True) != 0, y / y.sum(dim=1, keepdim=True), torch.zeros_like(y))
        x = self(x)

        loss = self.focal_loss(x, y_true)
        
        self.log('train_loss', loss)

        self.accuracy(x, y)
        self.log("train_acc", self.accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y = y.float()
        y_true = torch.where(y.sum(dim=1, keepdim=True) != 0, y / y.sum(dim=1, keepdim=True), torch.zeros_like(y))
        x = self(x)
        
        loss = self.focal_loss(x, y_true)
        myAucroc = self.aucRoc(x, y)
        combinedMetrics = myAucroc - loss
        
        self.log("combinedMetrics", combinedMetrics)
        self.log("auc_roc", myAucroc)
        self.log('val_loss', loss, sync_dist=True)
        self.accuracy(x, y)
        self.log("val_acc", self.accuracy, sync_dist=True)

    def forward(self, x):
        return self.encoder(x)