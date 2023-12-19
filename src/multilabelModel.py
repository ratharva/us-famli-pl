import pytorch_lightning as pl
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
import timm
import torchmetrics
from nets.classification_old import EfficientNet


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1., gamma=2.):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss()(inputs, targets) #, reduction='none'
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        return F_loss.mean()



class EfficientNetNew(pl.LightningModule):
    def __init__(self, args = None, base_encoder='efficientnet_b0', num_classes=4, lr=1e-3, weight_decay=1e-4, class_weights=None, ckpt_path = "./"):
        super().__init__()
        self.save_hyperparameters()
        self.ckpt_path = ckpt_path
        # self.efficientnet = timm.create_model("efficientnetv2_rw_m", pretrained=True)
        # # print(self.efficientnet)
        # self.efficientnet = nn.Sequential(*(list(self.efficientnet.children())[:-1]), nn.Linear(2152, num_classes))  # Adjust the output size to fit the number of classes
        # self.sigmoid = nn.Sigmoid()
        self.myLoss = FocalLoss(alpha=1., gamma=2.)

        # print("CKPT PATH: ", self.ckpt_path)
        self.efficientnet = EfficientNet(base_encoder="efficientnet-b0").load_from_checkpoint(self.ckpt_path)
        
        self.efficientnet.convnet.classifier[1] = nn.Linear(self.efficientnet.convnet.classifier[1].in_features, num_classes)

        # if(class_weights is not None):
        #     class_weights = torch.tensor(class_weights).to(torch.float32)
        # self.loss = nn.CrossEntropyLoss(weight=class_weights)
        # # self.bLoss = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        self.accuracy = torchmetrics.Accuracy(task="multilabel", num_labels=num_classes)
        # self.sigmoid = nn.Sigmoid()


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y = y.float()
        x = self(x)
        weights = torch.tensor([1.0, 1.0, 2.0, 3.0, 4.0, 1.0])
        # loss = nn.BCEWithLogitsLoss(weight=weights.cuda())(x, y)
        loss = self.myLoss(x, y)
        
        self.log('train_loss', loss)

        self.accuracy(x, y)
        self.log("train_acc", self.accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y = y.float()
        x = self(x)
        
        # loss = nn.BCEWithLogitsLoss()(x, y)
        loss = self.myLoss(x, y)
        
        self.log('val_loss', loss, sync_dist=True)
        self.accuracy(x, y)
        self.log("val_acc", self.accuracy, sync_dist=True)

    def forward(self, x):
        x = self.efficientnet(x)
        return x
        # return self.sigmoid(x)