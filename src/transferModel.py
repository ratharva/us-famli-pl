
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision 

import pytorch_lightning as pl

import torchmetrics
from nets.classification_old import EfficientNet

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


class EfficientNetTransfer(pl.LightningModule):
    def __init__(self, args = None, base_encoder='efficientnet_b0', num_classes=4, lr=1e-4, weight_decay=1e-4, class_weights=None, ckpt_path = "./"):
        super().__init__()
        self.save_hyperparameters()
        self.ckpt_path = ckpt_path

        print("CKPT PATH: ", self.ckpt_path)
        self.efficientnet = EfficientNet(base_encoder="efficientnet-b0").load_from_checkpoint(self.ckpt_path)
        
        self.efficientnet.convnet.classifier[1] = nn.Linear(self.efficientnet.convnet.classifier[1].in_features, num_classes)

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
        F_loss = self.alpha.cuda() * (1 - pt)**self.gamma * CE_loss
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
        return self.efficientnet(x)



class EfficientNetTransferSigmoid(pl.LightningModule):
    def __init__(self, args = None, base_encoder='efficientnet_b0', num_classes=4, lr=1e-3, weight_decay=1e-4, class_weights=None, ckpt_path = "/mnt/raid/C1_ML_Analysis/train_output/classification/extract_frames_blind_sweeps_c1_30082022_wscores_train_train_sample_clean_feat/epoch=9-val_loss=0.27.ckpt"):
        super().__init__()
        self.save_hyperparameters()
        self.ckpt_path = ckpt_path

        print("CKPT PATH: ", self.ckpt_path)
        self.efficientnet = EfficientNet(base_encoder="efficientnet-b0").load_from_checkpoint(self.ckpt_path) 
        # for param in self.efficientnet.parameters():
        #     param.requires_grad = False
        
        # self.efficientnet.convnet.classifier = nn.Sequential(Linear(in_features))
        # print("EfficientNet Model: ", self.efficientnet)
        # raise NotImplementedError
        # self.efficientnet.c
        
        # Replace the last fully connected layer with a new one for multi-label classification
        # print(self.efficientnet.convnet.classifier[1])
        # in_features = self.efficientnet.convnet.classifier.Squential.Linear(in_features, num_classes)
        # print("EfficientNet Model: ", self.efficientnet._fc.in_features)
        # in_features =  self.efficientnet.convnet(num_classes)
        # print("IN_FEATURES: ", in_features)
        # print("EfficientNet Model: ", self.efficientnet.convnet.classifier[1].in_features)
        # print("What is the model: ", self.efficientnet.loss)
        # self.efficientnet.loss = nn.BCEWithLogitsLoss()
        # print("What is the model2: ", self.efficientnet)
        # self.efficientnet.sigmoid = nn.Sigmoid()
        # print("What is the model3: ", self.efficientnet)
        # self.efficientnet.accuracy = nn.MultilabelAccuracy()
        
        self.efficientnet.convnet.classifier[1] = nn.Linear(self.efficientnet.convnet.classifier[1].in_features, num_classes)
        # print("What is this: ", self.efficientnet.convnet.classifier)
        # print("What is the model: ", self.efficientnet.convnet)
        
        # self.sigmoid = nn.Sigmoid()
        # self.myLoss = nn.B
        # print("EfficientNet Model: ", self.efficientnet.convnet.classifier)
        # raise NotImplementedError
        if(class_weights is not None):
            class_weights = torch.tensor(class_weights).to(torch.float32)
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        # self.bLoss = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        self.accuracy = torchmetrics.Accuracy(task="multilabel", num_labels=num_classes)
        self.sigmoid = nn.Sigmoid()
        # print("What is the model2: ", self.efficientnet)


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        # x, y = train_batch
        # x = self(x)
        # # loss = nn.BCEWithLogitsLoss()(y_pred, y.float())
        # y_true = nn.Softmax(y)
        # loss = self.loss(y_pred, y.fl)
        # self.log('train_loss', loss)
        # return loss
        x, y = train_batch
        # print("VALUE OF Y TRAIN: ", y)
        # print("Dimensions of Y: ", y.shape)
        y = y.float()
        # y_true = F.softmax(y, dim=1)
        # y_true = y / y.sum(dim=1, keepdim=True)
        y_true = torch.where(y.sum(dim=1, keepdim=True) != 0, y / y.sum(dim=1, keepdim=True), torch.zeros_like(y))
        # y_true = nn.Sigmoid()(y)
        # y_true = y
        # print("VALUE OF Y TRAIN: ", y_true)
        x = self(x)

        loss = self.loss(x, y_true)
        # loss = self.bLoss(x, y_true)
        
        self.log('train_loss', loss)

        self.accuracy(x, y)
        self.log("train_acc", self.accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):
        # x, y = val_batch
        # y_pred = self(x)
        # loss = nn.BCEWithLogitsLoss()(y_pred, y.float())
        # self.log('val_loss', loss)
        # return loss
        x, y = val_batch
        y = y.float()
        # print("VALUE OF Y VAL: ", y)
        # print("Dimensions of Y: ", y.shape)
        # y_true = F.softmax(y, dim=1)
        y_true = torch.where(y.sum(dim=1, keepdim=True) != 0, y / y.sum(dim=1, keepdim=True), torch.zeros_like(y))
        # y_true = nn.Sigmoid()(y)
        # y_true = y
        # print("VALUE OF Y VAL: ", y_true)
        x = self(x)
        
        loss = self.loss(x, y_true)
        # loss = self.bLoss(x, y_true)
        
        self.log('val_loss', loss, sync_dist=True)
        self.accuracy(x, y)
        self.log("val_acc", self.accuracy, sync_dist=True)

    def forward(self, x):
        # return self.sigmoid(self.efficientnet(x))
        return self.efficientnet(x)
    
    # def predict(self, x):
    #     y_pred = self(x)
    #     return self.sigmoid(y_pred)

    # def predict_thresholded(self, x, threshold=0.5):
    #     probabilities = self.predict(x)
    #     predictions = (probabilities > threshold).int()
    #     return predictions