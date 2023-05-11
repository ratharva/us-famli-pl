import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision 

import pytorch_lightning as pl

import torchmetrics

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=1280, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        )

    def forward(self, x):
        x = self.model(x)  
        x = torch.abs(x)      
        return F.normalize(x, dim=1)

class GaussianNoise(nn.Module):    
    def __init__(self, mean=0.0, std=0.1):
        super(GaussianNoise, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    def forward(self, x):
        return x + torch.normal(mean=self.mean, std=self.std, size=x.size(), device=x.device)

class EfficientNet(pl.LightningModule):
    def __init__(self, args = None, base_encoder='efficientnet_b0', num_classes=4, lr=1e-3, weight_decay=1e-4, class_weights=None):
        super().__init__()
        self.save_hyperparameters()

        template_model = getattr(torchvision.models, self.hparams.base_encoder)
        self.convnet = template_model(num_classes=num_classes)

        self.extract_features = False

        if hasattr(args, 'model_feat') and args.model_feat:
            classifier = self.convnet.classifier
            self.convnet.classifier = nn.Identity()
            self.convnet.load_state_dict(torch.load(args.model_feat))
            # for param in self.convnet.parameters():
            #     param.requires_grad = False
            self.convnet.classifier = classifier


        if(class_weights is not None):
            class_weights = torch.tensor(class_weights).to(torch.float32)
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        self.softmax = nn.Softmax(dim=1)

        self.noise_transform = torch.nn.Sequential(
            GaussianNoise()
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        
        x = self(x)

        loss = self.loss(x, y)
        
        self.log('train_loss', loss)

        self.accuracy(x, y)
        self.log("train_acc", self.accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        
        x = self(x)
        
        loss = self.loss(x, y)
        
        self.log('val_loss', loss, sync_dist=True)
        self.accuracy(x, y)
        self.log("val_acc", self.accuracy)

    def forward(self, x):        
        if self.extract_features:
            x_f = self.convnet.features(x)
            x_f = self.convnet.avgpool(x_f)            
            x = torch.flatten(x_f, 1)            
            return self.convnet.classifier(x), x_f
        else:
            return self.convnet(x)

class EfficientnetB0(pl.LightningModule):
    def __init__(self, args = None, out_features=4, class_weights=None, features=False):
        super(EfficientnetB0, self).__init__()        
        
        self.save_hyperparameters()        
        self.args = args

        self.class_weights = class_weights
        self.features = features

        if(class_weights is not None):
            class_weights = torch.tensor(class_weights).to(torch.float32)
            
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy()

        self.model = nn.Sequential(
            models.efficientnet_b0(pretrained=True).features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=1280, out_features=out_features, bias=True)
            )
        
        self.softmax = nn.Softmax(dim=1)
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer

    def forward(self, x):        
        if self.features:            
            x = self.model[0](x)
            x = self.model[1](x)
            x_f = self.model[2](x)
            x = self.model[3](x_f)
            x = self.softmax(x)
            return x, x_f
        else:
            x = self.model(x)
            x = self.softmax(x)
        return x

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        
        x = self.model(x)

        loss = self.loss(x, y)
        
        self.log('train_loss', loss)

        self.accuracy(x, y)
        self.log("train_acc", self.accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        
        x = self.test_transform(x)
        
        x = self.model(x)
        
        loss = self.loss(x, y)
        
        self.log('val_loss', loss)
        self.accuracy(x, y)
        self.log("val_acc", self.accuracy)
