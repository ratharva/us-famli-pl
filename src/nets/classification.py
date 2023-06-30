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
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        NN = getattr(torchvision.models, self.hparams.base_encoder)
        self.convnet = NN(num_classes=self.hparams.num_classes)

        self.extract_features = False

        if hasattr(self.hparams, 'model_feat') and self.hparams.model_feat is not None:
            classifier = self.convnet.classifier
            self.convnet.classifier = nn.Identity()
            self.convnet.load_state_dict(torch.load(args.model_feat))
            # for param in self.convnet.parameters():
            #     param.requires_grad = False
            self.convnet.classifier = classifier


        class_weights = None
        if(hasattr(self.hparams, 'class_weights')):
            class_weights = torch.tensor(self.hparams.class_weights).to(torch.float32)

        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)

        self.softmax = nn.Softmax(dim=1)

        self.noise_transform = torch.nn.Sequential(
            GaussianNoise(std=0.05)
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        
        x = self(self.noise_transform(x))

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

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        
        x = self(x)
        
        loss = self.loss(x, y)
        
        self.log('test_loss', loss, sync_dist=True)
        self.accuracy(x, y)
        self.log("test_acc", self.accuracy)

    def forward(self, x):        
        if self.extract_features:
            x_f = self.convnet.features(x)
            x_f = self.convnet.avgpool(x_f)            
            x = torch.flatten(x_f, 1)            
            return self.convnet.classifier(x), x_f
        else:
            return self.convnet(x)
