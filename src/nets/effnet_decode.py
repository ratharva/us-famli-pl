import math
import numpy as np 

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from torchvision import models
from torchvision import transforms

from monai.transforms import (RandGaussianNoise, Compose)
from monai.networks.nets import AutoEncoder
from monai.networks.blocks import Convolution

import pytorch_lightning as pl

class Feedforward(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Feedforward, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity
        self.tanh = nn.Tanh()
        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        # Linear function
        out = self.fc1(x)
        # Non-linearity
        out = self.tanh(out)
        # Linear function (readout)
        out = self.fc2(out)
        return out

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

class SaltAndPepper(nn.Module):    
    def __init__(self, prob=0.2):
        super(SaltAndPepper, self).__init__()
        self.prob = 0.2
    def forward(self, x):
        noise_tensor = torch.rand(x.shape).cuda()
        salt = torch.max(x)
        pepper = torch.min(x)
        x[noise_tensor < self.prob/2] = salt
        x[noise_tensor > 1-self.prob/2] = pepper
        return x

class GaussianNoise(nn.Module):    
    def __init__(self, mean=0.0, std=0.1):
        super(GaussianNoise, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    def forward(self, x):
        return x + torch.normal(mean=self.mean, std=self.std, size=x.size(), device=x.device)

class EffnetDecodeSimilarity(pl.LightningModule):
    def __init__(self, args = None):
        super(EffnetDecodeSimilarity, self).__init__()
        
        self.save_hyperparameters() 

        self.args = args
        self.encoder = nn.Sequential(models.efficientnet_b0(pretrained=True).features)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten(start_dim=1)
        
        self.output_dim = 128
        if hasattr(args, 'output_dim'):
            self.output_dim = args.output_dim

        self.projection = ProjectionHead(output_dim=self.output_dim)        
        self.decoder = nn.Sequential(
            Convolution(dimensions=2, in_channels=1280, out_channels=256, spatial_dims=2, strides=2, is_transposed=True),
            Convolution(dimensions=2, in_channels=256, out_channels=128, spatial_dims=2, strides=2, is_transposed=True),
            Convolution(dimensions=2, in_channels=128, out_channels=64, spatial_dims=2, strides=2, is_transposed=True),
            Convolution(dimensions=2, in_channels=64, out_channels=32, spatial_dims=2, strides=2, is_transposed=True),
            Convolution(dimensions=2, in_channels=32, out_channels=3, spatial_dims=2, strides=2, is_transposed=True, conv_only=True)
            )
        
        self.rec_loss = nn.MSELoss()
        self.sim_loss = nn.CosineSimilarity()
        
        self.intensity_transform = torch.nn.Sequential(
            transforms.RandomApply([transforms.ColorJitter(brightness=(0.3, 1.0), contrast=(0.3, 1.0), saturation=(0.3, 1.0), hue=(0, 0.4))], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.5),
            transforms.RandomApply([SaltAndPepper(prob=0.1)], p=0.5)
        )
        
        
        self.rotation_transform = torch.nn.Sequential(
            transforms.RandomApply([transforms.RandomResizedCrop(256, scale=(0.2, 1.))], p=0.5),
            transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=(0,180))], p=0.5)
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer

    def forward(self, x):        
        z = self.encoder(x)
        z = self.avg(z)
        z = self.flat(z)
        z = self.projection(z)
        return z

    def training_step(self, train_batch, batch_idx):
        x = train_batch
        batch_size = x.size(0)

        x = x.repeat(2,1,1,1).contiguous()
        x = self.rotation_transform(x)
        z = self.encoder(self.intensity_transform(x))
        x_hat = self.decoder(z)
        
        z = self.avg(z)
        z = self.flat(z)
        z = self.projection(z)
        
        z_0, z_1 = torch.split(z, batch_size)

        recon_loss = self.rec_loss(x_hat, x)
        sim_loss = torch.mean(1.0 - torch.abs(self.sim_loss(z_0, z_1)))

        idx = torch.randperm(batch_size)
        z_1 = z_1[idx]
        diff_loss = torch.mean(torch.abs(self.sim_loss(z_0, z_1)))

        loss = recon_loss + sim_loss + diff_loss
        
        self.log('recon_loss', recon_loss)
        self.log('sim_loss', sim_loss)
        self.log('diff_loss', diff_loss)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch
        batch_size = x.size(0)

        x = x.repeat(2,1,1,1).contiguous()
        x = self.rotation_transform(x)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        
        z = self.avg(z)
        z = self.flat(z)
        z = self.projection(z)
        
        z_0, z_1 = torch.split(z, batch_size)

        recon_loss = self.rec_loss(x_hat, x)
        sim_loss = torch.mean(1.0 - torch.abs(self.sim_loss(z_0, z_1)))
        
        idx = torch.randperm(batch_size)
        z_1 = z_1[idx]
        diff_loss = torch.mean(torch.abs(self.sim_loss(z_0, z_1)))

        loss = recon_loss + sim_loss + diff_loss
        
        self.log('val_loss', loss)
        return loss

class EffnetDecode(pl.LightningModule):
    def __init__(self, args = None, train_transform=None, valid_transform=None):
        super(EffnetDecode, self).__init__()

        self.args = args
        # self.encode = nn.Sequential(models.efficientnet_b0(pretrained=True).features)
        model = AutoEncoder(
            spatial_dims=2,
            in_channels=3,
            out_channels=3,
            channels=(128, 256, 512, 1024, 2048),
            strides=(2, 2, 2, 2, 2),
        )

        self.encode = model.encode
        
        self.projection = nn.Sequential(nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(start_dim=1),
            ProjectionHead(input_dim=2048, output_dim=args.output_dim)
        )
        
        self.decode = model.decode

        self.noise_transform = torch.nn.Sequential(
            GaussianNoise(),
            transforms.GaussianBlur(5, sigma=(0.1, 4.0)),
        )  

        self.loss = nn.MSELoss()
        self.sim_loss = nn.CosineSimilarity()

        self.train_transform = train_transform
        self.valid_transform = valid_transform

        self.k = args.k

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer

    def forward(self, x):        
        z_hat = self.encode(x)
        x_hat = self.decode(z_hat)
        z = self.projection(z_hat)        
        return x_hat, z

    def compute_loss(self, x_0, x_1, mode):
        batch_size = x_0.size(0)
        
        x = torch.cat([x_0, x_1], dim=0)

        x_hat, z = self(self.noise_transform(x))

        z_0, z_1 = torch.split(z, batch_size)
        
        loss_recon = self.loss(x_hat, x)
        
        loss_proj = torch.mean(1.0 - self.sim_loss(z_0, z_1))

        #randomize the batch        
        r = torch.randperm(batch_size)
        z_0_r = z_0[r]
        # Compute the cosine similarity between the shuffled batch z_0_r and z_1. 
        # 1. Get the most different ones. topk(1 - sim_loss)
        # 2. We want to further maximize their distance
        # 2.1 If the value is close to 1 then the log is 0
        # 2.2 If the value is close to 0 (similar) then the log is high
        # If all images in the batch are mapping to the same direction, this term should be really high.
        # If there are similar images in the batch and they are put together after randomization, they could map to a similar location and that is OK!
        # if mode == 'val':
        #     k = batch_size
        # else:
        #     k = min(batch_size, self.k)        
        # loss_proj_c = torch.mean(-torch.log(torch.topk(1.0 - self.sim_loss(z_0_r, z_1), k=k).values + 1e-8))
        k = min(batch_size, self.k)
        loss_proj_c = torch.mean(-torch.log(torch.topk(1.0 - self.sim_loss(z_0_r, z_1), k=k).values + 1e-8))

        loss =  loss_recon + loss_proj + loss_proj_c

        self.log(mode + '_loss_recon', loss_recon)
        self.log(mode + '_loss_proj', loss_proj)
        self.log(mode + '_loss_proj_c', loss_proj_c)
        self.log(mode + '_loss', loss)
        
        return loss

    def training_step(self, train_batch, batch_idx):        
        
        x_0, x_1 = self.train_transform(train_batch)
        return self.compute_loss(x_0, x_1, 'train')

    def validation_step(self, val_batch, batch_idx):
        
        x_0, x_1 = self.valid_transform(val_batch)
        self.compute_loss(x_0, x_1, 'val')

class MonaiAutoEncoder(pl.LightningModule):
    def __init__(self, output_dim=128, lr=1e-3, k=128):
        super(MonaiAutoEncoder, self).__init__()

        self.save_hyperparameters() 
        
        model = AutoEncoder(
            spatial_dims=2,
            in_channels=3,
            out_channels=3,
            channels=(128, 256, 512, 1024, 2048), #8*8*2048
            strides=(2, 2, 2, 2, 2),
        )

        self.encode = model.encode
        
        self.projection = nn.Sequential(nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(start_dim=1),
            ProjectionHead(input_dim=2048, output_dim=output_dim)
        )
        
        self.decode = model.decode
        self.sigmoid = nn.Sigmoid()

        self.noise_transform = torch.nn.Sequential(
            GaussianNoise()
        )

        # self.loss = nn.MSELoss()
        self.loss = nn.BCELoss()
        self.sim_loss = nn.CosineSimilarity()
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def forward(self, x):        
        z_hat = self.encode(x)
        x_hat = self.decode(z_hat)
        x_hat = self.sigmoid(x_hat)

        z = self.projection(z_hat)

        return x_hat, z

    def compute_loss(self, x_0, x_1, mode):
        batch_size = x_0.size(0)
        
        x = torch.cat([x_0, x_1], dim=0)

        x_hat, z = self(self.noise_transform(x))

        loss_recon = self.loss(x_hat, x)

        z_0, z_1 = torch.split(z, batch_size)
        
        loss_proj = torch.sum(torch.square(1.0 - self.sim_loss(z_0, z_1)))

        #randomize the batch        
        r = torch.randperm(batch_size)
        z_0_r = z_0[r]
        # Compute the cosine similarity between the shuffled batch z_0_r and z_1. 
        # 1. Get the most different ones. topk(1 - loss)
        # 2. We want to further maximize their distance
        # 2.1 We get the indices of those elements
        # 2.2 We sum those elements, we want that their cosine similarit is  0
        # If all images in the batch are mapping to the same direction, this term should be really high.
        # If there are similar images in the batch and they are put together after randomization, they could map to a similar location and that is OK!

        if mode == 'val':
            k = batch_size
        else:
            k = min(batch_size, self.hparams.k)        

        loss_proj_c = self.sim_loss(z_0_r, z_1)
        top_k_i = torch.topk(1.0 - loss_proj_c, k=k).indices
        loss_proj_c = torch.sum(torch.square(loss_proj_c[top_k_i]))
        
        loss =  loss_recon + loss_proj + loss_proj_c
        
        self.log(mode + '_loss_recon', loss_recon)
        self.log(mode + '_loss_proj', loss_proj)
        self.log(mode + '_loss_proj_c', loss_proj_c)
        self.log(mode + '_loss', loss)
        
        return loss

    def training_step(self, train_batch, batch_idx):
        x_0, x_1 = train_batch
        return self.compute_loss(x_0, x_1, 'train')

    def validation_step(self, val_batch, batch_idx):
        x_0, x_1 = val_batch
        self.compute_loss(x_0, x_1, 'val')
    
