import math
from typing import Optional, Tuple

import numpy as np 

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from torchvision import models
from torchvision import transforms
import torchmetrics

import monai
from monai.networks.nets import AutoEncoder
from monai.networks.blocks import Convolution

import pytorch_lightning as pl

from pl_bolts.transforms.dataset_normalizations import (
    imagenet_normalization
)

class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, query: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, hidden_dim, input_size = query.size(0), query.size(2), value.size(1)

        score = torch.bmm(query, value.transpose(1, 2))
        attn = F.softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        context = torch.bmm(attn, value).view(batch_size, -1, hidden_dim)
        context = torch.sum(context, dim=1)

        return context, attn

class SelfAttention(nn.Module):
    def __init__(self, in_units, out_units):
        super(SelfAttention, self).__init__()

        self.W1 = nn.Linear(in_units, out_units)
        self.V = nn.Linear(out_units, 1)

    def forward(self, query, values):

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(nn.Tanh()(self.W1(query)))

        # min_score = tf.reduce_min(tf.math.top_k(tf.reshape(score, [-1, tf.shape(score)[1]]), k=self.k, sorted=False, name=None)[0], axis=1, keepdims=True)
        # min_score = tf.reshape(min_score, [-1, 1, 1])
        # score_mask = tf.greater_equal(score, min_score)
        # score_mask = tf.cast(score_mask, tf.float32)
        # attention_weights = tf.multiply(tf.exp(score), score_mask) / tf.reduce_sum(tf.multiply(tf.exp(score), score_mask), axis=1, keepdims=True)

        # attention_weights shape == (batch_size, max_length, 1)
        score = nn.Sigmoid()(score)
        sum_score = torch.sum(score, 1, keepdim=True)
        attention_weights = score / sum_score

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector, score

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module
 
    def forward(self, input_seq):
        assert len(input_seq.size()) > 2
 
        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps

        size = input_seq.size()

        batch_size = size[0]
        time_steps = size[1]

        size_reshape = [batch_size*time_steps] + list(size[2:])
        reshaped_input = input_seq.contiguous().view(size_reshape)
 
        output = self.module(reshaped_input)
        
        output_size = output.size()
        output_size = [batch_size, time_steps] + list(output_size[1:])
        output = output.contiguous().view(output_size)

        return output

class GANet3D(pl.LightningModule):
    def __init__(self, args = None, out_features=1, class_weights=None):
        super(GANet3D, self).__init__()        
        
        self.save_hyperparameters()        
        self.args = args

        self.class_weights = class_weights

        if(class_weights is not None):
            class_weights = torch.tensor(class_weights).to(torch.float32)
            
        self.loss = nn.L1Loss()

        effnet = monai.networks.nets.EfficientNetBN("efficientnet-b0", spatial_dims=3, in_channels=1)
        effnet._fc = nn.Identity()

        self.features = TimeDistributed(effnet)
        self.WV = nn.Linear(1280, 128)
        self.Attention = SelfAttention(1280, 64)
        self.Prediction = nn.Linear(128, 1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer

    def forward(self, x):
        x = self.features(x)
        x_v = self.WV(x)
        x_a, x_s = self.Attention(x, x_v)
        x_p = self.Prediction(x_a)
        x_v_p = self.Prediction(x_v)

        return x_p, x_a, x_s, x_v, x_v_p
        

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        x = self.features(x)
        x_v = self.WV(x)
        x, x_s = self.Attention(x, x_v)
        x = self.Prediction(x)

        loss = self.loss(x, y)
        
        self.log('train_loss', loss)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        
        x = self.features(x)
        x_v = self.WV(x)
        x, x_s = self.Attention(x, x_v)
        x = self.Prediction(x)

        loss = self.loss(x, y)
        
        self.log('val_loss', loss)

class GANet3DV0(pl.LightningModule):
    def __init__(self, args = None, out_features=1, class_weights=None):
        super(GANet3DV0, self).__init__()        
        
        self.save_hyperparameters()        
        self.args = args

        self.class_weights = class_weights

        if(class_weights is not None):
            class_weights = torch.tensor(class_weights).to(torch.float32)
            
        self.loss = nn.MSELoss()

        effnet = monai.networks.nets.EfficientNetBN("efficientnet-b0", spatial_dims=3, in_channels=1)
        effnet._fc = nn.Identity()

        self.features = TimeDistributed(effnet)
        self.WV = nn.Linear(1280, 512)
        self.Attention = SelfAttention(1280, 128)
        self.Prediction = nn.Linear(512, 1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer

    def forward(self, x):
        x = self.features(x)
        x_v = self.WV(x)
        x_a, x_s = self.Attention(x, x_v)
        x_p = self.Prediction(x_a)
        x_v_p = self.Prediction(x_v)

        return x_p, x_a, x_s, x_v, x_v_p
        

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        x = self.features(x)
        x_v = self.WV(x)
        x, x_s = self.Attention(x, x_v)
        x = self.Prediction(x)

        loss = self.loss(x, y)
        
        self.log('train_loss', loss)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        
        x = self.features(x)
        x_v = self.WV(x)
        x, x_s = self.Attention(x, x_v)
        x = self.Prediction(x)

        loss = self.loss(x, y)
        
        self.log('val_loss', loss)

class GANet3DB3(pl.LightningModule):
    def __init__(self, args = None, out_features=1, class_weights=None):
        super(GANet3DB3, self).__init__()        
        
        self.save_hyperparameters()        
        self.args = args

        self.class_weights = class_weights

        if(class_weights is not None):
            class_weights = torch.tensor(class_weights).to(torch.float32)
            
        self.loss = nn.L1Loss()

        effnet = monai.networks.nets.EfficientNetBN("efficientnet-b3", spatial_dims=3, in_channels=1)
        effnet._fc = nn.Identity()

        self.features = TimeDistributed(effnet)
        self.WV = nn.Linear(1536, 128)
        self.Attention = SelfAttention(1536, 64)
        self.Prediction = nn.Linear(128, 1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer

    def forward(self, x):
        x = self.features(x)
        x_v = self.WV(x)
        x_a, x_s = self.Attention(x, x_v)
        x_p = self.Prediction(x_a)
        x_v_p = self.Prediction(x_v)

        return x_p, x_a, x_s, x_v, x_v_p
        

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        x, x_a, x_s, x_v, x_v_p = self(x)        

        loss = self.loss(x, y)
        
        self.log('train_loss', loss)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        
        x, x_a, x_s, x_v, x_v_p = self(x)        

        loss = self.loss(x, y)
        
        self.log('val_loss', loss)

class GANet3DDot(pl.LightningModule):
    def __init__(self, args = None, out_features=1, class_weights=None):
        super(GANet3DDot, self).__init__()        
        
        self.save_hyperparameters()        
        self.args = args

        self.class_weights = class_weights

        if(class_weights is not None):
            class_weights = torch.tensor(class_weights).to(torch.float32)
            
        self.loss = nn.L1Loss()

        effnet = monai.networks.nets.EfficientNetBN("efficientnet-b3", spatial_dims=3, in_channels=1)
        effnet._fc = nn.Identity()

        self.F = TimeDistributed(effnet)        
        self.V = nn.Sequential(nn.Linear(1536, 256), nn.ReLU(), nn.Linear(256, 1536))
        self.A = DotProductAttention()
        self.P = nn.Linear(in_features=1536, out_features=out_features)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer

    def forward(self, x):
        x = self.F(x)
        x_v = self.V(x)
        x_a, x_s = self.A(x, x_v)
        x_p = self.P(x_a)
        x_v_p = self.P(x_v)

        return x_p, x_a, x_s, x_v, x_v_p
        

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        x, x_a, x_s, x_v, x_v_p = self(x)        

        loss = self.loss(x, y)
        
        self.log('train_loss', loss)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        
        x, x_a, x_s, x_v, x_v_p = self(x)        

        loss = self.loss(x, y)
        
        self.log('val_loss', loss)