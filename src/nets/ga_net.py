import os
import time
import math
from math import ceil

import pandas as pd
import numpy as np
import nrrd

import torch
import torch.optim as optim
from torch import nn
from torchvision import models

import pytorch_lightning as pl

class Identity(nn.Module):    
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

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

        return context_vector, score, attention_weights

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


class GA_Net(nn.Module):
    def __init__(self):
        super(GA_Net, self).__init__()

        cnn = models.efficientnet_b0(pretrained=True)
        cnn.classifier = Identity()

        self.TimeDistributed = TimeDistributed(cnn)
        self.WV = nn.Linear(1280, 128)
        self.Attention = SelfAttention(1280, 64)
        self.Prediction = nn.Linear(128, 1)
 
    def forward(self, x):

        x = self.TimeDistributed(x)

        x_v = self.WV(x)

        x_a, x_s, w_a = self.Attention(x, x_v)

        x_v_p = self.Prediction(x_v)
        x = self.Prediction(x_a)

        return x, x_a, x_v, x_s, x_v_p, w_a

class GA_Net_features(nn.Module):
    def __init__(self, cnn_pretrained):
        super(GA_Net_features, self).__init__()

        cnn = cnn_pretrained

        self.TimeDistributed = TimeDistributed(cnn)

    def forward(self, x):
        
        x = self.TimeDistributed(x)

        return x
       
class GA_Net_attn_output(nn.Module):
    def __init__(self):
        super(GA_Net_attn_output, self).__init__()

        self.WV = nn.Linear(1280, 128)
        self.Attention = SelfAttention(1280, 64)
        self.Prediction = nn.Linear(128, 1)
 
    def forward(self, x):
        
        x_v = self.WV(x)
        x_a, w_a, score = self.Attention(x, x_v)

        x = self.Prediction(x_a)

        return x, w_a, score


class NormLayer(nn.Module):
    def __init__(self):
        super(NormLayer, self).__init__()
        
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        
    def forward(self, x):
        x = x.repeat(1, 1, 1, 3)
        return torch.div(torch.sub(torch.div(x, 255.0), self.mean), self.std).permute((0, 3, 1, 2))

class GA_Net_features_norm(nn.Module):
    def __init__(self, cnn):
        super(GA_Net_features_norm, self).__init__()
        self.norm = NormLayer()
        self.cnn = cnn
    
    def forward(self, x):
        x = self.norm(x)
        x = self.cnn(x)
        return x

class GA_NetL(pl.LightningModule):
    def __init__(self, lr = 1e-4):
        super(GA_NetL, self).__init__()
        
        self.save_hyperparameters() 
        
        self.model = GA_Net()
        
        self.loss_fn = nn.L1Loss(reduction='none')

        self.relu = nn.ReLU()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        
        # x, pred_by_frame, w_a, score = self(x)
        x, x_a, x_v, x_s, x_v_p, w_a = self(x)


        loss = self.compute_loss(x, y, x_s, x_v_p, w_a)        
        
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        
        x, x_a, x_v, x_s, x_v_p, w_a = self(x)

        loss = self.compute_loss(x, y, x_s, x_v_p, w_a)
        
        self.log('val_loss', loss, sync_dist=True)

    def compute_loss(self, x, y, x_s, x_v_p, w_a):
        # x_v_p = pred_by_frame
        # x_s = score
        # w_a = w_a

        raw_score_loss = self.relu(-torch.sum(torch.sum(x_s.squeeze(2), axis=1), axis=0) + 30) / 10
        w_a_detached = w_a.detach().clone()
        w_mse = 0.5 * self.WeightedMSE(x_v_p, w_a_detached, y)
        loss = torch.mean(self.loss_fn(x, y) + w_mse) + torch.mean(raw_score_loss)

        return loss

    def WeightedMSE(self, x, w, y):
        n = x.size(1)
        target = y.unsqueeze(1).repeat_interleave(n,dim=1)
        squared_errors = torch.square(x - target)
        weighted_mse = torch.sqrt(torch.sum(w * squared_errors, dim=1))
        return weighted_mse