import os
import time
from collections import Counter
import re
import math
from math import ceil
from collections import OrderedDict
import argparse

import pandas as pd
import numpy as np
import itk
import nrrd

import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Sampler
from torchvision import transforms as T
import torchvision.models as models
#from torchvision.transforms import ToTensor
from torchvision.transforms import Pad
from torchvision.transforms import Resize, ToPILImage

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data.distributed

from torch.distributions.utils import probs_to_logits
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed import ReduceOp

from monai.data import ITKReader, PILReader
from monai.transforms import (
    ToTensor, LoadImage, Lambda, AddChannel, RepeatChannel, ScaleIntensityRange, RandSpatialCrop, RandRotate,
    Resized, Compose, BorderPad, NormalizeIntensity
)
from monai.config import print_config

from sklearn.metrics import roc_auc_score


dataset_dir = "/mnt/raid/C1_ML_Analysis/"
#dataset_dir = "/mnt/famli_netapp_shared/C1_ML_Analysis/"
output_mount = "/mnt/famli_netapp_shared/C1_ML_Analysis/train_out/"

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model[0].state_dict(), self.path + '_features.pt')
        # torch.save(model[1].state_dict(), self.path + '_lstm.pt')
        # torch.save(model[2].state_dict(), self.path + '_output.pt')
        torch.save(model.state_dict(), self.path + '.pt')
        self.val_loss_min = val_loss

class TransformFrames:
    def __init__(self, training=True):
        if training:
            self.transform = Compose([
    #        Lambda(func=random_choice), 
            AddChannel(),
            BorderPad(spatial_border=[-1, 32, 32]),
            RandSpatialCrop(roi_size=[-1, 256, 256], random_size=False),
            #RandRotate(prob=1.0, range_x=1.5708), #random up to 45 degrees
            #Flip(spatial_axis=3),
            #ScaleIntensityRange(0.0, 255.0, 0, 1.0),
            #RepeatChannel(repeats=3),
            ToTensor(),
            #Lambda(lambda x: torch.transpose(x, 0, 1)),
            #NormalizeIntensity(subtrahend=np.array([0.485, 0.456, 0.406]), divisor=np.array([0.229, 0.224, 0.225])),
            #ToTensor(),
            #normalize 
            ])
        else:
            self.transform = Compose([
        #    Lambda(func=random_choice), 
            AddChannel(),
            #BorderPad(spatial_border=[-1, 24, 24]),
            #RandSpatialCrop(roi_size=[-1, 256, 256], random_size=False),
            #Flip(spatial_axis=3),
            #ScaleIntensityRange(0.0, 255.0, 0, 1.0),
            #RepeatChannel(repeats=3),
            ToTensor(),
            #Lambda(lambda x: torch.transpose(x, 0, 1)),
            #NormalizeIntensity(subtrahend=np.array([0.485, 0.456, 0.406]), divisor=np.array([0.229, 0.224, 0.225])),
            #ToTensor(),
            #normalize 
            ])

    def __call__(self, x):
        x = self.transform(x)
        x = torch.transpose(x, 0, 1)
        return x


class ITKImageDataset(Dataset):
    def __init__(self, df, mount_point,
                transform=None, num_sample_frames=None, device='cpu'):
        self.df = df
        self.mount = mount_point
        self.targets = self.df.loc[:, 'label'].to_numpy(dtype=np.int8)
        self.labels_str = self.df.loc[:, 'fetal_presentation_str']
        self.transform = transform
        self.num_sample_frames = num_sample_frames
        self.device = device

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        img_path = self.df.iloc[idx, 0]
        label_str = self.df.loc[idx, 'fetal_presentation_str']
        if label_str == 'Cephalic':
            label = np.array([0,0,np.nan], dtype=np.float)
        elif label_str == 'Transverse':
            label = np.array([1,np.nan,0], dtype=np.float)
        elif label_str == 'Breech':
            label = np.array([np.nan,1,1], dtype=np.float)
        else:
            label = np.array([np.nan,np.nan,np.nan], dtype=np.float)
        try:
            img, header = nrrd.read(os.path.join(self.mount, img_path), index_order='C')
            img = torch.tensor(img, dtype=torch.float)
            if (len(img.shape) == 4):
                img = img[:,:,:,0]
            assert(len(img.shape) == 3)
            assert(img.shape[1] == 256)
            assert(img.shape[2] == 256)
        except:
            print("Error reading cine: " + img_path)
            img = torch.zeros(200, 256, 256)
        if self.num_sample_frames:
            per_chunk_n = int(self.num_sample_frames / 5)
            if (img.size(0) < 25):
                print(img.size())
                print(img_path)
            idx_chunks = torch.tensor_split(torch.arange(img.size(0)), 5)
            idx = torch.cat([torch.randint(low=chunk[0].item(), high=chunk[-1].item(), size=(per_chunk_n,)) for chunk in idx_chunks], dim=0)
            sorted_idx, indices = torch.sort(idx)
            img = img[sorted_idx]
        if self.transform:
            img = self.transform(img)
        return img, label

class ITKImageDatasetByID(Dataset):
    def __init__(self, df, mount_point,
                transform=None, num_sample_frames=None, num_sample_sweeps=None):
        self.df = df
        self.mount = mount_point
        self.transform = transform
        label_map = self.df[["study_id", "label"]].drop_duplicates().reset_index(drop=True)
        self.study_ids = label_map.study_id.values
        self.targets = label_map.label.to_numpy()
        self.num_sample_frames = num_sample_frames
        self.num_sample_sweeps = num_sample_sweeps

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        relevant_rows = self.df.loc[self.df.study_id == study_id,:]
        if self.num_sample_sweeps:
            relevant_rows = relevant_rows.sample(n=self.num_sample_sweeps).reset_index(drop=True)
        else:
            #shuffle rows
            relevant_rows = relevant_rows.sample(frac=1).reset_index(drop=True)
        label = relevant_rows.loc[0, 'label']
        label_str = relevant_rows.loc[0, 'fetal_presentation_str']
        if label_str == 'Cephalic':
            label = np.array([0,0,np.nan], dtype=np.float)
        elif label_str == 'Transverse':
            label = np.array([1,np.nan,0], dtype=np.float)
        elif label_str == 'Breech':
            label = np.array([np.nan,1,1], dtype=np.float)
        else:
            label = np.array([np.nan,np.nan,np.nan], dtype=np.float)
        seq_im_array = []
        tag_list = []
        for idx, row in relevant_rows.iterrows():
            img_path = row['file_path']
            tag = row['tag']
            try:
                img, header = nrrd.read(os.path.join(self.mount, img_path), index_order='C')
                img = torch.tensor(img.astype(np.float32))
                if (len(img.shape) == 4):
                    img = img[:,:,:,0]
                assert(len(img.shape) == 3)
                assert(img.shape[1] == 256)
                assert(img.shape[2] == 256)
            except:
                print("Error reading cine: " + img_path)
                img = torch.zeros(200, 256, 256)
            if self.num_sample_frames:
                per_chunk_n = int(self.num_sample_frames / 5)
                idx_chunks = torch.tensor_split(torch.arange(img.size(0)), 5)
                idx = torch.cat([torch.randint(low=chunk[0].item(), high=chunk[-1].item(), size=(per_chunk_n,)) for chunk in idx_chunks], dim=0)
                sorted_idx, indices = torch.sort(idx)
                img = img[sorted_idx]
            if self.transform:
                img = self.transform(img)
            seq_im_array.append(img)
            tag_list.append(tag)
        im_array = torch.stack(seq_im_array, dim=0)
        return im_array, label


class ITKImageDatasetByID3SweepsChannels(Dataset):
    def __init__(self, df, mount_point,
                transform=None, num_sample_frames=None):
        self.df = df
        self.mount = mount_point
        self.transform = transform
        label_map = self.df[["study_id", "label", 'fetal_presentation_str']].drop_duplicates().reset_index(drop=True)
        self.study_ids = label_map.study_id.values
        self.targets = label_map.label.to_numpy()
        self.labels_str = label_map.loc[:, 'fetal_presentation_str']
        self.num_sample_frames = num_sample_frames
        self.cc_tag_val_dict = {'M':0, 
                        'AssumedM':0,
                        'ML':0, 'MR':0,
                        'L0':-1, 'L1':-1, 'L15':0, 'L2':-1, 'L45':0,
                        'R0':1, 'R1':1, 'R15':0, 'R2':1, 'R45':0,
                        'NL':-1, 'NM':0, 'NR':1,
                      'ASSBS_M':0,
                      'ASSBS_L0':-1, 'ASSBS_L1':-1, 
                      'ASSBS_R0':1,'ASSBS_R1':1}
        self.df[['cc_tag_val']] = self.df.tag.map(self.cc_tag_val_dict)

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        relevant_rows = self.df.loc[self.df.study_id == study_id,:]
        relevant_rows = relevant_rows.groupby('cc_tag_val').sample(n=1).reset_index(drop=True)
        #print(relevant_rows.tag)
        label = relevant_rows.loc[0, 'label']
        label_str = relevant_rows.loc[0, 'fetal_presentation_str']
        if label_str == 'Cephalic':
            label = np.array([0,0,np.nan], dtype=np.float)
        elif label_str == 'Transverse':
            label = np.array([1,np.nan,0], dtype=np.float)
        elif label_str == 'Breech':
            label = np.array([np.nan,1,1], dtype=np.float)
        else:
            label = np.array([np.nan,np.nan,np.nan], dtype=np.float)
        im_array = torch.zeros((self.num_sample_frames, 3, 256, 256)) #insert sweep to the channel dimension
        for idx, row in relevant_rows.iterrows():
            img_path = row['file_path']
            tag = row['tag']
            cc_tag_val = row['cc_tag_val']
            try:
                img, header = nrrd.read(os.path.join(self.mount, img_path), index_order='C')
                img = torch.tensor(img.astype(np.float32))
                if (len(img.shape) == 4):
                    img = img[:,:,:,0]
                assert(len(img.shape) == 3)
                assert(img.shape[1] == 256)
                assert(img.shape[2] == 256)
            except:
                print("Error reading cine: " + img_path)
                img = torch.zeros(200, 256, 256)
            if self.num_sample_frames:
                per_chunk_n = int(self.num_sample_frames / 5)
                idx_chunks = torch.tensor_split(torch.arange(img.size(0)), 5)
                idx = torch.cat([torch.randint(low=chunk[0].item(), high=chunk[-1].item(), size=(per_chunk_n,)) for chunk in idx_chunks], dim=0)
                sorted_idx, indices = torch.sort(idx)
                img = img[sorted_idx]
            if self.transform:
                img = self.transform(img)
            im_array[:,cc_tag_val+1,:,:] = img.squeeze(1) #insert sweep to the channel dimension
        return im_array, label

class ITKImageDatasetByID3Sweeps(Dataset):
    def __init__(self, df, mount_point,
                transform=None, num_sample_frames=None):
        self.df = df
        self.mount = mount_point
        self.transform = transform
        label_map = self.df[["study_id", "label", 'fetal_presentation_str']].drop_duplicates().reset_index(drop=True)
        self.study_ids = label_map.study_id.values
        self.targets = label_map.label.to_numpy()
        self.labels_str = label_map.loc[:, 'fetal_presentation_str']
        self.num_sample_frames = num_sample_frames
        self.cc_tag_val_dict = {'M':0, 
                        'AssumedM':0,
                        'ML':0, 'MR':0,
                        'L0':-1, 'L1':-1, 'L15':0, 'L2':-1, 'L45':0,
                        'R0':1, 'R1':1, 'R15':0, 'R2':1, 'R45':0,
                        'NL':-1, 'NM':0, 'NR':1,
                      'ASSBS_M':0,
                      'ASSBS_L0':-1, 'ASSBS_L1':-1, 
                      'ASSBS_R0':1,'ASSBS_R1':1}
        self.df[['cc_tag_val']] = self.df.tag.map(self.cc_tag_val_dict)

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        relevant_rows = self.df.loc[self.df.study_id == study_id,:]
        relevant_rows = relevant_rows.groupby('cc_tag_val').sample(n=1).reset_index(drop=True)
        #print(relevant_rows.tag)
        label = relevant_rows.loc[0, 'label']
        label_str = relevant_rows.loc[0, 'fetal_presentation_str']
        if label_str == 'Cephalic':
            label = np.array([0,0,np.nan], dtype=np.float)
        elif label_str == 'Transverse':
            label = np.array([1,np.nan,0], dtype=np.float)
        elif label_str == 'Breech':
            label = np.array([np.nan,1,1], dtype=np.float)
        else:
            label = np.array([np.nan,np.nan,np.nan], dtype=np.float)
        im_array = torch.zeros((3, self.num_sample_frames, 1, 256, 256)) #insert sweep to the channel dimension
        for idx, row in relevant_rows.iterrows():
            img_path = row['file_path']
            tag = row['tag']
            cc_tag_val = row['cc_tag_val']
            try:
                img, header = nrrd.read(os.path.join(self.mount, img_path), index_order='C')
                img = torch.tensor(img.astype(np.float32))
                if (len(img.shape) == 4):
                    img = img[:,:,:,0]
                assert(len(img.shape) == 3)
                assert(img.shape[1] == 256)
                assert(img.shape[2] == 256)
            except:
                print("Error reading cine: " + img_path)
                img = torch.zeros(200, 256, 256)
            if self.num_sample_frames:
                per_chunk_n = int(self.num_sample_frames / 5)
                idx_chunks = torch.tensor_split(torch.arange(img.size(0)), 5)
                idx = torch.cat([torch.randint(low=chunk[0].item(), high=chunk[-1].item(), size=(per_chunk_n,)) for chunk in idx_chunks], dim=0)
                sorted_idx, indices = torch.sort(idx)
                img = img[sorted_idx]
            if self.transform:
                img = self.transform(img)
            im_array[cc_tag_val+1,:,:,:,:] = img #insert sweep to the channel dimension
        return im_array, label

class ITKImageDatasetByIDTwoDim(Dataset):
    def __init__(self, df, mount_point,
                transform=None, num_sample_frames=None, num_sample_sweeps=None):
        self.df = df
        self.mount = mount_point
        self.transform = transform
        label_map = self.df[["study_id", "label", 'fetal_presentation_str']].drop_duplicates().reset_index(drop=True)
        self.study_ids = label_map.study_id.values
        self.targets = label_map.label.to_numpy()
        self.labels_str = label_map.loc[:, 'fetal_presentation_str']
        self.num_sample_frames = num_sample_frames
        self.num_sample_sweeps = num_sample_sweeps

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        relevant_rows = self.df.loc[self.df.study_id == study_id,:]
        if self.num_sample_sweeps:
            if (self.num_sample_sweeps > len(relevant_rows)):
                relevant_rows = relevant_rows.sample(n=self.num_sample_sweeps, replace=True).reset_index(drop=True)
            else:
                relevant_rows = relevant_rows.sample(n=self.num_sample_sweeps, replace=False).reset_index(drop=True)
        else:
            #shuffle rows
            relevant_rows = relevant_rows.sample(frac=1).reset_index(drop=True)
        #print(relevant_rows)
        label = relevant_rows.loc[0, 'label']
        label_str = relevant_rows.loc[0, 'fetal_presentation_str']
        if label_str == 'Cephalic':
            label = np.array([0,0,np.nan], dtype=np.float)
        elif label_str == 'Transverse':
            label = np.array([1,np.nan,0], dtype=np.float)
        elif label_str == 'Breech':
            label = np.array([np.nan,1,1], dtype=np.float)
        else:
            label = np.array([np.nan,np.nan,np.nan], dtype=np.float)
        im_array = torch.zeros((relevant_rows.shape[0], self.num_sample_frames, 1, 256, 256)) #insert sweep to the channel dimension
        sweep_i = 0
        tag_list = []
        for idx, row in relevant_rows.iterrows():
            img_path = row['file_path']
            tag = row['tag']
            try:
                img, header = nrrd.read(os.path.join(self.mount, img_path), index_order='C')
                img = torch.tensor(img.astype(np.float32))
                if (len(img.shape) == 4):
                    img = img[:,:,:,0]
                assert(len(img.shape) == 3)
                assert(img.shape[1] == 256)
                assert(img.shape[2] == 256)
            except:
                print("Error reading cine: " + img_path)
                img = torch.zeros(200, 256, 256)
            if self.num_sample_frames:
                per_chunk_n = int(self.num_sample_frames / 5)
                idx_chunks = torch.tensor_split(torch.arange(img.size(0)), 5)
                idx = torch.cat([torch.randint(low=chunk[0].item(), high=chunk[-1].item(), size=(per_chunk_n,)) for chunk in idx_chunks], dim=0)
                sorted_idx, indices = torch.sort(idx)
                img = img[sorted_idx]
            if self.transform:
                img = self.transform(img)
            im_array[sweep_i,:,:,:,:] = img #insert sweep to the first dimension
            tag_list.append(tag)
            sweep_i += 1
        #print(','.join(tag_list))
        return im_array, ','.join(tag_list), label


class DistributedWeightedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, replacement=True, shuffle=True):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement
        self.shuffle = shuffle

    def calculate_weights(self, labels_str):
        labels_str = np.array(labels_str)
        weights = np.ones(len(labels_str))
        for i in range(len(weights)):
            label = labels_str[i]
            if label == "Breech":
                weights[i] = 3
            elif label == "Transverse":
                weights[i] = 8

        samples_weight = torch.tensor(weights)
        return samples_weight

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # get targets (you can alternatively pass them in __init__, if this op is expensive)
        labels_str = self.dataset.labels_str
        # select only the wanted targets for this subsample
        labels_str = labels_str[indices]
        assert len(labels_str) == self.num_samples
        # randomly sample this subset, producing balanced classes
        weights = self.calculate_weights(labels_str)
        subsample_balanced_indicies = torch.multinomial(weights, self.num_samples, self.replacement)
        # now map these target indicies back to the original dataset index...
        dataset_indices = torch.tensor(indices)[subsample_balanced_indicies]

        return iter(dataset_indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

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


class BayesBeliefsUpdate(nn.Module):
    def __init__(self, prior_prob=0.5):
        super(BayesBeliefsUpdate, self).__init__()
        self.prior_noncephalic = prior_prob

    def forward(self, x):

        prior_p = self.prior_noncephalic
        for t in range(x.size(1)):
            updated_p = (x[:,t,1] * prior_p) / ((prior_p * x[:,t,1]) + ((1 - prior_p) * x[:,t,0]))
            updated_p = torch.clamp(updated_p, min=0.01, max=0.99)
            prior_p = updated_p

        return updated_p

class BayesOddsUpdate(nn.Module):
    def __init__(self, prior_odds=1.0):
        super(BayesOddsUpdate, self).__init__()
        self.prior_noncephalic = prior_odds
        self.dropout = nn.Dropout(p=0.5)
        self.threshold = nn.Threshold(threshold=0.10, value=0.0)

    def forward(self, x, prior_odds=None):
        if prior_odds is None:
            prior_log_odds = torch.log10(torch.tensor([self.prior_noncephalic]))
        else:
            prior_log_odds = torch.log10(prior_odds)
        if torch.cuda.is_available():
            prior_log_odds = prior_log_odds.to('cuda')
        eps = 1e-8
        log_bayes_factor_t_transverse_ceph = torch.log10(x[:,:,1] + eps) - torch.log10(x[:,:,0] + eps)
        log_bayes_factor_t_breech_ceph = torch.log10(x[:,:,2] + eps) - torch.log10(x[:,:,0] + eps)
        log_bayes_factor_t_breech_transverse = log_bayes_factor_t_breech_ceph - log_bayes_factor_t_transverse_ceph
        log_bayes_factor_t_transverse_breech = log_bayes_factor_t_transverse_ceph - log_bayes_factor_t_breech_ceph
        log_bayes_factor_t_transverse_ceph = torch.clamp(log_bayes_factor_t_transverse_ceph, min=-1.0, max=1.0)
        log_bayes_factor_t_breech_ceph = torch.clamp(log_bayes_factor_t_breech_ceph, min=-1.0, max=1.0)
        log_bayes_factor_t_breech_transverse = torch.clamp(log_bayes_factor_t_breech_transverse, min=-1.0, max=1.0)
        log_bayes_factor_t_transverse_breech = torch.clamp(log_bayes_factor_t_transverse_breech, min=-1.0, max=1.0)
        log_bayes_factor_t_transverse_ceph_ignore_small = self.threshold(log_bayes_factor_t_transverse_ceph) - self.threshold(-log_bayes_factor_t_transverse_ceph)
        log_bayes_factor_t_breech_ceph_ignore_small = self.threshold(log_bayes_factor_t_breech_ceph) - self.threshold(-log_bayes_factor_t_breech_ceph)
        log_bayes_factor_t_breech_transverse_ignore_small = self.threshold(log_bayes_factor_t_breech_transverse) - self.threshold(-log_bayes_factor_t_breech_transverse)
        log_bayes_factor_t_transverse_breech_ignore_small = self.threshold(log_bayes_factor_t_transverse_breech) - self.threshold(-log_bayes_factor_t_transverse_breech)
        # log_bayes_factor_t_transverse_ceph_ignore_small = log_bayes_factor_t_transverse_ceph_ignore_small[:,4:]
        # log_bayes_factor_t_breech_ceph_ignore_small = log_bayes_factor_t_breech_ceph_ignore_small[:,4:]
        # log_bayes_factor_t_breech_transverse_ignore_small = log_bayes_factor_t_breech_transverse_ignore_small[:,4:]
        # log_bayes_factor_t_transverse_breech_ignore_small = log_bayes_factor_t_transverse_breech_ignore_small[:,4:]
        ###########################
        updated_odds_transverse_ceph_ignore_small = 10 ** (prior_log_odds + torch.clamp(torch.sum(log_bayes_factor_t_transverse_ceph_ignore_small, dim=1), min=-2.0,max=2.0))
        updated_odds_breech_ceph_ignore_small = 10 ** (prior_log_odds + torch.clamp(torch.sum(log_bayes_factor_t_breech_ceph_ignore_small, dim=1), min=-2.0,max=2.0))
        updated_odds_breech_transverse_ignore_small = 10 ** (prior_log_odds + torch.clamp(torch.sum(log_bayes_factor_t_breech_transverse_ignore_small, dim=1), min=-2.0,max=2.0))
        updated_odds_transverse_breech_ignore_small = 10 ** (prior_log_odds + torch.clamp(torch.sum(log_bayes_factor_t_transverse_breech_ignore_small, dim=1), min=-2.0,max=2.0))
        ####
        updated_odds_transverse_ceph_tensor = 10 ** (prior_log_odds + torch.clamp(torch.cumsum(log_bayes_factor_t_transverse_ceph_ignore_small, dim=1), min=-2.0, max=2.0))
        updated_odds_breech_ceph_tensor = 10 ** (prior_log_odds + torch.clamp(torch.cumsum(log_bayes_factor_t_breech_ceph_ignore_small, dim=1), min=-2.0, max=2.0))
        updated_odds_breech_transverse_tensor = 10 ** (prior_log_odds + torch.clamp(torch.cumsum(log_bayes_factor_t_breech_transverse_ignore_small, dim=1), min=-2.0,max=2.0))
        updated_odds_transverse_breech_tensor = 10 ** (prior_log_odds + torch.clamp(torch.cumsum(log_bayes_factor_t_transverse_breech_ignore_small, dim=1), min=-2.0,max=2.0))
        ####
        odds_tensor_transverse_ceph_argmax = torch.argmax(torch.abs(torch.cumsum(log_bayes_factor_t_transverse_ceph_ignore_small, dim=1)), dim=1)
        odds_tensor_breech_ceph_argmax = torch.argmax(torch.abs(torch.cumsum(log_bayes_factor_t_breech_ceph_ignore_small, dim=1)), dim=1)
        odds_tensor_breech_transverse_argmax = torch.argmax(torch.abs(torch.cumsum(log_bayes_factor_t_breech_transverse_ignore_small, dim=1)), dim=1)
        odds_tensor_transverse_breech_argmax = torch.argmax(torch.abs(torch.cumsum(log_bayes_factor_t_transverse_breech_ignore_small, dim=1)), dim=1)
        ###########################
        updated_odds_transverse_ceph_max = 10 ** (prior_log_odds + torch.clamp(torch.gather(torch.cumsum(log_bayes_factor_t_transverse_ceph_ignore_small, dim=1), 1, 
                                                                            odds_tensor_transverse_ceph_argmax.view(-1,1)).view(-1), 
                                                                min=-2.0, max=2.0))
        updated_odds_breech_ceph_max = 10 ** (prior_log_odds + torch.clamp(torch.gather(torch.cumsum(log_bayes_factor_t_breech_ceph_ignore_small, dim=1), 1, 
                                                                            odds_tensor_breech_ceph_argmax.view(-1,1)).view(-1), 
                                                                min=-2.0, max=2.0))
        updated_odds_breech_transverse_max = 10 ** (prior_log_odds + torch.clamp(torch.gather(torch.cumsum(log_bayes_factor_t_breech_transverse_ignore_small, dim=1), 1, 
                                                                            odds_tensor_breech_transverse_argmax.view(-1,1)).view(-1), 
                                                                min=-2.0, max=2.0))
        updated_odds_transverse_breech_max = 10 ** (prior_log_odds + torch.clamp(torch.gather(torch.cumsum(log_bayes_factor_t_transverse_breech_ignore_small, dim=1), 1, 
                                                                            odds_tensor_transverse_breech_argmax.view(-1,1)).view(-1), 
                                                                min=-2.0, max=2.0))
        ####
        updated_transverse_ceph_p = updated_odds_transverse_ceph_ignore_small / (1 + updated_odds_transverse_ceph_ignore_small)
        updated_breech_ceph_p = updated_odds_breech_ceph_ignore_small / (1 + updated_odds_breech_ceph_ignore_small)
        updated_breech_transverse_p = updated_odds_breech_transverse_ignore_small / (1 + updated_odds_breech_transverse_ignore_small)
        updated_transverse_breech_p = updated_odds_transverse_breech_ignore_small / (1 + updated_odds_transverse_breech_ignore_small)
        updated_p = torch.stack([updated_transverse_ceph_p,
                                updated_breech_ceph_p, updated_breech_transverse_p], dim=1)
        log_bayes_factor_t = torch.stack([log_bayes_factor_t_transverse_ceph,
                                        log_bayes_factor_t_breech_ceph, log_bayes_factor_t_breech_transverse], dim=1)
        log_bayes_factor_t_ignore_small = torch.stack([log_bayes_factor_t_transverse_ceph_ignore_small,
                                                    log_bayes_factor_t_breech_ceph_ignore_small, log_bayes_factor_t_breech_transverse_ignore_small], dim=1)
        return updated_p, log_bayes_factor_t, log_bayes_factor_t_ignore_small

class FrameEmbedding(nn.Module):
    def __init__(self, d_model=128):
        super(FrameEmbedding, self).__init__()
        cnn = models.mobilenet_v3_large(pretrained=True)
        cnn.classifier = Identity()
        self.TimeDistributed = TimeDistributed(cnn)
        self.W_emb = nn.Linear(960, d_model)

    def forward(self, x):
        x = self.TimeDistributed(x)
        x = self.W_emb(x)

        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model=128, dropout=0.1, max_len=500):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class TwoDimPositionalEmbedding(nn.Module):
    ## Cite Translating Math Formula Images to LaTeX Sequences Using Deep Neural Networks with Sequence-level Training 2019
    def __init__(self, d_model=128, dropout=0.1, max_len_width=5, max_len_time=100):
        super(TwoDimPositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len_width, max_len_time, d_model)
        position_x = torch.arange(0, max_len_width).unsqueeze(1)
        position_y = torch.arange(0, max_len_time).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 4) *
                             -(math.log(100.0) / d_model))
        pe[:,:,torch.arange(0, d_model/2, 2).long()] = torch.repeat_interleave(torch.sin(position_x * div_term).unsqueeze(1), repeats=max_len_time, dim=1)
        pe[:,:,torch.arange(0, d_model/2, 2).long()+1] = torch.repeat_interleave(torch.cos(position_x * div_term).unsqueeze(1), repeats=max_len_time, dim=1)
        pe[:,:,torch.arange(d_model/2, d_model, 2).long()] = torch.repeat_interleave(torch.sin(position_y * div_term).unsqueeze(0), repeats=max_len_width, dim=0)
        pe[:,:,torch.arange(d_model/2, d_model, 2).long()+1] = torch.repeat_interleave(torch.cos(position_y * div_term).unsqueeze(0), repeats=max_len_width, dim=0)
        #pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.cc_tag_pos_dict = {'M':5, 
                        'AssumedM':5,
                        'ML':5, 'MR':5,
                        'L0':4, 'L1':2, 'L15':5, 'L2':0, 'L45':5,
                        'R0':6, 'R1':8, 'R15':5, 'R2':9, 'R45':5,
                        'NL':3, 'NM':5, 'NR':7,
                      'ASSBS_M':5,
                      'ASSBS_L0':4, 'ASSBS_L1':2, 
                      'ASSBS_R0':6,'ASSBS_R1':8}
        self.lat_tag_pos_dict = {'C1': 1, 'NC1': 1, 'ASSBS_C1': 1,
                                }
        
    def forward(self, x, tags):

        #tags is list of lists
        n_batch = len(tags)
        tags = np.array(tags).flatten()
        pos_encoding_matrix_list = []
        for (i,tag) in enumerate(tags):
            if tag in self.cc_tag_pos_dict:
                pos_encoding_matrix_list.append(self.pe[self.cc_tag_pos_dict[tag],:,:])
            else:
                pos_encoding_matrix_list.append(self.pe[:,self.lat_tag_pos_dict[tag],:])
        pos_encoding_matrix = torch.stack(pos_encoding_matrix_list, dim=0)
        pos_encoding_matrix = pos_encoding_matrix.view(n_batch, -1, pos_encoding_matrix.size(1), pos_encoding_matrix.size(2))
        x = x + Variable(pos_encoding_matrix[:,:,:x.size(2)], 
                         requires_grad=False)
        return self.dropout(x)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def attention2D(query, key, value, tags, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    #print(query.size(), key.size())
    batch_size = query.size(0)
    d_k = query.size(-1)
    sweep_dim = query.size(1)
    time_dim = query.size(2)
    tags= np.array(tags)
    #print(tags)
    C1_tags_mask = torch.zeros(tags.shape)
    C1_tags_mask = C1_tags_mask.masked_fill(torch.tensor(tags == 'C1'), 1)
    C1_tags_mask = C1_tags_mask.masked_fill(torch.tensor(tags == 'ASSBS_C1'), 1)
    C1_tags_mask = C1_tags_mask.masked_fill(torch.tensor(tags == 'NC1'), 1)
    #print(C1_tags_mask)
    if torch.cuda.is_available():
        C1_tags_mask = C1_tags_mask.to('cuda')
    scores = torch.matmul(query.view(query.size(0),-1,query.size(-1)), 
                        key.view(key.size(0),-1,key.size(-1)).transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        mask = mask.repeat(sweep_dim,sweep_dim)
        mask = mask.unsqueeze(0).repeat(batch_size,1,1)
        mask = mask.view(-1, mask.size(2))
        #C1 rows for mask
        mask[C1_tags_mask.view(-1).repeat_interleave(time_dim) == 1,:] = mask[C1_tags_mask.view(-1).repeat_interleave(time_dim) == 1,:] * \
                                                                        C1_tags_mask.repeat_interleave(torch.sum(C1_tags_mask == 1, dim=1), dim=0).repeat_interleave(time_dim,dim=1).repeat_interleave(time_dim,dim=0)
        #cc rows for mask
        mask[C1_tags_mask.view(-1).repeat_interleave(time_dim) == 0,:] = mask[C1_tags_mask.view(-1).repeat_interleave(time_dim) == 0,:].masked_fill(
                                                                        C1_tags_mask.repeat_interleave(time_dim, dim = 1).repeat_interleave(time_dim * 
                                                                                                                            torch.sum(C1_tags_mask == 0, dim = 1), dim=0) == 1, 0)
        mask = mask.view(batch_size, -1, mask.size(1))
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    value = torch.transpose(value, 1, 2).contiguous() #is this correct?
    value = value.view(value.size(0),-1, value.size(-1))
    #print(p_attn.size(), value.size())
    return torch.matmul(p_attn, value).view(-1,sweep_dim,time_dim, d_k), p_attn


class SelfAttention(nn.Module):
    def __init__(self, in_units, out_units):
        super(SelfAttention, self).__init__()
        self.W = nn.Linear(in_units, out_units)
        self.Q = nn.Linear(in_units, out_units)
        self.V = nn.Linear(in_units, out_units)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, tags, mask=None):
        key = self.W(x)
        query = self.Q(x)
        value = self.V(x)
        context, attn = attention2D(query,key,value, tags,mask, self.dropout)
        del query
        del key
        del value
        return context, attn

    # def __init__(self, hidden_dim: int):
    #     super(AdditiveAttention, self).__init__()
    #     self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
    #     self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
    #     self.bias = nn.Parameter(torch.rand(hidden_dim).uniform_(-0.1, 0.1))
    #     self.score_proj = nn.Linear(hidden_dim, 1)

    # def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
    #     score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query) + self.bias)).squeeze(-1)
    #     attn = F.softmax(score, dim=-1)
    #     context = torch.bmm(attn.unsqueeze(1), value)
    #     return context, attn

class PositionwiseFeedForward(nn.Module):
    "Implements FFN layer"

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w_1(x).relu())

class Fetal_Pres_Net(nn.Module):
    def __init__(self):
        super(Fetal_Pres_Net, self).__init__()
        self.frame_emb = FrameEmbedding(d_model=128)
        self.add_positional_emb = TwoDimPositionalEmbedding(d_model=128, dropout=0.5, max_len_width=10, max_len_time=10)
        self.attention_layer = SelfAttention(in_units=128, out_units=128)
        self.avgpool = nn.AvgPool1d(10, stride=10)
        #self.ffn = PositionwiseFeedForward(d_model=64, d_ff = 16, dropout=0.5)
        self.Prediction = nn.Linear(128, 3, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.BayesOddsUpdate = BayesOddsUpdate(prior_odds=1.0)
 
    def forward(self, x, tags, prior_odds=None):
        if (len(x.shape) == 6):
            multiple_cine_tags = True
            batch_size = x.size(0)
            num_cine_tags = x.size(1)
            x = x.view(x.size(0)*x.size(1), x.size(2), x.size(3), x.size(4), x.size(5))
        else:
            multiple_cine_tags = False
        x = self.frame_emb(x)
        x = self.avgpool(x.permute(0,2,1)).permute(0,2,1)
        dim_time = x.size(1)
        if multiple_cine_tags:
            x = x.view(batch_size, num_cine_tags, x.size(1), x.size(2))
        x_with_pos = self.add_positional_emb(x, tags)
        if torch.cuda.is_available():
            mask = torch.ones(dim_time, dim_time).to('cuda')
            src = torch.zeros(dim_time, 2).to('cuda') # 5 positions on both sides are masked
            index_tensor = torch.stack([torch.arange(i-1,i+1) for i in range(dim_time)], dim = 0).to('cuda')
        mask = torch.triu(mask,diagonal=1).T
        index_tensor[index_tensor < 0] = 0
        index_tensor[index_tensor >= dim_time] = dim_time - 1
        mask.scatter_(1, index_tensor, src)
        context_vector, attn = self.attention_layer(x_with_pos, tags, mask)
        #print(x_with_pos.size(), context_vector.size())
        #x_with_pos = torch.mean(x_with_pos, dim=1)
        out = x_with_pos + context_vector #residual connection
        #out = self.avgpool(out.permute(0,2,1)).permute(0,2,1)
        #out = self.ffn(out) # out = out + self.ffn(out)
        prob_evidence_given_H = self.sigmoid(self.Prediction(out))
        #print("prob_evidence_given_H shape", prob_evidence_given_H.shape)
        if multiple_cine_tags:
            prob_evidence_given_H = prob_evidence_given_H.view(batch_size, -1, prob_evidence_given_H.size(-1))
        #print("prob_evidence_given_H shape (after)", prob_evidence_given_H.shape)
        pred, log_bayes_factor_t, log_bayes_factor_t_ignore_small = self.BayesOddsUpdate(prob_evidence_given_H, prior_odds)
        return pred, log_bayes_factor_t, log_bayes_factor_t_ignore_small, prob_evidence_given_H, attn

class Fetal_Pres_Net_bottom_layers(nn.Module):
    def __init__(self):
        super(Fetal_Pres_Net_bottom_layers, self).__init__()
        self.frame_emb = FrameEmbedding(d_model=128)
    def forward(self, x):
        if (len(x.shape) == 6):
            multiple_cine_tags = True
            batch_size = x.size(0)
            num_cine_tags = x.size(1)
            x = x.view(x.size(0)*x.size(1), x.size(2), x.size(3), x.size(4), x.size(5))
        else:
            multiple_cine_tags = False
        x = self.frame_emb(x)
        if multiple_cine_tags:
            x = x.view(batch_size, num_cine_tags, x.size(1), x.size(2))
        return x

class Fetal_Pres_Net_upper_layers(nn.Module):
    def __init__(self):
        super(Fetal_Pres_Net_upper_layers, self).__init__()
        self.add_positional_emb = TwoDimPositionalEmbedding(d_model=128, dropout=0.5, max_len_width=10, max_len_time=10)
        self.attention_layer = SelfAttention(in_units=128, out_units=128)
        self.avgpool = nn.AvgPool1d(10, stride=10)
        #self.ffn = PositionwiseFeedForward(d_model=64, d_ff = 16, dropout=0.5)
        self.Prediction = nn.Linear(128, 3, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.BayesOddsUpdate = BayesOddsUpdate(prior_odds=1.0)
 
    def forward(self, x, tags, prior_odds=None):
        if (len(x.shape) == 4):
            multiple_cine_tags = True
            batch_size = x.size(0)
            num_cine_tags = x.size(1)
            x = x.view(x.size(0)*x.size(1), x.size(2), x.size(3))
        else:
            multiple_cine_tags = False
        x = self.avgpool(x.permute(0,2,1)).permute(0,2,1)
        dim_time = x.size(1)
        if multiple_cine_tags:
            x = x.view(batch_size, num_cine_tags, x.size(1), x.size(2))
        x_with_pos = self.add_positional_emb(x, tags)
        if torch.cuda.is_available():
            mask = torch.ones(dim_time, dim_time).to('cuda')
            src = torch.zeros(dim_time, 2).to('cuda') # 5 positions on both sides are masked
            index_tensor = torch.stack([torch.arange(i-1,i+1) for i in range(dim_time)], dim = 0).to('cuda')
        mask = torch.triu(mask,diagonal=1).T
        index_tensor[index_tensor < 0] = 0
        index_tensor[index_tensor >= dim_time] = dim_time - 1
        mask.scatter_(1, index_tensor, src)
        context_vector, attn = self.attention_layer(x_with_pos, tags, mask)
        #print(x_with_pos.size(), context_vector.size())
        #x_with_pos = torch.mean(x_with_pos, dim=1)
        out = x_with_pos + context_vector #residual connection
        #out = self.avgpool(out.permute(0,2,1)).permute(0,2,1)
        #out = self.ffn(out) # out = out + self.ffn(out)
        prob_evidence_given_H = self.sigmoid(self.Prediction(out))
        #print("prob_evidence_given_H shape", prob_evidence_given_H.shape)
        if multiple_cine_tags:
            prob_evidence_given_H = prob_evidence_given_H.view(batch_size, -1, prob_evidence_given_H.size(-1))
        #print("prob_evidence_given_H shape (after)", prob_evidence_given_H.shape)
        pred, log_bayes_factor_t, log_bayes_factor_t_ignore_small = self.BayesOddsUpdate(prob_evidence_given_H, prior_odds)
        return pred, log_bayes_factor_t, log_bayes_factor_t_ignore_small, prob_evidence_given_H, attn


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon:float=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = (-log_preds.sum(dim=-1)).mean()
        nll = F.nll_loss(log_preds, target, reduction='mean')
        return ((1-self.epsilon) * nll) + (self.epsilon * (loss/n))

def train(gpu, args):
    ############################################################
    rank = args.nr * args.gpus + gpu                              
    dist.init_process_group(                                   
        backend='nccl',                                         
        init_method='env://',                                   
        world_size=args.world_size,                              
        rank=rank                                               
    )
    torch.manual_seed(0)                                                          
    ############################################################
    ####### Model #############
    model = Fetal_Pres_Net()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    ###############################################################
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu])
    ###############################################################
    batch_size = 4
    NUM_ACCUMULATION_STEPS = 2
    # define loss function (criterion) and optimizer
    #loss_fn_label_smoothing = nn.CrossEntropyLoss(reduction='mean',label_smoothing=0.1)
    #loss_fn = nn.CrossEntropyLoss(reduction='mean')
    loss_fn = nn.BCELoss(reduction='mean')
    if torch.cuda.is_available():
        #loss_fn_label_smoothing = loss_fn_label_smoothing.to('cuda')
        loss_fn = loss_fn.to('cuda')
    optimizer = optim.Adam(model.parameters(), 
                            lr=1e-4)
    # Data loading code
    train_df = pd.read_csv('/mnt/famli_netapp_shared/C1_ML_Analysis/FullCineLists/train_fetal_pres_24weeks_onward_with_C1tag.csv')
    train_df.file_path = train_df.file_path.str.replace('\\','/', regex=False)
    train_df.file_path = train_df.file_path.str.replace('mp4','nrrd', regex=False)
    train_df.file_path = train_df.file_path.str.replace('.*Dataset_C/','Dataset_C_masked_resampled_256_spc075/')
    train_df.file_path = train_df.file_path.str.replace('dcm','nrrd', regex=False)
    train_df.file_path = train_df.file_path.str.replace('.*Dataset_C1/','dataset_C1_cines_masked_resampled_256_spc075/')
    print(len(set(train_df.study_id_orig)))
    train_df = train_df.loc[:, 
                            ["file_path", "fetal_presentation_str",'tag','study_id']].reset_index(drop=True)
    train_df[["label"]] = None
    train_df.loc[train_df.fetal_presentation_str == 'Cephalic' ,"label"] = 0
    train_df.loc[train_df.fetal_presentation_str != 'Cephalic' ,"label"] = 1
    print(np.unique(train_df.label, return_counts = True))
    print(np.unique(train_df.fetal_presentation_str, return_counts = True))
    label_map = train_df[["study_id", "label", 'fetal_presentation_str']].drop_duplicates().reset_index(drop=True)
    print(np.unique(label_map.fetal_presentation_str, return_counts = True))
    training_data = ITKImageDatasetByIDTwoDim(train_df, mount_point=dataset_dir,
                                        transform=TransformFrames(training=True), num_sample_frames = 100, num_sample_sweeps=2)
    train_sampler = DistributedWeightedSampler(training_data, num_replicas=args.world_size, 
                                        rank=rank, replacement=True, shuffle=True)
    train_dataloader = DataLoader(
        training_data, batch_size=batch_size, sampler=train_sampler, num_workers=8, pin_memory=True)
    val_df = pd.read_csv('/mnt/famli_netapp_shared/C1_ML_Analysis/FullCineLists/valid_fetal_pres_24weeks_onward_with_C1tag.csv')
    val_df.file_path = val_df.file_path.str.replace('\\','/', regex=False)
    val_df.file_path = val_df.file_path.str.replace('mp4','nrrd', regex=False)
    val_df.file_path = val_df.file_path.str.replace('.*Dataset_C/','Dataset_C_masked_resampled_256_spc075/')
    val_df.file_path = val_df.file_path.str.replace('dcm','nrrd', regex=False)
    val_df.file_path = val_df.file_path.str.replace('.*Dataset_C1/','dataset_C1_cines_masked_resampled_256_spc075/')
    print(len(set(val_df.study_id_orig)))
    val_df = val_df.loc[:, 
                        ["file_path", "fetal_presentation_str",'tag','study_id']].reset_index(drop=True)
    val_df[["label"]] = None
    val_df.loc[val_df.fetal_presentation_str == 'Cephalic' ,"label"] = 0
    val_df.loc[val_df.fetal_presentation_str != 'Cephalic' ,"label"] = 1

    val_data = ITKImageDatasetByIDTwoDim(val_df, mount_point=dataset_dir,
                                        transform=TransformFrames(training=False), num_sample_frames = 100, num_sample_sweeps=None)

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_data, num_replicas=args.world_size, rank=rank)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=1,
                                              sampler=val_sampler, num_workers=6, pin_memory=True,
                                              persistent_workers=True)
    num_epochs = 200
    early_stop = EarlyStopping(patience=3, verbose=True, 
                                    path=os.path.join(output_mount, 'model/model_fetal_pres_230120_bayes_label_smooth_mask_cc'))
    # Keep track of losses
    f_train_epoch_loss_history = open(os.path.join(output_mount,"log/train_model_fetal_pres_230120_bayes_label_smooth_mask_cc" + ".txt"),"w", buffering=1)
    f_validation_epoch_history = open(os.path.join(output_mount,"log/valid_model_fetal_pres_230120_bayes_label_smooth_mask_cc" + ".txt"),"w", buffering=1)

    n_batch = 1000 * NUM_ACCUMULATION_STEPS
    label_smoothing = 0.0
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float)[None, None, :, None, None].cuda()
    sd = torch.as_tensor([0.229, 0.224, 0.225], dtype=torch.float)[None, None, :, None, None].cuda()
    Relu = nn.ReLU()
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        running_loss1 = 0.0
        running_loss2 = 0.0
        running_loss3 = 0.0
        num_batches = 0
        while num_batches != n_batch:
            for batch, (X, tags,  y) in enumerate(train_dataloader):
                tags = [x.split(',') for x in tags]
                num_batches += 1
                X = X.cuda()
                y = y.cuda()
                X.div_(255)
                X = X.repeat_interleave(3,dim=3)
                X.sub_(mean).div_(sd)
                rotater = T.RandomRotation(degrees=(-15, 15))
                batch_size = X.size(0)
                num_sweeps = X.size(1)
                seq_len = X.size(2)
                X = rotater(X.view(batch_size * num_sweeps * seq_len, X.size(3), X.size(4), X.size(5)))
                X = X.view(batch_size, num_sweeps, seq_len, X.size(1), X.size(2), X.size(3))
                # random_prior_prob = (torch.rand(X.size(0)) * 0.6) + 0.2 #between 0.2 to 0.8
                # random_prior_odds = random_prior_prob / (1 - random_prior_prob)
                # out, log_bayes_factor_t, prob_evidence_given_H, attn = model(X, random_prior_odds)
                out, log_bayes_factor_t, log_bayes_factor_t_ignore_small, prob_evidence_given_H, attn = model(X, tags)
                #one_hot_y = F.one_hot(y.view(-1).long(), num_classes=2)
                #loss = loss_fn(prob, ((1 - label_smoothing) * one_hot_y.float()) + (label_smoothing * (1 - one_hot_y.float())))
                seq_len = log_bayes_factor_t_ignore_small.size(2)
                updated_odds = 10 ** (0.0 + torch.clamp(torch.sum(log_bayes_factor_t, dim=2), min=-5.0, max=5.0))
                out_p = updated_odds / (1 + updated_odds)
                opposite_evidence = Relu((-log_bayes_factor_t_ignore_small[~torch.isnan(y),:].view(-1,seq_len) * y[~torch.isnan(y)].view(-1,1).repeat(1,seq_len))) + \
                                         Relu((log_bayes_factor_t_ignore_small[~torch.isnan(y),:].view(-1,seq_len) * (1 - y[~torch.isnan(y)].view(-1,1).repeat(1,seq_len))))
                opposite_evidence_sum = torch.clamp(torch.sum(opposite_evidence, dim = 1), min=0.0, max=5.0)
                opposite_evidence_odds = 10 ** (opposite_evidence_sum)
                opposite_evidence_p = opposite_evidence_odds / (1 + opposite_evidence_odds)
                #print("log_bayes_factor_t_ignore_small: ", log_bayes_factor_t_ignore_small)
                #print("opposite_evidence: ", opposite_evidence_sum, opposite_evidence_p)
                opposite_evidence_p_entropy_loss = torch.mean(torch.distributions.Bernoulli(opposite_evidence_p).entropy())
                #opposite_evidence_p_entropy_loss = torch.tensor([0.0])
                print("y: ",y)
                print("out: ", out, out_p)
                out = out[~torch.isnan(y)]
                out_p = out_p[~torch.isnan(y)]
                y = y[~torch.isnan(y)]

                loss1 = loss_fn(out, ((1 - label_smoothing) * y.float()) + (label_smoothing * (1 - y.float())))
                loss2 = loss_fn(out_p, ((1 - label_smoothing) * y.float()) + (label_smoothing * (1 - y.float())))
                #print(loss1.item(), loss2.item(), opposite_evidence_p_entropy_loss.item())
                loss = loss1 + (0.1 * loss2) - (10.0 * opposite_evidence_p_entropy_loss)
                # Normalize the Gradients
                # loss = loss / NUM_ACCUMULATION_STEPS
                loss.backward()
                print("loss: ", loss1.item(), loss2.item(), -20.0 * opposite_evidence_p_entropy_loss.item())

                if ((batch + 1) % NUM_ACCUMULATION_STEPS == 0) or (num_batches == n_batch):
                    # Update Optimizer
                    optimizer.step()
                    optimizer.zero_grad()

                if y[~torch.isnan(y)].size(0) != 0:
                    running_loss += loss.item()
                    running_loss1 += loss1.item()
                    running_loss2 += loss2.item()
                    running_loss3 += opposite_evidence_p_entropy_loss.item()
                else:
                    running_loss += 0.0
                    running_loss1 += 0.0
                    running_loss2 += 0.0
                    running_loss3 += 0.0

                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{len(training_data):>5d}]")
                if num_batches == n_batch:
                    break

        train_loss = running_loss / num_batches
        train_loss1 = running_loss1 / num_batches
        train_loss2 = running_loss2 / num_batches
        train_loss3 = running_loss3 / num_batches
        print(f"average epoch loss: {train_loss:>7f}  [{epoch:>5d}/{num_epochs:>5d}]")
        train_loss = torch.tensor([train_loss]).cuda()
        train_loss1 = torch.tensor([train_loss1]).cuda()
        train_loss2 = torch.tensor([train_loss2]).cuda()
        train_loss3 = torch.tensor([train_loss3]).cuda()
        dist.all_reduce(train_loss, op=ReduceOp.SUM)
        dist.all_reduce(train_loss1, op=ReduceOp.SUM)
        dist.all_reduce(train_loss2, op=ReduceOp.SUM)
        dist.all_reduce(train_loss3, op=ReduceOp.SUM)
        train_loss = train_loss.cpu().item() / args.world_size
        train_loss1 = train_loss1.cpu().item() / args.world_size
        train_loss2 = train_loss2.cpu().item() / args.world_size
        train_loss3 = train_loss3.cpu().item() / args.world_size
        if rank == 0:
            f_train_epoch_loss_history.write("Epoch: " + str(epoch) + '\n')
            f_train_epoch_loss_history.write("Num batches: " + str(num_batches) + '\n')
            f_train_epoch_loss_history.write("Overall loss: " + str(train_loss) + '\n')
            f_train_epoch_loss_history.write("loss1: " + str(train_loss1) + '\n')
            f_train_epoch_loss_history.write("loss2: " + str(train_loss2) + '\n')
            f_train_epoch_loss_history.write("loss3: " + str(train_loss3) + '\n')
            f_train_epoch_loss_history.write("Training Loop Time (minutes): " + str(((time.time() - epoch_start)/60.0)) +'\n')
        #################################
        dist.barrier()
        model.eval()
        model_bottom_layers = Fetal_Pres_Net_bottom_layers()
        model_bottom_layers.frame_emb = model.module.frame_emb
        model_bottom_layers.cuda(gpu)
        model_bottom_layers.eval()

        model_upper_layers = Fetal_Pres_Net_upper_layers()
        model_upper_layers.add_positional_emb = model.module.add_positional_emb
        model_upper_layers.attention_layer = model.module.attention_layer
        #model_upper_layers.ffn = model.module.ffn
        model_upper_layers.Prediction = model.module.Prediction
        model_upper_layers.sigmoid = model.module.sigmoid
        model_upper_layers.BayesOddsUpdate = model.module.BayesOddsUpdate
        model_upper_layers.cuda(gpu)
        model_upper_layers.eval()
        with torch.no_grad():
            running_loss = 0.0
            correct = 0.0
            total = 0.0
            y_out_tensor_list = [torch.zeros((val_sampler.num_samples * 3, 2), dtype=torch.float32).cuda() for _ in range(args.world_size)]
            for batch, (X, tags, y) in enumerate(val_dataloader):
                tags = [x.split(',') for x in tags]
                y = y.cuda()
                batch_size = y.size(0)
                X_sweep_emb_list = []
                for X_sweep in torch.unbind(X, dim=1):
                    X_sweep = X_sweep.unsqueeze(1)
                    if torch.cuda.is_available():
                        X_sweep = X_sweep.cuda()
                    X_sweep.div_(255)
                    X_sweep = X_sweep.repeat_interleave(3,dim=3)
                    X_sweep.sub_(mean).div_(sd)
                    X_sweep_emb = model_bottom_layers(X_sweep)
                    X_sweep_emb_list.append(X_sweep_emb)
                X_emb = torch.cat(X_sweep_emb_list, dim=1)
                out, log_bayes_factor_t, log_bayes_factor_t_ignore_small, prob_evidence_given_H, attn = model_upper_layers(X_emb, tags)
                if batch == 0:
                    y_out_tensor = torch.cat((y.float().view(-1,1), out.view(-1,1)), dim=1)
                else:
                    y_out_tensor = torch.cat((y_out_tensor, torch.cat((y.float().view(-1,1), out.view(-1,1)), dim=1)), dim=0)
                out = out[~torch.isnan(y)]
                y = y[~torch.isnan(y)]
                loss = loss_fn(out, y.float())
                if len(y) != 0: #prevent y which is all nan
                    predicted = torch.round(out)
                    correct += (predicted.view(-1) == y.view(-1)).sum().item()
                    total += y.view(-1).size(0)
                    running_loss += loss.item()

        val_loss = torch.tensor([running_loss / len(val_sampler)]).cuda()
        val_accuracy = torch.tensor([correct / total]).cuda()
        dist.all_gather(y_out_tensor_list, y_out_tensor.cuda())
        dist.all_reduce(val_loss, op=ReduceOp.SUM)
        dist.all_reduce(val_accuracy, op=ReduceOp.SUM)
        val_loss = val_loss.cpu().item() / args.world_size
        val_accuracy = val_accuracy.cpu().item() / args.world_size
        if rank == 0:
            y_out_tensor_combined = torch.cat(y_out_tensor_list, dim=0)
            roc_auc = roc_auc_score(y_out_tensor_combined[~torch.isnan(y_out_tensor_combined[:,0]),0].cpu().numpy(), 
                                    y_out_tensor_combined[~torch.isnan(y_out_tensor_combined[:,0]),1].cpu().numpy())
            f_validation_epoch_history.write("epoch " + str(epoch) + '\n')
            f_validation_epoch_history.write(str(val_loss) + '\n')
            f_validation_epoch_history.write(str(val_accuracy) + '\n')
            f_validation_epoch_history.write(str(roc_auc) + '\n')
            # early_stop(-val_accuracy, model.module
            #    )
            # early_stop(val_loss, model.module
            #     )
            early_stop(-roc_auc, model.module
                )
            f_train_epoch_loss_history.write("Time (minutes): " + str(((time.time() - epoch_start)/60.0)) +'\n')
            if early_stop.early_stop:
                early_stop_indicator = torch.tensor([1.0]).cuda()
            else:
                early_stop_indicator = torch.tensor([0.0]).cuda()
        else:
            early_stop_indicator = torch.tensor([0.0]).cuda()
        dist.all_reduce(early_stop_indicator, op=ReduceOp.SUM)
        if early_stop_indicator.cpu() == torch.tensor([1.0]):
            print("Early stopping")            
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    #########################################################
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8889'         
    mp.spawn(train, nprocs=args.gpus, args=(args,))
    #########################################################

if __name__ == '__main__':
    main()

#to run
#nohup python train_fetal_pres_230120_bayes_label_smooth_mask_cc.py -n 1 -g 4 > train_fetal_pres_230120_bayes_label_smooth_mask_cc.out &


