
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import SimpleITK as sitk
from PIL import Image
import nrrd
import os
import sys
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


import pytorch_lightning as pl

class TensorDatasetModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, batch_size=256, num_workers=2, train_transform=None, valid_transform=None, test_transform=None):
        super().__init__()

        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform

        self.num_workers = num_workers

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = torch.utils.data.TensorDataset(*self.train_data)
        self.val_ds = torch.utils.data.TensorDataset(*self.val_data)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True)