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


class USDataset(Dataset):
    def __init__(self, df, label_column = None, mount_point = "./", transform=None, img_column="img_path", class_column=None, ga_column=None, scalar_column=None, repeat_channel=True, return_head=False):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.class_column = class_column
        self.ga_column = ga_column
        self.scalar_column = scalar_column
        self.repeat_channel = repeat_channel
        self.return_head = return_head
        self.label_column = label_column

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])
        
        try:
            if os.path.splitext(img_path)[1] == ".nrrd":
                img, head = nrrd.read(img_path, index_order="C")
                # img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
                img = torch.tensor(img, dtype=torch.float32)
                img = img.squeeze()
                if self.repeat_channel == False:
                    img = img.unsqueeze(0)
                # print("IMAGE SIZE1: ", img.shape)
                if self.repeat_channel:
                    img = img.unsqueeze(0).repeat(3,1,1)
            else:
                img = np.array(Image.open(img_path))
                img = torch.tensor(img, dtype=torch.float32)
                if len(img.shape) == 3:                    
                    img = torch.permute(img, [2, 0, 1])[0:3, :, :]
                else:                    
                    img = img.unsqueeze(0).repeat(3,1,1)           
                # print("IMAGE SIZE2: ", img.shape) 
        except:
            print("Error reading frame: " + img_path, file=sys.stderr)
            img = torch.tensor(np.zeros([3, 256, 256]), dtype=torch.float32)

        if(self.transform):
            img = self.transform(img)
        
        if self.label_column:
            labels = self.df.iloc[idx][self.label_column].values.astype(int)
            return img, labels
        if self.class_column:
            return img, torch.tensor(self.df.iloc[idx][self.class_column]).to(torch.long)

        if self.ga_column:
            ga = self.df.iloc[idx][self.ga_column]
            return img, torch.tensor([ga])

        if self.scalar_column:
            scalar = self.df.iloc[idx][self.scalar_column]
            return img, torch.tensor(scalar)            
        if self.return_head:
            return img, head
        return img

class USDataModule(pl.LightningDataModule):
    def __init__(self, df_train, df_val, df_test, label_columns, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", class_column=None, ga_column=None, scalar_column=None, train_transform=None, valid_transform=None, test_transform=None, drop_last=False, repeat_channel=True):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.class_column = class_column
        self.scalar_column = scalar_column
        self.ga_column = ga_column
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last
        self.repeat_channel = repeat_channel
        self.label_columns = label_columns

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = USDataset(self.df_train, label_column=self.label_columns, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column, ga_column=self.ga_column, scalar_column=self.scalar_column, transform=self.train_transform, repeat_channel=self.repeat_channel)
        print(len(self.train_ds))
        self.val_ds = USDataset(self.df_val, label_column=self.label_columns, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column, ga_column=self.ga_column, scalar_column=self.scalar_column, transform=self.valid_transform, repeat_channel=self.repeat_channel)
        print(len(self.val_ds))
        self.test_ds = USDataset(self.df_test, label_column=self.label_columns, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column, ga_column=self.ga_column, scalar_column=self.scalar_column, transform=self.test_transform, repeat_channel=self.repeat_channel)
        print(len(self.test_ds))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=False, drop_last=self.drop_last, shuffle=True, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=False, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=False, drop_last=self.drop_last)
    